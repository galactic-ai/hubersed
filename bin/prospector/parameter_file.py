import pickle
import sys
import time
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.utils.obsutils import fix_obs
from prospect.sources import FastStepBasis
from prospect import prospect_args
import numpy as np
import torch

from hubersed.paths import PATHS
from hubersed.prospector.utils import load_lines

from huggingface_hub import hffs

if not hasattr(np, "infty"):
    np.infty = np.inf  # compatibility shim for older code

DATA_PATH = PATHS['DATA']
RESULTS_PATH = PATHS['RESULTS']

WAVE_OBS = np.linspace(3600.0, 9824.0, 7781, dtype=np.float32)
OUTLIERS_IDX = torch.load(RESULTS_PATH / "desi_outliers.pt", weights_only=False)["outlier_indices"]
    
C_CGS = 2.99792458e10          # cm/s
FNU_PER_MAGGIE = 3631e-23       # erg/s/cm^2/Hz

EM_LINES_A = load_lines()['emission']['wave_vac']

# build obs
def build_obs(spec: np.ndarray, unc: np.ndarray, mask: np.ndarray) -> dict:
    """
    Build the observation dictionary for the fit. 
    This should include at least the spectrum and uncertainty, for DESI data.

    Parameters
    ----------
    spec : np.ndarray
        The observed spectrum. In maggies, make sure to convert it before hand.
    unc : np.ndarray
        The uncertainty on the observed spectrum. In maggies, make sure to convert it before hand.
        Should be sigma, not variance.
    mask: np.ndarray
        A boolean array indicating which pixels to use in the fit. False elements will be ignored
        in the likelihood calculation. This can be used to mask out bad pixels, sky lines, etc.
    """

    obs_dict  = {
        "wavelength": WAVE_OBS,
        "spectrum": spec,
        "unc": unc,
        "mask": mask,
        "filters": None,
        "maggies": None,
        "maggies_unc": None,
        "phot_mask" : None,
    }
    obs_dict = fix_obs(obs_dict)
    return obs_dict

# build sps
def build_sps():
    return FastStepBasis()


# DESI Spectra
def get_outlier_info(idx, streaming=True):
    """
    Get information about a specific outlier.

    Parameters    
    ----------
    idx : int
        Index of the outlier to retrieve (0 to len(OUTLIERS_IDX)-1)
    
    Returns
    -------
    spec : np.ndarray
        The observed spectrum for the outlier, in the original units (flambda).
    unc : np.ndarray
        The uncertainty on the observed spectrum, in the original units (flambda).
    redshift : float
        The redshift of the outlier.
    mask : np.ndarray
        A boolean array indicating which pixels are valid (True) or should be masked (False).
    id : int
        The original ID of the spectrum in the DESI dataset, for reference.
    """

    chunk_size = 1024
    chunk_indices = OUTLIERS_IDX // chunk_size
    chunk_files = [DATA_PATH / 'desi_spectra' / f"DESIchunk1024_{i}.pkl" for i in chunk_indices]
    chunk_indices_in_chunk = OUTLIERS_IDX % chunk_size

    chunk_file = chunk_files[idx]
    idx_in_chunk = chunk_indices_in_chunk[idx]    

    if not streaming:
        with open(chunk_file, "rb") as f:
            s, w, z, id, norm, *_ = pickle.load(f)
    else:
        with hffs.open(f"buckets/nikhil0504/hubersed-data/desi_spectra/{chunk_file.name}", "rb") as f:
            s, w, z, id, norm, *_ = pickle.load(f)
    
    # correct for normalization
    s = s * norm[:, None]
    w = w / norm[:, None]**2
    mask = np.isfinite(s) & np.isfinite(w) & (w > 0)

    # convert to numpy arrays if not already from torch tensors
    if isinstance(s, torch.Tensor):
        s = s.cpu().numpy()
    if isinstance(w, torch.Tensor):
        w = w.cpu().numpy()
    if isinstance(z, torch.Tensor):
        z = z.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(id, torch.Tensor):
        id = id.cpu().numpy()



    return s[idx_in_chunk], w[idx_in_chunk], z[idx_in_chunk], mask[idx_in_chunk], id[idx_in_chunk]


def flambda_to_maggies(wave_A, flambda):
    # flambda: erg/s/cm^2/Å ; wave_A: Å
    flambda_cgs = flambda * 1e-17 # DESI spectra are in 1e-17 erg/s/cm^2/Å, convert to cgs
    return flambda_cgs * (wave_A**2) * 1e-8 / C_CGS / FNU_PER_MAGGIE

def ivar_flambda_to_ivar_maggies(wave_A, ivar_flambda):
    # maggies = flambda * K  => ivar_maggies = ivar_flambda / K^2
    # because DESI spectra are in 1e-17 erg/s/cm^2/Å, 
    # we need to include that factor in the conversion from flambda to maggies
    K = (wave_A**2) * 1e-8 * 1e-17 / C_CGS / FNU_PER_MAGGIE 
    return ivar_flambda / (K**2)

def mask_spectral_lines(wave_obs, mask, z, line_waves=EM_LINES_A, halfwidth_kms=500.0):
    """Mask spectral lines using a velocity-based window.
    
    Parameters
    ----------
    wave_obs : np.ndarray
        Observed wavelength array [Å].
    mask : np.ndarray
        Boolean array; True = good pixel, False = masked.
    z : float
        Redshift of the object.
    line_waves : np.ndarray
        Rest-frame vacuum wavelengths of lines to mask [Å].
    halfwidth_kms : float, optional
        Half-width of mask window in km/s. Default is 500 km/s.
    
    Returns
    -------
    np.ndarray
        Updated boolean mask.
    """
    c_kms = 299792.458
    mask = mask.copy()
    wave_rest = wave_obs / (1.0 + z)
    
    for line in line_waves:
        dwave = line * halfwidth_kms / c_kms
        mask &= np.abs(wave_rest - line) > dwave
    
    return mask

if __name__ == "__main__":
    # Get the default argument parser
    parser = prospect_args.get_parser()

    # add id argument to specify which outlier to fit
    parser.add_argument("--id", type=int, default=0, help="ID of the outlier to fit (0 to {0})".format(len(OUTLIERS_IDX)-1))
    parser.add_argument("--mask", action="store_true", help="Whether to mask emission lines in the fit")

    # Parse the supplied arguments, convert to a dictionary, and add this file for logging purposes
    args = parser.parse_args()
    run_params = vars(args)
    run_params["param_file"] = __file__

    # get outlier info
    spec, w, redshift, mask, id = get_outlier_info(args.id)

    # convert to maggies
    spec = flambda_to_maggies(wave_A=WAVE_OBS, flambda=spec)
    w = ivar_flambda_to_ivar_maggies(wave_A=WAVE_OBS, ivar_flambda=w)
    unc = 1 / np.sqrt(w)

    # mask sky lines if arguments specify
    if args.mask:
        mask = mask_spectral_lines(WAVE_OBS, mask, redshift, EM_LINES_A, halfwidth_kms=500.0)

    # build the fit ingredients
    obs, model, sps, noise = build_all(spec, unc, mask=mask, redshift=redshift)
    run_params["sps_libraries"] = sps.ssp.libraries

    # Set up MPI communication
    try:
        import mpi4py
        from mpi4py import MPI
        from schwimmbad import MPIPool

        mpi4py.rc.threads = False
        mpi4py.rc.recv_mprobe = False

        comm = MPI.COMM_WORLD
        size = comm.Get_size()

        withmpi = comm.Get_size() > 1
    except ImportError:
        print('Failed to start MPI; are mpi4py and schwimmbad installed? Proceeding without MPI.')
        withmpi = False

    # Set up an output file name and run the fit
    # output = fit_model(obs, model, sps, noise, **run_params)
    
    # Evaluate SPS over logzsol grid in order to get necessary data in cache/memory
    # for each MPI process. Otherwise, you risk creating a lag between the MPI tasks
    # caching SSPs which can slow down the parallelization
    if (withmpi) & ('logzsol' in model.free_params):
        dummy_obs = dict(filters=None, wavelength=None)

        logzsol_prior = model.config_dict["logzsol"]['prior']
        lo, hi = logzsol_prior.range
        logzsol_grid = np.around(np.arange(lo, hi, step=0.1), decimals=2)

        sps.update(**model.params)  # make sure we are caching the correct IMF / SFH / etc
        for logzsol in logzsol_grid:
            model.params["logzsol"] = np.array([logzsol])
            _ = model.predict(model.theta, obs=dummy_obs, sps=sps)

    # ensure that each processor runs its own version of FSPS
    # this ensures no cross-over memory usage
    from prospect.fitting import lnprobfn
    from functools import partial
    lnprobfn_fixed = partial(lnprobfn, sps=sps)

    if withmpi:
        run_params["using_mpi"] = True
        with MPIPool() as pool:

            # The dependent processes will run up to this point in the code
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            nprocs = pool.size
            # The parent process will oversee the fitting
            output = fit_model(obs, model, sps, noise, pool=pool, queue_size=nprocs, lnprobfn=lnprobfn_fixed, **run_params)
    else:
        # without MPI we don't pass the pool
        output = fit_model(obs, model, sps, noise, lnprobfn=lnprobfn_fixed, **run_params)

    ts = time.strftime("%y%b%d-%H.%M", time.localtime())
    hfile = "{0}_{1}_mcmc.h5".format(args.outfile, ts)
    
    # Write results to output file
    writer.write_hdf5(hfile, run_params, model, obs,
                        output["sampling"][0], output["optimization"][0],
                        tsample=output["sampling"][1],
                        toptimize=output["optimization"][1],
                        sps=sps)