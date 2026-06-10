import pickle
from prospect.observation import Spectrum
from prospect.sources import FastStepBasis, SSPBasis
import numpy as np
import torch

from hubersed.paths import PATHS
from hubersed.prospector.utils import load_lines
from hubersed.prospector.lsf import DESI_WAV, desi_resolution, C_KMS

from huggingface_hub import hffs

if not hasattr(np, "infty"):
    np.infty = np.inf  # compatibility shim for older code

DATA_PATH = PATHS['DATA']
RESULTS_PATH = PATHS['RESULTS']

WAVE_OBS = DESI_WAV.astype(np.float32)
OUTLIERS_IDX = torch.load(RESULTS_PATH / "desi_outliers.pt", weights_only=False)["outlier_indices"]

EM_LINES_A = load_lines()['emission']['wave_vac']

# build obs
def build_obs(spec: np.ndarray, unc: np.ndarray, mask: np.ndarray,
              resolution: np.ndarray = None) -> list:
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
    resolution: np.ndarray, optional
        Instrumental resolution (sigma) at each wavelength, in km/s (prospect
        Spectrum convention). When given, prospect smooths the model to the DESI
        LSF. Pass C_KMS/(2.355*R(lambda)). Safe because build_sps zeroes the
        library resolution (no `data higher resolution than library` assert).
    """
    spec_obs = Spectrum(
        wavelength=WAVE_OBS,
        flux=spec,
        uncertainty=unc,
        mask=mask,
        resolution=resolution,
    )
    spec_obs.rectify()
    return [spec_obs]

# build sps
def build_sps():
    sps = FastStepBasis()
    SSPBasis.spectral_resolution = property(lambda self: np.zeros_like(self.ssp.wavelengths))
    return sps


# use Cue instead
def build_cue_sps():
    """
    SPS with Cue (Li+24) nebular emulator instead of FSPS+Cloudy lines.
    """
    from prospect.sources import NebStepBasis
    return NebStepBasis()

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

    # recomputes all the chunk files
    # TODO: optimise it to only compute for the file we care about
    
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
