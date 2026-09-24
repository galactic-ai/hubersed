"""Make mock DESI spectra from a prior sample and write them to one h5 file.

The h5 holds the fluxes in maggies on the DESI grid, the drawn logsfr_ratios, the nebular
line luminosities and a copy of every prior array. The command line entry point is
``scripts/make_model_seds.py``.
"""

import copy
import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from pathlib import Path

import h5py
import numpy as np
import scipy.stats
from prospect.models import priors, transforms
from prospect.models.sedmodel import HyperSpecModel
from prospect.models.templates import TemplateLibrary, adjust_stochastic_params
from prospect.observation import Spectrum
from tqdm.auto import tqdm

from hubersed.sps.lsf import DESI_WAV, build_desi_resolution_matrix
from hubersed.sps.utils import make_stochastic_agebins

N_RATIOS = 9  # 9 logsfr_ratios for 10 age bins

# per-worker state, populated by _init_worker(). Never touched at module level.
_S = {}


def _setup_process():
    """Hide RuntimeWarnings from zero inverse variance and run numerics on one thread.

    Called by ``write_mock_seds`` before the worker pool starts, so the spawned workers inherit the
    thread settings, and by each worker. Importing this module changes nothing process wide.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # One thread per process for numpy and for JAX on the Cue path, to avoid
    # oversubscription across workers.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")


def build_base_template(nebular):
    """Return the prospector template shared by every mock, before the per-mock values.

    Parameters
    ----------
    nebular : {"cue", "fsps"}
        Nebular emission model.

    Returns
    -------
    dict
        Stochastic SFH, dust emission and nebular template with fixed smoothing.
    """
    t = copy.deepcopy(TemplateLibrary["stochastic_sfh"])
    t.update(copy.deepcopy(TemplateLibrary["dust_emission"]))

    if nebular == "cue":
        # Cue (Li+24): ionizing spectrum tied to FSPS stars; free gas_logz/u/nH/no/co
        t.update(copy.deepcopy(TemplateLibrary["cue_stellar_nebular"]))
    else:
        # FSPS / Byler+2017 CLOUDY grids; free gas_logz/u only
        t.update(copy.deepcopy(TemplateLibrary["nebular"]))
        # prospect's nebular template ties gas_logz to logzsol. Untie it so each mock uses
        # its drawn gas_logz, as the Cue mocks and both fit models do.
        del t["gas_logz"]["depends_on"]

    t["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}

    t["dust_type"]["init"] = 4
    t["mass"]["init"] = 10**10.7
    t["dust1"] = {
        "N": 1,
        "isfree": False,
        "depends_on": transforms.dustratio_to_dust1,
        "init": 0.0,
        "units": "optical depth towards young stars",
    }
    t["dust_ratio"] = {
        "N": 1,
        "isfree": True,
        "init": 1.0,
        "units": "ratio of birth-cloud to diffuse dust",
        "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3),
    }
    t["dust_index"] = {
        "N": 1,
        "isfree": True,
        "init": np.float64(0.0),
        "units": "power-law multiplication of Calzetti",
        "prior": priors.TopHat(mini=-2.5, maxi=0.4),
    }

    # velocity dispersion
    t["sigma_smooth"] = {"N": 1, "isfree": False, "init": 200.0, "units": "km/s"}
    t["smoothtype"] = {"N": 1, "isfree": False, "init": "vel"}
    t["fftsmooth"] = {"N": 1, "isfree": False, "init": True}
    t["eline_sigma"] = {"N": 1, "isfree": False, "init": 100.0, "units": "km/s"}
    return t


def load_priors(path, nebular):
    """Load the prior arrays as a dict.

    Raises
    ------
    SystemExit
        If the npz is missing, or a Cue sample lacks the Cue gas parameters.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"no priors at {path}; run scripts/get_stochastic_priors.py first")
    npz = np.load(path, allow_pickle=True)
    d = {k: npz[k] for k in npz.files}

    if nebular == "cue":
        for need in ("gas_lognHs", "gas_lognos", "gas_logcos"):
            if need not in d:
                raise SystemExit(
                    f"missing Cue prior {need} in {path.name}; regenerate with: "
                    f"uv run python scripts/get_stochastic_priors.py --cue"
                )
    return d


def sample_logsfr_ratios(parset, index, seed):
    """Draw logsfr_ratios from the template's multivariate normal prior.

    The generator is seeded with ``[seed, index]``, so each mock gets the same draw
    whatever the number of workers or the order they finish in.
    """
    p = parset["logsfr_ratios"]["prior"]
    rng = np.random.default_rng([seed, index])
    dist = scipy.stats.multivariate_normal(mean=p.loc, cov=p.scale)
    return np.asarray(dist.rvs(random_state=rng), dtype=float).reshape(-1)


def build_parset_for_index(i):
    """Fill the base template with prior sample ``i`` and draw its logsfr_ratios.

    Returns
    -------
    t : dict
        Template for mock ``i``.
    ratios : np.ndarray
        The drawn logsfr_ratios, also set as ``t["logsfr_ratios"]["init"]``.
    """
    nebular = _S["nebular"]
    priors_dict = _S["priors"]
    t = copy.deepcopy(_S["base_template"])

    t["logmass"]["init"] = priors_dict["stellar_masses"][i]
    t["logzsol"]["init"] = priors_dict["stellar_metallicities"][i]

    # tau_dust_1s is used as the dust_ratio init
    t["dust_index"]["init"] = priors_dict["ns"][i]
    t["dust_ratio"]["init"] = priors_dict["tau_dust_1s"][i]
    t["dust2"]["init"] = priors_dict["tau_dust_2s"][i]

    # dust emission (Draine & Li 2007)
    t["duste_umin"]["init"] = priors_dict["u_mins"][i]
    t["duste_qpah"]["init"] = priors_dict["q_pahs"][i]
    t["duste_gamma"]["init"] = priors_dict["gamma_es"][i]

    # nebular gas
    t["gas_logz"]["init"] = priors_dict["gas_metallicities"][i]
    t["gas_logu"]["init"] = priors_dict["gas_ionization_parameters"][i]
    if nebular == "cue":
        t["gas_lognH"]["init"] = priors_dict["gas_lognHs"][i]
        t["gas_logno"]["init"] = priors_dict["gas_lognos"][i]
        t["gas_logco"]["init"] = priors_dict["gas_logcos"][i]

    # redshift & agebins
    z = priors_dict["redshifts"][i]
    t["zred"]["init"] = z
    t["agebins"]["init"] = make_stochastic_agebins(z=z)

    # stochastic SFH hyperparameters
    t["sigma_reg"]["init"] = priors_dict["sigma_regs"][i]
    t["tau_eq"]["init"] = priors_dict["tau_eqs"][i]
    t["tau_in"]["init"] = priors_dict["tau_ins"][i]
    t["sigma_dyn"]["init"] = priors_dict["sigma_dyns"][i]
    t["tau_dyn"]["init"] = priors_dict["tau_dyns"][i]

    t["sigma_smooth"]["init"] = priors_dict["sigma_smooths"][i]
    t["eline_sigma"]["init"] = priors_dict["sigma_gass"][i]

    # builds the MVN(0, Sigma_ACF) prior object on logsfr_ratios
    t = adjust_stochastic_params(t)

    ratios = sample_logsfr_ratios(t, i, _S["seed"])
    t["logsfr_ratios"]["init"] = ratios
    return t, ratios


@cache
def _get_sps(nebular):
    """Build the stellar source once per process, NebStepBasis for Cue."""
    if nebular == "cue":
        from prospect.sources import NebStepBasis

        return NebStepBasis()
    from prospect.sources import FastStepBasis

    return FastStepBasis()


@cache
def _get_resolution_matrix():
    """Build the DESI resolution matrix once per process."""
    return build_desi_resolution_matrix(DESI_WAV)


def make_obs(n_wave):
    """Return a placeholder prospector Spectrum on the DESI grid with unit flux and errors."""
    obs = Spectrum(
        wavelength=np.asarray(DESI_WAV, dtype=np.float64),
        flux=np.ones(n_wave, dtype=np.float64),
        uncertainty=np.ones(n_wave, dtype=np.float64),
        mask=np.ones(n_wave, dtype=bool),
    )
    obs.rectify()
    return obs


def _init_worker(priors_file, nebular, seed):
    """Set up this process and load the priors and base template into its ``_S``."""
    _setup_process()
    _S["nebular"] = nebular
    _S["seed"] = seed
    _S["priors"] = load_priors(priors_file, nebular)
    _S["base_template"] = build_base_template(nebular)


def worker_block(start, stop):
    """Compute mocks ``start`` to ``stop`` in this process.

    Returns
    -------
    start, stop : int
        The block's rows.
    block : np.ndarray
        Fluxes in maggies on the DESI grid, float32.
    ratios_block : np.ndarray
        The drawn logsfr_ratios.
    lum_block : np.ndarray
        Dust attenuated nebular line luminosities in erg/s.
    """
    nebular = _S["nebular"]
    n_wave = DESI_WAV.size

    sps = _get_sps(nebular)
    R_mat = _get_resolution_matrix()
    obs = make_obs(n_wave)

    block = np.empty((stop - start, n_wave), dtype=np.float32)
    ratios_block = np.empty((stop - start, N_RATIOS), dtype=np.float32)
    lum_block = None  # (stop-start, n_lines), allocated once n_lines is known

    for j, i in enumerate(range(start, stop)):
        parset, ratios = build_parset_for_index(i)
        ratios_block[j, :] = ratios.astype(np.float32)
        model = HyperSpecModel(configuration=parset)
        # sigma_smooth applied by Prospector; lines injected because obs carries flux
        preds, _ = model.predict(model.theta, [obs], sps=sps)
        # The DESI LSF is applied here because MILES is coarser than DESI, which
        # prospect's own obs.resolution path refuses.
        spec = R_mat.dot(preds[0])
        block[j, :] = spec.astype(np.float32)
        # Dust attenuated nebular line luminosities in erg/s.
        lum = np.asarray(model._eline_lum, dtype=np.float32)
        if lum_block is None:
            lum_block = np.empty((stop - start, lum.size), dtype=np.float32)
        lum_block[j, :] = lum

    return start, stop, block, ratios_block, lum_block


def write_mock_seds(priors_file, out_file, nebular, seed, workers=8, chunk_size=500):
    """Compute every mock in worker processes and write the h5 file.

    Parameters
    ----------
    priors_file : str or pathlib.Path
        Prior npz written by scripts/get_stochastic_priors.py.
    out_file : str or pathlib.Path
        Output h5. It is overwritten.
    nebular : {"cue", "fsps"}
        Nebular emission model.
    seed : int
        Seed for the logsfr_ratios draw. It is keyed per index, so the result does not
        depend on the number of workers or the completion order.
    workers : int, optional
        Number of worker processes.
    chunk_size : int, optional
        Mocks per task.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    _setup_process()
    # parent needs the priors too: to size the datasets and to copy them into the h5
    _init_worker(priors_file, nebular, seed)
    priors_dict = _S["priors"]
    n_spectra = len(priors_dict["redshifts"])
    n_wave = DESI_WAV.size

    print(f"nebular={nebular}  n={n_spectra}  seed={seed}  priors={Path(priors_file).name}")

    with h5py.File(out_file, "w") as hf:
        hf.create_dataset("wavelength", data=DESI_WAV, compression="gzip")
        flux_dset = hf.create_dataset(
            "fluxes", shape=(n_spectra, n_wave), dtype=np.float32, compression="gzip"
        )
        ratios_dset = hf.create_dataset(
            "priors/logsfr_ratios",
            shape=(n_spectra, N_RATIOS),
            dtype=np.float32,
            compression="gzip",
        )

        line_wave = _get_line_wave(nebular, n_wave)
        n_lines = line_wave.size
        hf.create_dataset("line_wave", data=line_wave, compression="gzip")
        lum_dset = hf.create_dataset(
            "priors/line_lum",
            shape=(n_spectra, n_lines),
            dtype=np.float32,
            compression="gzip",
        )
        print(f"{nebular} line list: {n_lines} lines")

        for key, arr in priors_dict.items():
            # get_stochastic_priors writes _seed, _sample_size and _cue as 0-d arrays.
            # h5py refuses compression on scalars, so they go in as attributes.
            if np.ndim(arr) == 0:
                hf.attrs[f"priors{key}" if key.startswith("_") else f"priors_{key}"] = arr.item()
            else:
                hf.create_dataset(f"priors/{key}", data=arr, compression="gzip")

        hf.attrs["nebular"] = nebular
        hf.attrs["sample_size"] = n_spectra
        hf.attrs["seed"] = seed
        hf.attrs["sfh_mean"] = "zero"  # the logsfr_ratios prior has zero mean
        hf.attrs["priors_file"] = Path(priors_file).name

        blocks = [(s, min(s + chunk_size, n_spectra)) for s in range(0, n_spectra, chunk_size)]

        # spawn, not fork. A forked child can deadlock in JAX, which Cue uses.
        ctx = mp.get_context("spawn")
        with (
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(priors_file, nebular, seed),
            ) as ex,
            tqdm(total=n_spectra, desc=f"{nebular} spectra") as pbar,
        ):
            futures = {ex.submit(worker_block, s, e): (s, e) for (s, e) in blocks}
            for fut in as_completed(futures):
                start, stop, block, ratios_block, lum_block = fut.result()
                flux_dset[start:stop, :] = block
                ratios_dset[start:stop, :] = ratios_block
                lum_dset[start:stop, :] = lum_block
                pbar.update(stop - start)

    print(f"Saved {nebular} model SEDs to {out_file}")
    return Path(out_file)


def _get_line_wave(nebular, n_wave):
    """Return the rest frame nebular line wavelengths in Angstrom that label line_lum."""
    parset, _ = build_parset_for_index(0)
    m = HyperSpecModel(configuration=parset)
    m.predict(m.theta, [make_obs(n_wave)], sps=_get_sps(nebular))
    return np.asarray(m._eline_wave, dtype=np.float64)
