"""Helpers and the nautilus run for the 2026-09-24_nautilus_tauin experiment.

Run from the repository root with
``uv run python experiments/2026-09-24_nautilus_tauin/fit.py --pool 24 --n-batch 480``.
"""

import argparse
import multiprocessing as mp
import pickle
from functools import partial
from pathlib import Path

import astropy.units as u
import numpy as np
from nautilus import Sampler
from prospect.fitting import lnprobfn
from prospect.models.priors import LogUniform
from prospect.models.sedmodel import HyperSpecModel

from hubersed.conversion import ivar_to_maggies, to_maggies
from hubersed.fitting.chi2 import WAVE_OBS
from hubersed.fitting.map_fits import LSF, chi2_parts, get_sps
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.sps.config import build_continuum_model, build_full_cue_model
from hubersed.sps.parameter_file import build_obs
from hubersed.sps.utils import universe_age_gyr

MILES_REST = (3601.8, 7400.8)  # native 0.9 A sampling in FSPS miles.lambda


def load_data(tid):
    """Load one DESI spectrum in maggies.

    Parameters
    ----------
    tid : int
        DESI TARGETID.

    Returns
    -------
    z : float
        Redshift.
    flux, unc : np.ndarray
        Flux and uncertainty in maggies on ``WAVE_OBS``. Masked pixels have infinite
        uncertainty.
    good : np.ndarray of bool
        Pixels not masked by the pipeline.
    in_miles : np.ndarray of bool
        Pixels whose rest wavelength falls in the MILES window.
    """
    s = load_spectrum(tid)
    z = float(s.redshift.value)
    flux = to_maggies(WAVE_OBS * u.AA, s.flux).value
    iv = ivar_to_maggies(WAVE_OBS * u.AA, s.uncertainty.quantity).value
    good = ~s.mask
    iv = np.where(good, iv, 0.0)
    unc = 1.0 / np.sqrt(np.where(iv > 0, iv, np.inf))
    rest = WAVE_OBS / (1 + z)
    in_miles = (rest >= MILES_REST[0]) & (rest <= MILES_REST[1])
    return z, flux, unc, good, in_miles


def make_model(z):
    """Build the full Cue model with this experiment's star formation history priors.

    tau_in is fixed at the age of the universe at ``z``. sigma_dyn, tau_dyn, sigma_reg and
    tau_eq get log-uniform priors. Every other parameter keeps the bounds of the v2 MAP fits.

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    prospect.models.sedmodel.HyperSpecModel
        The model.
    """
    cont_model, cont_tmpl = build_continuum_model(z)
    _, tmpl = build_full_cue_model(cont_tmpl, cont_model.theta, cont_model)
    tmpl["tau_in"]["isfree"] = False
    tmpl["tau_in"]["init"] = universe_age_gyr(z)
    tmpl["sigma_dyn"]["prior"] = LogUniform(mini=0.01, maxi=1.0)
    tmpl["tau_dyn"]["prior"] = LogUniform(mini=0.001, maxi=0.02)
    tmpl["sigma_reg"]["prior"] = LogUniform(mini=0.1, maxi=3.0)
    tmpl["tau_eq"]["prior"] = LogUniform(mini=0.01, maxi=1.0)
    tmpl["sigma_dyn"]["init"] = 0.05
    tmpl["tau_dyn"]["init"] = 0.005
    tmpl["tau_eq"]["init"] = 0.1
    return HyperSpecModel(tmpl)


def make_obs(flux, unc, mask):
    """Wrap a spectrum as a prospect observation with the DESI line spread function.

    Parameters
    ----------
    flux, unc : np.ndarray
        Flux and uncertainty in maggies on ``WAVE_OBS``.
    mask : np.ndarray of bool
        Pixels to fit.

    Returns
    -------
    list
        The prospect observations.
    """
    return build_obs(spec=flux, unc=unc, mask=mask, resolution=LSF, wavelength=WAVE_OBS)


_STATE = {}


def loglike(theta, tid):
    """Return the log likelihood of ``theta`` for one galaxy.

    Each process builds its own model, observation and Cue source on its first call and keeps
    them in ``_STATE``. Pool workers cannot receive them from the main process, because the Cue
    emulator cannot be pickled and JAX is not fork safe, so the pool uses spawn.

    Parameters
    ----------
    theta : np.ndarray
        Free parameters in the order of ``make_model(z).theta_labels()``.
    tid : int
        DESI TARGETID.

    Returns
    -------
    float
        The log likelihood, with the prior left to the sampler.
    """
    if tid not in _STATE:
        z, flux, unc, good, in_miles = load_data(tid)
        _STATE[tid] = (make_model(z), make_obs(flux, unc, good & in_miles), get_sps()["cue"])
    model, obs, cue = _STATE[tid]
    return lnprobfn(theta, model=model, observations=obs, sps=cue, nested=True)


def main(tid, pool, n_batch, timeout, out):
    """Check the setup against the v2 MAP fit, then run or resume nautilus.

    Parameters
    ----------
    tid : int
        DESI TARGETID.
    pool : int
        Number of spawned worker processes.
    n_batch : int or None
        Likelihood calls per batch. None uses the nautilus default.
    timeout : float
        Seconds before the sampler stops and saves. Rerunning resumes from the checkpoint.
    out : pathlib.Path
        Directory for the checkpoint.
    """
    out.mkdir(parents=True, exist_ok=True)
    z, flux, unc, good, in_miles = load_data(tid)
    model = make_model(z)

    # The v2 model on the full wavelength range must reproduce the saved v2 chi2.
    cont_model, cont_tmpl = build_continuum_model(z)
    model_v2, _ = build_full_cue_model(cont_tmpl, cont_model.theta, cont_model)
    with open(PATHS["RESULTS"] / "map_fits_outliers310_v2" / f"{tid}.pkl", "rb") as f:
        saved = pickle.load(f)
    theta_v2 = model_v2.theta.copy()
    for k, v in saved["theta_dict"].items():
        theta_v2[model_v2.theta_index[k]] = v
    _, stats = chi2_parts(
        model_v2, theta_v2, make_obs(flux, unc, good), get_sps()["cue"], saved["line_pix"]
    )
    assert np.isclose(stats["chi2"], saved["stats"]["chi2"], rtol=1e-4)

    sampler = Sampler(
        model.prior_transform,
        partial(loglike, tid=tid),
        n_dim=model.ndim,
        n_live=1000,
        pool=pool,
        n_batch=n_batch,
        seed=0,
        filepath=str(out / f"{tid}_miles_seed0.h5"),
        resume=True,
    )
    done = sampler.run(n_eff=2000, discard_exploration=True, timeout=timeout, verbose=True)
    print("log Z", sampler.log_z, "N_eff", sampler.n_eff, "calls", sampler.n_like)
    if not done:
        print("Stopped before convergence. Rerun the same command to resume.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tid", type=int, default=39627770174637084)
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--n-batch", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=np.inf)
    parser.add_argument("--out", type=Path, default=PATHS["RESULTS"] / "2026-09-24_nautilus_tauin")
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)
    main(args.tid, args.pool, args.n_batch, args.timeout, args.out)
