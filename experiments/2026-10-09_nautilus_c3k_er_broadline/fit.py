"""Helpers and the nautilus run for the 2026-10-09_nautilus_c3k_er_broadline experiment.

Same model, priors and sampler settings as 2026-09-30_nautilus_c3k_broadline_split45, with two
changes:

- the stellar library is C3K_ER (R = 10000 at rest 3300.5-10000 A) instead of C3K_HR. ER is
  sharper than DESI everywhere, so the data are no longer degraded. The model is smoothed to the
  DESI resolution, and prospect subtracts the 12.7 km/s library resolution in quadrature.
  Fitted pixels are at rest 3320 A or redder, away from the R = 200 part of ER below 3300.5 A;
- the inverse variance is used as the pipeline gives it, so spender's skyline mask stays on in the
  [OI] and [SII] windows (no SPARCL unmasking).

FSPS comes from the C3K_ER build with AFE_FLAG (build_er_fsps.sh in 2026-10-09_c3k_er), put
first on PYTHONPATH, with SPS_HOME holding the ER spectra. [alpha/Fe] stays at 0 here.

Run from the repository root on an ls6 compute node with
``PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe/fsps_site
SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er uv run --no-sync python
experiments/2026-10-09_nautilus_c3k_er_broadline/fit.py --tid 39627770174637084
--forbidden-broad shared --pool 10 --n-batch 480``. ``--check-only`` runs the setup checks and
exits without sampling.
"""

import argparse
import json
import multiprocessing as mp
from functools import partial
from pathlib import Path

import astropy.units as u
import numpy as np
from nautilus import Sampler
from prospect.fitting import lnprobfn
from prospect.models.priors import LogUniform

from hubersed.conversion import ivar_to_maggies, to_maggies
from hubersed.fitting.chi2 import WAVE_OBS
from hubersed.fitting.map_fits import LSF, get_sps
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.sps import broadline
from hubersed.sps.config import build_continuum_model, build_full_cue_model
from hubersed.sps.parameter_file import build_obs
from hubersed.sps.utils import universe_age_gyr

# The data stay at the DESI resolution. Prospector smooths the model by sqrt(res^2 - lib^2),
# which is at least 12 km/s with the 12.7 km/s ER library and DESI at 17.5 km/s or more.
RES = LSF
REST_MIN = 3320.0
N_LIVE = 1000
N_EFF = 2000


def load_data(tid, rest_max=9000.0):
    """Load one DESI spectrum in maggies at the DESI resolution.

    Parameters
    ----------
    tid : int
        DESI TARGETID.
    rest_max : float
        Reddest rest-frame wavelength fitted, in Angstrom.

    Returns
    -------
    z : float
        Redshift.
    flux, unc : np.ndarray
        Flux and uncertainty in maggies on ``WAVE_OBS``. Pixels outside ``good`` have infinite
        uncertainty.
    good : np.ndarray of bool
        Pixels with a positive inverse variance, at rest ``REST_MIN`` to ``rest_max``.
    """
    s = load_spectrum(tid)
    z = float(s.redshift.value)
    flux = to_maggies(WAVE_OBS * u.AA, s.flux).value
    iv = ivar_to_maggies(WAVE_OBS * u.AA, s.uncertainty.quantity).value
    rest = WAVE_OBS / (1 + z)
    good = np.isfinite(flux) & np.isfinite(iv) & (iv > 0) & (rest >= REST_MIN) & (rest < rest_max)
    unc = 1.0 / np.sqrt(np.where(good, iv, np.inf))
    return z, flux, unc, good


def make_params(z, forbidden_broad, sigma_split):
    """Build the parameter specification of the two-component line model.

    tau_in is fixed at the age of the universe at ``z``. sigma_dyn, tau_dyn, sigma_reg and
    tau_eq get log-uniform priors. The broad-line parameters, the forbidden-line width and the
    forbidden broad fraction come from ``broadline.add_broad_params``. Every other parameter keeps
    the bounds of the v2 MAP fits.

    Parameters
    ----------
    z : float
        Redshift.
    forbidden_broad : {"free", "shared"}
        See ``broadline.add_broad_params``.
    sigma_split : float
        Upper bound of the narrow widths and lower bound of the broad width, in km/s.

    Returns
    -------
    dict
        The parameter specification.
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
    params = broadline.add_broad_params(
        tmpl,
        separate_forbidden_width=True,
        forbidden_broad=forbidden_broad,
        sigma_split=sigma_split,
    )
    # the template init (100 km/s) lies outside the narrow prior; nautilus draws from the prior,
    # so this only matters for the start-point checks
    params["eline_sigma"]["init"] = 0.5 * (10.0 + sigma_split)
    return params


def make_model(z, forbidden_broad, sigma_split):
    """Build the two-component line model, see ``make_params``."""
    return broadline.TwoCompLineModel(make_params(z, forbidden_broad, sigma_split))


def make_obs(flux, unc, mask):
    """Wrap a spectrum as a prospect observation at the DESI resolution.

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
    return build_obs(spec=flux, unc=unc, mask=mask, resolution=RES, wavelength=WAVE_OBS)


_STATE = {}


def loglike(theta, tid, forbidden_broad, sigma_split, rest_max=9000.0):
    """Return the log likelihood of ``theta`` for one galaxy.

    Each process builds its own model, observation and Cue source on its first call and keeps
    them in ``_STATE``. Pool workers cannot receive them from the main process, because the Cue
    emulator cannot be pickled and JAX is not fork safe, so the pool uses spawn. The run settings
    are arguments, not module constants, because spawned workers re-import this file.

    Parameters
    ----------
    theta : np.ndarray
        Free parameters in the order of ``make_model(...).theta_labels()``.
    tid : int
        DESI TARGETID.
    forbidden_broad : {"free", "shared"}
        See ``make_params``.
    sigma_split : float
        See ``make_params``.
    rest_max : float
        Reddest rest-frame wavelength fitted, see ``load_data``.

    Returns
    -------
    float
        The log likelihood, with the prior left to the sampler.
    """
    key = (tid, forbidden_broad, sigma_split, rest_max)
    if key not in _STATE:
        z, flux, unc, good = load_data(tid, rest_max)
        cue = get_sps(zero_library_resolution=False)["cue"]
        _STATE[key] = (make_model(z, forbidden_broad, sigma_split), make_obs(flux, unc, good), cue)
    model, obs, cue = _STATE[key]
    return lnprobfn(theta, model=model, observations=obs, sps=cue, nested=True)


def main(args):
    """Check the library and resolution setup, then run or resume nautilus.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line options, see the parser below.
    """
    tid, fb, split, rest_max = args.tid, args.forbidden_broad, args.sigma_split, args.rest_max
    z, flux, unc, good = load_data(tid, rest_max)
    model = make_model(z, fb, split)
    stem = f"{tid}_er_broad_{fb}_split{split:g}_rest{rest_max:g}_seed0"
    # the self-check takes most of the setup (its prior draws build SSPs at every metallicity),
    # so a run that resumes a checkpoint skips it; it passed when that run started
    resuming = (args.out / f"{stem}.h5").exists()

    cue = get_sps(zero_library_resolution=False)["cue"]
    assert cue.ssp.libraries[1] in ("c3k_er", b"c3k_er"), cue.ssp.libraries
    spec = make_obs(flux, unc, good)[0]
    lib = np.interp(spec.padded_wavelength, cue.wavelengths * (1 + z), cue.spectral_resolution)
    # every padded pixel must sit in the R = 10000 part of ER, or the margin below fails
    assert np.all(lib > 12) and np.all(lib < 13), "library resolution is not C3K_ER R = 10000"
    assert np.all(np.sqrt(spec.padded_resolution**2 - lib**2) > 9.9)
    # with eline_fbroad = 0 and one line width the model must equal the one-component model
    if resuming:
        print("resuming a checkpoint, broadline.self_check skipped")
    else:
        broadline.self_check(make_params(z, fb, split), make_obs(flux, unc, good), cue, model.theta)
    assert "eline_fbroad_forb" in model.params
    assert ("eline_fbroad_forb" in model.theta_labels()) == (fb == "free")
    lnl = loglike(model.theta, tid, fb, split, rest_max)
    assert np.isfinite(lnl)
    rest = WAVE_OBS / (1 + z)
    margin = np.sqrt(spec.padded_resolution**2 - lib**2)
    print(
        f"tid {tid}, z {z:.5f}, forbidden_broad {fb}, split {split:g} km/s, "
        f"{good.sum()} pixels, rest {rest[good].min():.0f}-{rest[good].max():.0f} A, "
        f"{model.ndim} free parameters, res {RES.min():.1f}-{RES.max():.1f} km/s, "
        f"smoothing {margin.min():.1f}-{margin.max():.1f} km/s, lnL0 {lnl:.1f}"
    )
    print("free:", ", ".join(model.theta_labels()))
    if args.check_only:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(
        tid=tid,
        z=z,
        library="c3k_er",
        rest_min=REST_MIN,
        rest_max=rest_max,
        forbidden_broad=fb,
        sigma_split=split,
        n_pix=int(good.sum()),
        n_live=N_LIVE,
        n_eff=N_EFF,
        labels=list(model.theta_labels()),
    )
    (args.out / f"{stem}.json").write_text(json.dumps(config, indent=1))

    sampler = Sampler(
        model.prior_transform,
        partial(loglike, tid=tid, forbidden_broad=fb, sigma_split=split, rest_max=rest_max),
        n_dim=model.ndim,
        n_live=N_LIVE,
        pool=args.pool,
        n_batch=args.n_batch,
        seed=0,
        filepath=str(args.out / f"{stem}.h5"),
        resume=True,
    )
    done = sampler.run(n_eff=N_EFF, discard_exploration=True, timeout=args.timeout, verbose=True)
    print("log Z", sampler.log_z, "N_eff", sampler.n_eff, "calls", sampler.n_like)
    if not done:
        print("Stopped before convergence. Rerun the same command to resume.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tid", type=int, default=39627770174637084)
    parser.add_argument("--forbidden-broad", choices=["free", "shared"], required=True)
    parser.add_argument("--sigma-split", type=float, default=45.0)
    parser.add_argument("--rest-max", type=float, default=9000.0)
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--n-batch", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=np.inf)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline"
    )
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)
    main(args)
