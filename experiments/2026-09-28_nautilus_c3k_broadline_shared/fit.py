"""Helpers and the nautilus run for the 2026-09-28_nautilus_c3k_broadline_shared experiment.

Run from the repository root with
``uv run python experiments/2026-09-28_nautilus_c3k_broadline_shared/fit.py --pool 24
--n-batch 480``. ``--check-only`` runs the setup checks and exits without sampling.
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
from hubersed.sps import broadline, rebin
from hubersed.sps.config import build_continuum_model, build_full_cue_model
from hubersed.sps.parameter_file import build_obs
from hubersed.sps.utils import universe_age_gyr

# Prospector smooths the model by sqrt(res^2 - lib^2) and fails when that is zero or imaginary,
# so the data are matched to the C3K_HR resolution plus 10 km/s in quadrature (43.6 km/s).
TARGET_KMS = np.hypot(rebin.C_KMS / (2.355 * 3000.0), 10.0)
TARGET_LIB = dict(R=rebin.C_KMS / (2.355 * TARGET_KMS), window=rebin.LIBRARIES["c3k_hr"]["window"])
RES = np.maximum(LSF, TARGET_KMS)
FORBIDDEN_BROAD = "shared"
SIGMA_SPLIT = 50.0
N_LIVE = 1000
N_EFF = 2000


def load_data(tid, rest_max=9000.0):
    """Load one DESI spectrum in maggies, degraded to ``TARGET_KMS`` where DESI is sharper.

    The whole spectrum is degraded before pixels redward of ``rest_max`` are masked.

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
        Degraded flux and the original uncertainty in maggies on ``WAVE_OBS``. Pixels outside
        ``good`` have zero uncertainty.
    good : np.ndarray of bool
        Pixels not masked by the pipeline, inside the C3K_HR window and blueward of
        ``rest_max``.
    """
    s = load_spectrum(tid)
    z = float(s.redshift.value)
    flux = to_maggies(WAVE_OBS * u.AA, s.flux).value
    iv = ivar_to_maggies(WAVE_OBS * u.AA, s.uncertainty.quantity).value
    ok = ~s.mask & np.isfinite(flux) & np.isfinite(iv) & (iv > 0)
    # zero the masked pixels first, or a NaN spreads through the convolution
    flux, iv, in_c3k = rebin.degrade_to_library(
        WAVE_OBS, np.where(ok, flux, 0.0), np.where(ok, iv, 0.0), z, TARGET_LIB, keep_ivar=True
    )
    good = ok & in_c3k & (WAVE_OBS / (1 + z) < rest_max)
    unc = 1.0 / np.sqrt(np.where(good, iv, np.inf))
    return z, flux, unc, good


def make_params(z):
    """Build the parameter specification of the two-component line model.

    tau_in is fixed at the age of the universe at ``z``. sigma_dyn, tau_dyn, sigma_reg and
    tau_eq get log-uniform priors. The broad-line parameters, the forbidden-line width and the
    shared forbidden broad fraction come from ``broadline.add_broad_params`` with a split of
    ``SIGMA_SPLIT``, which narrows the eline_sigma prior to 10-50 km/s. Every other parameter keeps
    the bounds of the v2 MAP fits.

    Parameters
    ----------
    z : float
        Redshift.

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
        forbidden_broad=FORBIDDEN_BROAD,
        sigma_split=SIGMA_SPLIT,
    )
    # the template init (100 km/s) lies outside the 10-50 km/s narrow prior; nautilus draws from
    # the prior, so this only matters for the start-point checks
    params["eline_sigma"]["init"] = 30.0
    return params


def make_model(z):
    """Build the two-component line model.

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    hubersed.sps.broadline.TwoCompLineModel
        The model.
    """
    return broadline.TwoCompLineModel(make_params(z))


def make_obs(flux, unc, mask):
    """Wrap a spectrum as a prospect observation with the resolution of the degraded data.

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


def loglike(theta, tid, rest_max=9000.0):
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
    rest_max : float
        Reddest rest-frame wavelength fitted, see ``load_data``.

    Returns
    -------
    float
        The log likelihood, with the prior left to the sampler.
    """
    key = (tid, rest_max)
    if key not in _STATE:
        z, flux, unc, good = load_data(tid, rest_max)
        cue = get_sps(zero_library_resolution=False)["cue"]
        _STATE[key] = (make_model(z), make_obs(flux, unc, good), cue)
    model, obs, cue = _STATE[key]
    return lnprobfn(theta, model=model, observations=obs, sps=cue, nested=True)


def main(tid, rest_max, pool, n_batch, timeout, out, check_only=False):
    """Check the library and resolution setup, then run or resume nautilus.

    Parameters
    ----------
    tid : int
        DESI TARGETID.
    rest_max : float
        Reddest rest-frame wavelength fitted, see ``load_data``.
    pool : int
        Number of spawned worker processes.
    n_batch : int or None
        Likelihood calls per batch. None uses the nautilus default.
    timeout : float
        Seconds before the sampler stops and saves. Rerunning resumes from the checkpoint.
    out : pathlib.Path
        Directory for the checkpoint and its run configuration.
    check_only : bool
        Run the setup checks and return before sampling.
    """
    z, flux, unc, good = load_data(tid, rest_max)
    model = make_model(z)

    cue = get_sps(zero_library_resolution=False)["cue"]
    assert cue.ssp.libraries[1] in ("c3k_hr", b"c3k_hr"), cue.ssp.libraries
    spec = make_obs(flux, unc, good)[0]
    lib = np.interp(spec.padded_wavelength, cue.wavelengths * (1 + z), cue.spectral_resolution)
    assert np.all(lib > 40), "library resolution is zeroed or not C3K_HR on the padded grid"
    assert np.all(np.sqrt(spec.padded_resolution**2 - lib**2) > 9.9)
    # with eline_fbroad = 0 and one line width the model must equal the one-component model
    broadline.self_check(make_params(z), make_obs(flux, unc, good), cue, model.theta)
    # the forbidden broad fraction follows the Balmer one instead of being sampled
    assert "eline_fbroad_forb" in model.params and "eline_fbroad_forb" not in model.theta_labels()
    lnl = loglike(model.theta, tid, rest_max)
    assert np.isfinite(lnl)
    rest = WAVE_OBS / (1 + z)
    print(
        f"tid {tid}, z {z:.5f}, {good.sum()} pixels, rest {rest[good].min():.0f}-"
        f"{rest[good].max():.0f} A, {model.ndim} free parameters, "
        f"res {RES.min():.1f}-{RES.max():.1f} km/s, lnL0 {lnl:.1f}"
    )
    print("free:", ", ".join(model.theta_labels()))
    if check_only:
        return

    out.mkdir(parents=True, exist_ok=True)
    stem = f"{tid}_broad_full_rest{rest_max:g}_seed0"
    config = dict(
        tid=tid,
        z=z,
        rest_max=rest_max,
        forbidden_broad=FORBIDDEN_BROAD,
        sigma_split=SIGMA_SPLIT,
        n_pix=int(good.sum()),
        n_live=N_LIVE,
        n_eff=N_EFF,
        labels=list(model.theta_labels()),
    )
    (out / f"{stem}.json").write_text(json.dumps(config, indent=1))

    sampler = Sampler(
        model.prior_transform,
        partial(loglike, tid=tid, rest_max=rest_max),
        n_dim=model.ndim,
        n_live=N_LIVE,
        pool=pool,
        n_batch=n_batch,
        seed=0,
        filepath=str(out / f"{stem}.h5"),
        resume=True,
    )
    done = sampler.run(n_eff=N_EFF, discard_exploration=True, timeout=timeout, verbose=True)
    print("log Z", sampler.log_z, "N_eff", sampler.n_eff, "calls", sampler.n_like)
    if not done:
        print("Stopped before convergence. Rerun the same command to resume.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tid", type=int, default=39628357133926597)
    parser.add_argument("--rest-max", type=float, default=9000.0)
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--n-batch", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=np.inf)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=PATHS["RESULTS"] / "2026-09-28_nautilus_c3k_broadline_shared"
    )
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)
    main(args.tid, args.rest_max, args.pool, args.n_batch, args.timeout, args.out, args.check_only)
