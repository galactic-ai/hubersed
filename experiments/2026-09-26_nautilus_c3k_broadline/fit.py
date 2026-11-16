"""Helpers and the nautilus run for the 2026-09-26_nautilus_c3k_broadline experiment.

SUPERSEDED BY: experiments/2026-09-30_nautilus_c3k_broadline_split45.
Refit of 37084 with the narrow and broad split at 45 km/s.

As 2026-09-25_nautilus_tauin_c3k_hr, but Balmer lines get a narrow plus broad profile, the other
lines their own width (hubersed.sps.broadline.TwoCompLineModel) and eline_sigma a 10-100 km/s prior.
Run from the repository root with
``uv run python experiments/2026-09-26_nautilus_c3k_broadline/fit.py --window miles --pool 24
--n-batch 480``.
"""

import argparse
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
MILES_REST = (3601.8, 7400.8)  # the fitted window of 2026-09-24_nautilus_tauin


def load_data(tid, window="full"):
    """Return z, flux, unc and good in maggies, degraded to ``TARGET_KMS`` where DESI is sharper."""
    s = load_spectrum(tid)
    z = float(s.redshift.value)
    flux = to_maggies(WAVE_OBS * u.AA, s.flux).value
    iv = ivar_to_maggies(WAVE_OBS * u.AA, s.uncertainty.quantity).value
    ok = ~s.mask & np.isfinite(flux) & np.isfinite(iv) & (iv > 0)
    # zero the masked pixels first, or a NaN spreads through the convolution
    flux, iv, in_c3k = rebin.degrade_to_library(
        WAVE_OBS, np.where(ok, flux, 0.0), np.where(ok, iv, 0.0), z, TARGET_LIB, keep_ivar=True
    )
    good = ok & in_c3k
    if window == "miles":
        rest = WAVE_OBS / (1 + z)
        good &= (rest >= MILES_REST[0]) & (rest <= MILES_REST[1])
    unc = 1.0 / np.sqrt(np.where(good, iv, np.inf))
    return z, flux, unc, good


def make_params(z):
    """Build the parameter specification of the two-component line model."""
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
    return broadline.add_broad_params(tmpl, separate_forbidden_width=True)


def make_model(z):
    """Build the two-component line model."""
    return broadline.TwoCompLineModel(make_params(z))


def make_obs(flux, unc, mask):
    """Wrap a spectrum as a prospect observation with the resolution of the degraded data."""
    return build_obs(spec=flux, unc=unc, mask=mask, resolution=RES, wavelength=WAVE_OBS)


_STATE = {}


def loglike(theta, tid, window="full"):
    """Return the log likelihood of ``theta``, with the model cached per spawned process."""
    key = (tid, window)
    if key not in _STATE:
        z, flux, unc, good = load_data(tid, window)
        cue = get_sps(zero_library_resolution=False)["cue"]
        _STATE[key] = (make_model(z), make_obs(flux, unc, good), cue)
    model, obs, cue = _STATE[key]
    return lnprobfn(theta, model=model, observations=obs, sps=cue, nested=True)


def main(tid, window, pool, n_batch, timeout, out):
    """Check the library and resolution setup, then run or resume nautilus."""
    out.mkdir(parents=True, exist_ok=True)
    z, flux, unc, good = load_data(tid, window)
    model = make_model(z)

    # The v2 MAP chi2 check of 2026-09-24_nautilus_tauin needs MILES, so check the C3K setup:
    # the library is C3K_HR with its resolution not zeroed, the data are coarser than it on the
    # whole padded grid (prospect smooths the model by sqrt(res^2 - lib^2), here 9.98 km/s or
    # more), and the start is finite.
    cue = get_sps(zero_library_resolution=False)["cue"]
    assert cue.ssp.libraries[1] in ("c3k_hr", b"c3k_hr"), cue.ssp.libraries
    spec = make_obs(flux, unc, good)[0]
    lib = np.interp(spec.padded_wavelength, cue.wavelengths * (1 + z), cue.spectral_resolution)
    assert np.all(lib > 40), "library resolution is zeroed or not C3K_HR on the padded grid"
    assert np.all(np.sqrt(spec.padded_resolution**2 - lib**2) > 9.9)
    # with eline_fbroad = 0 and one line width the model must equal the one-component model
    broadline.self_check(make_params(z), make_obs(flux, unc, good), cue, model.theta)
    lnl = loglike(model.theta, tid, window)
    assert np.isfinite(lnl)
    print(
        f"z {z:.5f}, window {window}, {good.sum()} pixels, "
        f"res {RES.min():.1f}-{RES.max():.1f} km/s, lnL0 {lnl:.1f}"
    )

    sampler = Sampler(
        model.prior_transform,
        partial(loglike, tid=tid, window=window),
        n_dim=model.ndim,
        n_live=1000,
        pool=pool,
        n_batch=n_batch,
        seed=0,
        filepath=str(out / f"{tid}_broad_{window}_seed0.h5"),
        resume=True,
    )
    done = sampler.run(n_eff=2000, discard_exploration=True, timeout=timeout, verbose=True)
    print("log Z", sampler.log_z, "N_eff", sampler.n_eff, "calls", sampler.n_like)
    if not done:
        print("Stopped before convergence. Rerun the same command to resume.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tid", type=int, default=39627770174637084)
    parser.add_argument("--window", choices=["full", "miles"], default="miles")
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--n-batch", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=np.inf)
    parser.add_argument(
        "--out", type=Path, default=PATHS["RESULTS"] / "2026-09-26_nautilus_c3k_broadline"
    )
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)
    main(args.tid, args.window, args.pool, args.n_batch, args.timeout, args.out)
