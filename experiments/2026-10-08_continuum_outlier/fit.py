"""Nautilus fits of the continuum outlier 39627757533007793 (issue #17).

SUPERSEDED BY: experiments/2026-10-28_continuum_c3k_er_rmatrix_map.
Continuum MAP with C3K_ER and the DR1 resolution matrix instead of C3K_HR.

As 2026-09-25_nautilus_tauin_c3k_hr, with switches ``--lib``, ``--window``, ``--nebular``, ``--afe``
and ``--degrade`` (off zeroes the library resolution, as in 2026-09-24_nautilus_tauin).
Run from the repository root with
``uv run python experiments/2026-10-08_continuum_outlier/fit.py --lib c3k_hr --window full
--nebular on --pool 24 --n-batch 480``.
"""

import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path

import astropy.units as u
import numpy as np
from nautilus import Sampler
from prospect.fitting import lnprobfn
from prospect.models.priors import LogUniform, TopHat
from prospect.models.sedmodel import HyperSpecModel
from prospect.sources.galaxy_basis import SSPBasis

from hubersed.conversion import ivar_to_maggies, to_maggies
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.sps import rebin
from hubersed.sps.config import build_continuum_model, build_full_cue_model
from hubersed.sps.lsf import desi_resolution
from hubersed.sps.parameter_file import WAVE_OBS, build_cue_sps, build_obs, build_sps
from hubersed.sps.utils import universe_age_gyr

C = rebin.C_KMS
MILES_REST = (3601.8, 7400.8)  # the MILES window of 2026-09-24 and 2026-09-25
LSF = C / (2.355 * desi_resolution(WAVE_OBS.astype(np.float64)))
HYPERS = {
    "sigma_dyn": (LogUniform(mini=0.01, maxi=1.0), 0.05),
    "tau_dyn": (LogUniform(mini=0.001, maxi=0.02), 0.005),
    "sigma_reg": (LogUniform(mini=0.1, maxi=3.0), None),
    "tau_eq": (LogUniform(mini=0.01, maxi=1.0), 0.1),
}

# Data targets are the library resolution plus 10 km/s in quadrature, because prospect smooths
# the model by sqrt(res^2 - lib^2). MILES has a constant FWHM in Angstrom, so 10 km/s is added
# at the red edge of the window, where it is smallest in Angstrom.
TARGETS = {
    "c3k_hr": dict(
        R=C / (2.355 * np.hypot(C / (2.355 * 3000.0), 10.0)),
        window=rebin.LIBRARIES["c3k_hr"]["window"],
    ),
    "miles": dict(
        fwhm_A=np.hypot(rebin.LIBRARIES["miles"]["fwhm_A"], 2.355 * 10.0 * MILES_REST[1] / C),
        window=MILES_REST,
    ),
}


def load_data(tid, lib, window, degrade):
    """Return z, flux, unc, good and target_kms, degraded to the ``lib`` target if ``degrade``."""
    s = load_spectrum(tid)
    z = float(s.redshift.value)
    flux = to_maggies(WAVE_OBS * u.AA, s.flux).value
    iv = ivar_to_maggies(WAVE_OBS * u.AA, s.uncertainty.quantity).value
    ok = ~s.mask & np.isfinite(flux) & np.isfinite(iv) & (iv > 0)
    if not degrade:
        good = ok.copy()
        if window == "miles":
            rest = WAVE_OBS / (1 + z)
            good &= (rest >= MILES_REST[0]) & (rest <= MILES_REST[1])
        unc = 1.0 / np.sqrt(np.where(good, iv, np.inf))
        return z, flux, unc, good, np.full_like(LSF, np.inf)
    # zero the masked pixels first so a NaN does not spread through the convolution
    flux, iv, in_lib = rebin.degrade_to_library(
        WAVE_OBS, np.where(ok, flux, 0.0), np.where(ok, iv, 0.0), z, TARGETS[lib], keep_ivar=True
    )
    good = ok & in_lib
    if window == "miles":
        rest = WAVE_OBS / (1 + z)
        good &= (rest >= MILES_REST[0]) & (rest <= MILES_REST[1])
    unc = 1.0 / np.sqrt(np.where(good, iv, np.inf))
    sig_A, _ = rebin._library_sigma_obs_A(WAVE_OBS, z, TARGETS[lib])
    return z, flux, unc, good, sig_A / WAVE_OBS * C


def make_model(z, nebular, afe="none"):
    """Build the 2026-09-25 model. With ``afe`` free or zero, logzsol is [Fe/H], not [Z/H]."""
    cont_model, tmpl = build_continuum_model(z)
    if nebular:
        _, tmpl = build_full_cue_model(tmpl, cont_model.theta, cont_model)
    else:
        tmpl["add_neb_emission"] = {"N": 1, "isfree": False, "init": False}
    for name, (prior, init) in HYPERS.items():
        tmpl[name]["isfree"] = True
        tmpl[name]["prior"] = prior
        if init is not None:
            tmpl[name]["init"] = init
    tmpl["tau_in"]["isfree"] = False
    tmpl["tau_in"]["init"] = universe_age_gyr(z)
    if afe != "none":
        tmpl["afe"] = {
            "N": 1,
            "isfree": afe == "free",
            "init": 0.0,
            "prior": TopHat(mini=-0.2, maxi=0.6),
        }
    return HyperSpecModel(tmpl)


def make_sps(nebular, degrade):
    """Return the Cue or plain FSPS source, library resolution zeroed unless ``degrade``."""
    if not degrade:
        SSPBasis.spectral_resolution = property(lambda self: np.zeros_like(self.ssp.wavelengths))
    if nebular:
        return build_cue_sps()
    return build_sps(zero_library_resolution=not degrade)


def model_resolution(z, target_kms, good, sps):
    """Return the resolution prospect smooths the model to, in km/s on ``WAVE_OBS``."""
    lib = np.interp(WAVE_OBS, sps.wavelengths * (1 + z), sps.spectral_resolution)
    res = np.maximum(LSF, np.where(np.isfinite(target_kms), target_kms, 0.0))
    return np.where(good, res, np.maximum(res, np.hypot(lib, 10.0)))


def make_obs(z, flux, unc, good, target_kms, sps):
    """Wrap the spectrum as a prospect observation, with library plus 10 km/s in the padding."""
    res = model_resolution(z, target_kms, good, sps)
    obs = build_obs(spec=flux, unc=unc, mask=good, resolution=res, wavelength=WAVE_OBS)
    for spec in obs:
        wave = spec.padded_wavelength
        lib = np.interp(wave, sps.wavelengths * (1 + z), sps.spectral_resolution)
        pad = (wave < WAVE_OBS[0]) | (wave > WAVE_OBS[-1])
        spec.padded_resolution[pad] = np.maximum(
            spec.padded_resolution[pad], np.hypot(lib[pad], 10.0)
        )
    return obs


_STATE = {}


def loglike(theta, tid, lib, window, nebular, degrade, afe="none"):
    """Return the log likelihood of ``theta``, with the model cached per spawned process."""
    key = (tid, lib, window, nebular, degrade, afe)
    if key not in _STATE:
        z, flux, unc, good, target_kms = load_data(tid, lib, window, degrade)
        sps = make_sps(nebular, degrade)
        obs = make_obs(z, flux, unc, good, target_kms, sps)
        _STATE[key] = (make_model(z, nebular, afe), obs, sps)
    model, obs, sps = _STATE[key]
    return lnprobfn(theta, model=model, observations=obs, sps=sps, nested=True)


def check_setup(tid, lib, window, nebular, degrade, afe="none"):
    """Check the library and resolution setup and return z, the model and the pixel count."""
    z, flux, unc, good, target_kms = load_data(tid, lib, window, degrade)
    model = make_model(z, nebular, afe)
    sps = make_sps(nebular, degrade)
    assert sps.ssp.libraries[1] in (lib, lib.encode()), sps.ssp.libraries
    if afe != "none":
        # a build without AFE_FLAG ignores afe without an error
        ssp = sps.ssp
        _, s0 = ssp.get_spectrum(tage=10.0, peraa=False)
        ssp.params["afe"] = 0.4
        _, s4 = ssp.get_spectrum(tage=10.0, peraa=False)
        ssp.params["afe"] = 0.0
        # compare ratios: the spectra are about 1e-14 Lsun/Hz or less, below allclose's atol
        ratio = np.median(s4[s0 > 0] / s0[s0 > 0])
        print(f"afe 0.4 vs 0 at 10 Gyr: median ratio {ratio:.3f}")
        assert abs(ratio - 1) > 0.01, "afe changes nothing, FSPS lacks AFE_FLAG"
    lib_kms = np.interp(WAVE_OBS, sps.wavelengths * (1 + z), sps.spectral_resolution)
    if not degrade:
        assert np.all(lib_kms == 0), "library resolution is not zeroed"
        assert np.isfinite(loglike(model.theta, tid, lib, window, nebular, degrade, afe))
        print("data not degraded, library resolution zeroed")
        return z, model, good.sum()
    # FSPS marks wavelengths outside the library window with a negative resolution
    assert np.all(lib_kms[good] > 0), "library resolution is zeroed on a fitted pixel"
    # prospect smooths by sqrt(res^2 - lib^2), which should be about 10 km/s or more
    res = model_resolution(z, target_kms, good, sps)
    margin = np.sqrt(res[good] ** 2 - lib_kms[good] ** 2)
    print(f"smoothing on fitted pixels {margin.min():.2f}-{margin.max():.2f} km/s")
    assert margin.min() > 9.9, "data are sharper than the library plus 10 km/s"
    assert np.isfinite(loglike(model.theta, tid, lib, window, nebular, degrade, afe))
    return z, model, good.sum()


def main(tid, lib, window, nebular, degrade, afe, pool, n_batch, timeout, out):
    """Check the setup, then run or resume nautilus."""
    out.mkdir(parents=True, exist_ok=True)
    z, model, npix = check_setup(tid, lib, window, nebular, degrade, afe)
    neb = "on" if nebular else "off"
    tag = ("" if degrade else "_nodeg") + ("" if afe == "none" else f"_afe{afe}")
    print(
        f"z {z:.5f}, lib {lib}, window {window}, nebular {neb}, degrade {degrade}, afe {afe}, "
        f"{npix} pixels, ndim {model.ndim}"
    )

    sampler = Sampler(
        model.prior_transform,
        partial(
            loglike, tid=tid, lib=lib, window=window, nebular=nebular, degrade=degrade, afe=afe
        ),
        n_dim=model.ndim,
        n_live=1000,
        pool=pool,
        n_batch=n_batch,
        seed=0,
        filepath=str(out / f"{tid}_{lib}_{window}_neb{neb}{tag}_seed0.h5"),
        resume=True,
    )
    done = sampler.run(n_eff=2000, discard_exploration=True, timeout=timeout, verbose=True)
    print("log Z", sampler.log_z, "N_eff", sampler.n_eff, "calls", sampler.n_like)
    if not done:
        print("Stopped before convergence. Rerun the same command to resume.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tid", type=int, default=39627757533007793)
    parser.add_argument("--lib", choices=["miles", "c3k_hr"], required=True)
    parser.add_argument("--window", choices=["miles", "full"], required=True)
    parser.add_argument("--nebular", choices=["on", "off"], required=True)
    parser.add_argument("--degrade", choices=["on", "off"], default="on")
    parser.add_argument("--afe", choices=["none", "free", "zero"], default="none")
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--n-batch", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=np.inf)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=PATHS["RESULTS"] / "2026-10-08_continuum_outlier"
    )
    args = parser.parse_args()
    if args.lib == "miles" and args.window == "full":
        parser.error("MILES only covers the MILES window")
    if args.afe != "none" and args.lib != "c3k_hr":
        parser.error("the alpha grid is C3K_HR")
    mp.set_start_method("spawn", force=True)
    nebular = args.nebular == "on"
    degrade = args.degrade == "on"
    if args.check_only:
        print(check_setup(args.tid, args.lib, args.window, nebular, degrade, args.afe))
    else:
        main(
            args.tid,
            args.lib,
            args.window,
            nebular,
            degrade,
            args.afe,
            args.pool,
            args.n_batch,
            args.timeout,
            args.out,
        )
