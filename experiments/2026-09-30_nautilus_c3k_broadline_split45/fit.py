"""Helpers and the nautilus run for the 2026-09-30_nautilus_c3k_broadline_split45 experiment.

SUPERSEDED BY: experiments/2026-10-09_nautilus_c3k_er_broadline.
Same fit with C3K_ER and data that are no longer degraded.

As 2026-09-28_nautilus_c3k_broadline_shared, plus ``--sigma-split`` (default 45 km/s),
``--forbidden-broad`` and SPARCL ivar (``sparcl_<tid5>.npz``) on [OI]/[SII] sky pixels by default.
Run from the repository root with
``uv run python experiments/2026-09-30_nautilus_c3k_broadline_split45/fit.py --tid 39627770174637084
--forbidden-broad free --pool 24 --n-batch 480``. ``--check-only`` runs the setup checks and exits
without sampling.
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

HERE = Path(__file__).resolve().parent
# Prospector smooths the model by sqrt(res^2 - lib^2) and fails when that is zero or imaginary,
# so the data are matched to the C3K_HR resolution plus 10 km/s in quadrature (43.6 km/s).
TARGET_KMS = np.hypot(rebin.C_KMS / (2.355 * 3000.0), 10.0)
TARGET_LIB = dict(R=rebin.C_KMS / (2.355 * TARGET_KMS), window=rebin.LIBRARIES["c3k_hr"]["window"])
RES = np.maximum(LSF, TARGET_KMS)
N_LIVE = 1000
N_EFF = 2000
# rest-frame vacuum wavelengths (Angstrom) of the lines whose windows are unmasked, and the
# half-width of each window; the spender masks there are 5.3-5.9 A wide in the observed frame
UNMASK_LINES = {
    "[OI]6300": 6302.05,
    "[OI]6363": 6365.54,
    "[SII]6716": 6718.29,
    "[SII]6731": 6732.67,
}
UNMASK_HALF = 10.0


def load_data(tid, rest_max=9000.0, unmask=True):
    """Return z, flux, unc, good and n_unmasked, with SPARCL ivar in sky windows if ``unmask``."""
    s = load_spectrum(tid)
    z = float(s.redshift.value)
    iv_flam = np.asarray(s.uncertainty.array, dtype=float)
    n_unmasked = 0
    if unmask:
        sp = np.load(HERE / f"sparcl_{str(tid)[-5:]}.npz")
        assert abs(float(sp["z"]) - z) < 1e-5, "SPARCL file is for another object"
        both = (iv_flam > 0) & (sp["ivar"] > 0)
        # the chunk files are the same DR1 coadd, so flux and ivar agree wherever both are set
        assert np.allclose(sp["flux"], s.flux.value, rtol=1e-5, atol=1e-6)
        assert np.allclose(sp["ivar"][both], iv_flam[both], rtol=1e-5)
        rest = WAVE_OBS / (1 + z)
        near = np.zeros(len(WAVE_OBS), bool)
        for lam in UNMASK_LINES.values():
            near |= np.abs(rest - lam) < UNMASK_HALF
        back = near & (iv_flam == 0) & (sp["mask"] == 0) & (sp["ivar"] > 0)
        iv_flam = np.where(back, sp["ivar"], iv_flam)
        n_unmasked = int(back.sum())
    flux = to_maggies(WAVE_OBS * u.AA, s.flux).value
    iv = ivar_to_maggies(WAVE_OBS * u.AA, iv_flam * s.uncertainty.unit).value
    ok = np.isfinite(flux) & np.isfinite(iv) & (iv > 0)
    # zero the masked pixels first, or a NaN spreads through the convolution
    flux, iv, in_c3k = rebin.degrade_to_library(
        WAVE_OBS, np.where(ok, flux, 0.0), np.where(ok, iv, 0.0), z, TARGET_LIB, keep_ivar=True
    )
    good = ok & in_c3k & (WAVE_OBS / (1 + z) < rest_max)
    unc = 1.0 / np.sqrt(np.where(good, iv, np.inf))
    return z, flux, unc, good, n_unmasked


def make_params(z, forbidden_broad, sigma_split):
    """Build the line model parameters for ``forbidden_broad`` and ``sigma_split`` (km/s)."""
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
    """Wrap a spectrum as a prospect observation with the resolution of the degraded data."""
    return build_obs(spec=flux, unc=unc, mask=mask, resolution=RES, wavelength=WAVE_OBS)


_STATE = {}


def loglike(theta, tid, forbidden_broad, sigma_split, unmask, rest_max=9000.0):
    """Return the log likelihood of ``theta``, with the model cached per spawned process."""
    key = (tid, forbidden_broad, sigma_split, unmask, rest_max)
    if key not in _STATE:
        z, flux, unc, good, _ = load_data(tid, rest_max, unmask)
        cue = get_sps(zero_library_resolution=False)["cue"]
        _STATE[key] = (make_model(z, forbidden_broad, sigma_split), make_obs(flux, unc, good), cue)
    model, obs, cue = _STATE[key]
    return lnprobfn(theta, model=model, observations=obs, sps=cue, nested=True)


def main(args):
    """Check the library and resolution setup, then run or resume nautilus."""
    tid, fb, split, unmask, rest_max = (
        args.tid,
        args.forbidden_broad,
        args.sigma_split,
        not args.keep_sky_mask,
        args.rest_max,
    )
    z, flux, unc, good, n_unmasked = load_data(tid, rest_max, unmask)
    model = make_model(z, fb, split)

    cue = get_sps(zero_library_resolution=False)["cue"]
    assert cue.ssp.libraries[1] in ("c3k_hr", b"c3k_hr"), cue.ssp.libraries
    spec = make_obs(flux, unc, good)[0]
    lib = np.interp(spec.padded_wavelength, cue.wavelengths * (1 + z), cue.spectral_resolution)
    assert np.all(lib > 40), "library resolution is zeroed or not C3K_HR on the padded grid"
    assert np.all(np.sqrt(spec.padded_resolution**2 - lib**2) > 9.9)
    # with eline_fbroad = 0 and one line width the model must equal the one-component model
    broadline.self_check(make_params(z, fb, split), make_obs(flux, unc, good), cue, model.theta)
    assert "eline_fbroad_forb" in model.params
    assert ("eline_fbroad_forb" in model.theta_labels()) == (fb == "free")
    lnl = loglike(model.theta, tid, fb, split, unmask, rest_max)
    assert np.isfinite(lnl)
    rest = WAVE_OBS / (1 + z)
    print(
        f"tid {tid}, z {z:.5f}, forbidden_broad {fb}, split {split:g} km/s, "
        f"{good.sum()} pixels ({n_unmasked} sky pixels unmasked), rest {rest[good].min():.0f}-"
        f"{rest[good].max():.0f} A, {model.ndim} free parameters, "
        f"res {RES.min():.1f}-{RES.max():.1f} km/s, lnL0 {lnl:.1f}"
    )
    print("free:", ", ".join(model.theta_labels()))
    if args.check_only:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    mask_tag = "unmask" if unmask else "skymask"
    stem = f"{tid}_broad_{fb}_split{split:g}_{mask_tag}_rest{rest_max:g}_seed0"
    config = dict(
        tid=tid,
        z=z,
        rest_max=rest_max,
        forbidden_broad=fb,
        sigma_split=split,
        unmask_sky=unmask,
        n_unmasked=n_unmasked,
        n_pix=int(good.sum()),
        n_live=N_LIVE,
        n_eff=N_EFF,
        labels=list(model.theta_labels()),
    )
    (args.out / f"{stem}.json").write_text(json.dumps(config, indent=1))

    sampler = Sampler(
        model.prior_transform,
        partial(
            loglike,
            tid=tid,
            forbidden_broad=fb,
            sigma_split=split,
            unmask=unmask,
            rest_max=rest_max,
        ),
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
    parser.add_argument("--keep-sky-mask", action="store_true")
    parser.add_argument("--rest-max", type=float, default=9000.0)
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--n-batch", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=np.inf)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=PATHS["RESULTS"] / "2026-09-30_nautilus_c3k_broadline_split45"
    )
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)
    main(args)
