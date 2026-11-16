"""Helpers and the nautilus run for the 2026-10-10_nautilus_c3k_er_cotie_split experiment.

New since 2026-10-09_nautilus_c3k_er_broadline are a C/O tie (Nicholls et al. 2017, 0.17 dex),
a second forbidden width eline_sigma_forb_hi, He I on the Balmer profile, a resume at ``--f-live 0.3``.
Run from the repository root through chain.slurm (fit_fork.py, c3k_er_afe_v3 build), or directly with
``PYTHONPATH=<site with the C3K_ER AFE_FLAG fsps>
SPS_HOME=<SPS_HOME with the ER spectra> .venv/bin/python
experiments/2026-10-10_nautilus_c3k_er_cotie_split/fit.py --tid 39627770174637084
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
from prospect.fitting.nested import lnlike_of_unit_cube, unit_cube_identity
from prospect.models.priors import LogUniform

from hubersed.conversion import ivar_to_maggies, to_maggies
from hubersed.fitting.chi2 import WAVE_OBS
from hubersed.fitting.map_fits import LSF, get_sps
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.sps import broadline, nebular_ties
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
    """Load one DESI spectrum in maggies at the DESI resolution."""
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

    Parameters
    ----------
    sigma_split : float
        Upper bound of the narrow widths and lower bound of the broad width, in km/s.
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
        split_forbidden=True,
        he_balmer_profile=True,
    )
    nebular_ties.add_co_tie(params)
    # the template init (100 km/s) lies outside the narrow prior; nautilus draws from the prior,
    # so this only matters for the start-point checks
    params["eline_sigma"]["init"] = 0.5 * (10.0 + sigma_split)
    return params


def make_model(z, forbidden_broad, sigma_split):
    """Build the two-component line model, see ``make_params``."""
    return broadline.TwoCompLineModel(make_params(z, forbidden_broad, sigma_split))


def make_obs(flux, unc, mask):
    """Wrap a spectrum as a prospect observation at the DESI resolution."""
    return build_obs(spec=flux, unc=unc, mask=mask, resolution=RES, wavelength=WAVE_OBS)


_STATE = {}


def loglike(theta, tid, forbidden_broad, sigma_split, rest_max=9000.0):
    """Return the log likelihood of ``theta`` for one galaxy."""
    model, obs, cue = process_state(tid, forbidden_broad, sigma_split, rest_max)
    return lnprobfn(theta, model=model, observations=obs, sps=cue, nested=True)


def process_state(tid, forbidden_broad, sigma_split, rest_max):
    """Return this process's model, observation and Cue source, building them on the first call."""
    key = (tid, forbidden_broad, sigma_split, rest_max)
    if key not in _STATE:
        z, flux, unc, good = load_data(tid, rest_max)
        cue = get_sps(zero_library_resolution=False)["cue"]
        _STATE[key] = (make_model(z, forbidden_broad, sigma_split), make_obs(flux, unc, good), cue)
    return _STATE[key]


def check_setup(args, z, flux, unc, good, model, resuming):
    """Check the library, the resolution margin and the likelihood, and print the setup."""
    tid, fb, split, rest_max = args.tid, args.forbidden_broad, args.sigma_split, args.rest_max
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
    labels = list(model.theta_labels())
    assert "gas_dlogco" in labels and "gas_logco" not in labels and "eline_sigma_forb_hi" in labels
    # the line groups of the lines inside the fitted range
    lines = model.emline_info["name"].astype(str)
    w = model.emline_info["wave"]
    inside = (w >= REST_MIN) & (w <= rest_max)
    balmer, high = broadline.line_groups(lines, he_balmer_profile=True)
    for name, m in [("Balmer profile", balmer), ("high", high), ("low", ~balmer & ~high)]:
        print(f"{name}:", "; ".join(np.char.strip(lines[m & inside])))
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


def main(args, pool=None):
    """Check the library and resolution setup, then run or resume nautilus.

    Parameters
    ----------
    pool : multiprocessing.pool.Pool or None
        None makes a pool of ``args.pool`` workers. ``fit_fork.py`` passes a forked one.
    """
    tid, fb, split, rest_max = args.tid, args.forbidden_broad, args.sigma_split, args.rest_max
    z, flux, unc, good = load_data(tid, rest_max)
    model = make_model(z, fb, split)
    stem = f"{tid}_er_cotie_split_broad_{fb}_split{split:g}_rest{rest_max:g}_seed0"
    # check_setup passed when the run started, and each pool worker builds its own Cue, so a run
    # that resumes a checkpoint skips it and the main process's Cue build (10 min)
    resuming = (args.out / f"{stem}.h5").exists()
    if resuming and not args.check_only:
        print("resuming a checkpoint, setup checks skipped")
    else:
        check_setup(args, z, flux, unc, good, model, resuming)
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
        co_tie="nicholls17 + gas_dlogco ~ N(0, 0.17)",
        split_forbidden=True,
        he_balmer_profile=True,
        n_pix=int(good.sum()),
        n_live=N_LIVE,
        n_eff=N_EFF,
        f_live=args.f_live,
        labels=list(model.theta_labels()),
    )
    (args.out / f"{stem}.json").write_text(json.dumps(config, indent=1))

    # The prior transform runs in the workers, not serially in this process, so nautilus gets the
    # identity as its prior and its posterior() returns unit-cube points. To read the posterior,
    # make a Sampler with model.prior_transform from the checkpoint instead.
    lnl = partial(loglike, tid=tid, forbidden_broad=fb, sigma_split=split, rest_max=rest_max)
    sampler = Sampler(
        unit_cube_identity,
        partial(
            lnlike_of_unit_cube, prior_transform=model.prior_transform, likelihood_function=lnl
        ),
        n_dim=model.ndim,
        n_live=N_LIVE,
        pool=args.pool if pool is None else pool,
        n_batch=args.n_batch,
        seed=0,
        filepath=str(args.out / f"{stem}.h5"),
        resume=True,
    )
    done = sampler.run(
        f_live=args.f_live,
        n_eff=N_EFF,
        discard_exploration=True,
        timeout=args.timeout,
        verbose=True,
    )
    print("log Z", sampler.log_z, "N_eff", sampler.n_eff, "calls", sampler.n_like)
    if not done:
        print("Stopped before convergence. Rerun the same command to resume.")


def parse_args():
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tid", type=int, default=39627770174637084)
    parser.add_argument("--forbidden-broad", choices=["free", "shared"], required=True)
    parser.add_argument("--sigma-split", type=float, default=45.0)
    parser.add_argument("--rest-max", type=float, default=9000.0)
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--n-batch", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=np.inf)
    parser.add_argument("--f-live", type=float, default=0.01)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=PATHS["RESULTS"] / "2026-10-10_nautilus_c3k_er_cotie_split"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    main(args)
