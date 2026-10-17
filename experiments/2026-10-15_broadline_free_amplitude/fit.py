"""Nautilus run of 39627770174637084 with the free-amplitude broadline model.

Same data, mask, priors, library, resolution matrix and sampler settings as the rmatrix mask run
of 2026-10-13_nautilus_resolution_matrix (its model_setup.py, n_eff 2000), with the
line model of hubersed.sps.free_lines (spec in UberSED
knowledge/broadline_free_amplitude_spec_2026-10-15.md): Cue makes the narrow lines only, narrow
[OII], [OIII] and [NII] have a 0.15 prior width about Cue, [SII], [OI] and [ArIII] are free, the
auroral lines, [NeIII] and [SIII] are free (``--variant free``) or have width 0.3
(``--variant free_aur0.3``), and broad H, [OII], [OIII], [NII], [SII] and [OI] have free
amplitudes marginalized analytically. eline_fbroad is replaced by eline_sigma_broad_forb, so the
run has the same 29 free parameters as the 10-13 run. ``--variant free_dust`` is free with
dust_ratio Uniform(0, 2) instead of ClippedNormal(1, 0.3) and frac_nodust free, Uniform(0, 0.9),
for 30 parameters.

The setup check requires those labels (against the 10-13 run's .json) and, where the 10-13
``_pred.npz`` exists (the Mac), prints the chi2 at its max-L point and, if map.py's
``map_<variant>.npz`` is in the output folder, at its MAP point with the SFH hyperparameters at the
10-13 max-L (map.txt: free 5887.3, free_aur0.3 5896.0).

FSPS comes from the C3K_ER build with AFE_FLAG, put first on PYTHONPATH with the 10-13 folder (for
model_setup.py), and SPS_HOME holds the ER spectra. Run from the repository root on an ls6 compute
node through run.slurm (fit_fork.py), or with ``--check-only`` to run the setup checks only.

COMMAND: on the Mac, ``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-15_broadline_free_amplitude/fit.py --check-only --variant free`` (passed
2026-10-15: 29 parameters, MAP chi2 5887.3). On ls6, ``sbatch -J freeline --export=ALL,VARIANT=free
experiments/2026-10-15_broadline_free_amplitude/run.slurm``.

OUTPUT: <stem>.h5 and <stem>.json in results/2026-10-15_broadline_free_amplitude/.
"""

import argparse
import json
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path

import numpy as np
from model_setup import (
    COADD,
    MISSING_LINES,
    REST_MAX,
    REST_MIN,
    SPLIT,
    TID,
    get_cue,
    load_data,
    make_obs,
    make_params,
)
from nautilus import Sampler
from prospect.fitting import lnprobfn
from prospect.fitting.nested import lnlike_of_unit_cube, unit_cube_identity
from prospect.models.priors import TopHat

from hubersed.paths import PATHS
from hubersed.sps.free_lines import FreeLineModel, add_free_line_params

OUT = PATHS["RESULTS"] / "2026-10-15_broadline_free_amplitude"
START = (
    PATHS["RESULTS"]
    / "2026-10-13_nautilus_resolution_matrix"
    / f"{TID}_rmatrix_mask_er_broad_shared_split45_rest9000_seed0_pred.npz"
)
START_JSON = START.with_name(START.name.replace("_pred.npz", ".json"))
VARIANTS = {"free": np.inf, "free_aur0.3": 0.3, "free_dust": np.inf}
STRONG_WIDTH = 0.15
HYPERS = ("sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn")
N_LIVE = 2000  # 1000 in the 10-13 run, whose chain stalled at N_eff 400-700
N_EFF = 2000


def stem(variant, n_live):
    """Return the file stem of one run."""
    return f"{TID}_rmatrix_mask_{variant}_nlive{n_live}_split{SPLIT:g}_rest{REST_MAX:g}_seed0"


def make_model(z, variant):
    """Return the free-line model of ``variant``."""
    params = add_free_line_params(
        make_params(z),
        sigma_split=SPLIT,
        strong_width=STRONG_WIDTH,
        auroral_width=VARIANTS[variant],
    )
    if variant == "free_dust":
        params["dust_ratio"]["prior"] = TopHat(mini=0.0, maxi=2.0)
        params["frac_nodust"] = dict(N=1, isfree=True, init=0.1, prior=TopHat(mini=0.0, maxi=0.9))
    return FreeLineModel(params)


_STATE = {}


def process_state(variant):
    """Return this process's model, observation and Cue source, building them on the first call.

    Pool workers build their own, because the Cue emulator cannot be pickled and JAX is not fork
    safe, as in the 10-13 fit.
    """
    if variant not in _STATE:
        z, flux, unc, good = load_data(tuple(MISSING_LINES))
        _STATE[variant] = (make_model(z, variant), make_obs(flux, unc, good, "rmatrix"), get_cue())
    return _STATE[variant]


def loglike(theta, variant):
    """Return the log likelihood of ``theta``, with the prior left to the sampler."""
    model, obs, cue = process_state(variant)
    return lnprobfn(theta, model=model, observations=obs, sps=cue, nested=True)


def chi2_at(model, obs, cue, theta):
    """Return the chi2 of ``theta`` over the fitted pixels."""
    spec = model.predict(theta, observations=obs, sps=cue)[0][0]
    o = obs[0]
    return float(np.sum(((o.flux - spec)[o.mask] / o.uncertainty[o.mask]) ** 2))


def check_setup(args):
    """Check the labels, the likelihood and the MAP chi2, and print the setup."""
    model, obs, cue = process_state(args.variant)
    labels = list(model.theta_labels())
    old = json.loads(START_JSON.read_text())["labels"]
    expect = [k for k in old if k != "eline_fbroad"] + ["eline_sigma_broad_forb"]
    expect += ["frac_nodust"] if args.variant == "free_dust" else []
    assert sorted(labels) == sorted(expect), (labels, expect)
    lnl = loglike(model.theta, args.variant)  # the first call also compiles
    assert np.isfinite(lnl)
    o = obs[0]
    print(
        f"tid {TID}, variant {args.variant}, {int(o.mask.sum())} pixels, "
        f"{model.ndim} free parameters, lnL0 {lnl:.1f}"
    )
    if not START.exists():  # predict.py output, kept on the Mac only
        print(f"{START.name} not found, max-L and MAP chi2 checks skipped")
        print("free:", ", ".join(labels))
        return
    pred = np.load(START, allow_pickle=True)
    best = dict(zip([str(x) for x in pred["labels"]], pred["best"], strict=True))
    best["eline_sigma_broad_forb"] = best["eline_sigma_broad"]
    best["frac_nodust"] = 0.0
    theta = np.array([best[k] for k in labels])
    t0 = time.time()
    chi2 = chi2_at(model, obs, cue, theta)
    print(f"chi2 at the 10-13 max-L point {chi2:.1f}, one prediction {time.time() - t0:.2f} s")
    path = args.out / f"map_{args.variant}.npz"
    if path.exists():
        m = np.load(path)
        point = dict(zip([str(x) for x in m["labels"]], m["theta"], strict=True))
        point.update({k: best[k] for k in HYPERS})
        point.setdefault("frac_nodust", 0.0)
        theta = np.array([point[k] for k in labels])
        print(f"chi2 at the MAP point of {path.name} {chi2_at(model, obs, cue, theta):.1f}")
    else:
        print(f"{path.name} not found, MAP chi2 check skipped")
    print("free:", ", ".join(labels))


def main(args, pool=None):
    """Check the setup, then run or resume nautilus.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line options, see ``parse_args``.
    pool : multiprocessing.pool.Pool or None
        Pool for nautilus. None makes a pool of ``args.pool`` workers with the spawn start
        method. ``fit_fork.py`` passes a forked one.
    """
    name = stem(args.variant, args.n_live)
    resuming = (args.out / f"{name}.h5").exists()
    if resuming and not args.check_only:
        print("resuming a checkpoint, setup checks skipped")
        z, _, _, good = load_data(tuple(MISSING_LINES))
        model = make_model(z, args.variant)
    else:
        check_setup(args)
        model = process_state(args.variant)[0]
        z = float(np.atleast_1d(model.params["zred"])[0])
        good = load_data(tuple(MISSING_LINES))[3]
    if args.check_only:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(
        tid=TID,
        z=z,
        library="c3k_er",
        lsf="rmatrix",
        mask="mask",
        masked_lines=list(MISSING_LINES),
        coadd=str(COADD.relative_to(PATHS["ROOT"])),
        rest_min=REST_MIN,
        rest_max=REST_MAX,
        line_model="free_lines",
        variant=args.variant,
        strong_width=STRONG_WIDTH,
        auroral_width=VARIANTS[args.variant],
        sigma_split=SPLIT,
        n_pix=int(good.sum()),
        n_live=args.n_live,
        n_eff=N_EFF,
        f_live=args.f_live,
        labels=list(model.theta_labels()),
    )
    (args.out / f"{name}.json").write_text(json.dumps(config, indent=1))

    # The prior transform runs in the workers, so nautilus gets the identity as its prior and its
    # posterior() returns unit-cube points. Map them with model.prior_transform to read them.
    lnl = partial(loglike, variant=args.variant)
    sampler = Sampler(
        unit_cube_identity,
        partial(
            lnlike_of_unit_cube, prior_transform=model.prior_transform, likelihood_function=lnl
        ),
        n_dim=model.ndim,
        n_live=args.n_live,
        pool=args.pool if pool is None else pool,
        n_batch=args.n_batch,
        seed=0,
        filepath=str(args.out / f"{name}.h5"),
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
    """Parse the command-line options.

    Returns
    -------
    argparse.Namespace
        The options. ``fit_fork.py`` takes the same ones.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=list(VARIANTS), default="free")
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--n-batch", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=np.inf)
    parser.add_argument("--f-live", type=float, default=0.01)
    parser.add_argument("--n-live", type=int, default=N_LIVE)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    main(args)
