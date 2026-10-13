"""Nautilus run of 39627770174637084 with its DESI resolution matrix and the corrected ivar.

Same model, priors, library and sampler settings as 2026-10-09_nautilus_c3k_er_broadline (C3K_ER,
Cue broadline model, forbidden broad fraction shared, split 45 km/s, rest 3320-9000 A, n_live 1000,
n_eff 2000), with two changes, both from model_setup.py:

- the data are the row-weighted recoadd with the corrected calibration ivar
  (resolution_matrix_rows_ivar.npz of the 10-09 folder) instead of the DR1 coadd. The fitted
  pixels are those of the 10-09 fit where the recoadded ivar is positive;
- the line spread is the target's own resolution matrix (``--lsf rmatrix``,
  ResolutionMatrixModel) instead of the median Gaussian LSF. ``--lsf median`` runs the control.

``--mask mask``, the default, also masks the lines missing from the Cue line list (H11, H12,
He I 5016, He I 7281, Pa11 to Pa16) at rest +-4 A. This is the setup of the rmatrix mask MAP of
map.py. ``--mask nomask`` keeps those pixels.

The setup check builds the model from the same make_params as the 10-09 run and requires the same
29 parameter labels, so the posteriors can be compared. If map.py's ``map_<lsf>_<mask>.npz`` is in
the output folder, it also prints the chi2 at that MAP point (SFH hyperparameters at the 10-09
max-L), which should equal the value in map.txt.

FSPS comes from the C3K_ER build with AFE_FLAG, put first on PYTHONPATH, with SPS_HOME holding the
ER spectra. Run from the repository root on an ls6 compute node through chain.slurm (fit_fork.py,
c3k_er_afe_v3), or directly with
``PYTHONPATH=/work/11006/nikhilgaruda/ls6/research/fsps_builds/c3k_er_afe_v3/fsps_site
SPS_HOME=/work/11006/nikhilgaruda/ls6/research/sps_home_er .venv/bin/python
experiments/2026-10-13_nautilus_resolution_matrix/fit.py --pool 10 --n-batch 480``.
``--check-only`` runs the setup checks and exits without sampling.
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
    ER_RESULTS,
    FB,
    MISSING_LINES,
    REST_MAX,
    REST_MIN,
    SPLIT,
    TID,
    get_cue,
    load_data,
    make_model,
    make_obs,
    make_params,
)
from nautilus import Sampler
from prospect.fitting import lnprobfn
from prospect.fitting.nested import lnlike_of_unit_cube, unit_cube_identity

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-13_nautilus_resolution_matrix"
MASKS = {"nomask": (), "mask": tuple(MISSING_LINES)}
ER_STEM = f"{TID}_er_broad_{FB}_split{SPLIT:g}_rest{REST_MAX:g}_seed0"
HYPERS = ("sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn")
N_LIVE = 1000
N_EFF = 2000


def stem(lsf, mask):
    """Return the file stem of one run."""
    return f"{TID}_{lsf}_{mask}_er_broad_{FB}_split{SPLIT:g}_rest{REST_MAX:g}_seed0"


_STATE = {}


def process_state(lsf, mask):
    """Return this process's model, observation and Cue source, building them on the first call.

    Pool workers cannot receive them from the main process, because the Cue emulator cannot be
    pickled and JAX is not fork safe, so each process builds its own, as in the 10-09 fit.

    Parameters
    ----------
    lsf : {"rmatrix", "median"}
        Line spread, see ``model_setup.make_model``.
    mask : {"nomask", "mask"}
        Key of ``MASKS``.

    Returns
    -------
    tuple
        The model, the observation list and the Cue source.
    """
    key = (lsf, mask)
    if key not in _STATE:
        z, flux, unc, good = load_data(MASKS[mask])
        model = make_model(make_params(z), lsf)
        _STATE[key] = (model, make_obs(flux, unc, good, lsf), get_cue())
    return _STATE[key]


def loglike(theta, lsf, mask):
    """Return the log likelihood of ``theta``, with the prior left to the sampler."""
    model, obs, cue = process_state(lsf, mask)
    return lnprobfn(theta, model=model, observations=obs, sps=cue, nested=True)


def chi2_at(model, obs, cue, theta):
    """Return the chi2 of ``theta`` over the fitted pixels."""
    spec = model.predict(theta, observations=obs, sps=cue)[0][0]
    o = obs[0]
    return float(np.sum(((o.flux - spec)[o.mask] / o.uncertainty[o.mask]) ** 2))


def check_setup(args):
    """Check the labels, the likelihood and the MAP chi2, and print the setup.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line options, see ``parse_args``.
    """
    model, obs, cue = process_state(args.lsf, args.mask)
    labels = list(model.theta_labels())
    er = json.loads((ER_RESULTS / f"{ER_STEM}.json").read_text())
    assert labels == er["labels"], "parameters differ from the 10-09 ER run"
    lnl = loglike(model.theta, args.lsf, args.mask)  # the first call also compiles
    assert np.isfinite(lnl)
    z = float(np.atleast_1d(model.params["zred"])[0])
    o = obs[0]
    rest = o.wavelength[o.mask] / (1 + z)
    print(
        f"tid {TID}, z {z:.5f}, lsf {args.lsf}, mask {args.mask}, {int(o.mask.sum())} pixels "
        f"(10-09 run {er['n_pix']}), rest {rest.min():.0f}-{rest.max():.0f} A, "
        f"{model.ndim} free parameters, lnL0 {lnl:.1f}"
    )
    # chi2 at the 10-09 max-L point and, if map.py ran here, at the MAP point of this run. The
    # max-L call is timed: it is a new theta, so nothing is reused from the call above.
    pred = np.load(ER_RESULTS / f"{ER_STEM}_pred.npz")
    best = dict(zip([str(x) for x in pred["labels"]], pred["best"], strict=True))
    theta = np.array([best[k] for k in labels])
    t0 = time.time()
    chi2 = chi2_at(model, obs, cue, theta)
    print(f"chi2 at the 10-09 max-L point {chi2:.1f}, one prediction {time.time() - t0:.2f} s")
    path = args.out / f"map_{args.lsf}_{args.mask}.npz"
    if path.exists():
        m = np.load(path)
        point = dict(zip([str(x) for x in m["labels"]], m["theta"], strict=True))
        point.update({k: best[k] for k in HYPERS})
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
        Pool for nautilus. None makes a pool of ``args.pool`` workers with the global start
        method, which ``__main__`` sets to spawn. ``fit_fork.py`` passes a forked one.
    """
    name = stem(args.lsf, args.mask)
    # the checks passed when the run started, and each pool worker builds its own Cue, so a run
    # that resumes a checkpoint skips them and the main process's Cue build
    resuming = (args.out / f"{name}.h5").exists()
    if resuming and not args.check_only:
        print("resuming a checkpoint, setup checks skipped")
        z, _, _, good = load_data(MASKS[args.mask])
        model = make_model(make_params(z), args.lsf)
    else:
        check_setup(args)
        model = process_state(args.lsf, args.mask)[0]
        z = float(np.atleast_1d(model.params["zred"])[0])
        good = load_data(MASKS[args.mask])[3]
    if args.check_only:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(
        tid=TID,
        z=z,
        library="c3k_er",
        lsf=args.lsf,
        mask=args.mask,
        masked_lines=list(MASKS[args.mask]),
        coadd=str(COADD.relative_to(PATHS["ROOT"])),
        rest_min=REST_MIN,
        rest_max=REST_MAX,
        forbidden_broad=FB,
        sigma_split=SPLIT,
        n_pix=int(good.sum()),
        n_live=N_LIVE,
        n_eff=N_EFF,
        f_live=args.f_live,
        labels=list(model.theta_labels()),
    )
    (args.out / f"{name}.json").write_text(json.dumps(config, indent=1))

    # The prior transform runs in the workers, so nautilus gets the identity as its prior and its
    # posterior() returns unit-cube points. To read the posterior, map them with
    # model.prior_transform, as analysis.py does.
    lnl = partial(loglike, lsf=args.lsf, mask=args.mask)
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
    parser.add_argument("--lsf", choices=["rmatrix", "median"], default="rmatrix")
    parser.add_argument("--mask", choices=list(MASKS), default="mask")
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--n-batch", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=np.inf)
    parser.add_argument("--f-live", type=float, default=0.01)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    main(args)
