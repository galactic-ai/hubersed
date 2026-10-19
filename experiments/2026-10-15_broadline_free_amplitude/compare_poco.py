"""Summarize a pocoMC state of the free-line fit and compare it with the MAP point.

Loads a poco.py state without running the sampler. Prints the weighted 16/50/84% of each parameter
and of 12+log(O/H) = gas_logz + 8.71, the ESS, beta, calls and log Z (final states only). The best
particle log L is compared with the map.py MAP point (map_free.npz), or with the MAP given by --map.
For another target set FIT_TID and pass its 2026-10-18_candidate_map npz as --map.

INPUTS: a pmc_<i>.state or pmc_final.state copied from ls6; map_free.npz or --map, and the 10-13
_pred.npz of 39627770174637084 for the SFH hyperparameters. SEED: none.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix:experiments/2026-10-15_broadline_free_amplitude
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-15_broadline_free_amplitude/compare_poco.py --state <path>``.

OUTPUT: compare_poco_<state name>.txt next to the state file.
"""

import argparse
from pathlib import Path

import dill
import fit
import numpy as np
import pocomc as pc
from model_setup import REFERENCE

CUE_OH_ZERO = 8.71


def load_sampler(path):
    """Return a pocoMC Sampler holding the state in ``path``, without its pool."""
    sampler = pc.Sampler.__new__(pc.Sampler)
    with open(path, "rb") as f:
        sampler.__dict__.update(dill.load(f))
    return sampler


def quantiles(x, w, q=(0.16, 0.5, 0.84)):
    """Return the weighted quantiles ``q`` of samples ``x`` with weights ``w``."""
    order = np.argsort(x)
    cdf = np.cumsum(w[order]) / np.sum(w)
    return np.interp(q, cdf, x[order])


def edges(model, name, x, w, frac=0.02):
    """Return the prior range of ``name`` and the posterior weight within ``frac`` of each edge.

    Vector parameters such as logsfr_ratios_1 get blanks.
    """
    if name not in model.config_dict:
        return ""
    a, b = np.ravel(model.config_dict[name]["prior"].range)[:2]
    if not (np.isfinite(a) and np.isfinite(b)):
        return f" {a:9.3g} {b:9.3g}"
    d = frac * (b - a)
    return f" {a:9.3g} {b:9.3g} {np.sum(w[x < a + d]):6.3f} {np.sum(w[x > b - d]):6.3f}"


def map_point(model, path=None):
    """Return a MAP theta with the SFH hyperparameters at the 10-13 max-L of 39627770174637084.

    ``path`` is a MAP npz, by default map_free.npz of map.py. The candidate MAPs fixed the
    hyperparameters at these values too.
    """
    start = fit.START.with_name(fit.START.name.replace(str(fit.TID), str(REFERENCE)))
    pred = np.load(start, allow_pickle=True)
    best = dict(zip([str(x) for x in pred["labels"]], pred["best"], strict=True))
    m = np.load(path if path is not None else fit.OUT / "map_free.npz")
    point = dict(zip([str(x) for x in m["labels"]], m["theta"], strict=True))
    point.update({k: best[k] for k in fit.HYPERS})
    point.setdefault("frac_nodust", 0.0)  # fixed at 0 in the free variant, as in fit.py
    return np.array([point[k] for k in model.theta_labels()])


def main():
    """Print and write the summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--variant", default="free")
    parser.add_argument("--map", type=Path, default=None, help="MAP npz, default map_free.npz")
    args = parser.parse_args()

    s = load_sampler(args.state)
    beta = float(s.current_particles["beta"]) if s.current_particles else np.nan
    u, w, logl, _ = s.posterior()
    w = w / w.sum()
    ess = 1.0 / np.sum(w**2)
    model, obs, cue = fit.process_state(args.variant)
    labels = list(model.theta_labels())
    theta = np.array([model.prior_transform(ui) for ui in u])

    lnl_map = fit.loglike(map_point(model, args.map), args.variant)
    lines = [
        f"{args.state}: iteration {s.t}, beta {beta:.4g}, calls {s.calls}, "
        f"log Z {getattr(s, 'logz', None)}",
        f"posterior samples {len(u)}, ESS {ess:.0f}",
        f"best log L of the particles {np.max(logl):.2f}, at the MAP point {lnl_map:.2f}",
        "",
        f"{'parameter':>24} {'16%':>10} {'50%':>10} {'84%':>10} {'prior lo':>9} {'prior hi':>9}"
        f" {'w lo':>6} {'w hi':>6}",
    ]
    for i, k in enumerate(labels):
        lo, med, hi = quantiles(theta[:, i], w)
        lines.append(f"{k:>24} {lo:10.4g} {med:10.4g} {hi:10.4g}" + edges(model, k, theta[:, i], w))
    oh = theta[:, labels.index("gas_logz")] + CUE_OH_ZERO
    lo, med, hi = quantiles(oh, w)
    lines.append(f"{'12+log(O/H)':>24} {lo:10.4g} {med:10.4g} {hi:10.4g}")
    text = "\n".join(lines)
    print(text)
    out = args.state.with_name(f"compare_poco_{args.state.stem}.txt")
    out.write_text(text + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
