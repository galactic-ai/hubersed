"""Why does ladder step 5 reject about 3% of prior draws?

QUESTION: the step 5 checks gave 0.967 (Mac) and 0.969 (skx) finite first pocoMC batches, against
1.000 in steps 3 and 4. prospect's lnprobfn turns any ValueError in predict into -inf, so the cause
is hidden. Which exception, and in which part of the prior?
INPUTS: fit.build(5) (same data, model and Cue as the step 5 fit), 1024 unit-cube draws.
SEED: numpy default_rng(1).
COMMAND: on stampede3 through reject.slurm (one skx-dev node, the fit.py node pool).
RESULT: pending.
FIGURES: results/2026-11-07_emline_ladder_39627770174637084/reject_step5.txt.
"""

import multiprocessing as mp
import time
import traceback
from collections import Counter

import fit
import numpy as np

N_DRAWS = 1024
OUT = fit.OUT / "reject_step5.txt"
_STATE = {}


def evaluate(u):
    """Return (lnL or None, exception summary or None, theta) for one unit-cube point."""
    from prospect.fitting import lnprobfn

    if "fit" not in _STATE:
        _STATE["fit"] = fit.build(5)
    observations, model, cue = _STATE["fit"]
    theta = model.prior_transform(u)
    try:
        model.set_parameters(theta)
        model.predict(theta, observations, sps=cue)
    except Exception as e:  # noqa: BLE001, the point is to see every kind
        last = traceback.extract_tb(e.__traceback__)[-1]
        where = f"{last.filename.split('site-packages/')[-1]}:{last.lineno}"
        return None, f"{type(e).__name__} at {where}: {str(e)[:120]}", theta
    lnl = lnprobfn(theta, model=model, observations=observations, sps=cue, nested=True)
    return float(lnl), None, theta


def main():
    """Evaluate the draws on the node pool and write the summary."""
    t = time.time()
    with fit.node_pool(0) as pool:
        _, model, _ = fit.build(5)
        labels = model.theta_labels()
        u = np.random.default_rng(1).uniform(size=(N_DRAWS, model.ndim))
        res = pool.map(evaluate, list(u))
    lnl = np.array([np.nan if r[0] is None else r[0] for r in res])
    bad = np.array([r[1] is not None for r in res])
    nonfinite = ~bad & ~np.isfinite(lnl)
    theta = np.array([r[2] for r in res])
    lines = [
        f"step 5, {N_DRAWS} prior draws (seed 1), {time.time() - t:.0f} s",
        f"exception in predict: {bad.sum()}; non-finite lnL without exception: {nonfinite.sum()}",
        "",
        "exceptions (count, kind, where, message):",
    ]
    lines += [f"{n:5d}  {msg}" for msg, n in Counter(r[1] for r in res if r[1]).most_common()]
    lines += ["", "parameter: median of rejected / median of accepted (prior units)"]
    rej = bad | nonfinite
    if rej.any():
        for j, name in enumerate(labels):
            lines.append(
                f"{name:24s} {np.median(theta[rej, j]):10.4g} {np.median(theta[~rej, j]):10.4g}"
            )
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
