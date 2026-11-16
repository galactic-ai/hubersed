"""Profile the free_dust fit of 39627770174637084 along the dust slope.

QUESTION: free_dust piles up at the dust_index floor of -2.5. Does ln L keep rising below -2.5,
or is the ridge flat below a physical slope, so that the pile-up at the floor is prior volume?
INPUTS: pmc_final.state of the free_dust poco.py run.
SEED: none.
COMMAND: from the repository root with the C3K_ER build,
``PYTHONPATH=experiments/2026-10-15_broadline_free_amplitude:<er_path>:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=<fsps> uv run --no-sync python
experiments/2026-10-15_broadline_free_amplitude/ridge_profile.py --state <path>``.
RESULT: 2026-10-18. No optimum inside -4 to 0.4, ln L rises as n falls. Delta chi2 against n = -4 is
+84.4 at 0.4, +80.9 at 0, +15.0 at -2.5. Any lower bound on n sets the posterior, A_V and mass.
Assumptions: local refits with gas and SFH hyperparameters fixed. Pipeline errors understate noise.
OUTPUT: ridge_profile.txt next to the state file.
"""

import argparse
import time
from pathlib import Path

import fit
import numpy as np
from mask_refit import best_point
from prospect.models.priors import TopHat
from scipy.optimize import minimize

VARIANT = "free_dust"
FREE = ["dust2", "dust_ratio", "frac_nodust", "logmass", "logzsol"]
FREE += [f"logsfr_ratios_{i}" for i in range(1, 10)]
WIDE = (-4.0, 1.5)
DOWN = (-2.5, -3.0, -3.5, -4.0)
UP = (-2.0, -1.5, -1.0, -0.7, -0.3, 0.0, 0.4)
SHOW = ("dust2", "dust_ratio", "frac_nodust", "logmass", "logzsol")


def refit(model, theta0, free):
    """Maximise ln L + ln prior over ``free`` from ``theta0``.

    Returns
    -------
    theta : np.ndarray
        The best point.
    lnl, lnp : float
        Its ln L and ln prior.
    """
    labels = list(model.theta_labels())
    idx = [labels.index(k) for k in free]

    def theta_of(x):
        th = theta0.copy()
        th[idx] = x
        return th

    def parts(th):
        lp = model.prior_product(th)
        if not np.isfinite(lp):
            return -np.inf, lp
        return fit.loglike(th, VARIANT), lp

    def neg(x):
        v, lp = parts(theta_of(x))
        return -(v + lp) if np.isfinite(v + lp) else 1e30

    x0 = theta0[idx]
    r = minimize(neg, x0, method="Powell", options=dict(maxfev=3000, xtol=1e-3, ftol=1e-8))
    step = np.where(np.abs(r.x) > 0.1, 0.05 * np.abs(r.x), 0.02)
    simplex = np.vstack([r.x] + [r.x + np.eye(len(r.x))[i] * step[i] for i in range(len(r.x))])
    r2 = minimize(neg, r.x, method="Nelder-Mead",
                  options=dict(initial_simplex=simplex, maxfev=2000, fatol=0.01))  # fmt: skip
    th = theta_of((r2 if r2.fun < r.fun else r).x)
    lnl, lnp = parts(th)
    return th, lnl, lnp


def main():
    """Run the profile and write the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    model = fit.process_state(VARIANT)[0]
    labels = list(model.theta_labels())
    i_n, i_f = labels.index("dust_index"), labels.index("frac_nodust")
    theta0 = best_point(args.state, model)  # before the prior changes prior_transform
    model.config_dict["dust_index"]["prior"] = TopHat(mini=WIDE[0], maxi=WIDE[1])

    out = args.state.with_name("ridge_profile.txt")
    lines = [f"start: pocoMC best point of {args.state}, dust_index {theta0[i_n]:.3f}, "
             f"dust_index prior TopHat{WIDE}"]  # fmt: skip
    lines.append(f"{'case':>10} {'n':>6} {'lnL':>10} {'lnL+lnp':>10} {'d chi2':>7} {'d lnP':>7} "
                 + " ".join(f"{k[:7]:>7}" for k in SHOW))  # fmt: skip
    ref = {}

    def record(case, th, lnl, lnp, t):
        ref.setdefault("lnl", lnl)
        ref.setdefault("lnP", lnl + lnp)
        g = " ".join(f"{th[labels.index(k)]:7.3f}" for k in SHOW)
        lines.append(
            f"{case:>10} {th[i_n]:6.2f} {lnl:10.2f} {lnl + lnp:10.2f} {-2 * (lnl - ref['lnl']):7.1f} "
            f"{lnl + lnp - ref['lnP']:7.1f} {g}   ({time.time() - t:.0f} s)"
        )
        print(lines[-1], flush=True)
        out.write_text("\n".join(lines) + "\n")

    t = time.time()
    th_free, lnl, lnp = refit(model, theta0, FREE + ["dust_index"])
    record("n free", th_free, lnl, lnp, t)
    for grid in (DOWN, UP):
        start = th_free.copy()
        for n in grid:
            t = time.time()
            start[i_n] = n
            start, lnl, lnp = refit(model, start, FREE)
            record("n fixed", start, lnl, lnp, t)
    t = time.time()
    start = theta0.copy()
    start[i_f], start[i_n] = 0.0, 0.3
    th, lnl, lnp = refit(model, start, [k for k in FREE if k != "frac_nodust"] + ["dust_index"])
    record("f = 0", th, lnl, lnp, t)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
