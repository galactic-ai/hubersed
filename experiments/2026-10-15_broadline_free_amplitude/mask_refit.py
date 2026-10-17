"""Refit the free-line model of 39627770174637084 with the spectrum ends masked: what holds dust_index at 0.4?

QUESTION: the pocoMC posterior puts dust_index on its 0.4 ceiling. Is it held there by the blue end, the red
end, or the whole spectrum? A calibration error at one end would show as n leaving the ceiling, or as a much
smaller cost of forcing n down, once that end is masked.

Starts at the pocoMC best point (highest ln L + ln prior of the 300 particles with the highest ln L in a
poco.py state) and maximises ln L + model.prior_product over dust2, dust_ratio, dust_index, logmass, logzsol
and logsfr_ratios_1-9 (gas, line and SFH hyperparameters fixed; they are uncorrelated with dust_index in the
posterior, |r| <= 0.1). Powell, then a Nelder-Mead polish. Local maximisation, not a posterior. Masks: none,
rest < 3800 A, rest > 7600 A, both. For each mask, n free, n fixed at 0 and n fixed at -0.7.

RESULT: 2026-10-17, mask_refit.txt. Neither end holds n up. n free: 0.365 (none), 0.365 (rest < 3800),
0.384 (rest > 7600), 0.394 (both). Cost of n = 0: +21.7, +21.3, +36.9, +35.1; of n = -0.7: +131, +135, +153,
+143. Masking the red end makes a steeper curve costlier, and at n = 0 dust_ratio goes to its lower bound
0 (0.000 red-masked, 0.021 both). The grey slope is set by the 3800-7600 A continuum together with the
emission lines: with a steeper curve the continuum wants more reddening than the Balmer lines allow once
the lines see at least the diffuse dust. logmass falls 0.10-0.19 dex at n = 0 and 0.20-0.33 dex at -0.7.

INPUTS: pmc_final.state of poco.py. SEED: none.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=experiments/2026-10-15_broadline_free_amplitude:~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-15_broadline_free_amplitude/mask_refit.py --state <path>``.

OUTPUT: mask_refit.txt next to the state file.
"""

import argparse
import time
from pathlib import Path

import fit
import numpy as np
from compare_poco import load_sampler
from scipy.optimize import minimize

FREE = ["dust2", "dust_ratio", "logmass", "logzsol"] + [f"logsfr_ratios_{i}" for i in range(1, 10)]
MASKS = {"none": (0.0, np.inf), "rest<3800": (3800.0, np.inf), "rest>7600": (0.0, 7600.0),
         "both": (3800.0, 7600.0)}  # fmt: skip
SLOPES = (None, 0.0, -0.7)
N_TOP = 300


def best_point(path, model):
    """Return the particle of highest ln L + ln prior among the N_TOP of highest ln L."""
    u, _, logl, _ = load_sampler(path).posterior()
    top = np.argsort(logl)[-N_TOP:]
    theta = [model.prior_transform(u[i]) for i in top]
    lnp = [logl[i] + model.prior_product(t) for i, t in zip(top, theta, strict=True)]
    return theta[int(np.argmax(lnp))]


def refit(model, theta0, free):
    """Maximise ln L + ln prior over ``free`` from ``theta0``; return theta and ln L + ln prior."""
    labels = list(model.theta_labels())
    idx = [labels.index(k) for k in free]

    def theta_of(x):
        th = theta0.copy()
        th[idx] = x
        return th

    def neg(x):
        th = theta_of(x)
        lp = model.prior_product(th)
        if not np.isfinite(lp):
            return 1e30
        v = fit.loglike(th, "free")
        return -(v + lp) if np.isfinite(v) else 1e30

    x0 = theta0[idx]
    r = minimize(neg, x0, method="Powell", options=dict(maxfev=3000, xtol=1e-3, ftol=1e-8))
    step = np.where(np.abs(r.x) > 0.1, 0.05 * np.abs(r.x), 0.02)
    simplex = np.vstack([r.x] + [r.x + np.eye(len(r.x))[i] * step[i] for i in range(len(r.x))])
    r2 = minimize(neg, r.x, method="Nelder-Mead",
                  options=dict(initial_simplex=simplex, maxfev=2000, fatol=0.01))  # fmt: skip
    best = r2 if r2.fun < r.fun else r
    return theta_of(best.x), -best.fun


def main():
    """Run every mask and slope case and write the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    model, obs, _ = fit.process_state("free")
    labels = list(model.theta_labels())
    o = obs[0]
    z = float(np.atleast_1d(model.params["zred"])[0])
    rest = o.wavelength / (1 + z)
    mask0 = o.mask.copy()
    theta0 = best_point(args.state, model)
    i_n = labels.index("dust_index")

    lines = [f"start: pocoMC best point of {args.state.name}, dust_index {theta0[i_n]:.3f}"]
    lines.append(f"{'mask':>10} {'n':>6} {'pixels':>6} {'lnL+lnp':>11} {'d chi2':>7} {'n fit':>6} "
                 f"{'dust2':>6} {'ratio':>6} {'logM':>6} {'logZ*':>6}")  # fmt: skip
    out = args.state.with_name("mask_refit.txt")
    for name, (lo, hi) in MASKS.items():
        o.mask = mask0 & (rest >= lo) & (rest <= hi)
        ref = None
        for n in SLOPES:
            t = time.time()
            start = theta0.copy()
            free = FREE + ["dust_index"] if n is None else FREE
            if n is not None:
                start[i_n] = n
            th, val = refit(model, start, free)
            ref = val if n is None else ref
            dchi2 = -2 * (val - ref)
            g = {k: th[labels.index(k)] for k in ("dust2", "dust_ratio", "logmass", "logzsol")}
            lines.append(
                f"{name:>10} {'free' if n is None else n:>6} {int(o.mask.sum()):6d} {val:11.2f} "
                f"{dchi2:7.1f} {th[i_n]:6.3f} {g['dust2']:6.3f} {g['dust_ratio']:6.3f} "
                f"{g['logmass']:6.3f} {g['logzsol']:6.3f}   ({time.time() - t:.0f} s)"
            )
            print(lines[-1], flush=True)
            out.write_text("\n".join(lines) + "\n")
    o.mask = mask0
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
