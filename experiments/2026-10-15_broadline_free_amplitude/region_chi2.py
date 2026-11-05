"""Where does forcing dust_index = 0 cost chi2 in the free-line fit of 39627770174637084?

QUESTION: n = 0 costs +21.7 (no mask) and +35.1 (rest 3800-7600 A) against n free (mask_refit.py),
and the Balmer decrement matches to 0.5% (balmer_check.py). Is the cost a broad continuum colour
mismatch or local features?

Repeats mask_refit's n-free and n = 0 refits for those two masks, splits the chi2 difference into
200 A rest bins for continuum and line pixels, and gives the n = 0 to n-free continuum ratio per
bin. SEED: none.

RESULT: 2026-10-17, region_chi2.txt. The cost is a broad continuum shape. No mask: continuum +18.9,
lines -2.0, prior about +4.8, largest bin 4600-4800 A (+9.9). 3800-7600 A: continuum +32.0, lines
-3.1, spread over several bins. The two continua differ by a smooth bow under 1% (0.993 at 4000 A,
1.003 at 5800-6200 A, 0.994 at 7200-7600 A).

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=experiments/2026-10-15_broadline_free_amplitude:~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-15_broadline_free_amplitude/region_chi2.py --state <path>``.

OUTPUT: region_chi2.txt next to the state file.
"""

import argparse
from pathlib import Path

import fit
import numpy as np
from mask_refit import FREE, MASKS, best_point, refit

EDGES = np.arange(3400.0, 9201.0, 200.0)
LINE_FRAC = 0.02


def components(model, obs, cue, theta):
    """Return the model spectrum, its continuum and its line spectrum at ``theta``."""
    spec = model.predict(theta, observations=obs, sps=cue)[0][0]
    cont = model._resolved_continuum.copy()
    return spec, cont, spec - cont


def main():
    """Run the two masks and write the binned table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    model, obs, cue = fit.process_state("free")
    labels = list(model.theta_labels())
    o = obs[0]
    z = float(np.atleast_1d(model.params["zred"])[0])
    rest = o.wavelength / (1 + z)
    mask0 = o.mask.copy()
    theta0 = best_point(args.state, model)
    i_n = labels.index("dust_index")
    lines = []
    for name in ("none", "both"):
        lo, hi = MASKS[name]
        o.mask = mask0 & (rest >= lo) & (rest <= hi)
        th_free, v_free = refit(model, theta0.copy(), FREE + ["dust_index"])
        start = theta0.copy()
        start[i_n] = 0.0
        th_0, v_0 = refit(model, start, FREE)
        s1, c1, l1 = components(model, obs, cue, th_free)
        s0, c0, _ = components(model, obs, cue, th_0)
        chi_free = ((o.flux - s1) / o.uncertainty) ** 2
        chi_0 = ((o.flux - s0) / o.uncertainty) ** 2
        is_line = np.abs(l1) > LINE_FRAC * np.abs(c1)
        lines += [
            f"mask {name}: n free {th_free[i_n]:.3f} (lnL+lnp {v_free:.2f}), n 0 (lnL+lnp {v_0:.2f}), "
            f"total d chi2 {-2 * (v_0 - v_free):.1f} (incl. prior)",
            f"{'rest bin':>11} {'npix c':>6} {'dchi2 c':>8} {'npix l':>6} {'dchi2 l':>8} {'cont0/free':>10}",
        ]
        for a, b in zip(EDGES[:-1], EDGES[1:], strict=True):
            sel = o.mask & (rest >= a) & (rest < b)
            if not sel.any():
                continue
            c, ln = sel & ~is_line, sel & is_line
            ratio = np.median(c0[c] / c1[c]) if c.any() else np.nan
            lines.append(
                f"{a:5.0f}-{b:5.0f} {int(c.sum()):6d} {np.sum(chi_0[c] - chi_free[c]):8.1f} "
                f"{int(ln.sum()):6d} {np.sum(chi_0[ln] - chi_free[ln]):8.1f} {ratio:10.4f}"
            )
        c, ln = o.mask & ~is_line, o.mask & is_line
        lines.append(
            f"{'all':>11} {int(c.sum()):6d} {np.sum(chi_0[c] - chi_free[c]):8.1f} "
            f"{int(ln.sum()):6d} {np.sum(chi_0[ln] - chi_free[ln]):8.1f}"
        )
        lines.append("")
        print("\n".join(lines[-40:]), flush=True)
    o.mask = mask0
    out = args.state.with_name("region_chi2.txt")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
