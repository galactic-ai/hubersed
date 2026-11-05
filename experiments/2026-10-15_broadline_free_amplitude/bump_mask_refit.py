"""Does the grey dust slope of 39627770174637084 survive masking the 4590-4720 A excess?

QUESTION: he2_window.py finds a 2-8% excess over rest ~4590-4700 A that carries +10.0 of the +21.7
no-mask cost of dust_index = 0. With it masked, how far does n move and what do n = 0 and n = -0.7
cost?

Repeats mask_refit.py for the masks none and both, with rest 4590-4720 A removed too. SEED: none.

RESULT: 2026-10-17, bump_mask_refit.txt. Inconclusive. No mask: n free 0.365 to 0.324, cost of n = 0
+21.7 to +12.0, of n = -0.7 +131 to +115. 3800-7600 A: n free 0.394 to 0.400, n = 0 +35.1 to +23.5,
n = -0.7 +143 to +130. The excess explains about half of the no-mask cost and the rest is the
continuum bow. These local refits use pipeline errors, which understate the noise on 50-200 A
scales, so this spectrum does not decide between a grey and a Calzetti-like slope.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=experiments/2026-10-15_broadline_free_amplitude:~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-15_broadline_free_amplitude/bump_mask_refit.py --state <path>``.

OUTPUT: bump_mask_refit.txt next to the state file.
"""

import argparse
import time
from pathlib import Path

import fit
import numpy as np
from mask_refit import FREE, MASKS, SLOPES, best_point, refit

BUMP = (4590.0, 4720.0)


def main():
    """Run both masks and every slope and write the table."""
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
    keep = (rest < BUMP[0]) | (rest > BUMP[1])

    lines = [f"rest {BUMP[0]:.0f}-{BUMP[1]:.0f} A masked; start dust_index {theta0[i_n]:.3f}"]
    lines.append(f"{'mask':>10} {'n':>6} {'pixels':>6} {'lnL+lnp':>11} {'d chi2':>7} {'n fit':>6} "
                 f"{'dust2':>6} {'ratio':>6} {'logM':>6} {'logZ*':>6}")  # fmt: skip
    out = args.state.with_name("bump_mask_refit.txt")
    for name in ("none", "both"):
        lo, hi = MASKS[name]
        o.mask = mask0 & (rest >= lo) & (rest <= hi) & keep
        ref = None
        for n in SLOPES:
            t = time.time()
            start = theta0.copy()
            free = FREE + ["dust_index"] if n is None else FREE
            if n is not None:
                start[i_n] = n
            th, val = refit(model, start, free)
            ref = val if n is None else ref
            g = {k: th[labels.index(k)] for k in ("dust2", "dust_ratio", "logmass", "logzsol")}
            lines.append(
                f"{name:>10} {'free' if n is None else n:>6} {int(o.mask.sum()):6d} {val:11.2f} "
                f"{-2 * (val - ref):7.1f} {th[i_n]:6.3f} {g['dust2']:6.3f} {g['dust_ratio']:6.3f} "
                f"{g['logmass']:6.3f} {g['logzsol']:6.3f}   ({time.time() - t:.0f} s)"
            )
            print(lines[-1], flush=True)
            out.write_text("\n".join(lines) + "\n")
    o.mask = mask0
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
