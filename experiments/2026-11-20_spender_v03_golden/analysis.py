"""Compare spender outputs from the pinned version and the version merged with upstream v0.3.

QUESTION: do latents, rest-frame decodes and noise-flow log p stay the same after the merge?
HYPOTHESIS: identical, since encode and decode do not change and the flow change is a dtype cast.
INPUTS: results/2026-11-20_spender_v03_golden/golden_{pinned,merged}.npz (golden.py).
SEED: 0, CPU, one thread.
COMMAND: ``uv run python experiments/2026-11-20_spender_v03_golden/analysis.py``.
RESULT: latents, decodes and flow log p are bit-identical for 256 DESI and 256 mock spectra. The
full forward pass (interpolation) differs by at most 1.9e-6 (encoder) and 1.5e-5 (noise spender).
Assumptions: one spectrum encoder stands for all; both runs used torch 2.9.0 on one CPU thread.
FIGURES: none (analysis.txt).
"""

from pathlib import Path

import numpy as np

OUT = Path("results/2026-11-20_spender_v03_golden")
TOL = 1e-6  # pass threshold on the max absolute difference, except forward outputs


def main():
    """Print the max absolute and relative difference per array and whether it passes."""
    a = np.load(OUT / "golden_pinned.npz")
    b = np.load(OUT / "golden_merged.npz")
    lines = [
        f"pinned spender {a['spender_version']}, torch {a['torch']}",
        f"merged spender {b['spender_version']}, torch {b['torch']}",
        f"{'array':55s} {'shape':>12s} {'max abs':>10s} {'max rel':>10s} result",
    ]
    keys = sorted(k for k in a.files if "/" in k)
    assert keys == sorted(k for k in b.files if "/" in k), "the two runs saved different arrays"
    for k in keys:
        x, y = a[k], b[k]
        d = np.abs(x - y)
        rel = d / np.maximum(np.abs(x), 1e-30)
        verdict = "report" if "/forward" in k else ("pass" if d.max() <= TOL else "FAIL")
        shape = "x".join(map(str, x.shape))
        lines.append(f"{k:55s} {shape:>12s} {d.max():10.3e} {rel.max():10.3e} {verdict}")
    text = "\n".join(lines)
    (OUT / "analysis.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
