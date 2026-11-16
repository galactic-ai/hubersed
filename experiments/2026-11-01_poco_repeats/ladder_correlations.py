"""Weighted posterior correlations of selected parameter pairs in each ladder run of one target.

Weighted Pearson r of the ladder_corner.py posteriors, and the weight share with sigma_dyn < 0.05.
COMMAND: ``uv run --no-sync python experiments/2026-11-01_poco_repeats/ladder_correlations.py --tid <TID>``.
OUTPUT: results/2026-11-01_poco_repeats/<TID>/ladder_correlations.txt.
"""

# %%
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAIRS = [
    ("dust2", "logzsol"), ("dust2", "afe"), ("logzsol", "afe"), ("dust2", "dust_index"),
    ("dust_index", "logzsol"), ("dust2", "logsfr_ratios_9"), ("logzsol", "logsfr_ratios_9"),
    ("afe", "logsfr_ratios_9"), ("sigma_dyn", "logsfr_ratios_1"), ("sigma_dyn", "logsfr_ratios_2"),
    ("sigma_dyn", "logsfr_ratios_3"), ("logmass", "dust2"), ("logmass", "logzsol"),
    ("logmass", "logsfr_ratios_9"), ("sigma_smooth", "logzsol"),
]  # fmt: skip


def load(name, path):
    """Load the file at ``path`` as module ``name``."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def wcorr(x, y, w):
    """Return the weighted Pearson correlation of x and y."""
    mx, my = np.sum(w * x), np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    return float(cov / np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2)))


def main(tid):
    """Write the correlation table of ``tid``."""
    lc = load("ladder_corner", HERE / "ladder_corner.py")
    _, rr = lc.runs(tid)
    short = ["MILES", "HR", "HR+afe", "DR1", "recoadd"]
    lines = [f"TARGETID {tid}: weighted Pearson r of posterior samples",
             f"{'pair':34s} " + " ".join(f"{s:>8s}" for s in short)]  # fmt: skip
    for a, b in PAIRS:
        cells = []
        for _, _, _, cols, w, _ in rr:
            cells.append(
                f"{wcorr(cols[a], cols[b], w):+8.2f}" if a in cols and b in cols else f"{'-':>8s}"
            )
        lines.append(f"{a + ' - ' + b:34s} " + " ".join(cells))
    lines.append("")
    lines.append(f"{'weight with sigma_dyn < 0.05':34s} "
                 + " ".join(f"{np.sum(w[cols['sigma_dyn'] < 0.05]):8.2f}" for _, _, _, cols, w, _ in rr))  # fmt: skip
    out = Path(lc.RES) / str(tid) / "ladder_correlations.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tid", type=int, required=True)
    main(ap.parse_args().tid)
