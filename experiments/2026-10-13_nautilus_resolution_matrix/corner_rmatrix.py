"""Corner plots of the rmatrix fit from its _pred.npz, physical and gas parameters separately.

Writes corner_rmatrix_phys.png and corner_rmatrix_gas.png in
results/2026-10-13_nautilus_resolution_matrix/.

COMMAND: ``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_nautilus_resolution_matrix/corner_rmatrix.py`` from the repository root.
"""

import corner
import matplotlib.pyplot as plt
import numpy as np
from fit import OUT, stem

d = np.load(OUT / f"{stem('rmatrix', 'mask')}_pred.npz")
labels = [str(x) for x in d["labels"]]
keep = d["log_w"] > d["log_w"].max() - 30
w = np.exp(d["log_w"][keep] - d["log_w"][keep].max())
w /= w.sum()
pts = d["points"][keep]
n_eff = float(d["n_eff"])

gas = [k for k in labels if k.startswith(("gas_", "eline_"))]
groups = {"phys": [k for k in labels if k not in gas], "gas": gas}
for name, cols in groups.items():
    idx = [labels.index(k) for k in cols]
    fig = corner.corner(
        pts[:, idx],
        weights=w,
        labels=cols,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3g",
        smooth=1.0,
        plot_datapoints=False,
        levels=(0.39, 0.86),
        color="#1b7837",
    )
    fig.suptitle(f"rmatrix fit, N_eff {n_eff:.0f} (snapshot, not converged)", y=1.01)
    fig.savefig(OUT / f"corner_rmatrix_{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote corner_rmatrix_{name}.png, {len(cols)} parameters")
