"""Posterior correlations of the line profile parameters with the gas and dust parameters.

QUESTION: in the converged C3K_ER broadline fit of 39627770174637084, do the line profile parameters
(eline_fbroad, eline_sigma_broad, eline_vbroad, eline_sigma, eline_sigma_forb) correlate with the
Cue gas parameters (gas_logz, gas_logu, gas_lognH, gas_logno, gas_logco) and the dust parameters?
HYPOTHESIS: the profile only moves flux within each line, so it couples to the gas parameters
only through blended lines (Halpha + [NII], [SII], [OII]). Correlations should be weak.
INPUTS: results/2026-10-09_nautilus_c3k_er_broadline/<stem>_pred.npz from predict.py, with
<stem> = 39627770174637084_er_broad_shared_split45_rest9000_seed0. Points and log_w are physical
parameter values and nautilus log weights.
SEED: none.
COMMAND: from the repository root,
uv run python experiments/2026-10-09_nautilus_c3k_er_broadline/line_gas_corr.py
RESULT: see line_gas_corr.txt. Two parameters sit on prior edges (eline_sigma_forb at its 10 km/s
floor, gas_logco near its log 5.4 ceiling), so their correlations are truncated.
FIGURES: results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/line_gas_corr.txt and
line_gas_corr.png.

Pearson is the weighted correlation of each pair. Partial is the correlation of each pair with all
other 27 parameters held fixed, from the inverse of the weighted covariance of all 29. Both are
linear measures and miss curved degeneracies.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39627770174637084
DIR = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline"
STEM = f"{TID}_er_broad_shared_split45_rest9000_seed0"
OUT = DIR / str(TID)
LINE = ["eline_fbroad", "eline_sigma_broad", "eline_vbroad", "eline_sigma", "eline_sigma_forb"]
GAS = ["gas_logz", "gas_logu", "gas_lognH", "gas_logno", "gas_logco"]
OTHER = ["dust2", "dust_ratio", "dust_index", "sigma_smooth", "logzsol"]

d = np.load(DIR / f"{STEM}_pred.npz")
labels = list(d["labels"])
w = np.exp(d["log_w"] - d["log_w"].max())
w /= w.sum()
x = d["points"]
print(f"{len(w)} points, N_eff {1 / np.sum(w**2):.0f}")

# %% weighted covariance, Pearson and partial correlations of all 29 parameters
mean = w @ x
dx = x - mean
cov = (dx * w[:, None]).T @ dx
sd = np.sqrt(np.diag(cov))
pearson = cov / np.outer(sd, sd)
prec = np.linalg.inv(cov)
partial = -prec / np.sqrt(np.outer(np.diag(prec), np.diag(prec)))
np.fill_diagonal(partial, 1.0)

# %% tables
rows = LINE
cols = GAS + OTHER
idx = {k: labels.index(k) for k in rows + cols}
lines = [
    f"Posterior correlations, {STEM}, N_eff {1 / np.sum(w**2):.0f}.",
    "eline_sigma_forb is at its 10 km/s floor and gas_logco near its ceiling: truncated.",
    "",
]
for name, mat in [("Pearson (marginal)", pearson), ("Partial (all other 27 fixed)", partial)]:
    lines.append(name)
    lines.append(f"{'':>18s}" + "".join(f"{c:>13s}" for c in cols))
    for r in rows:
        vals = "".join(f"{mat[idx[r], idx[c]]:13.2f}" for c in cols)
        lines.append(f"{r:>18s}{vals}")
    lines.append("")
lines.append("Pearson among the line parameters")
lines.append(f"{'':>18s}" + "".join(f"{c:>18s}" for c in rows))
for r in rows:
    lines.append(f"{r:>18s}" + "".join(f"{pearson[idx[r], idx[c]]:18.2f}" for c in rows))
lines.append("")
lines.append("Strongest Pearson partner of each line parameter among all 28 others")
for r in rows:
    c = pearson[idx[r]].copy()
    c[idx[r]] = 0
    order = np.argsort(-np.abs(c))[:4]
    lines.append(f"{r:>18s}  " + ", ".join(f"{labels[j]} {c[j]:+.2f}" for j in order))
text = "\n".join(lines)
print(text)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "line_gas_corr.txt").write_text(text + "\n")

# %% figure: Pearson and partial side by side
use_apj_style()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for ax, (name, mat) in zip(axes, [("Pearson", pearson), ("Partial", partial)], strict=True):
    sub = mat[np.ix_([idx[r] for r in rows], [idx[c] for c in cols])]
    im = ax.imshow(sub, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{sub[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(name)
fig.colorbar(im, ax=axes, shrink=0.8)
fig.savefig(OUT / "line_gas_corr.png", dpi=200)
print(f"wrote {OUT / 'line_gas_corr.txt'} and line_gas_corr.png")
