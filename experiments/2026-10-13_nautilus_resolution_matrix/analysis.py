"""Compare the posteriors of three nautilus fits of 39627770174637084.

QUESTION: how do the posteriors move from HR split 45 (09-30) to ER (10-09) to rmatrix (this
folder: target R, corrected ivar, missing-line mask)? Same 29 parameters and priors. Pixels differ:
5621, 5597, 5530.

HYPOTHESIS: from the MAP runs, rmatrix has wider lines than ER (eline_sigma_forb ~22 km/s,
eline_sigma ~25 km/s), lower dust_ratio (~0.37 vs 0.46), sigma_smooth ~21 km/s. SFH barely moves.

INPUTS: the HR and ER _pred.npz, and the rmatrix _pred.npz of predict.py (or its .h5 if the npz is
missing). Points more than e^-30 below the heaviest are dropped. SEED: none.

COMMAND: from the repository root on the Mac
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_nautilus_resolution_matrix/analysis.py``.

RESULT: not run yet.

FIGURES: compare_posteriors.txt (median, 16-84%, shift from ER in ER half widths) and
corner_{lines,dust_gas,stars}.png in results/2026-10-13_nautilus_resolution_matrix/.
"""

# %%
import json

import corner
import matplotlib.pyplot as plt
import numpy as np
from fit import stem
from model_setup import TID, make_model, make_params
from nautilus import Sampler

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-13_nautilus_resolution_matrix"
HR_PRED = (
    PATHS["RESULTS"]
    / "2026-09-30_nautilus_c3k_broadline_split45"
    / f"{TID}_broad_shared_split45_unmask_rest9000_seed0_pred.npz"
)
ER_PRED = (
    PATHS["RESULTS"]
    / "2026-10-09_nautilus_c3k_er_broadline"
    / f"{TID}_er_broad_shared_split45_rest9000_seed0_pred.npz"
)
RM_STEM = stem("rmatrix", "mask")
COLORS = {"HR": "#2166ac", "ER": "#b2182b", "rmatrix": "#1b7837"}
NAMES = {
    "HR": "HR split 45 (degraded to C3K_HR)",
    "ER": "ER (DESI resolution, median LSF)",
    "rmatrix": "ER, target R, corrected ivar, line mask",
}
GROUPS = {
    "lines": [
        "eline_sigma",
        "eline_sigma_forb",
        "eline_fbroad",
        "eline_sigma_broad",
        "eline_vbroad",
    ],
    "dust_gas": ["dust2", "dust_ratio", "dust_index", "gas_logz", "gas_logu", "gas_logno"],
    "stars": ["logzsol", "logmass", "sigma_smooth", "logsfr_ratios_1", "logsfr_ratios_2"],
}


def wquantile(x, w, q):
    """Weighted quantile of samples x with weights w that sum to 1."""
    o = np.argsort(x)
    return np.interp(q, np.cumsum(w[o]), x[o])


def keep_heavy(points, log_w):
    """Drop points more than e^-30 below the heaviest; return points and normalized weights."""
    keep = log_w > log_w.max() - 30
    w = np.exp(log_w[keep] - log_w[keep].max())
    return points[keep], w / w.sum()


def from_pred(path):
    """Return the posterior points, weights and labels of a ``_pred.npz``."""
    d = np.load(path)
    pts, w = keep_heavy(d["points"], d["log_w"])
    return pts, w, [str(x) for x in d["labels"]]


def from_checkpoint(model, name):
    """Return the posterior points, weights and labels of a checkpoint of fit.py.

    The fit gave nautilus the identity prior, so the checkpoint holds unit-cube points. They are
    read with the identity and only the kept ones go through ``model.prior_transform``.
    """
    s = Sampler(
        lambda x: x,
        lambda x: 0.0,
        n_dim=model.ndim,
        n_live=1000,
        filepath=str(OUT / f"{name}.h5"),
        resume=True,
    )
    unit, log_w, _ = s.posterior()
    unit, w = keep_heavy(unit, log_w)
    pts = np.array([model.prior_transform(x) for x in unit])
    return pts, w, list(model.theta_labels())


# %% load the three posteriors
runs = {}
pts, w, labels = from_pred(HR_PRED)
runs["HR"] = (pts, w)
pts, w, labels_er = from_pred(ER_PRED)
assert labels_er == labels
runs["ER"] = (pts, w)
z = float(np.load(ER_PRED)["z"])
model = make_model(make_params(z), "rmatrix")
assert list(model.theta_labels()) == labels
if (OUT / f"{RM_STEM}_pred.npz").exists():
    pts, w, labels_rm = from_pred(OUT / f"{RM_STEM}_pred.npz")
    assert labels_rm == labels
    runs["rmatrix"] = (pts, w)
elif (OUT / f"{RM_STEM}.h5").exists():
    cfg = json.loads((OUT / f"{RM_STEM}.json").read_text())
    assert cfg["lsf"] == "rmatrix" and cfg["mask"] == "mask" and cfg["labels"] == labels
    pts, w, _ = from_checkpoint(model, RM_STEM)
    runs["rmatrix"] = (pts, w)
else:
    print(f"{RM_STEM}_pred.npz and .h5 not found, comparing HR and ER only")

# %% table of every parameter
lines = [f"{TID} posteriors: weighted median [16%, 84%]; shift from ER in ER half widths"]
for name, (_, w) in runs.items():
    lines.append(f"{name}: {NAMES[name]}, {len(w)} points, N_eff {1 / np.sum(w**2):.0f}")
lines.append("")
lines.append(
    f"{'parameter':18s}"
    + "".join(f"{n:>30s}" for n in runs)
    + "".join(f"{'d' + n:>10s}" for n in runs if n != "ER")
)
for j, lab in enumerate(labels):
    q = {n: [wquantile(p[:, j], w, v) for v in (0.16, 0.5, 0.84)] for n, (p, w) in runs.items()}
    half = 0.5 * (q["ER"][2] - q["ER"][0])
    row = f"{lab:18s}" + "".join(
        f"{f'{v[1]:.4g} [{v[0]:.4g}, {v[2]:.4g}]':>30s}" for v in q.values()
    )
    row += "".join(f"{(q[n][1] - q['ER'][1]) / half:+10.2f}" for n in runs if n != "ER")
    lines.append(row)
text = "\n".join(lines)
print(text)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "compare_posteriors.txt").write_text(text + "\n")

# %% corner plots, one per parameter group
for group, pars in GROUPS.items():
    idx = [labels.index(p) for p in pars]
    rng = []
    for j in idx:
        lo = min(wquantile(p[:, j], w, 0.001) for p, w in runs.values())
        hi = max(wquantile(p[:, j], w, 0.999) for p, w in runs.values())
        pad = (hi - lo) * 0.05 or 1e-3
        rng.append((lo - pad, hi + pad))
    k = len(pars)
    fig = plt.figure(figsize=(2.6 * k, 2.6 * k))
    for name, (p, w) in runs.items():
        corner.corner(
            p[:, idx],
            weights=w,
            fig=fig,
            color=COLORS[name],
            labels=pars,
            range=rng,
            plot_datapoints=False,
            plot_density=False,
            smooth=1.0,
            bins=30,
            levels=(0.393, 0.865),  # 1 and 2 sigma for a 2-D Gaussian
            max_n_ticks=3,
            labelpad=0.12,
            label_kwargs={"fontsize": 18},
            hist_kwargs={"density": True},
        )
    axes = np.array(fig.axes).reshape(k, k)
    for ax in fig.axes:
        ax.tick_params(labelsize=12)
    for j, lab in enumerate(pars):
        # dashed lines at prior edges inside the axis; vector parameters such as
        # logsfr_ratios_1 have no entry of their own and get none
        if lab not in model.config_dict:
            continue
        for edge in np.ravel(model.config_dict[lab]["prior"].range):
            if axes[j, j].get_xlim()[0] <= edge <= axes[j, j].get_xlim()[1]:
                axes[j, j].axvline(edge, color="0.3", ls="--", lw=1.2)
    handles = [plt.Line2D([], [], color=COLORS[n], lw=3) for n in runs]
    fig.legend(
        handles,
        [NAMES[n] for n in runs],
        loc="upper right",
        fontsize=16,
        frameon=False,
        bbox_to_anchor=(0.97, 0.92),
    )
    fig.suptitle(f"{TID}  {group}", fontsize=22, y=1.02)
    path = OUT / f"corner_{group}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
