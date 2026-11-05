"""Corner plots and the SFH of run A (37084, C3K_HR, forbidden broad shared, split 45 km/s), for slides.

QUESTION: what do the posteriors of run A look like without the 09-26 run drawn over them?
INPUTS: results/2026-09-30_nautilus_c3k_broadline_split45/39627770174637084_broad_shared_split45_unmask_rest9000_seed0.h5
and the .json next to it.
SEED: 0, for the 2000 weighted posterior draws of the SFH.
COMMAND: uv run python experiments/2026-09-30_nautilus_c3k_broadline_split45/corner_a.py
FIGURES: corner_phys_A_39627770174637084.png, corner_sfh_A_39627770174637084.png,
corner_lines_A_39627770174637084.png and sfh_A_39627770174637084.png in the same results folder.

The parameter groups are those of the 2026-10-10 C/O tie analysis. Fonts are large so the figures
can go on a slide. Dashed lines mark prior edges inside an axis.
"""

import importlib.util
import json
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np
from nautilus import Sampler

from hubersed.fitting.map_fits import sfh_from_theta
from hubersed.paths import PATHS
from hubersed.plotting.sfh import sfh_figure

HERE = Path(__file__).resolve().parent
OUT = PATHS["RESULTS"] / "2026-09-30_nautilus_c3k_broadline_split45"
TID = 39627770174637084
STEM = f"{TID}_broad_shared_split45_unmask_rest9000_seed0"
COLOR = "C3"
GEN = np.random.default_rng(0)
LINE_PARS = [
    "eline_sigma",
    "eline_sigma_forb",
    "eline_fbroad",
    "eline_sigma_broad",
    "eline_vbroad",
    "gas_logz",
    "gas_logu",
    "gas_lognH",
    "gas_logno",
    "gas_logco",
]


def load_module(path, name):
    """Import a fit.py under its own module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wquantile(x, w, q):
    """Weighted quantile of samples x with weights w that sum to 1."""
    o = np.argsort(x)
    return np.interp(q, np.cumsum(w[o]), x[o])


def read_posterior(path, model):
    """Return the physical posterior points and their normalized weights from a checkpoint.

    The fit gave nautilus the identity prior, so the checkpoint holds unit cube points. Points
    more than e^-30 below the heaviest one are dropped, as in the predict scripts.
    """
    s = Sampler(
        lambda x: x, lambda x: 0.0, n_dim=model.ndim, n_live=1000, filepath=str(path), resume=True
    )
    unit, log_w, _ = s.posterior()
    keep = log_w > log_w.max() - 30
    w = np.exp(log_w[keep] - log_w[keep].max())
    points = np.array([model.prior_transform(x) for x in unit[keep]])
    print(f"{len(unit)} points, {keep.sum()} kept, N_eff {s.n_eff:.0f}, log Z {s.log_z:.2f}")
    return points, w / w.sum()


fit = load_module(HERE / "fit.py", "fit_0930")
cfg = json.loads((OUT / f"{STEM}.json").read_text())
model = fit.make_model(cfg["z"], cfg["forbidden_broad"], cfg["sigma_split"])
labels = list(model.theta_labels())
assert labels == cfg["labels"]
points, w = read_posterior(OUT / f"{STEM}.h5", model)

groups = {
    "phys": [lab for lab in labels if not lab.startswith("logsfr_ratios") and lab not in LINE_PARS],
    "sfh": [lab for lab in labels if lab.startswith("logsfr_ratios")],
    "lines": [lab for lab in labels if lab in LINE_PARS],
}
for tag, labs in groups.items():
    cols = np.column_stack([points[:, labels.index(lab)] for lab in labs])
    rng = []
    for c in cols.T:
        lo, hi = wquantile(c, w, 0.001), wquantile(c, w, 0.999)
        p = (hi - lo) * 0.05 or 1e-3
        rng.append((lo - p, hi + p))
    k = len(labs)
    fig = plt.figure(figsize=(2.6 * k, 2.6 * k))
    corner.corner(
        cols,
        weights=w,
        fig=fig,
        color=COLOR,
        labels=labs,
        range=rng,
        plot_datapoints=False,
        plot_density=False,
        smooth=1.0,
        bins=30,
        levels=(0.393, 0.865),  # 1 and 2 sigma for a 2-D Gaussian
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        max_n_ticks=3,
        labelpad=0.12,
        label_kwargs={"fontsize": 20},
        title_kwargs={"fontsize": 15},
        hist_kwargs={"density": True},
    )
    axes = np.array(fig.axes).reshape(k, k)
    for ax in fig.axes:
        ax.tick_params(labelsize=13)
    for j, lab in enumerate(labs):
        # the logsfr_ratios_N labels are entries of one vector parameter and get no edges
        if lab not in model.config_dict:
            continue
        for edge in np.ravel(model.config_dict[lab]["prior"].range):
            if axes[j, j].get_xlim()[0] <= edge <= axes[j, j].get_xlim()[1]:
                axes[j, j].axvline(edge, color="0.3", ls="--", lw=1.2)
    fig.suptitle(f"{TID}  C3K_HR, split 45 km/s, forbidden broad shared: {tag}", fontsize=24)
    fig.savefig(OUT / f"corner_{tag}_A_{TID}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote corner_{tag}_A_{TID}.png")

# SFH: median and 16-84% per bin from 2000 weighted draws, as in the 2026-10-09 ER analysis
draws = [sfh_from_theta(model, points[i]) for i in GEN.choice(len(w), 2000, p=w)]
edges = draws[0]["edges_gyr"]
ssfr = np.percentile([d["ssfr"] for d in draws], [16, 50, 84], axis=0)
cmf = np.percentile([d["cmf"] for d in draws], [16, 50, 84], axis=0)
fig, ax = sfh_figure(
    edges,
    ssfr[1],
    cmf[1],
    figsize=(10, 8),
    ssfr_kwargs={"label": r"median, 16--84\%", "ssfr_kw": {"color": COLOR, "lw": 2.5}},
    cmf_kwargs={"color": COLOR, "lw": 2.5},
)
# the SFH style renders text with LaTeX, so the percent sign is escaped in the label above
for a, q in zip(ax, (ssfr, cmf), strict=True):
    a.stairs(q[2], edges, baseline=q[0], fill=True, color=COLOR, alpha=0.25, lw=0)
    a.tick_params(labelsize=16)
    a.xaxis.label.set_size(20)
    a.yaxis.label.set_size(20)
ax[0].legend(frameon=True, fontsize=16, loc="lower left")
ax[0].set_title(f"{TID}  C3K_HR, split 45 km/s, forbidden broad shared", fontsize=18)
fig.savefig(OUT / f"sfh_A_{TID}.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote sfh_A_{TID}.png")
