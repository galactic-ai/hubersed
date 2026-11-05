"""Corner plot of dust_ratio and the Cue gas parameters of the ER and HR split 45 fits of 37084.

QUESTION: is dust_ratio degenerate with the gas parameters (gas_logz, gas_logu, gas_lognH,
gas_logno, gas_logco) in the ER and HR posteriors? The Fisher matrix at the ER max-L point
(dust_reading.py) puts no gas parameter among its six largest partners.

INPUTS: the ER ``_pred.npz`` of predict.py (posterior points and log weights), and
results/2026-09-30_nautilus_c3k_broadline_split45/39627770174637084_broad_shared_split45_unmask_rest9000_seed0.h5
with its .json. SEED: none.

Run from the repository root with the C3K_ER build, as hr_point.py (the model is built only for
the prior edges). Command:
``PYTHONPATH=<site with the C3K_ER fsps>:experiments/2026-10-09_nautilus_c3k_er_broadline
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-09_nautilus_c3k_er_broadline/corner_dust_gas.py``.

OUTPUT: results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/corner_dust_gas.png and
corner_dust_gas.txt (weighted posterior correlations of dust_ratio with every parameter, per run).
Both runs have the same model and parameters but different data handling, see hr_point.py.
Fonts are large for slides. Dashed lines mark prior edges inside an axis.
"""

import importlib.util
import json

import corner
import matplotlib.pyplot as plt
import numpy as np
from fit import make_model
from nautilus import Sampler

from hubersed.paths import PATHS

TID = 39627770174637084
FB, SPLIT = "shared", 45.0
OUT = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline"
FIG = OUT / str(TID)
STEM = f"{TID}_er_broad_{FB}_split{SPLIT:g}_rest9000_seed0"
PARS = ["dust_ratio", "gas_logz", "gas_logu", "gas_lognH", "gas_logno", "gas_logco"]
COLORS = {"ER": "#b2182b", "HR": "#2166ac"}
HR_DIR = PATHS["RESULTS"] / "2026-09-30_nautilus_c3k_broadline_split45"
HR_STEM = f"{TID}_broad_shared_split45_unmask_rest9000_seed0"
HR_FIT = PATHS["ROOT"] / "experiments" / "2026-09-30_nautilus_c3k_broadline_split45" / "fit.py"


def wquantile(x, w, q):
    """Weighted quantile of samples x with weights w that sum to 1."""
    o = np.argsort(x)
    return np.interp(q, np.cumsum(w[o]), x[o])


def wcorr(points, w):
    """Weighted correlation matrix of the columns of points."""
    mean = w @ points
    d = points - mean
    cov = (d * w[:, None]).T @ d
    s = np.sqrt(np.diag(cov))
    return cov / np.outer(s, s)


def load_module(path, name):
    """Import a fit.py from another experiment under its own module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def er_posterior():
    """Return the ER posterior points, normalized weights and labels from the _pred.npz."""
    d = np.load(OUT / f"{STEM}_pred.npz")
    keep = d["log_w"] > d["log_w"].max() - 30
    w = np.exp(d["log_w"][keep] - d["log_w"][keep].max())
    return d["points"][keep], w / w.sum(), [str(x) for x in d["labels"]], float(d["z"])


def hr_posterior():
    """Return the HR posterior points, normalized weights and labels from its checkpoint.

    The fit gave nautilus the identity prior, so the checkpoint holds unit cube points. Points
    more than e^-30 below the heaviest one are dropped, as in corner_a.py.
    """
    f30 = load_module(HR_FIT, "fit_0930")
    cfg = json.loads((HR_DIR / f"{HR_STEM}.json").read_text())
    m = f30.make_model(cfg["z"], cfg["forbidden_broad"], cfg["sigma_split"])
    s = Sampler(
        lambda x: x,
        lambda x: 0.0,
        n_dim=m.ndim,
        n_live=1000,
        filepath=str(HR_DIR / f"{HR_STEM}.h5"),
        resume=True,
    )
    unit, log_w, _ = s.posterior()
    keep = log_w > log_w.max() - 30
    w = np.exp(log_w[keep] - log_w[keep].max())
    points = np.array([m.prior_transform(x) for x in unit[keep]])
    return points, w / w.sum(), list(m.theta_labels())


def main():
    p_er, w_er, labels, z = er_posterior()
    p_hr, w_hr, labels_hr = hr_posterior()
    assert labels_hr == labels
    runs = {"ER": (p_er, w_er), "HR": (p_hr, w_hr)}
    idx = [labels.index(p) for p in PARS]
    i = labels.index("dust_ratio")

    corr, lines = {}, []
    for name, (pts, w) in runs.items():
        corr[name] = wcorr(pts, w)
        c = corr[name]
        lines.append(f"{name}: {len(w)} points, N_eff {1 / np.sum(w**2):.0f}")
        lines.append(f"{name}: weighted posterior correlation of dust_ratio, largest first")
        order = [j for j in np.argsort(-np.abs(c[i])) if j != i][:12]
        lines += [f"   {labels[j]:18s} {c[i, j]:+.3f}" for j in order]
        lines.append(f"{name}: correlation matrix of the plotted parameters")
        lines.append(" " * 12 + "".join(f"{p[:10]:>11s}" for p in PARS))
        for a in idx:
            lines.append(f"{labels[a][:10]:12s}" + "".join(f"{c[a, b]:+11.3f}" for b in idx))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (FIG / "corner_dust_gas.txt").write_text(text + "\n")

    model = make_model(z, FB, SPLIT)
    rng = []
    for j in idx:
        lo = min(wquantile(pts[:, j], w, 0.001) for pts, w in runs.values())
        hi = max(wquantile(pts[:, j], w, 0.999) for pts, w in runs.values())
        p = (hi - lo) * 0.05 or 1e-3
        rng.append((lo - p, hi + p))
    k = len(PARS)
    fig = plt.figure(figsize=(2.6 * k, 2.6 * k))
    for name, (pts, w) in runs.items():
        corner.corner(
            pts[:, idx],
            weights=w,
            fig=fig,
            color=COLORS[name],
            labels=PARS,
            range=rng,
            plot_datapoints=False,
            plot_density=False,
            smooth=1.0,
            bins=30,
            levels=(0.393, 0.865),  # 1 and 2 sigma for a 2-D Gaussian
            max_n_ticks=3,
            labelpad=0.12,
            label_kwargs={"fontsize": 20},
            hist_kwargs={"density": True},
        )
    axes = np.array(fig.axes).reshape(k, k)
    for ax in fig.axes:
        ax.tick_params(labelsize=13)
    for j, lab in enumerate(PARS):
        for edge in np.ravel(model.config_dict[lab]["prior"].range):
            if axes[j, j].get_xlim()[0] <= edge <= axes[j, j].get_xlim()[1]:
                axes[j, j].axvline(edge, color="0.3", ls="--", lw=1.2)
        # weighted median and 16-84% of each run above the diagonal panel
        title = []
        for name, (pts, w) in runs.items():
            q = [wquantile(pts[:, idx[j]], w, v) for v in (0.16, 0.5, 0.84)]
            title.append(f"{name} {q[1]:.3f}$^{{+{q[2] - q[1]:.3f}}}_{{-{q[1] - q[0]:.3f}}}$")
        axes[j, j].set_title("\n".join(title), fontsize=13)
    handles = [plt.Line2D([], [], color=COLORS[n], lw=3) for n in runs]
    fig.legend(
        handles,
        ["ER (DESI resolution)", "HR split 45 (degraded to C3K_HR)"],
        loc="upper right",
        fontsize=20,
        frameon=False,
        bbox_to_anchor=(0.95, 0.9),
    )
    fig.suptitle(f"{TID}  dust_ratio and gas, ER and HR split 45 posteriors", fontsize=24, y=1.04)
    path = FIG / "corner_dust_gas.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
