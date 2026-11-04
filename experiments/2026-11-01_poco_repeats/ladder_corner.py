"""Overlaid pocoMC posteriors of one target's model ladder steps, in the four corners of analysis.py.

The runs are the ladder steps with a pocoMC posterior: MILES, C3K HR, C3K HR + afe (fit_ladder.py)
and C3K ER + afe + R matrix on the DR1 coadd (fit.py), and on the recoadd once it exists. Each
step's model is rebuilt without FSPS for its prior transform: 2026-10-08_continuum_outlier/fit.py
make_model for the first three, the fit.py model for the R-matrix steps. A_V, E(B-V), quantiles
and prior ranges are those of analysis.py. A parameter a run does not have (afe in MILES and C3K
HR, where afe is absent or fixed at 0) is left out of that run. With afe free logzsol is [Fe/H],
without it [Z/H] (10-08 fit.py make_model), so logzsol differs in meaning across the runs.
Contours at 39.3 and 86.5 per cent, unfilled, one colour per run as ladder_plot.py; dashed lines
mark prior bounds inside the plotted range.
COMMAND: ``uv run --no-sync python experiments/2026-11-01_poco_repeats/ladder_corner.py --tid <TARGETID>``.
OUTPUT: results/2026-11-01_poco_repeats/<TARGETID>/ladder_corner.png, ladder_corner_stars_dust.png,
ladder_corner_sfh.png, ladder_corner_all_ssfr.png, and ladder_posterior.txt (16, 50 and 84 per
cent quantiles of every parameter of every run).
"""

# %%
import argparse
import importlib.util
import sys
from pathlib import Path

import corner
import dill
import matplotlib.pyplot as plt
import numpy as np
import pocomc as pc
from matplotlib.lines import Line2D

from hubersed.fitting.map_fits import sfh_from_theta
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

HERE = Path(__file__).resolve().parent
RES = PATHS["RESULTS"] / HERE.name
SETS = {
    "ladder_corner": ["dust2", "dust_index", "A_V", "E(B-V)", "logzsol", "afe"],
    "ladder_corner_stars_dust": ["logzsol", "afe", "logmass", "dust2", "dust_ratio", "dust_index",
                                 "sigma_smooth", "sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn"],
    "ladder_corner_sfh": [f"logsfr_ratios_{i}" for i in range(1, 10)],
}  # fmt: skip


def load(name, path):
    """Load the file at ``path`` as module ``name``."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def runs(tid):
    """Return (label, colour, model, posterior columns, weights, thetas) of each finished run."""
    ana = load("poco_analysis", HERE / "analysis.py")
    f08 = load("fit_1008", HERE.parent / "2026-10-08_continuum_outlier" / "fit.py")
    poco = load("quiescent_fit", HERE / "fit.py").load_poco(tid)
    d = RES / str(tid)
    z = float(np.load(d / f"{tid}_poco_maxl.npz")["z"])
    spec = (
        ("MILES", "#b2182b", d / "miles", lambda: f08.make_model(z, False, "none")),
        ("C3K HR (degraded)", "#2166ac", d / "c3k_hr", lambda: f08.make_model(z, False, "zero")),
        ("C3K HR + afe (degraded)", "#e66101", d / "c3k_hr_afe",
         lambda: f08.make_model(z, False, "free")),
        ("C3K ER + afe + R matrix, DR1 coadd", "#762a83", d, lambda: poco.make_model(z)),
        ("C3K ER + afe + R matrix, recoadd", "#1b9e77", d / "recoadd", lambda: poco.make_model(z)),
    )  # fmt: skip
    out = []
    for label, colour, folder, make in spec:
        path = folder / "poco_neff1024" / "pmc_final.state"
        if not path.exists():
            print(f"{label}: no posterior yet")
            continue
        model = make()
        s = pc.Sampler.__new__(pc.Sampler)
        with open(path, "rb") as f:
            s.__dict__.update(dill.load(f))
        usamp, w, _, _ = s.posterior()
        assert usamp.shape[1] == model.ndim, f"{label}: {usamp.shape[1]} != {model.ndim} free"
        theta = np.array([model.prior_transform(x) for x in usamp])
        w = w / w.sum()
        out.append((label, colour, model, ana.columns(model, theta), w, theta))
        print(f"{label}: {len(w)} samples, ESS {1 / np.sum(w**2):.0f}")
    return ana, out


def prior_range(model, name):
    """Return the prior bounds of a free parameter, or None for a derived one."""
    if name in model.config_dict and model.config_dict[name].get("isfree"):
        lo, hi = np.ravel(model.config_dict[name]["prior"].range)[:2]
        return float(lo), float(hi)
    return None


def overlay(ana, rr, names, path, inset=False, tid=None):
    """Draw one corner of ``names`` with every run of ``rr`` overlaid and save it to ``path``."""
    k = len(names)
    ranges = []
    for p in names:
        lo, hi = np.inf, -np.inf
        for _, _, _, cols, w, _ in rr:
            if p in cols:
                a, b = ana.wquantile(cols[p], w, np.array([0.001, 0.999]))
                lo, hi = min(lo, a), max(hi, b)
        pad = (hi - lo) * 0.05 or 1e-3
        ranges.append((lo - pad, hi + pad))
    # drawn panel by panel with corner.hist2d, so a run that lacks a parameter skips its panels
    fig, axes = plt.subplots(k, k, figsize=(2.8 * k, 2.8 * k),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.05})  # fmt: skip
    axes = np.atleast_2d(axes)
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.set_axis_off()
                continue
            for _, colour, _, cols, w, _ in rr:
                if names[j] not in cols or (i != j and names[i] not in cols):
                    continue
                if i == j:
                    ax.hist(cols[names[j]], bins=30, range=ranges[j], weights=w, density=True,
                            histtype="step", color=colour, lw=2)  # fmt: skip
                else:
                    corner.hist2d(cols[names[j]], cols[names[i]], ax=ax, weights=w,
                                  range=[ranges[j], ranges[i]], color=colour, smooth=1.0, bins=30,
                                  levels=(0.393, 0.865), plot_datapoints=False, plot_density=False,
                                  fill_contours=False, contour_kwargs={"linewidths": 2})  # fmt: skip
            ax.set_xlim(ranges[j])
            if i != j:
                ax.set_ylim(ranges[i])
            else:
                ax.set_yticks([])
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            if i < k - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(names[j], fontsize=34)
            if j > 0 or i == 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(names[i], fontsize=34)
    for j, p in enumerate(names):
        pr = prior_range(rr[0][2], p) or next((r for r in (prior_range(m, p) for _, _, m, *_ in rr)
                                               if r), None)  # fmt: skip
        for e in pr or ():
            if ranges[j][0] <= e <= ranges[j][1]:
                axes[j, j].axvline(e, color="k", ls="--", lw=2)
    for ax in axes.ravel():
        ax.tick_params(labelsize=22)
    for ax in axes[-1]:
        ax.xaxis.label.set_rotation(35)
        ax.xaxis.label.set_ha("right")
        ax.xaxis.set_label_coords(0.5, -0.35)
    for ax in axes[1:, 0]:
        ax.yaxis.label.set_rotation(55)
        ax.yaxis.label.set_va("top")
        ax.yaxis.set_label_coords(-0.45, 0.5)
    handles = [Line2D([], [], color=c, lw=4) for _, c, *_ in rr]
    fig.legend(handles, [lab for lab, *_ in rr], loc="upper right", fontsize=34, frameon=False,
               bbox_to_anchor=(0.98, 0.98) if not inset else (0.98, 0.60))  # fmt: skip
    if inset:
        ax = fig.add_axes([0.53, 0.66, 0.44, 0.31])
        for _, colour, model, _, w, theta in rr:
            r = np.random.default_rng(0)
            sfhs = [sfh_from_theta(model, theta[i]) for i in r.choice(len(w), size=2000, p=w)]
            edges = sfhs[0]["edges_gyr"]
            lo16, med50, hi84 = np.percentile([sf["ssfr"] for sf in sfhs], [16, 50, 84], axis=0)
            ax.stairs(hi84, edges, baseline=lo16, fill=True, color=colour, alpha=0.2, lw=0)
            ax.stairs(med50, edges, color=colour, lw=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("lookback time [Gyr]", fontsize=44)
        ax.set_ylabel(r"sSFR [yr$^{-1}$]", fontsize=44)
        ax.tick_params(labelsize=34)
        ax.set_title(f"{tid}, median and 16--84\\%", fontsize=40)
    fig.savefig(path, dpi=60, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def table(ana, rr, path):
    """Write the 16, 50 and 84 per cent quantiles of every parameter of every run to ``path``."""
    names = []
    for _, _, _, cols, _, _ in rr:
        names += [p for p in cols if p not in names]
    lines = [f"{'parameter':>16} " + " ".join(f"{lab[:28]:>30}" for lab, *_ in rr)]
    for p in names:
        cells = []
        for _, _, _, cols, w, _ in rr:
            if p in cols:
                a, m, b = ana.wquantile(cols[p], w, np.array([0.16, 0.5, 0.84]))
                cells.append(f"{m:10.4f} -{m - a:8.4f} +{b - m:8.4f}")
            else:
                cells.append(f"{'-':>30}")
        lines.append(f"{p:>16} " + " ".join(cells))
    lines += ["", "median -(median - q16) +(q84 - median), weighted pocoMC samples"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main(tid):
    """Write the four overlaid corners of ``tid`` and the quantile table of its runs."""
    ana, rr = runs(tid)
    use_apj_style()
    out = RES / str(tid)
    table(ana, rr, out / "ladder_posterior.txt")
    for name, cols in SETS.items():
        overlay(ana, rr, cols, out / f"{name}.png")
    names = [p for p in rr[-1][2].theta_labels() if not p.startswith("logsfr_ratios")]
    names += [p for p in ("afe",) if p not in names]
    overlay(ana, rr, names, out / "ladder_corner_all_ssfr.png", inset=True, tid=tid)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tid", type=int, required=True)
    main(ap.parse_args().tid)
