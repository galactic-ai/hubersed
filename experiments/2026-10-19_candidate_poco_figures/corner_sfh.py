"""QUESTION: Does any parameter of the pocoMC free_dust fit of a candidate sit at its prior bound?
HYPOTHESIS: None registered before this analysis.
INPUTS: pmc_final.state of poco.py in
results/2026-10-15_broadline_free_amplitude/poco_<TID>_free_dust_neff1024/.
METHOD: corner plot of every free parameter but logsfr_ratios, with median and 16-84% titles and the
sSFR posterior at top right. Axes show the central 99.8% plus 5%. Prior bounds inside an axis are
dashed lines. The table gives the 0.5% and 99.5% quantiles and their distance to each bound as a
fraction of the prior width.
SEED: 0 for the SFH draws.
COMMAND: from the repository root on the Mac with the C3K_ER build,
FIT_TID=<TID> PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-15_broadline_free_amplitude:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-19_candidate_poco_figures/corner_sfh.py
RESULT: 2026-10-21, 39633278973513279. Within 1% of the prior width of a bound: tau_eq and dust_ratio
at their lower bounds, sigma_reg at 0.8%, gas_logco at 1.2% of its upper bound. dust2 and sigma_dyn
come within 4-5% of their upper bounds. logzsol, dust_index and frac_nodust stay clear of theirs.
FIGURES: <TID>_corner_sfh.png and <TID>_prior_edges.txt in results/2026-10-19_candidate_poco_figures/.
"""

# %%

import argparse

import numpy as np

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-10-19_candidate_poco_figures"
VARIANT = "free_dust"
RED = "#b2182b"


def wquantile(x, w, q):
    """Return weighted quantiles of ``x`` with weights ``w`` that sum to 1."""
    o = np.argsort(x)
    return np.interp(q, np.cumsum(w[o]), x[o])


def axis_range(x, w, q=(0.001, 0.999), pad=0.05):
    """Return the central 99.8% of the posterior plus 5% padding."""
    lo, hi = wquantile(x, w, np.array(q))
    d = (hi - lo) * pad or 1e-3
    return (lo - d, hi + d)


def prior_range(model, name):
    """Return the (low, high) bounds of the prior of ``name``."""
    lo, hi = model.config_dict[name]["prior"].range
    return float(np.squeeze(lo)), float(np.squeeze(hi))


def load(tid):
    """Return the model, the posterior samples and their weights."""
    import fit
    from compare_poco import load_sampler

    assert fit.TID == tid, "set FIT_TID"
    state = RES / "2026-10-15_broadline_free_amplitude" / f"poco_{tid}_{VARIANT}_neff1024"
    s = load_sampler(state / "pmc_final.state")
    usamp, w, _, _ = s.posterior()
    w = w / w.sum()
    model, _, _ = fit.process_state(VARIANT)
    theta = np.array([model.prior_transform(x) for x in usamp])
    return model, theta, w


def edge_table(model, theta, w, names):
    """Return the text table of quantiles and distances to the prior bounds."""
    labels = list(model.theta_labels())
    lines = [
        f"{'parameter':>24} {'prior lo':>9} {'prior hi':>9} {'q0.5%':>9} {'median':>9} "
        f"{'q99.5%':>9} {'lo dist':>7} {'hi dist':>7}"
    ]
    for p in names:
        x = theta[:, labels.index(p)]
        lo, hi = prior_range(model, p)
        a, m, b = wquantile(x, w, np.array([0.005, 0.5, 0.995]))
        width = hi - lo
        lines.append(
            f"{p:>24} {lo:9.3g} {hi:9.3g} {a:9.4g} {m:9.4g} {b:9.4g} "
            f"{(a - lo) / width:7.3f} {(hi - b) / width:7.3f}"
        )
    lines.append("")
    lines.append(
        "dist = distance of the 0.5% or 99.5% quantile to the bound, in units of prior width."
    )
    return lines


def corner_figure(tid, model, theta, w, names, path):
    """Draw the corner plot with median +- titles and the sSFR posterior at top right."""
    import corner
    import matplotlib.pyplot as plt

    from hubersed.fitting.map_fits import sfh_from_theta

    labels = list(model.theta_labels())
    idx = [labels.index(p) for p in names]
    k = len(idx)
    fig = plt.figure(figsize=(2.8 * k, 2.8 * k))
    corner.corner(
        theta[:, idx],
        weights=w,
        fig=fig,
        labels=names,
        color=RED,
        range=[axis_range(theta[:, i], w) for i in idx],
        plot_datapoints=False,
        fill_contours=True,
        smooth=1.0,
        bins=30,
        levels=(0.393, 0.865),  # 1 and 2 sigma for a 2-D Gaussian
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        max_n_ticks=3,
        label_kwargs={"fontsize": 34},
        title_kwargs={"fontsize": 30},
        labelpad=0.3,
        hist_kwargs={"color": RED, "lw": 2},
    )
    axes = np.array(fig.axes).reshape(k, k)
    for j, p in enumerate(names):
        ax = axes[j, j]
        for edge in prior_range(model, p):
            if ax.get_xlim()[0] <= edge <= ax.get_xlim()[1]:
                ax.axvline(edge, color="k", ls="--", lw=2)
    for ax in axes.ravel():
        ax.tick_params(labelsize=22)
    for ax in axes[-1]:
        ax.xaxis.label.set_rotation(35)
        ax.xaxis.label.set_ha("right")
    for ax in axes[:, 0]:
        ax.yaxis.label.set_rotation(55)
        ax.yaxis.label.set_va("top")

    rng = np.random.default_rng(0)
    sfhs = [sfh_from_theta(model, theta[i]) for i in rng.choice(len(w), size=2000, p=w)]
    edges = sfhs[0]["edges_gyr"]
    vals = np.array([s["ssfr"] for s in sfhs])
    lo, med, hi = np.percentile(vals, [16, 50, 84], axis=0)
    ax = fig.add_axes([0.53, 0.62, 0.44, 0.33])
    ax.stairs(hi, edges, baseline=lo, fill=True, color=RED, alpha=0.35, lw=0, label=r"16--84\%")
    ax.stairs(med, edges, color=RED, lw=3, label="median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lookback time [Gyr]", fontsize=44)
    ax.set_ylabel(r"sSFR [yr$^{-1}$]", fontsize=44)
    ax.tick_params(labelsize=34)
    ax.legend(frameon=False, fontsize=38, loc="lower left")
    ax.set_title(f"{tid}, pocoMC {VARIANT.replace('_', ' ')}", fontsize=46)
    fig.savefig(path, dpi=60, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# %%
def main():
    """Write the corner figure and the prior edge table of the target in FIT_TID."""
    import fit

    from hubersed.plotting.style import use_apj_style

    argparse.ArgumentParser(description=__doc__).parse_args()
    use_apj_style()
    tid = fit.TID
    model, theta, w = load(tid)
    names = [p for p in model.theta_labels() if not p.startswith("logsfr_ratios")]
    lines = edge_table(model, theta, w, names)
    (OUT / f"{tid}_prior_edges.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    corner_figure(tid, model, theta, w, names, OUT / f"{tid}_corner_sfh.png")


if __name__ == "__main__":
    main()
