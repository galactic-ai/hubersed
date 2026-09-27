"""QUESTION:
HYPOTHESIS:
INPUTS:
SEED: 0 for nautilus, 0 for the posterior draws in the figures.
COMMAND:
RESULT:
FIGURES:
"""

# %%
import astropy.units as u
import corner
import matplotlib.pyplot as plt
import numpy as np
from fit import load_data, loglike, make_model, make_obs
from matplotlib.lines import Line2D
from nautilus import Sampler

from hubersed.conversion import to_flambda
from hubersed.fitting.chi2 import WAVE_OBS
from hubersed.fitting.map_fits import chi2_parts, get_sps, sfh_from_theta
from hubersed.paths import PATHS
from hubersed.plotting.sfh import sfh_figure
from hubersed.plotting.spectra import plot_residual, residual_chi, spectrum_figure

# %%
TID = 39627770174637084
OUT = PATHS["RESULTS"] / "2026-09-26_nautilus_c3k_broadline"
PREV = PATHS["RESULTS"] / "2026-09-25_nautilus_tauin_c3k_hr"  # plot data from its save.py
COLOR = {"MILES": "C0", "C3K": "C1", "C3K broad": "C2"}

z, flux, unc, good = load_data(TID, "miles")
model = make_model(z)

# %% MILES and C3K: posterior, max-L point and SFH draws saved by 2026-09-25 save.py
post = {}
for run in ["MILES", "C3K"]:
    d = dict(np.load(PREV / f"{TID}_{run}_plotdata.npz"))
    assert np.isclose(d["z"], z) and np.array_equal(d["good"], good), run
    d["labels"] = list(d["labels"])
    d["prior"] = dict(zip(d["prior_labels"], d["prior_range"], strict=True))
    post[run] = d
    print(f"{run}: {len(d['w'])} points, N_eff {d['n_eff']:.0f}, max lnL {d['max_lnl']:.1f}")

# %% C3K broad: the same quantities from its checkpoint
# Only reads the checkpoint. The likelihood is never called.
s = Sampler(
    model.prior_transform,
    lambda x: 0.0,
    n_dim=model.ndim,
    n_live=1000,
    filepath=str(OUT / "run2" / f"{TID}_broad_miles_seed0.h5"),
    resume=True,
)
# an unfinished run gives all points so far, with a small N_eff
points, log_w, log_l = s.posterior()  # about 2 min
w = np.exp(log_w)
gen = np.random.default_rng(0)
draws = [sfh_from_theta(model, points[i]) for i in gen.choice(len(w), size=2000, p=w)]
labels = model.theta_labels()
post["C3K broad"] = dict(
    points=points,
    w=w,
    labels=labels,
    best=points[np.argmax(log_l)],
    max_lnl=log_l.max(),
    prior={
        lab: model.config_dict[lab]["prior"].range
        for lab in labels
        if not lab.startswith("logsfr_ratios")
    },
    sfh_edges=draws[0]["edges_gyr"],
    sfh_ssfr=np.array([d["ssfr"] for d in draws]),
    sfh_cmf=np.array([d["cmf"] for d in draws]),
)
print(
    f"C3K broad: {len(points)} points, explored {s.explored}, N_eff {s.n_eff:.0f}, "
    f"calls {s.n_like}, max lnL {log_l.max():.1f}"
)


def wquantile(x, w, q):
    """Weighted quantiles of samples x (weights w, summing to 1)."""
    o = np.argsort(x)
    return np.interp(q, np.cumsum(w[o]), x[o])


def column(run, lab):
    """Samples of parameter ``lab`` in ``run``."""
    p = post[run]
    return p["points"][:, p["labels"].index(lab)]


# %% table: weighted 16/50/84 per run and the C3K broad - C3K median shift in combined sigma
q = {
    r: {lab: wquantile(column(r, lab), p["w"], [0.16, 0.5, 0.84]) for lab in p["labels"]}
    for r, p in post.items()
}
print(f"{'parameter':18s}" + "".join(f"{r:>24s}" for r in post) + f"{'shift/sigma':>12s}")
for lab in labels:
    row = f"{lab:18s}"
    for r in post:
        if lab in q[r]:
            a, m, b = q[r][lab]
            row += f"{m:10.3f} +{b - m:.3f} -{m - a:.3f}"
        else:
            row += " " * 24
    if lab in q["C3K"]:
        (c1, c5, c8), (b1, b5, b8) = q["C3K"][lab], q["C3K broad"][lab]
        row += f"{(b5 - c5) / np.hypot((c8 - c1) / 2, (b8 - b1) / 2):12.2f}"
    print(row)

# %% corner plots: posteriors overlaid, dashed lines at prior edges inside an axis
# The broad-line parameters exist only in that run, so they get their own corner plot.
common = [lab for lab in post["C3K"]["labels"] if lab in labels]
panels = [
    ("phys", [lab for lab in common if not lab.startswith("logsfr_ratios")], list(post)),
    ("sfh", [lab for lab in common if lab.startswith("logsfr_ratios")], list(post)),
    (
        "lines",
        ["eline_sigma", "eline_sigma_forb", "eline_fbroad", "eline_sigma_broad", "eline_vbroad"],
        ["C3K broad"],
    ),
]

for tag, labs, runs in panels:
    # axis range: union of the runs' central 99.8%, plus 5% padding
    rng = []
    for lab in labs:
        a = min(wquantile(column(r, lab), post[r]["w"], 0.001) for r in runs)
        b = max(wquantile(column(r, lab), post[r]["w"], 0.999) for r in runs)
        d = (b - a) * 0.05 or 1e-3
        rng.append((a - d, b + d))
    fig = None
    for r in runs:
        fig = corner.corner(
            np.column_stack([column(r, lab) for lab in labs]),
            weights=post[r]["w"],
            fig=fig,
            color=COLOR[r],
            labels=labs,
            range=rng,
            plot_datapoints=False,
            plot_density=False,
            smooth=1.0,
            bins=30,
            levels=(0.393, 0.865),  # 1 and 2 sigma for a 2-D Gaussian
            label_kwargs={"fontsize": 9},
            hist_kwargs={"density": True},
        )
    axes = np.array(fig.axes).reshape(len(labs), len(labs))
    if tag != "sfh":
        for j, lab in enumerate(labs):
            ax = axes[j, j]
            for edge in {e for r in runs for e in np.atleast_1d(post[r]["prior"][lab])}:
                if ax.get_xlim()[0] <= edge <= ax.get_xlim()[1]:
                    ax.axvline(edge, color="0.4", ls="--", lw=1)
    handles = [Line2D([], [], color=COLOR[r], label=f"{r} nautilus") for r in runs]
    fig.legend(handles=handles, loc="upper right", frameon=False)
    fig.savefig(OUT / f"corner_{tag}.png", dpi=120)
    plt.close(fig)

# %% max-L spectra of the C3K runs, with and without the broad Balmer component
p = post["C3K broad"]
lnl = loglike(p["best"], TID, "miles")
assert np.isclose(lnl, p["max_lnl"], atol=0.1), (lnl, p["max_lnl"])
cue = get_sps(zero_library_resolution=False)["cue"]
p["sp_best"], _ = chi2_parts(model, p["best"], make_obs(flux, unc, good), cue, np.zeros_like(good))

best = {}
for run in ["C3K", "C3K broad"]:
    chi = residual_chi(flux, post[run]["sp_best"], unc, good)
    ndim = len(post[run]["labels"])
    best[run] = dict(
        sp=post[run]["sp_best"], chi=chi, chi2_red=np.nansum(chi**2) / (good.sum() - ndim)
    )
    print(f"{run} max-L chi2_nu {best[run]['chi2_red']:.3f} with the fit uncertainty")


def flam(maggies):
    """Convert maggies on WAVE_OBS to DESI f_lambda units, NaN outside the fitted pixels."""
    f = to_flambda(WAVE_OBS * u.AA, np.asarray(maggies, float) * u.mgy).value
    return np.where(good, f, np.nan)


rest = WAVE_OBS / (1 + z)

# %% spectrum: degraded DESI data, both max-L models, chi residuals
fig, ax = spectrum_figure(
    WAVE_OBS,
    z=z,
    figsize=(11, 6),
    data=flam(flux),
    unc=flam(unc),
    band_kw={},
    data_kw={"label": "DESI spectrum, degraded"},
    models=[
        {
            "flux": flam(b["sp"]),
            "color": COLOR[run],
            "lw": 0.8,
            "label": rf"{run} nautilus max-L, $\chi^2_\nu$ = {b['chi2_red']:.2f}",
        }
        for run, b in best.items()
    ],
)
for run, b in best.items():
    plot_residual(ax[1], WAVE_OBS, z=z, chi=b["chi"], lw=0.5, color=COLOR[run], alpha=0.8)
ax[1].set_ylim(-8, 8)
ax[1].set_xlim(rest[good].min(), rest[good].max())
fig.savefig(OUT / "spectrum.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% zooms on the Dn4000 and Hdelta region and on Halpha with [NII]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for a, (w_lo, w_hi) in zip(axes, [(3820, 4180), (6520, 6610)], strict=True):
    k = (rest > w_lo) & (rest < w_hi)
    a.plot(rest[k], flam(flux)[k], color="0.3", lw=0.7, label="DESI spectrum, degraded")
    for run, b in best.items():
        a.plot(rest[k], flam(b["sp"])[k], color=COLOR[run], lw=0.8, label=f"{run} nautilus max-L")
    a.set_xlabel(r"rest wavelength [$\AA$]")
axes[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
axes[0].legend(frameon=False, fontsize="small")
fig.savefig(OUT / "spectrum_zoom.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% SFH: median and 16-84% per bin from the 2000 weighted draws per run
hist = {
    run: {key: np.percentile(p[f"sfh_{key}"], [16, 50, 84], axis=0) for key in ["ssfr", "cmf"]}
    for run, p in post.items()
}
edges = post["C3K broad"]["sfh_edges"]
assert all(np.allclose(p["sfh_edges"], edges) for p in post.values())

fig, ax = sfh_figure(
    edges,
    hist["MILES"]["ssfr"][1],
    hist["MILES"]["cmf"][1],
    ssfr_kwargs={"label": "MILES median", "ssfr_kw": {"color": COLOR["MILES"]}},
    cmf_kwargs={"color": COLOR["MILES"]},
)
for run in ["C3K", "C3K broad"]:
    ax[0].stairs(hist[run]["ssfr"][1], edges, color=COLOR[run], lw=1.8, label=f"{run} median")
    ax[1].stairs(hist[run]["cmf"][1], edges, color=COLOR[run], lw=1.8)
for run, h in hist.items():
    for a, key in zip(ax, ["ssfr", "cmf"], strict=False):
        a.stairs(h[key][2], edges, baseline=h[key][0], fill=True, color=COLOR[run], alpha=0.2, lw=0)

ax[0].legend(frameon=True, fontsize="small", loc="lower left")
fig.savefig(OUT / "sfh.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %%
