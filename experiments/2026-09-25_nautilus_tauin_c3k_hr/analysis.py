"""QUESTION: Do the rails of 2026-09-24_nautilus_tauin (sigma_smooth at its 10 km/s floor, gas_logco
at -1) and the missing 0.15-0.9 Gyr stars go away with FSPS C3K_HR and correct library resolution?
HYPOTHESIS: sigma_smooth railed because the zeroed MILES resolution over-broadened the model, so
with C3K_HR and a consistent LSF it moves off the floor.
INPUTS: DESI DR1 spectrum from hubersed.io.desi.load_spectrum, degraded to 43.6 km/s,
MILES window 3601.8-7400.8 A rest, 4448 pixels.
SEED: 0 for nautilus, 0 for the posterior draws in the figures.
COMMAND: uv run python experiments/2026-09-25_nautilus_tauin_c3k_hr/fit.py --window miles --pool 24
--n-batch 480 on LS6 (SPS_HOME at FSPS 7572834), then uv run python
experiments/2026-09-25_nautilus_tauin_c3k_hr/analysis.py
RESULT: Converged (N_eff 2000, log Z 79788.06, 1.43M calls).
sigma_smooth still piles at 10 km/s. gas_logco moved to its upper edge (+0.72) and dust_index to its
upper edge (0.4). logzsol +0.32 -> -0.23.
FIGURES: results/2026-09-25_nautilus_tauin_c3k_hr/spectrum.png, spectrum_zoom.png, corner_phys.png,
corner_sfh.png and sfh.png.
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
OUT = PATHS["RESULTS"] / "2026-09-25_nautilus_tauin_c3k_hr"
CKPT = {
    "MILES": PATHS["RESULTS"] / "2026-09-24_nautilus_tauin" / f"{TID}_miles_seed0.h5",
    "C3K": OUT / f"{TID}_c3k_miles_seed0.h5",
}
COLOR = {"MILES": "C0", "C3K": "C1"}

z, flux, unc, good = load_data(TID, "miles")
model = make_model(z)
labels = model.theta_labels()

# %% load posterior samples from checkpoints
post = {}
for run, path in CKPT.items():
    # Only reads the checkpoint. The likelihood is never called.
    s = Sampler(
        model.prior_transform,
        lambda x: 0.0,
        n_dim=model.ndim,
        n_live=1000,
        filepath=str(path),
        resume=True,
    )
    assert s.explored, path
    points, log_w, log_l = s.posterior()  # about 2 min each
    post[run] = dict(
        points=points, w=np.exp(log_w), best=points[np.argmax(log_l)], max_lnl=log_l.max()
    )
    print(
        f"{run}: {len(points)} points, N_eff {s.n_eff:.0f}, calls {s.n_like}, max lnL {log_l.max():.1f}"
    )


def wquantile(x, w, q):
    """Weighted quantiles of samples x (weights w, summing to 1)."""
    o = np.argsort(x)
    return np.interp(q, np.cumsum(w[o]), x[o])


# %% shift table: weighted 16/50/84 per run and the median shift in units of the combined sigma
q = {
    r: np.array(
        [wquantile(p["points"][:, i], p["w"], [0.16, 0.5, 0.84]) for i in range(len(labels))]
    )
    for r, p in post.items()
}
print(f"{'parameter':18s} {'MILES':>24s} {'C3K':>24s} {'shift/sigma':>11s}")
for i, lab in enumerate(labels):
    (m1, m5, m8), (c1, c5, c8) = q["MILES"][i], q["C3K"][i]
    sig = np.hypot((m8 - m1) / 2, (c8 - c1) / 2)
    print(
        f"{lab:18s} {m5:10.3f} +{m8 - m5:.3f} -{m5 - m1:.3f} {c5:10.3f} +{c8 - c5:.3f} -{c5 - c1:.3f} {(c5 - m5) / sig:11.2f}"
    )

# %% corner plots: both posteriors overlaid, dashed lines at prior edges inside an axis
phys = [i for i, lab in enumerate(labels) if not lab.startswith("logsfr_ratios")]
sfh = [i for i, lab in enumerate(labels) if lab.startswith("logsfr_ratios")]

for tag, idx in [("phys", phys), ("sfh", sfh)]:
    # axis range: union of both runs' central 99.8%, plus 5% padding
    rng = []
    for i in idx:
        a = min(wquantile(p["points"][:, i], p["w"], 0.001) for p in post.values())
        b = max(wquantile(p["points"][:, i], p["w"], 0.999) for p in post.values())
        d = (b - a) * 0.05 or 1e-3
        rng.append((a - d, b + d))
    fig = None
    for run, p in post.items():
        fig = corner.corner(
            p["points"][:, idx],
            weights=p["w"],
            fig=fig,
            color=COLOR[run],
            labels=[labels[i] for i in idx],
            range=rng,
            plot_datapoints=False,
            plot_density=False,
            smooth=1.0,
            bins=30,
            levels=(0.393, 0.865),  # 1 and 2 sigma for a 2-D Gaussian
            label_kwargs={"fontsize": 9},
            hist_kwargs={"density": True},
        )
    axes = np.array(fig.axes).reshape(len(idx), len(idx))
    if tag == "phys":
        for j, i in enumerate(idx):
            ax = axes[j, j]
            for edge in np.atleast_1d(model.config_dict[labels[i]]["prior"].range):
                if ax.get_xlim()[0] <= edge <= ax.get_xlim()[1]:
                    ax.axvline(edge, color="0.4", ls="--", lw=1)
    handles = [Line2D([], [], color=COLOR[r], label=f"{r} nautilus") for r in post]
    fig.legend(handles=handles, loc="upper right", frameon=False)
    fig.savefig(OUT / f"corner_{tag}.png", dpi=120)
    plt.close(fig)

# %% C3K max-L spectrum and a 16-84% band from 100 weighted draws
p = post["C3K"]
lnl = loglike(p["best"], TID, "miles")  # the likelihood of the C3K run
assert np.isclose(lnl, p["max_lnl"], atol=0.1), (lnl, p["max_lnl"])
cue = get_sps(zero_library_resolution=False)["cue"]
obs = make_obs(flux, unc, good)
sp_best, _ = chi2_parts(model, p["best"], obs, cue, np.zeros_like(good))
gen = np.random.default_rng(0)
draw = gen.choice(len(p["w"]), size=100, p=p["w"])
spec = np.array([chi2_parts(model, p["points"][i], obs, cue, np.zeros_like(good))[0] for i in draw])
lo, hi = np.percentile(spec, [16, 84], axis=0)

# Chi uses the uncertainty the fit used (raw DESI ivar kept after degrading, keep_ivar=True in
# load_data), so the plot matches the likelihood. It overstates the per-pixel noise of the
# smoothed flux, so chi2_nu here is not in noise units and is not comparable with the MILES run.
chi = residual_chi(flux, sp_best, unc, good)
chi2_red = np.nansum(chi**2) / (good.sum() - model.ndim)
print(f"C3K max-L chi2_nu {chi2_red:.2f} with the fit uncertainty")


def flam(maggies):
    """Convert maggies on WAVE_OBS to DESI f_lambda units, NaN outside the fitted pixels."""
    f = to_flambda(WAVE_OBS * u.AA, np.asarray(maggies, float) * u.mgy).value
    return np.where(good, f, np.nan)


rest = WAVE_OBS / (1 + z)

# %% spectrum: degraded DESI data, C3K max-L and band, chi residual
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
            "flux": flam(sp_best),
            "color": "#b2182b",
            "lw": 0.8,
            "label": rf"C3K nautilus max-L, $\chi^2_\nu$ = {chi2_red:.2f}",
        },
    ],
)
ax[0].fill_between(rest, flam(lo), flam(hi), color="#b2182b", alpha=0.3, lw=0)
plot_residual(ax[1], WAVE_OBS, z=z, chi=chi, lw=0.5)
ax[1].set_ylim(-8, 8)
ax[1].set_xlim(rest[good].min(), rest[good].max())
fig.savefig(OUT / "spectrum.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% zooms on the Dn4000 and Hdelta region and on Halpha with [NII]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for a, (w_lo, w_hi) in zip(axes, [(3820, 4180), (6520, 6610)], strict=True):
    k = (rest > w_lo) & (rest < w_hi)
    a.plot(rest[k], flam(flux)[k], color="0.3", lw=0.7, label="DESI spectrum, degraded")
    a.fill_between(rest[k], flam(lo)[k], flam(hi)[k], color="#b2182b", alpha=0.3, lw=0)
    a.plot(rest[k], flam(sp_best)[k], color="#b2182b", lw=0.8, label="C3K nautilus max-L")
    a.set_xlabel(r"rest wavelength [$\AA$]")
axes[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
axes[0].legend(frameon=False, fontsize="small")
fig.savefig(OUT / "spectrum_zoom.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% SFH: median and 16-84% per bin from 2000 weighted draws per run
hist = {}
for run, pr in post.items():
    draws = [
        sfh_from_theta(model, pr["points"][i])
        for i in gen.choice(len(pr["w"]), size=2000, p=pr["w"])
    ]
    hist[run] = dict(
        ssfr=np.percentile([d["ssfr"] for d in draws], [16, 50, 84], axis=0),
        cmf=np.percentile([d["cmf"] for d in draws], [16, 50, 84], axis=0),
    )
edges = draws[0]["edges_gyr"]

fig, ax = sfh_figure(
    edges,
    hist["MILES"]["ssfr"][1],
    hist["MILES"]["cmf"][1],
    ssfr_kwargs={"label": "MILES median", "ssfr_kw": {"color": COLOR["MILES"]}},
    cmf_kwargs={"color": COLOR["MILES"]},
)
ax[0].stairs(hist["C3K"]["ssfr"][1], edges, color=COLOR["C3K"], lw=1.8, label="C3K median")
ax[1].stairs(hist["C3K"]["cmf"][1], edges, color=COLOR["C3K"], lw=1.8)
for run, h in hist.items():
    for a, key in zip(ax, ["ssfr", "cmf"], strict=False):
        a.stairs(h[key][2], edges, baseline=h[key][0], fill=True, color=COLOR[run], alpha=0.2, lw=0)

ax[0].legend(frameon=True, fontsize="small", loc="lower left")
fig.savefig(OUT / "sfh.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %%
