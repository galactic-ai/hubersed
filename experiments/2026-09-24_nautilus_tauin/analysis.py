"""QUESTION: Do the v2 MAP rails of 39627770174637084 (sigma_dyn, tau_eq, tau_in, dust_ratio, dust_index,
sigma_smooth at prior bounds) disappear when tau_in is fixed at t_H and the posterior is sampled with nautilus?
HYPOTHESIS: The rails are MAP artefacts (the SFH prior's -1/2 ln det Sigma term drives the hyperparameters
to their bounds), so the posterior piles up at no prior edge and chi2 beats v2.
INPUTS: DESI DR1 spectrum from hubersed.io.desi.load_spectrum. v2 MAP fit
results/map_fits_outliers310_v2/39627770174637084.pkl. Checkpoint
results/2026-09-24_nautilus_tauin/39627770174637084_miles_seed0.h5
SEED: 0 for nautilus, 0 for the posterior draws in the figures.
COMMAND: uv run python experiments/2026-09-24_nautilus_tauin/fit.py --pool 24 --n-batch 480, then
uv run python experiments/2026-09-24_nautilus_tauin/analysis.py
RESULT: Converged (N_eff 2006, log Z 78155.05, 1.56M calls); chi2_red 2.52 vs 3.10 for v2 on the same 4448 pixels.
The rails are gone (dust_ratio, dust_index, tau_eq, sigma_dyn; sigma_reg leans to 0.1).
Still railed: sigma_smooth at its 10 km/s floor (zeroed library resolution over-broadens the model) and gas_logco at -1.
Open: Z* +0.32 vs gas_logz -0.39, and too few 0.15-0.9 Gyr stars (model HdeltaA 2.9 vs 6.0-7.8 emission-subtracted).
FIGURES: results/2026-09-24_nautilus_tauin/spectrum.png, spectrum_zoom.png, corner_phys.png,
corner_sfh.png and sfh.png.
"""

# %%
import pickle

import astropy.units as u
import corner
import matplotlib.pyplot as plt
import numpy as np
from fit import load_data, make_model, make_obs
from nautilus import Sampler

from hubersed.conversion import to_flambda
from hubersed.fitting.chi2 import WAVE_OBS
from hubersed.fitting.map_fits import chi2_parts, get_sps, sfh_from_theta
from hubersed.paths import PATHS
from hubersed.plotting.sfh import sfh_figure
from hubersed.plotting.spectra import plot_residual, residual_chi, spectrum_figure
from hubersed.sps.config import build_continuum_model, build_full_cue_model

TID = 39627770174637084
OUT = PATHS["RESULTS"] / "2026-09-24_nautilus_tauin"
CKPT = OUT / f"{TID}_miles_seed0.h5"

# %%
z, flux, unc, good, in_miles = load_data(TID)
mask = good & in_miles
model = make_model(z)
labels = model.theta_labels()

# Only reads the checkpoint. The likelihood is never called.
sampler = Sampler(
    model.prior_transform,
    lambda x: 0.0,
    n_dim=model.ndim,
    n_live=1000,
    filepath=str(CKPT),
    resume=True,
)
assert sampler.explored
points, log_w, log_l = sampler.posterior()  # about 2 min, prior_transform runs once per point
w = np.exp(log_w)
print(len(points), "points  N_eff", sampler.n_eff, " log Z", sampler.log_z)

# %% corner plots: v2 MAP as truths, dashed lines at prior edges that fall inside an axis
with open(PATHS["RESULTS"] / "map_fits_outliers310_v2" / f"{TID}.pkl", "rb") as f:
    saved = pickle.load(f)
cont_model, cont_tmpl = build_continuum_model(z)
model_v2, _ = build_full_cue_model(cont_tmpl, cont_model.theta, cont_model)
theta_v2 = model_v2.theta.copy()
for k, v in saved["theta_dict"].items():
    theta_v2[model_v2.theta_index[k]] = v
v2 = dict(zip(model_v2.theta_labels(), theta_v2, strict=True))

phys = [i for i, lab in enumerate(labels) if not lab.startswith("logsfr_ratios")]
sfh = [i for i, lab in enumerate(labels) if lab.startswith("logsfr_ratios")]


def wquantile(x, w, q):
    """Weighted quantiles of samples x (weights w, summing to 1)."""
    o = np.argsort(x)
    return np.interp(q, np.cumsum(w[o]), x[o])


def axis_range(x, w, q=(0.001, 0.999), pad=0.05):
    """Central 99.8% of the posterior plus 5% padding."""
    lo, hi = wquantile(x, w, np.array(q))
    d = (hi - lo) * pad or 1e-3
    return (lo - d, hi + d)


for tag, idx in [("phys", phys), ("sfh", sfh)]:
    truths = [v2.get(labels[i]) for i in idx]
    fig = corner.corner(
        points[:, idx],
        weights=w,
        labels=[labels[i] for i in idx],
        truths=truths,
        truth_color="C1",
        range=[axis_range(points[:, i], w) for i in idx],
        plot_datapoints=False,
        fill_contours=True,
        smooth=1.0,
        bins=30,
        levels=(0.393, 0.865),  # 1 and 2 sigma for a 2-D Gaussian
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        label_kwargs={"fontsize": 9},
        title_kwargs={"fontsize": 8},
    )
    axes = np.array(fig.axes).reshape(len(idx), len(idx))
    if tag == "phys":
        for j, i in enumerate(idx):
            ax = axes[j, j]
            for edge in np.atleast_1d(model.config_dict[labels[i]]["prior"].range):
                if ax.get_xlim()[0] <= edge <= ax.get_xlim()[1]:
                    ax.axvline(edge, color="0.4", ls="--", lw=1)
    # v2 values outside the axis are written in the corner of the diagonal panel instead.
    for j, t in enumerate(truths):
        ax = axes[j, j]
        lo, hi = ax.get_xlim()
        if t is not None and not lo <= t <= hi:
            right = t > hi
            ax.text(
                0.97 if right else 0.03,
                0.92,
                f"v2 {t:.3g}",
                transform=ax.transAxes,
                ha="right" if right else "left",
                color="C1",
                fontsize=7,
            )
    fig.savefig(OUT / f"corner_{tag}.png", dpi=120)
    plt.close(fig)

# %% spectrum: data, max-L model, 16-84% band from 100 weighted draws, v2 MAP, residuals
rng = np.random.default_rng(0)
draw = rng.choice(len(w), size=100, p=w)
best = points[np.argmax(log_l)]
cue = get_sps()["cue"]
obs = make_obs(flux, unc, mask)
line_pix = saved["line_pix"] & in_miles

# chi2_parts calls the same model.predict as the likelihood.
sp_best, stats = chi2_parts(model, best, obs, cue, line_pix)
spec = np.array([chi2_parts(model, points[i], obs, cue, line_pix)[0] for i in draw])
lo, hi = np.percentile(spec, [16, 84], axis=0)

# v2 on the same pixels and the same uncertainties, for the RESULT comparison.
r_v2 = residual_chi(flux, saved["model"], unc, mask)
chi2_v2 = np.nansum(r_v2**2)
chi2_red_v2 = chi2_v2 / (mask.sum() - saved["ndim"])  # v2 had its own 26 free parameters
print("nautilus max-L", stats)
print("v2 on the same pixels: chi2", chi2_v2, "chi2_red", chi2_red_v2)
print("v2 error_floor", saved["error_floor"])


def flam(maggies):
    """Convert maggies on WAVE_OBS to DESI f_lambda units, NaN outside the fitted pixels."""
    f = to_flambda(WAVE_OBS * u.AA, np.asarray(maggies) * u.mgy).value
    return np.where(mask, f, np.nan)


rest = WAVE_OBS / (1 + z)
fig, ax = spectrum_figure(
    WAVE_OBS,
    z=z,
    figsize=(11, 6),
    data=flam(flux),
    unc=flam(unc),
    band_kw={},
    models=[
        {
            "flux": flam(sp_best),
            "color": "#b2182b",
            "lw": 0.8,
            "label": rf"nautilus max-L, $\chi^2_\nu$ = {stats['chi2_red']:.2f}",
        },
        {
            "flux": flam(saved["model"]),
            "color": "C0",
            "lw": 0.6,
            "label": rf"v2 MAP, $\chi^2_\nu$ = {chi2_red_v2:.2f}",
        },
    ],
)
ax[0].fill_between(rest, flam(lo), flam(hi), color="#b2182b", alpha=0.3, lw=0)
plot_residual(ax[1], WAVE_OBS, z=z, chi=residual_chi(flux, sp_best, unc, mask), lw=0.5)
ax[1].set_ylim(-8, 8)
ax[1].set_xlim(rest[mask].min(), rest[mask].max())
fig.savefig(OUT / "spectrum.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Zooms on the Dn4000 and Hdelta region and on Halpha with [NII].
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for a, (w_lo, w_hi) in zip(axes, [(3820, 4180), (6520, 6610)], strict=True):
    k = (rest > w_lo) & (rest < w_hi)
    a.plot(rest[k], flam(flux)[k], color="0.3", lw=0.7, label="DESI spectrum")
    a.fill_between(rest[k], flam(lo)[k], flam(hi)[k], color="#b2182b", alpha=0.3, lw=0)
    a.plot(rest[k], flam(sp_best)[k], color="#b2182b", lw=0.8, label="nautilus max-L")
    a.plot(rest[k], flam(saved["model"])[k], color="C0", lw=0.6, label="v2 MAP")
    a.set_xlabel(r"rest wavelength [$\AA$]")
axes[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
axes[0].legend(frameon=False, fontsize="small")
fig.savefig(OUT / "spectrum_zoom.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% SFH posterior: median and 16-84% per bin from 2000 weighted draws, v2 MAP dashed
sfhs = [sfh_from_theta(model, points[i]) for i in rng.choice(len(w), size=2000, p=w)]
edges = sfhs[0]["edges_gyr"]
ssfr = np.array([s["ssfr"] for s in sfhs])
cmf = np.array([s["cmf"] for s in sfhs])
ssfr_lo, ssfr_med, ssfr_hi = np.percentile(ssfr, [16, 50, 84], axis=0)
cmf_lo, cmf_med, cmf_hi = np.percentile(cmf, [16, 50, 84], axis=0)
sfh_v2 = sfh_from_theta(model_v2, theta_v2)

fig, ax = sfh_figure(
    edges,
    ssfr_med,
    cmf_med,
    ssfr_kwargs={"label": "nautilus median"},
    cmf_kwargs={"color": "C3"},
)
ax[0].stairs(ssfr_hi, edges, baseline=ssfr_lo, fill=True, color="C3", alpha=0.25, lw=0)
ax[1].stairs(cmf_hi, edges, baseline=cmf_lo, fill=True, color="C3", alpha=0.25, lw=0)
ax[0].stairs(sfh_v2["ssfr"], edges, color="C1", ls="--", lw=1.2, label="v2 MAP")
ax[1].stairs(sfh_v2["cmf"], edges, color="C1", ls="--", lw=1.2)
# Bins 6 and 7, 146-873 Myr, where the fit keeps too few A stars.
for a in ax:
    a.axvspan(edges[5], edges[7], color="0.85", zorder=0)
ax[0].legend(frameon=True, fontsize="small", loc="lower left")
fig.savefig(OUT / "sfh.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %%
