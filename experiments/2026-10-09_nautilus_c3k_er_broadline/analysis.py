"""QUESTION: What does the converged C3K_ER broadline fit of 39627770174637084 look like (posterior,
SFH, spectrum and line residuals), and how do its posteriors differ from the C3K_HR split45 fit of the
same object with the same model?
HYPOTHESIS: None registered before this analysis.
INPUTS: results/2026-10-09_nautilus_c3k_er_broadline/<stem>_pred.npz from predict.py, with
<stem> = 39627770174637084_er_broad_shared_split45_rest9000_seed0. For comparison the checkpoint
results/2026-09-30_nautilus_c3k_broadline_split45/39627770174637084_broad_shared_split45_unmask_rest9000_seed0.h5.
The HR run fitted different data (degraded to C3K_HR plus 10 km/s, [OI]/[SII] sky pixels unmasked),
so its log Z and chi2 are not comparable with the ER run. Only posteriors are compared.
SEED: 0 for the SFH draws here, 0 for the 100 spectrum draws in predict.py.
COMMAND: on an ls6 compute node, after predict.py,
.venv/bin/python experiments/2026-10-09_nautilus_c3k_er_broadline/analysis.py
RESULT: see analysis_summary.txt next to the figures.
FIGURES: results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/. params_table.txt,
analysis_summary.txt, corner_phys.png, corner_sfh.png, corner_lines.png (ER filled, HR grey),
sfh.png (ER band, HR median), spectrum.png, spectrum_zoom.png and line_windows.png.
"""

# %%
import argparse
import importlib.util

import astropy.units as u
import corner
import matplotlib.pyplot as plt
import numpy as np
from fit import make_model
from matplotlib.lines import Line2D
from nautilus import Sampler

from hubersed.conversion import to_flambda
from hubersed.fitting.map_fits import sfh_from_theta
from hubersed.paths import PATHS
from hubersed.plotting.sfh import plot_cumulative_mass, plot_ssfr, sfh_figure
from hubersed.plotting.spectra import plot_residual, residual_chi, spectrum_figure

# parse_known_args, so the cells also run in an interactive session
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tid", type=int, default=39627770174637084)
TID = parser.parse_known_args()[0].tid
OUT = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline"
FIG = OUT / str(TID)
FIG.mkdir(parents=True, exist_ok=True)
STEM = f"{TID}_er_broad_shared_split45_rest9000_seed0"
HR_DIR = "2026-09-30_nautilus_c3k_broadline_split45"
HR_H5 = PATHS["RESULTS"] / HR_DIR / f"{TID}_broad_shared_split45_unmask_rest9000_seed0.h5"
GEN = np.random.default_rng(0)
ER_COL, HR_COL = "#b2182b", "0.45"
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
# rest-frame (vacuum) windows for the line residual figure, as in the split45 analysis
WINDOWS = {
    "Hbeta": (4845, 4880),
    "[OIII]5007": (4995, 5022),
    "[OI]6300": (6292, 6312),
    "Halpha+[NII]": (6540, 6595),
    "[SII]": (6708, 6742),
}


def wquantile(x, w, q):
    """Weighted quantiles of samples x (weights w, summing to 1)."""
    o = np.argsort(x)
    return np.interp(q, np.cumsum(w[o]), x[o])


def load_module(path, name):
    """Import a fit.py from another experiment under its own module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# %% the ER posterior and spectra from predict.py, and the HR posterior from its checkpoint
er = dict(np.load(OUT / f"{STEM}_pred.npz"))
er["labels"] = [str(x) for x in er["labels"]]
er["w"] = np.exp(er["log_w"] - er["log_w"].max())
er["w"] /= er["w"].sum()
z = float(er["z"])
model = make_model(z, "shared", 45.0)
assert list(model.theta_labels()) == er["labels"]

f30 = load_module(PATHS["ROOT"] / "experiments" / HR_DIR / "fit.py", "fit0930")
model_hr = f30.make_model(z, "shared", 45.0)
assert list(model_hr.theta_labels()) == er["labels"]
s = Sampler(
    lambda x: x, lambda x: 0.0, n_dim=model_hr.ndim, n_live=1000, filepath=str(HR_H5), resume=True
)
unit, log_w, log_l = s.posterior()
keep = log_w > log_w.max() - 30
w = np.exp(log_w[keep] - log_w[keep].max())
hr = dict(
    labels=er["labels"],
    points=np.array([model_hr.prior_transform(x) for x in unit[keep]]),
    w=w / w.sum(),
    log_z=float(s.log_z),
    n_eff=float(s.n_eff),
    n_like=int(s.n_like),
    max_lnl=float(log_l.max()),
)
labels = er["labels"]


def column(d, lab):
    return d["points"][:, labels.index(lab)]


def prior_range(m, lab):
    """Prior edges of a scalar parameter, NaN for the entries of a vector one (logsfr_ratios)."""
    if lab not in m.config_dict:
        return np.array([np.nan, np.nan])
    return np.ravel(m.config_dict[lab]["prior"].range)[:2]


# %% fit quality of the ER run
chi = residual_chi(er["flux"], er["sp_best"], er["unc"], er["good"])
n_pix = int(np.isfinite(chi).sum())
chi2_nu = float(np.nansum(chi**2) / (n_pix - len(labels)))
summary = [
    f"ER: N_eff {float(er['n_eff']):.0f}, calls {int(er['n_like'])}, log Z {float(er['log_z']):.2f}, "
    f"max lnL {float(er['max_lnl']):.2f}, chi2_nu {chi2_nu:.3f} over {n_pix} pixels, "
    f"{len(er['w'])} points kept",
    f"HR (other data, not comparable in log Z): N_eff {hr['n_eff']:.0f}, calls {hr['n_like']}, "
    f"log Z {hr['log_z']:.2f}, max lnL {hr['max_lnl']:.2f}, {len(hr['w'])} points kept",
]
print("\n".join(summary))

# %% parameter table: 16/50/84 for ER and HR, flagged where the median sits within 2% of a prior edge
rows = [f"{'parameter':20s}{'ER median [16, 84]':>40s}{'HR median [16, 84]':>40s}"]
for lab in labels:
    row = f"{lab:20s}"
    lo, hi = prior_range(model, lab)
    for d in (er, hr):
        a, mid, b = wquantile(column(d, lab), d["w"], [0.16, 0.5, 0.84])
        rail = ""
        if (mid - lo) < 0.02 * (hi - lo):
            rail = " FLOOR"
        elif (hi - mid) < 0.02 * (hi - lo):
            rail = " CEIL"
        row += f"{mid:12.4f} [{a:10.4f},{b:10.4f}]{rail:6s}".rjust(40)
    rows.append(row)
table = "\n".join(rows)
print(table)
(FIG / "params_table.txt").write_text(table + "\n")

# %% corner plots, ER filled and HR as grey contours, with dashed lines at prior edges inside an axis


def axis_range(lab, q=(0.001, 0.999), pad=0.05):
    """Central 99.8% of both posteriors plus 5% padding."""
    lo = min(wquantile(column(d, lab), d["w"], q[0]) for d in (er, hr))
    hi = max(wquantile(column(d, lab), d["w"], q[1]) for d in (er, hr))
    p = (hi - lo) * pad or 1e-3
    return (lo - p, hi + p)


groups = {
    "phys": [lab for lab in labels if not lab.startswith("logsfr_ratios") and lab not in LINE_PARS],
    "sfh": [lab for lab in labels if lab.startswith("logsfr_ratios")],
    "lines": [lab for lab in labels if lab in LINE_PARS],
}
for tag, labs in groups.items():
    rng = [axis_range(lab) for lab in labs]
    fig = corner.corner(
        np.column_stack([column(hr, lab) for lab in labs]),
        weights=hr["w"],
        color=HR_COL,
        range=rng,
        plot_datapoints=False,
        plot_density=False,
        smooth=1.0,
        bins=30,
        levels=(0.393, 0.865),  # 1 and 2 sigma for a 2-D Gaussian
        hist_kwargs={"density": True},
    )
    corner.corner(
        np.column_stack([column(er, lab) for lab in labs]),
        weights=er["w"],
        fig=fig,
        color=ER_COL,
        labels=labs,
        range=rng,
        plot_datapoints=False,
        fill_contours=True,
        smooth=1.0,
        bins=30,
        levels=(0.393, 0.865),
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        label_kwargs={"fontsize": 9},
        title_kwargs={"fontsize": 8},
        hist_kwargs={"density": True},
    )
    axes = np.array(fig.axes).reshape(len(labs), len(labs))
    for j, lab in enumerate(labs):
        for edge in prior_range(model, lab):
            if axes[j, j].get_xlim()[0] <= edge <= axes[j, j].get_xlim()[1]:
                axes[j, j].axvline(edge, color="0.3", ls="--", lw=0.8)
    fig.legend(
        handles=[
            Line2D([], [], color=ER_COL, label="C3K_ER (2026-10-09), titles"),
            Line2D([], [], color=HR_COL, label="C3K_HR split45 (2026-09-30)"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=12,
    )
    fig.suptitle(f"{TID} {tag}", fontsize=12)
    fig.savefig(FIG / f"corner_{tag}.png", dpi=110)
    plt.close(fig)

# %% SFH: ER median and 16-84% per bin from 2000 weighted draws, HR median for comparison
sfh = {}
for key, d, m in [("ER", er, model), ("HR", hr, model_hr)]:
    draws = [sfh_from_theta(m, d["points"][i]) for i in GEN.choice(len(d["w"]), 2000, p=d["w"])]
    sfh[key] = dict(
        edges=draws[0]["edges_gyr"],
        ssfr=np.percentile([x["ssfr"] for x in draws], [16, 50, 84], axis=0),
        cmf=np.percentile([x["cmf"] for x in draws], [16, 50, 84], axis=0),
    )
edges = sfh["ER"]["edges"]
fig, ax = sfh_figure(
    edges,
    sfh["ER"]["ssfr"][1],
    sfh["ER"]["cmf"][1],
    ssfr_kwargs={"label": "C3K_ER median", "ssfr_kw": {"color": ER_COL}},
    cmf_kwargs={"color": ER_COL},
)
ax[0].stairs(
    sfh["ER"]["ssfr"][2],
    edges,
    baseline=sfh["ER"]["ssfr"][0],
    fill=True,
    color=ER_COL,
    alpha=0.25,
    lw=0,
)
ax[1].stairs(
    sfh["ER"]["cmf"][2],
    edges,
    baseline=sfh["ER"]["cmf"][0],
    fill=True,
    color=ER_COL,
    alpha=0.25,
    lw=0,
)
plot_ssfr(
    ax[0],
    sfh["HR"]["edges"],
    sfh["HR"]["ssfr"][1],
    guides=(),
    label="C3K_HR median",
    ssfr_kw={"color": HR_COL},
)
plot_cumulative_mass(ax[1], sfh["HR"]["edges"], sfh["HR"]["cmf"][1], guides=(), color=HR_COL)
ax[0].legend(frameon=True, fontsize="small", loc="lower left")
fig.savefig(FIG / "sfh.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% ER spectrum: data, max-L model and a 16-84% band from 100 draws, chi below
rest = er["wave"] / (1 + z)


def flam(maggies):
    """Convert maggies to DESI f_lambda units, NaN outside the fitted pixels."""
    f = to_flambda(er["wave"] * u.AA, np.asarray(maggies, float) * u.mgy).value
    return np.where(er["good"], f, np.nan)


lo, hi = np.percentile(er["draw_spec"], [16, 84], axis=0)
fig, ax = spectrum_figure(
    er["wave"],
    z=z,
    figsize=(11, 6),
    data=flam(er["flux"]),
    unc=flam(er["unc"]),
    band_kw={},
    data_kw={"label": "DESI spectrum (DESI resolution)"},
    models=[
        {
            "flux": flam(er["sp_best"]),
            "color": ER_COL,
            "lw": 0.8,
            "label": rf"C3K_ER max-L, $\chi^2_\nu$ = {chi2_nu:.2f}",
        }
    ],
)
ax[0].fill_between(rest, flam(lo), flam(hi), color=ER_COL, alpha=0.3, lw=0)
plot_residual(ax[1], er["wave"], z=z, chi=chi, lw=0.5, color=ER_COL, alpha=0.8)
ax[1].set_ylim(-8, 8)
ax[1].set_xlim(rest[er["good"]].min(), rest[er["good"]].max())
fig.savefig(FIG / "spectrum.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# zooms on Dn4000 and Hdelta, on Hbeta to Fe5335, and on Halpha, with chi below
regions = [(3820, 4180), (4800, 5400), (6520, 6610)]
fig, axes = plt.subplots(
    2, 3, figsize=(15, 5), sharex="col", gridspec_kw=dict(height_ratios=[3, 1.2])
)
for j, (w_lo, w_hi) in enumerate(regions):
    k = (rest > w_lo) & (rest < w_hi)
    a, ar = axes[0, j], axes[1, j]
    a.plot(rest[k], flam(er["flux"])[k], color="0.3", lw=0.7, label="DESI spectrum")
    a.fill_between(rest[k], flam(lo)[k], flam(hi)[k], color=ER_COL, alpha=0.3, lw=0)
    a.plot(rest[k], flam(er["sp_best"])[k], color=ER_COL, lw=0.8, label="C3K_ER max-L")
    ar.step(rest[k], chi[k], color=ER_COL, where="mid", lw=0.7)
    ar.axhline(0, color="0.3", lw=0.5)
    for h in (-3, 3):
        ar.axhline(h, color="0.6", lw=0.5, ls=":")
    ar.set_xlabel(r"rest wavelength [$\AA$]")
axes[0, 0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
axes[1, 0].set_ylabel(r"$\chi$")
axes[0, 0].legend(frameon=False, fontsize="small")
fig.tight_layout()
fig.savefig(FIG / "spectrum_zoom.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% line windows: data, max-L model, the same point with f_b = 0, chi below
lines = []
fig, axes = plt.subplots(
    2, len(WINDOWS), figsize=(16, 4.5), gridspec_kw=dict(height_ratios=[3, 1.2])
)
for j, (name, (w_lo, w_hi)) in enumerate(WINDOWS.items()):
    k = (rest > w_lo) & (rest < w_hi)
    a, ar = axes[0, j], axes[1, j]
    d, e = flam(er["flux"])[k], flam(er["unc"])[k]
    a.fill_between(rest[k], d - e, d + e, color="0.8", step="mid", lw=0)
    a.step(rest[k], d, "k", where="mid", lw=0.7, label="DESI")
    a.plot(rest[k], flam(er["sp_best"])[k], color=ER_COL, lw=1, label="max-L")
    a.plot(rest[k], flam(er["sp_narrow"])[k], "C0--", lw=0.8, label="max-L, f_b = 0")
    ar.step(rest[k], chi[k], color=ER_COL, where="mid", lw=0.8)
    ar.axhline(0, color="0.3", lw=0.5)
    for h in (-3, 3):
        ar.axhline(h, color="0.6", lw=0.5, ls=":")
    a.set_title(name, fontsize=9)
    a.set_xlim(w_lo, w_hi)
    ar.set_xlim(w_lo, w_hi)
    ar.set_xlabel(r"rest wavelength [$\AA$]")
    c = chi[k]
    c = c[np.isfinite(c)]
    if c.size:
        lines.append(
            f"{name:13s} n={len(c):3d} rms chi {np.sqrt(np.mean(c**2)):6.2f}  "
            f"mean chi {np.mean(c):+6.2f}  min {c.min():+7.2f}  max {c.max():+7.2f}"
        )
    else:
        lines.append(f"{name:13s} no fitted pixels")
axes[0, 0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "line_windows.png", dpi=140)
plt.close(fig)
print("\n".join(lines))

(FIG / "analysis_summary.txt").write_text("\n".join(summary + ["", table, ""] + lines) + "\n")
