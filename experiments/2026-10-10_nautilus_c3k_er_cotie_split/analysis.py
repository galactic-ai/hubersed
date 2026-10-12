"""QUESTION: How does the converged C/O tie fit of 39627770174637084 (soft C/O tie, two forbidden
widths, He I on the Balmer profile) differ from the ER fit of the same data, in evidence, posteriors,
SFH and the line-window misfits that motivated the three changes?
HYPOTHESIS: None registered before this analysis.
INPUTS: results/2026-10-10_nautilus_c3k_er_cotie_split/<stem>_pred.npz from predict.py, with
<stem> = 39627770174637084_er_cotie_split_broad_shared_split45_rest9000_seed0, and the ER run's
results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084_er_broad_shared_split45_rest9000_seed0_pred.npz.
Both fits used the same flux, uncertainty and fitted pixels (asserted below), so log Z and chi2
compare directly. The vacuum wavelengths behind the extra windows are from
cuejax/data/cue_emlines_info.dat.
SEED: 0 for the SFH draws here, 0 for the 100 spectrum draws in predict.py.
COMMAND: from the repository root, on the Mac or an ls6 compute node,
uv run python experiments/2026-10-10_nautilus_c3k_er_cotie_split/analysis.py
RESULT: see analysis_summary.txt next to the figures.
FIGURES: results/2026-10-10_nautilus_c3k_er_cotie_split/39627770174637084/. analysis_summary.txt,
params_table.txt, line_flux_table.txt, corner_phys.png, corner_sfh.png, corner_lines.png (C/O tie
filled, ER contours), sfh.png and line_windows.png.
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

from hubersed.conversion import to_flambda
from hubersed.fitting.map_fits import sfh_from_theta
from hubersed.paths import PATHS
from hubersed.plotting.sfh import plot_cumulative_mass, plot_ssfr, sfh_figure
from hubersed.plotting.spectra import residual_chi
from hubersed.sps.nebular_ties import CUE_LOGCO_RANGE, co_from_oh

# parse_known_args, so the cells also run in an interactive session
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tid", type=int, default=39627770174637084)
TID = parser.parse_known_args()[0].tid
OUT = PATHS["RESULTS"] / "2026-10-10_nautilus_c3k_er_cotie_split"
FIG = OUT / str(TID)
FIG.mkdir(parents=True, exist_ok=True)
ER_DIR = "2026-10-09_nautilus_c3k_er_broadline"
STEMS = {
    "ER": PATHS["RESULTS"] / ER_DIR / f"{TID}_er_broad_shared_split45_rest9000_seed0_pred.npz",
    "C/O tie": OUT / f"{TID}_er_cotie_split_broad_shared_split45_rest9000_seed0_pred.npz",
}
GEN = np.random.default_rng(0)
COL = {"ER": "#b2182b", "C/O tie": "#2166ac"}
LINE_PARS = [
    "eline_sigma",
    "eline_sigma_forb",
    "eline_sigma_forb_hi",
    "eline_fbroad",
    "eline_sigma_broad",
    "eline_vbroad",
    "gas_logz",
    "gas_logu",
    "gas_lognH",
    "gas_logno",
    "gas_logco",
    "gas_dlogco",
]
# Rest-frame vacuum windows. The first fifteen are those of line_flux_check.py in the ER experiment.
# The last three are added here around the Cue lines at [SIII] 6313.81, He I 6679.99 and
# [OII] 7321.79 and 7332.27. The left sideband of [SIII]6312 holds [OI] 6302.04, so its R/F and
# scale are biased there; R/err does not use the baseline.
EXTRA_WINDOWS = {
    "[SIII]6312": (6309, 6319),
    "HeI6678": (6672, 6688),
    "[OII]7320/30": (7314, 7340),
}
# windows of the line figure
PANELS = {
    "[NeIII]3869": (3862, 3876),
    "[OIII]4363": (4359, 4370),
    "Hbeta": (4845, 4880),
    "[OIII]5007": (4995, 5022),
    "HeI5876": (5868, 5885),
    "Halpha+[NII]": (6540, 6595),
    "[SII]": (6708, 6742),
}


def wquantile(x, w, q):
    """Weighted quantiles of samples x (weights w, summing to 1)."""
    o = np.argsort(x)
    return np.interp(q, np.cumsum(w[o]), x[o])


def load_module(path, name):
    """Import a script from another experiment under its own module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# %% both posteriors and spectra from predict.py, with the parameters each run lacks derived
er_fit = load_module(PATHS["ROOT"] / "experiments" / ER_DIR / "fit.py", "fit1009")
lfc = load_module(PATHS["ROOT"] / "experiments" / ER_DIR / "line_flux_check.py", "lfc1009")
WINDOWS = {**lfc.WINDOWS, **EXTRA_WINDOWS}

runs = {}
for key, path in STEMS.items():
    d = dict(np.load(path))
    d["labels"] = [str(x) for x in d["labels"]]
    d["w"] = np.exp(d["log_w"] - d["log_w"].max())
    d["w"] /= d["w"].sum()
    runs[key] = d
er, co = runs["ER"], runs["C/O tie"]
for k in ("wave", "z", "flux", "unc", "good"):
    assert np.array_equal(er[k], co[k]), k
z = float(er["z"])
models = {"ER": er_fit.make_model(z, "shared", 45.0), "C/O tie": make_model(z, "shared", 45.0)}
for key, d in runs.items():
    assert list(models[key].theta_labels()) == d["labels"], key
    d["cols"] = {lab: d["points"][:, i] for i, lab in enumerate(d["labels"])}

# C/O tie: gas_logco follows from gas_logz and the offset, as in the fit.
co["cols"]["gas_logco"] = co_from_oh(co["cols"]["gas_logz"], co["cols"]["gas_dlogco"])
# ER: the offset of its free gas_logco from the Nicholls et al. (2017) relation at its gas_logz.
# The ER prior on gas_logco was flat, so this is not drawn from the N(0, 0.17) prior of the tie.
er["cols"]["gas_dlogco"] = er["cols"]["gas_logco"] - co_from_oh(er["cols"]["gas_logz"], 0.0)
# ER has one width for all forbidden lines, so it also holds for the high-ionization group.
er["cols"]["eline_sigma_forb_hi"] = er["cols"]["eline_sigma_forb"]
labels = list(co["labels"])
labels.insert(labels.index("gas_dlogco"), "gas_logco")


def column(d, lab):
    return d["cols"][lab]


def prior_range(m, lab):
    """Prior edges of a free scalar parameter, NaN otherwise.

    gas_logco gets the edges of Cue's grid, where the ER prior ends and co_from_oh clips.
    A Normal prior has no edges.
    """
    if lab == "gas_logco":
        return np.array(CUE_LOGCO_RANGE)
    c = m.config_dict.get(lab, {})
    p = c.get("prior")
    if not c.get("isfree") or p is None or c.get("N", 1) > 1 or type(p).__name__ == "Normal":
        return np.array([np.nan, np.nan])
    return np.ravel(p.range)[:2]


# %% fit quality and posterior weights of both runs
summary = []
for key, d in runs.items():
    d["chi"] = residual_chi(d["flux"], d["sp_best"], d["unc"], d["good"])
    n_pix = int(np.isfinite(d["chi"]).sum())
    d["chi2_nu"] = float(np.nansum(d["chi"] ** 2) / (n_pix - len(d["labels"])))
    w = d["w"]
    summary.append(
        f"{key}: {len(d['labels'])} free, N_eff {float(d['n_eff']):.0f}, calls {int(d['n_like'])}, "
        f"log Z {float(d['log_z']):.2f}, max lnL {float(d['max_lnl']):.2f}, chi2_nu "
        f"{d['chi2_nu']:.3f} over {n_pix} pixels, {len(w)} points kept, largest weight share "
        f"{w.max():.4f}, ten largest {np.sort(w)[-10:].sum():.4f}, Kish ESS {1 / np.sum(w**2):.0f}"
    )
d_lnl = float(co["max_lnl"]) - float(er["max_lnl"])
summary.append(
    f"C/O tie - ER: Delta log Z {float(co['log_z']) - float(er['log_z']):+.2f}, Delta max lnL "
    f"{d_lnl:+.2f} (Delta chi2 {-2 * d_lnl:+.1f}) at the best sampled points, not optimized"
)
print("\n".join(summary))

# %% parameter table: 16/50/84 for both runs, flagged where the median sits within 2% of a prior edge
rows = [f"{'parameter':20s}{'ER median [16, 84]':>40s}{'C/O tie median [16, 84]':>40s}"]
for lab in labels:
    row = f"{lab:20s}"
    for key, d in runs.items():
        a, mid, b = wquantile(column(d, lab), d["w"], [0.16, 0.5, 0.84])
        lo, hi = prior_range(models[key], lab)
        rail = ""
        if (mid - lo) < 0.02 * (hi - lo):
            rail = " FLOOR"
        elif (hi - mid) < 0.02 * (hi - lo):
            rail = " CEIL"
        row += f"{mid:12.4f} [{a:10.4f},{b:10.4f}]{rail:6s}".rjust(40)
    rows.append(row)
notes = [
    "ER gas_dlogco is derived (gas_logco minus Nicholls+17 at gas_logz), ER eline_sigma_forb_hi",
    "is its single eline_sigma_forb, and C/O tie gas_logco is derived from the tie.",
]
table = "\n".join(rows + [""] + notes)
print(table)
(FIG / "params_table.txt").write_text(table + "\n")

# %% corner plots, C/O tie filled and ER as contours, with dashed lines at prior edges inside an axis


def axis_range(lab, q=(0.001, 0.999), pad=0.05):
    """Central 99.8% of both posteriors plus 5% padding."""
    lo = min(wquantile(column(d, lab), d["w"], q[0]) for d in runs.values())
    hi = max(wquantile(column(d, lab), d["w"], q[1]) for d in runs.values())
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
        np.column_stack([column(er, lab) for lab in labs]),
        weights=er["w"],
        color=COL["ER"],
        range=rng,
        plot_datapoints=False,
        plot_density=False,
        smooth=1.0,
        bins=30,
        levels=(0.393, 0.865),  # 1 and 2 sigma for a 2-D Gaussian
        hist_kwargs={"density": True},
    )
    corner.corner(
        np.column_stack([column(co, lab) for lab in labs]),
        weights=co["w"],
        fig=fig,
        color=COL["C/O tie"],
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
        for edge in prior_range(models["C/O tie"], lab):
            if axes[j, j].get_xlim()[0] <= edge <= axes[j, j].get_xlim()[1]:
                axes[j, j].axvline(edge, color="0.3", ls="--", lw=0.8)
    fig.legend(
        handles=[
            Line2D([], [], color=COL["C/O tie"], label="C/O tie (2026-10-10), titles"),
            Line2D([], [], color=COL["ER"], label="ER (2026-10-09)"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=12,
    )
    fig.suptitle(f"{TID} {tag}", fontsize=12)
    fig.savefig(FIG / f"corner_{tag}.png", dpi=110)
    plt.close(fig)

# %% SFH: C/O tie median and 16-84% per bin from 2000 weighted draws, ER median for comparison
sfh = {}
for key, d in runs.items():
    m = models[key]
    draws = [sfh_from_theta(m, d["points"][i]) for i in GEN.choice(len(d["w"]), 2000, p=d["w"])]
    sfh[key] = dict(
        edges=draws[0]["edges_gyr"],
        ssfr=np.percentile([x["ssfr"] for x in draws], [16, 50, 84], axis=0),
        cmf=np.percentile([x["cmf"] for x in draws], [16, 50, 84], axis=0),
    )
s = sfh["C/O tie"]
fig, ax = sfh_figure(
    s["edges"],
    s["ssfr"][1],
    s["cmf"][1],
    ssfr_kwargs={"label": "C/O tie median", "ssfr_kw": {"color": COL["C/O tie"]}},
    cmf_kwargs={"color": COL["C/O tie"]},
)
for i, k in enumerate(("ssfr", "cmf")):
    ax[i].stairs(
        s[k][2], s["edges"], baseline=s[k][0], fill=True, color=COL["C/O tie"], alpha=0.25, lw=0
    )
plot_ssfr(
    ax[0],
    sfh["ER"]["edges"],
    sfh["ER"]["ssfr"][1],
    guides=(),
    label="ER median",
    ssfr_kw={"color": COL["ER"]},
)
plot_cumulative_mass(ax[1], sfh["ER"]["edges"], sfh["ER"]["cmf"][1], guides=(), color=COL["ER"])
ax[0].legend(frameon=True, fontsize="small", loc="lower left")
fig.savefig(FIG / "sfh.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% line windows: flux and profile split of the misfit at the max-L spectrum, as in line_flux_check
rest = er["wave"] / (1 + z)
dlam = np.gradient(er["wave"])
flux, unc, good = er["flux"], er["unc"], er["good"]


def window_stats(d, a, b):
    """Misfit of one rest-frame window at the max-L spectrum of run d, None without a baseline.

    R is the integrated data minus model, F the model line flux above the sideband baseline,
    scale the best single rescale of the model line, flux_pct the share of the window chi2 that
    rescale removes, and draw_sd half the 16-84% range of F over the posterior draws, over F.
    """
    m = good & (rest >= a) & (rest < b)
    model = d["sp_best"]
    base = lfc.baseline(rest, model, good, a, b)
    if m.sum() < 3 or base is None:
        return None
    resid = (flux - model)[m]
    r = np.sum(resid * dlam[m])
    line = (model - base)[m]
    f_line = np.sum(line * dlam[m])
    chi2 = np.sum((resid / unc[m]) ** 2)
    iv = 1 / unc[m] ** 2
    scale = np.sum(iv * line * (flux - base)[m]) / np.sum(iv * line**2)
    chi2_scaled = np.sum((((flux - base)[m] - scale * line) / unc[m]) ** 2)
    f_draws = [np.sum((sp - base)[m] * dlam[m]) for sp in d["draw_spec"]]
    lo, hi = np.percentile(f_draws, [16, 84])
    return dict(
        n=int(m.sum()),
        r_err=r / np.sqrt(np.sum((unc[m] * dlam[m]) ** 2)),
        r_f=r / f_line,
        f=f_line,
        scale=scale,
        chi2_px=chi2 / m.sum(),
        scaled_px=chi2_scaled / m.sum(),
        flux_pct=100 * (chi2 - chi2_scaled) / chi2,
        draw_sd=(hi - lo) / 2 / abs(f_line),
    )


stats = {
    key: {name: window_stats(d, *ab) for name, ab in WINDOWS.items()} for key, d in runs.items()
}
cols = f"{'R/err':>7s} {'R/F':>7s} {'chi2/px':>8s} {'flux%':>6s}"
rows = [
    "Line-window misfit at the max-L spectrum of each run (definitions in line_flux_check.py of",
    "the ER experiment). R/err: integrated data - model over its noise. R/F: as a fraction of the",
    "model line flux above the sideband baseline. chi2/px in the window. flux%: share of that chi2",
    "removed by the best single rescale of the model line, the rest is profile.",
    "",
    f"{'window':>13s} {'npix':>4s} | ER {cols} | C/O tie {cols}",
]
for name in WINDOWS:
    row = f"{name:>13s} "
    if stats["ER"][name] is None:
        a, b = WINDOWS[name]
        n = int((good & (rest >= a) & (rest < b)).sum())
        rows.append(row + f"{n:4d} fitted pixels, too few or a sideband fully masked")
        continue
    row += f"{stats['ER'][name]['n']:4d}"
    for key in runs:
        x = stats[key][name]
        row += (
            f" | {'':{3 if key == 'ER' else 8}s}{x['r_err']:+7.2f} {x['r_f']:+7.3f} "
            f"{x['chi2_px']:8.2f} {x['flux_pct']:6.1f}"
        )
    rows.append(row)

# Balmer decrement: data/model of each line is 1 + R/F, so their ratio is data/model of Ha/Hb.
# The sideband baseline does not remove stellar Balmer absorption and the Halpha left sideband
# holds [NII]6548, the same in both runs.
rows.append("")
for key in runs:
    ha, hb = stats[key]["Halpha"], stats[key]["Hbeta"]
    rows.append(
        f"{key}: data/model Halpha {1 + ha['r_f']:.3f}, Hbeta {1 + hb['r_f']:.3f}, so data/model "
        f"Halpha/Hbeta {(1 + ha['r_f']) / (1 + hb['r_f']):.3f}; model Halpha/Hbeta above the "
        f"baseline {ha['f'] / hb['f']:.3f}"
    )
flux_table = "\n".join(rows)
print(flux_table)
(FIG / "line_flux_table.txt").write_text(flux_table + "\n")


# %% line figure: data, both max-L spectra, chi of each run below
def flam(maggies):
    """Convert maggies to DESI f_lambda units, NaN outside the fitted pixels."""
    f = to_flambda(er["wave"] * u.AA, np.asarray(maggies, float) * u.mgy).value
    return np.where(good, f, np.nan)


lines = []
fig, axes = plt.subplots(
    2, len(PANELS), figsize=(18, 4.8), gridspec_kw=dict(height_ratios=[3, 1.4])
)
for j, (name, (w_lo, w_hi)) in enumerate(PANELS.items()):
    k = (rest > w_lo) & (rest < w_hi)
    a, ar = axes[0, j], axes[1, j]
    dd, e = flam(flux)[k], flam(unc)[k]
    a.fill_between(rest[k], dd - e, dd + e, color="0.8", step="mid", lw=0)
    a.step(rest[k], dd, "k", where="mid", lw=0.7, label="DESI")
    row = f"{name:13s}"
    for key, d in runs.items():
        a.plot(rest[k], flam(d["sp_best"])[k], color=COL[key], lw=1, label=f"{key} max-L")
        ar.step(rest[k], d["chi"][k], color=COL[key], where="mid", lw=0.8)
        c = d["chi"][k]
        c = c[np.isfinite(c)]
        row += (
            f"  {key}: n={len(c):3d} rms chi {np.sqrt(np.mean(c**2)):5.2f} mean {np.mean(c):+5.2f}"
        )
    lines.append(row)
    ar.axhline(0, color="0.3", lw=0.5)
    for h in (-3, 3):
        ar.axhline(h, color="0.6", lw=0.5, ls=":")
    a.set_title(name, fontsize=9)
    a.set_xlim(w_lo, w_hi)
    ar.set_xlim(w_lo, w_hi)
    ar.set_xlabel(r"rest wavelength [$\AA$]")
axes[0, 0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
axes[1, 0].set_ylabel(r"$\chi$")
axes[0, 0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "line_windows.png", dpi=140)
plt.close(fig)
print("\n".join(lines))

(FIG / "analysis_summary.txt").write_text(
    "\n".join(summary + ["", table, "", flux_table, ""] + lines) + "\n"
)
