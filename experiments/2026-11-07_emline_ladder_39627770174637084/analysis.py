"""Figures and tables of the emission-line ladder of 39627770174637084.

QUESTION: how do the fit and the posterior change along ladder steps 1-8 (MILES, C3K HR, C3K ER with
Gaussian LSF, own R, free line columns, recoadd, ratio broad widths)?
HYPOTHESIS: none fixed. Steps 3, 4 and 5 share pixels and compare by chi2 and log Z.
INPUTS: results/2026-11-07_emline_ladder_39627770174637084/<STEM>/predict.npz (fit.py predict),
<STEM>.h5 (fit.py run), pmc_final.state and mfrac.npz (mfrac.py); LOGMSTAR of FastSpecFit v4.0 and
LOGM of CIGALE v1.2 (SV3 bright) in data/.
SEED: none here (posterior draws in fit.py predict and mfrac.py, seed 0).
COMMAND: ``SPS_HOME=<fsps> uv run --no-sync python
experiments/2026-11-07_emline_ladder_39627770174637084/analysis.py``.
RESULT: max-likelihood chi2 / 5597 px is 8277.7 (step 5), 7418.6 (6), 6791.8 (7), 6383.6 (8).
log Z 101724.31 +- 0.55 (7), 101805.56 +- 0.28 (8).
Assumptions: steps 6 and 8 fit the recoadd, so they differ from 5 and 7 in the data too.
FIGURES: ladder.png, windows.png, ladder_table.txt, step5_line_ew.txt and
corner_{gas,stars_dust}_{1_4,7_8}.png in results/2026-11-07_emline_ladder_39627770174637084/.
Steps 7, 8 carry the labels of 5, 6.
"""

# %%
import importlib.util
from pathlib import Path

import astropy.units as u
import corner
import dill
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from prospect.io.read_results import results_from
from scipy.signal import medfilt

from hubersed.conversion import to_flambda
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style
from hubersed.sps.config import build_continuum_model, build_full_cue_model

TID = 39627770174637084
OUT = PATHS["RESULTS"] / "2026-11-07_emline_ladder_39627770174637084"
STEPS = {  # step: (stem, label, colour); colours of the 11-01 ladder (steps 5, 6: its unused ones)
    1: ("step1_dr1_miles_zeroed_cue_neff1024", "1 MILES", "#b2182b"),
    2: ("step2_dr1_c3khr_degraded_cue_neff1024", "2 C3K HR (degraded)", "#2166ac"),
    3: ("step3_dr1_median_cue_neff1024", "3 C3K ER, median LSF", "#1b7837"),
    4: ("step4_dr1_rmatrix_cue_neff1024", "4 C3K ER + R matrix", "#762a83"),
    5: ("step5_dr1_rmatrix_cols_neff1024", "5 C3K ER + R matrix + free lines", "#1b9e77"),
    6: ("step6_recoadd_rmatrix_cols_neff1024", "6 as 5, recoadd", "#e66101"),
    7: ("step7_dr1_rmatrix_cols_ratio_neff1024", "7 as 5, ratio widths", "#7570b3"),
    8: ("step8_recoadd_rmatrix_cols_ratio_neff1024", "8 as 6, ratio widths", "#e7298a"),
}
# the figures show steps 1-4 and, as 5 and 6, the ratio-width reruns 7 and 8;
# the tables keep all steps
FIG_STEPS = {s: STEPS[s] for s in (1, 2, 3, 4)} | {
    7: (STEPS[7][0], STEPS[5][1], STEPS[5][2]),
    8: (STEPS[8][0], STEPS[6][1], STEPS[6][2]),
}
# 2026-10-15_broadline_free_amplitude/map.py WINDOWS (rest vacuum A), used by the #14 windows figure
WINDOWS = {
    "[OII]3727": (3718, 3740),
    "[NeIII]3869": (3860, 3878),
    "[OIII]4363": (4355, 4372),
    "Hbeta": (4850, 4876),
    "[OIII]5007": (4997, 5020),
    "[OI]6300": (6290, 6312),
    "Halpha+[NII]": (6540, 6595),
    "[SII]": (6705, 6745),
    "[OII]7325": (7312, 7340),
}
N_MEDIAN = 9


def load_module(name, path):
    """Return the module at ``path``."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ladder = load_module(
    "ladder_plot_1101",
    Path(__file__).resolve().parents[1] / "2026-11-01_poco_repeats" / "ladder_plot.py",
)
ladder.CUTOUTS = OUT / "cutouts"  # cache this target's cutout here, not in 11-01's results
use_apj_style()
z = float(load_spectrum(TID).redshift.value)


def flam(wave, maggies):
    """Return maggies as DESI flux density, 1e-17 erg/s/cm^2/A."""
    return to_flambda(wave * u.AA, np.asarray(maggies, float) * u.mgy).value


# %% load what exists
pred, post, logz = {}, {}, {}
for s, (stem, _, _) in STEPS.items():
    d = OUT / stem
    if (d / "predict.npz").exists():
        pred[s] = dict(np.load(d / "predict.npz", allow_pickle=True))
    if (d / f"{stem}.h5").exists():
        res, _, _ = results_from(str(d / f"{stem}.h5"))
        labels = []
        for name in res["chain"].dtype.names:  # vector parameters become name_1 ... name_n
            n = int(np.prod(res["chain"].dtype[name].shape or (1,)))
            labels += [name] if n == 1 else [f"{name}_{i + 1}" for i in range(n)]
        post[s] = (res["unstructured_chain"], res["weights"] / res["weights"].sum(), labels)
    if (d / "pmc_final.state").exists():
        with open(d / "pmc_final.state", "rb") as f:
            st = dill.load(f)
        logz[s] = (float(st["logz"]), float(st["logz_err"]))
print("predict:", sorted(pred), "posterior:", sorted(post), "log Z:", sorted(logz))
wave0 = pred[min(pred)]["wave"]
for s, p in pred.items():
    assert np.allclose(p["wave"], wave0), f"step {s} is on another grid"
    p["rest"] = p["wave"] / (1 + z)
    p["chi"] = np.where(p["mask"], p["chi_ml"], np.nan)

# %% model ladder, as 2026-11-01_poco_repeats/ladder_plot.py
top = max(
    s for s in pred if s >= 3 and s in FIG_STEPS
)  # newest figure step on 3320-9000 A (8: recoadd)
p = pred[top]
rest = p["rest"]
data = np.where(p["mask"], flam(p["wave"], p["flux"]), np.nan)
med = np.full(data.size, np.nan)
med[p["mask"]] = medfilt(data[p["mask"]], N_MEDIAN)
lo = min(pred[s]["rest"][pred[s]["mask"]].min() for s in pred)
hi = max(pred[s]["rest"][pred[s]["mask"]].max() for s in pred)
fig, ax = plt.subplots(1 + len(FIG_STEPS), 1, figsize=(14, 4 + 1.6 * len(FIG_STEPS)), sharex=True,
                       gridspec_kw={"height_ratios": [3] + [1] * len(FIG_STEPS), "hspace": 0.06})  # fmt: skip
ax[0].plot(rest, data, color="0.6", lw=0.6, label="DESI Spectrum")
ax[0].plot(rest, med, color="k", lw=0.5, label="Median Filter")
ax[0].plot(rest, np.where(p["mask"], flam(p["wave"], p["spec_ml"]), np.nan), color=FIG_STEPS[top][2],
           lw=0.9, label="Model Fit")  # fmt: skip
ax[0].set_ylim(bottom=0)
ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
ax[0].legend(frameon=False, fontsize=16, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3)
for a, (s, (_, label, colour)) in zip(ax[1:], FIG_STEPS.items(), strict=True):
    if s in pred:
        a.plot(pred[s]["rest"], pred[s]["chi"], color=colour, lw=0.5)
        g = np.isfinite(pred[s]["chi"])
        print(f"{label:36s} chi2 {np.sum(pred[s]['chi'][g] ** 2):8.1f} on {g.sum()} px, "
              f"chi2/N {np.mean(pred[s]['chi'][g] ** 2):.3f}")  # fmt: skip
    else:
        label = f"{label} (not run yet)"
    for h in (-2, 2):
        a.axhline(h, color="0.3", ls="--", lw=1)
    a.axhline(0, color="0.3", lw=0.6)
    a.set_ylim(-5, 5)
    a.set_ylabel(r"$\chi$", fontsize=14)
    a.text(0.01, 0.93, label, transform=a.transAxes, ha="left", va="top", fontsize=11,
           bbox=dict(facecolor="white", edgecolor=colour, boxstyle="round,pad=0.25"))  # fmt: skip
ax[-1].set_xlim(lo, hi)
# Emission lines: a short blue tick at the top of the spectrum panel per line (vacuum rest
# wavelengths, as the axis), and one label above the axes per group of close lines.
EMLINES = {  # label: (label position, tick wavelengths)
    "[OII]": (3728.5, (3727.09, 3729.88)),
    "[NeIII]": (3869.86, (3869.86,)),
    r"H$\delta$": (4102.89, (4102.89,)),
    r"H$\gamma$": (4341.69, (4341.69, 4364.44)),
    r"H$\beta$": (4862.68, (4862.68,)),
    "[OIII]": (4984.3, (4960.30, 5008.24)),
    "HeI": (5877.25, (5877.25,)),
    "[OI]": (6302.05, (6302.05,)),
    r"H$\alpha$+[NII]": (6564.61, (6549.86, 6564.61, 6585.27)),
    "[SII]": (6725.5, (6718.29, 6732.67)),
    "[ArIII]": (7137.77, (7137.77,)),
    "[OII]7325": (7327.1, (7321.94, 7332.21)),
}
for name, (mid, waves) in EMLINES.items():
    for w in waves:
        ax[0].axvline(w, ymin=0.96, ymax=1.0, color="tab:blue", lw=1.2)
    if lo < mid < hi:
        ax[0].text(mid, 1.01, name, transform=ax[0].get_xaxis_transform(), rotation=90, fontsize=9,
                   ha="center", va="bottom", color="tab:blue")  # fmt: skip
ins = ax[0].inset_axes([0.885, 0.64, 0.11, 0.34])
img = ladder.cutout(TID)
ins.imshow(img, origin="upper")
c = ((img.width - 1) / 2, (img.height - 1) / 2)
ins.add_patch(Circle(c, 0.5 * ladder.FIBRE_DIAM / ladder.PIXSCALE, fill=False, ec="lime", lw=1.0))
ins.set_axis_off()
ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
fig.savefig(OUT / "ladder.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% line windows, as 2026-10-18_issue14_figures windows_figure: each step against its own data
fig, ax = plt.subplots(2, len(WINDOWS), figsize=(3.2 * len(WINDOWS), 6.0), sharex="col",
                       gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05})  # fmt: skip
for i, (key, (a, b)) in enumerate(WINDOWS.items()):
    win = (rest >= a) & (rest < b)
    ax[0, i].plot(rest[win], data[win], color="0.2", lw=1.0)
    handles = []
    for s, (_, _, colour) in FIG_STEPS.items():
        if s not in pred:
            continue
        r = pred[s]
        w = (r["rest"] >= a) & (r["rest"] < b)
        m = w & r["mask"]
        if not m.any():
            continue
        c2n = float(np.sum(r["chi"][m] ** 2) / m.sum())
        (h,) = ax[0, i].plot(r["rest"][w], np.where(r["mask"][w], flam(r["wave"][w], r["spec_ml"][w]), np.nan),
                             color=colour, lw=1.1, label=rf"$\chi^2/N$ = {c2n:.1f}")  # fmt: skip
        handles.append(h)
        ax[1, i].plot(r["rest"][w], r["chi"][w], color=colour, lw=0.9)
    ax[0, i].set_title(key, fontsize=14)
    ylo, yhi = ax[0, i].get_ylim()
    ax[0, i].set_ylim(ylo, ylo + 1.45 * (yhi - ylo))  # room for the chi2 legend above the lines
    ax[0, i].legend(handles=handles, frameon=False, fontsize=10, loc="upper left")
    for h in (-2, 2):
        ax[1, i].axhline(h, color="0.3", ls=":", lw=0.8)
    ax[1, i].axhline(0, color="0.3", lw=0.6)
    ax[1, i].set_ylim(-5, 5)
    ax[1, i].set_xlabel(r"rest wavelength [$\AA$]", fontsize=12)
ax[0, 0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ cgs $\AA^{-1}$]", fontsize=13)
ax[1, 0].set_ylabel(r"$\chi$", fontsize=13)
handles = [plt.Line2D([], [], color="0.2", lw=2)] + [
    plt.Line2D([], [], color=c, lw=3) for s, (_, _, c) in FIG_STEPS.items() if s in pred
]
coadd = "recoadd" if top in (6, 8) else "DR1 coadd"
names = [f"DESI {coadd} (the C3K HR fit saw a degraded copy)"] + [
    label for s, (_, label, _) in FIG_STEPS.items() if s in pred
]
fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.915), ncol=7, fontsize=13,
           frameon=False)  # fmt: skip
fig.savefig(OUT / "windows.png", dpi=130, bbox_inches="tight")
plt.close(fig)


# %% table: log Z, chi2, posterior medians and 16/84
def wquantile(x, w, q):
    """Return weighted quantiles ``q`` of ``x``."""
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return np.interp(q, c / c[-1], x[o])


def column(s, name):
    """Return the samples of ``name`` in step ``s``, or None if the step does not fit it."""
    chain, _, labels = post[s]
    return chain[:, labels.index(name)] if name in labels else None


lines = ["step | log Z | ML chi2 / n_pix"]
for s, (_, label, _) in STEPS.items():
    lz = f"{logz[s][0]:.2f} +- {logz[s][1]:.2f}" if s in logz else "-"
    ch = f"{float(pred[s]['chi2_ml']):.1f} / {int(pred[s]['n_pix'])}" if s in pred else "-"
    lines.append(f"{label} | {lz} | {ch}")
lines += [
    "",
    "parameter | " + " | ".join(STEPS[s][1] for s in sorted(post)) + "  (median [16, 84])",
]
all_names = []
for s in sorted(post):
    all_names += [n for n in post[s][2] if n not in all_names]
for name in all_names:
    cells = []
    for s in sorted(post):
        x = column(s, name)
        if x is None:
            cells.append("-")
            continue
        q = wquantile(x, post[s][1], [0.16, 0.5, 0.84])
        cells.append(f"{q[1]:.4g} [{q[0]:.4g}, {q[2]:.4g}]")
    lines.append(f"{name} | " + " | ".join(cells))
(OUT / "ladder_table.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines[: 2 + len(STEPS)]))

# %% step 5 free-column equivalent widths (truncated-mean amplitudes, Angstrom)
if 5 in pred:
    p5 = pred[5]
    rows = ["column | W max | EW max-likelihood | EW median [16, 84] of 100 draws"]
    for i, name in enumerate(p5["column_names"]):
        rows.append(f"{name} | {float(p5['column_wmax'][i]):.1f} | {float(p5['ew_ml'][i]):.3g} | "
                    f"{float(p5['ew_p50'][i]):.3g} [{float(p5['ew_p16'][i]):.3g}, "
                    f"{float(p5['ew_p84'][i]):.3g}]")  # fmt: skip
    (OUT / "step5_line_ew.txt").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))

# %% corners of steps 1 and 4, and 7 and 8: gas, and stars and dust; 68 and 95% contours
# Gas uses the weighted chain. Stars and dust use the 1000 draws of mfrac.py, which carry the
# surviving mass. Grey dashed lines are prior edges inside the plotted range, black lines catalog
# masses as catalogued (FastSpecFit v4.0 dashed, CIGALE v1.2 dotted). The fits assume WMAP9, CIGALE
# H0 = 70.4. FastSpecFit's data model says h = 1, but its masses match CIGALE's to a median of
# 0.00 dex over the SV3 bright sample, so its h convention is unresolved.
CORNER_STEPS = ((1, 4), (7, 8))
GAS = {
    "gas_logz": r"$\log Z_{\rm gas}/Z_\odot$",
    "gas_logno": r"[N/O]",
    "gas_logu": r"$\log U$",
    "gas_lognH": r"$\log n_{\rm H}$",
    "gas_logco": r"[C/O]",
}
STARS = {
    "logzsol": r"$\log Z_\star/Z_\odot$",
    "logmass_surv": r"$\log M_\star/M_\odot$",
    "dust2": r"$\tau_2$",
    "dust_ratio": r"$\tau_1/\tau_2$",
}
with fits.open(PATHS["DATA"] / "fastspecfit_v4.0" / "fastspec-iron-sv3-bright.fits") as h:
    d = h["SPECPHOT"].data
    m_fsf = float(d["LOGMSTAR"][d["TARGETID"] == TID][0])
with fits.open(PATHS["DATA"] / "cigale_dr1" / "IronPhysProp_v1.2.fits", memmap=True) as h:
    d = h[1].data
    m_cig = float(d["LOGM"][(d["TARGETID"] == TID) & (d["PROGRAM"] == "bright")][0])
CATALOG = {"FastSpecFit": (m_fsf, "--"), "CIGALE": (m_cig, ":")}
print(f"catalog masses: FastSpecFit {m_fsf:.3f}, CIGALE {m_cig:.3f}")
cont_model, cont_tmpl = build_continuum_model(z)
_, prior_tmpl = build_full_cue_model(cont_tmpl, cont_model.theta, cont_model)


def prior_edges(name):
    """Return the prior bounds of ``name``, or none for a derived quantity."""
    if name not in prior_tmpl:
        return ()
    params = prior_tmpl[name]["prior"].params
    return tuple(params[k] for k in ("mini", "maxi") if k in params)


def corner_samples(s, names):
    """Return the samples of ``names`` in step ``s`` and their weights."""
    chain, w, labels = post[s]
    if "logmass_surv" not in names:
        return np.column_stack([chain[:, labels.index(n)] for n in names]), w
    mf = np.load(OUT / STEPS[s][0] / "mfrac.npz")
    cols = [
        mf["logmass_surv"] if n == "logmass_surv" else chain[mf["index"], labels.index(n)]
        for n in names
    ]
    return np.column_stack(cols), np.full(len(mf["index"]), 1 / len(mf["index"]))


for group, names in (("gas", GAS), ("stars_dust", STARS)):
    for pair in CORNER_STEPS:
        if group == "stars_dust" and not all(
            (OUT / STEPS[s][0] / "mfrac.npz").exists() for s in pair
        ):
            print(f"corner {group} {pair}: no mfrac.npz yet, run mfrac.py")
            continue
        samp = {s: corner_samples(s, list(names)) for s in pair}
        ranges = []
        for j, n in enumerate(names):
            lo = min(wquantile(x[:, j], w, 0.001) for x, w in samp.values())
            hi = max(wquantile(x[:, j], w, 0.999) for x, w in samp.values())
            if n == "logmass_surv":
                masses = [v for v, _ in CATALOG.values()]
                lo, hi = min(lo, *masses), max(hi, *masses)
            ranges.append((lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)))
        fig = None
        for s in pair:
            x, w = samp[s]
            fig = corner.corner(x, weights=w, range=ranges, labels=list(names.values()),
                                color=FIG_STEPS[s][2], fig=fig, bins=30, smooth=1.0, smooth1d=1.0,
                                levels=(0.68, 0.95), fill_contours=True, plot_datapoints=False,
                                plot_density=False, max_n_ticks=3,
                                label_kwargs={"fontsize": 10})  # fmt: skip
        k = len(names)
        axes = np.array(fig.axes).reshape(k, k)
        for i in range(k):
            axes[i, i].relim()
            axes[i, i].autoscale(axis="y")
            axes[i, i].set_ylim(bottom=0)
        handles = [Patch(color=FIG_STEPS[s][2], label=FIG_STEPS[s][1]) for s in pair]
        edge_drawn = False
        for j, n in enumerate(names):
            marks = [(e, "0.5", "--") for e in prior_edges(n) if ranges[j][0] < e < ranges[j][1]]
            edge_drawn |= bool(marks)
            if n == "logmass_surv":
                marks += [(v, "k", ls) for v, ls in CATALOG.values()]
            for v, colour, ls in marks:
                for i in range(j, k):
                    axes[i, j].axvline(v, color=colour, ls=ls, lw=0.8)
                for c in range(j):
                    axes[j, c].axhline(v, color=colour, ls=ls, lw=0.8)
        if edge_drawn:
            handles.append(Line2D([], [], color="0.5", ls="--", lw=0.8, label="prior edge"))
        if group == "stars_dust":
            handles += [Line2D([], [], color="k", ls=ls, lw=0.8, label=f"{name} mass")
                        for name, (_, ls) in CATALOG.items()]  # fmt: skip
        fig.set_size_inches(1.1 * k + 0.6, 1.1 * k + 0.6)
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        fig.savefig(OUT / f"corner_{group}_{pair[0]}_{pair[1]}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote corner_{group}_{pair[0]}_{pair[1]}.png")
