"""Figures and tables of the emission-line ladder of 39627770174637084.

QUESTION: how do the fit and the posterior change along the ladder (1 MILES, 2 C3K HR, 3 C3K ER with
a Gaussian LSF, 4 with the own R, 5 with free weak and broad line columns, 6 as 5 on the recoadd,
7-8 as 5-6 with broad widths as a ratio of the narrow width)?
HYPOTHESIS: none fixed; steps 3, 4 and 5 share pixels and compare by chi2 and log Z, steps 1 and 2
fit other pixels and are shown, not compared.
INPUTS: results/2026-11-07_emline_ladder_39627770174637084/<STEM>/predict.npz (fit.py predict),
<STEM>.h5 (fit.py run) and pmc_final.state (log Z), copied from stampede3. Missing steps are marked
"not run yet". Figures reuse two designs: the model ladder of 2026-11-01_poco_repeats/ladder_plot.py
(its cutout and fibre circle are imported from it; emission lines marked instead of its index
bands, Nikhil 11-11) and the line windows of
2026-10-18_issue14_figures windows_figure (windows of 2026-10-15_broadline_free_amplitude/map.py).
SEED: none here (posterior draws in fit.py predict, seed 0).
COMMAND: ``SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-11-07_emline_ladder_39627770174637084/analysis.py``.
RESULT: max-likelihood chi2 / 5597 px: step 5 8277.7 (1.48), 6 7418.6 (1.33); with ratio broad
widths 7 6791.8 (1.21), 8 6383.6 (1.14); log Z 101724.31 +- 0.55 (7), 101805.56 +- 0.28 (8).
Windows chi2/N, step 8: Halpha+[NII] 2.1, [OIII]5007 4.2, Hbeta 1.6 (step 6: 13.3, 8.2, 2.5). The
figures show steps 1-4 and 7, 8 under the labels of 5, 6 (Nikhil 11-10). Assumption: steps 6 and
8 fit the recoadd, so they differ from 5 and 7 in the data, not only the model.
FIGURES: ladder.png, windows.png, ladder_table.txt, step5_line_ew.txt in
results/2026-11-07_emline_ladder_39627770174637084/.
"""

# %%
import importlib.util
from pathlib import Path

import astropy.units as u
import dill
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from prospect.io.read_results import results_from
from scipy.signal import medfilt

from hubersed.conversion import to_flambda
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

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
# the figures show steps 1-4 and, as 5 and 6, the ratio-width reruns 7 and 8 (Nikhil 11-10);
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
