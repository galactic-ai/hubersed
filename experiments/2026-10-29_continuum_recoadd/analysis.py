"""DR1 coadd against PR 2377 recoadd continuum MAPs of the continuum outlier 39627757533007793 (#17).

QUESTION: how do the continuum MAP parameters and residuals change when the SV3 and main spectra
are recoadded with the PR 2377 row-weighted resolution matrix and the fixed flux-calibration ivar,
as for the #14 line outlier?
HYPOTHESIS: none registered.
INPUTS: <TID>_map.npz of the DR1-coadd MAPs (2026-10-28_continuum_c3k_er_rmatrix_map for SV3,
2026-10-29_continuum_main_survey_map for main) and of the recoadd MAPs in
results/2026-10-29_continuum_recoadd/<survey>/, each with the data it fitted, copied from
stampede3. Index bands from $SPS_HOME/data/allindices.dat.
SEED: none.
COMMAND: from the repository root on the Mac,
``SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-29_continuum_recoadd/analysis.py``.
RESULT: 2026-10-29, dr1_vs_recoadd.txt. chi2/(N-16) drops from 1.874 to 1.047 (SV3) and from
1.152 to 0.916 (main). The SV3 red end (rest 6300-8800 A, chi2/N 3.0-4.4) falls to 0.84-1.39.
The Mgb deficit stays (mean chi -0.42 SV3, -0.47 main), the main NaD deficit stays (-0.61) and the
SV3 Halpha-core deficit shrinks (-1.16 to -0.71). SV3 and main still disagree: afe -0.03 vs 0.30,
logzsol +0.19 vs -0.13, dust_index -2.5 (edge) vs -1.05. sigma_smooth is 275 vs 277 km/s.
Assumptions: MAP only (7 Powell starts), SFH hyperparameters fixed, no calibration polynomial.
FIGURES: results/2026-10-29_continuum_recoadd/<survey>_dr1_vs_recoadd.png, and
dr1_vs_recoadd.txt.
"""

# %%
import os
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

from hubersed.conversion import to_flambda
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39627757533007793
RES = PATHS["RESULTS"]
OUT = RES / "2026-10-29_continuum_recoadd"
RUNS = {
    "sv3": {
        "DR1": RES / "2026-10-28_continuum_c3k_er_rmatrix_map" / f"{TID}_map.npz",
        "recoadd": OUT / "sv3" / f"{TID}_map.npz",
    },
    "main": {
        "DR1": RES / "2026-10-29_continuum_main_survey_map" / f"{TID}_map.npz",
        "recoadd": OUT / "main" / f"{TID}_map.npz",
    },
}
COLORS = {"DR1": "#e6c35c", "recoadd": "#b2182b"}  # DR1 fit light gold, recoadd fit red
BINS = [(lo, lo + 500) for lo in range(3300, 8800, 500)]
# rest-frame windows, Angstrom: the Mgb and NaD feature bands of allindices.dat and the Ha core
WINDOWS = {"Mgb": (5160.1, 5192.6), "NaD": (5876.9, 5909.4), "Ha core": (6555.0, 6575.0)}
INDEX_NAMES = "Dn4000 HdeltaA CN1 Ca4227 G4300 C4668 Hbeta Mgb Fe5270 Fe5335 NaD TiO1 TiO2".split()
N_MEDIAN = 9


def flam(d, x):
    """Return maggies on the fit's wavelength grid in 1e-17 erg/s/cm^2/A."""
    return to_flambda(d["wave"] * u.AA, np.asarray(x, float) * u.mgy).value


def rest(d):
    """Return the rest-frame wavelength of a fit."""
    return d["wave"] / (1 + float(d["z"]))


def chi(d):
    """Return (data - model) / sigma on the fitted pixels, NaN elsewhere."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d["good"], (d["flux"] - d["sp_best"]) / d["unc"], np.nan)


def index_bands():
    """Return each index's shaded rest-frame spans, as in 2026-10-28 analysis.py."""
    table = np.genfromtxt(
        Path(os.environ["SPS_HOME"]) / "data" / "allindices.dat",
        dtype=None,
        encoding=None,
        comments="#",
    )
    out = {}
    for r in table:
        name = str(r[-1])
        if name in INDEX_NAMES:
            f0, f1, b0, b1, r0, r1, kind = (r[0], r[1], r[2], r[3], r[4], r[5], int(r[6]))
            out[name] = [(b0, b1), (r0, r1)] if kind == 3 else [(f0, f1)]
    return out


# %% tables
fits = {s: {k: np.load(p) for k, p in v.items()} for s, v in RUNS.items()}
lines = []
for survey, f in fits.items():
    a, b = f["DR1"], f["recoadd"]
    labels = [str(x) for x in a["labels"]]
    assert labels == [str(x) for x in b["labels"]]
    lines += [f"== {survey}", f"{'parameter':>16} {'DR1':>10} {'recoadd':>10} {'change':>10}"]
    for i, name in enumerate(labels):
        x, y = float(a["best"][i]), float(b["best"][i])
        lines.append(f"{name:>16} {x:10.4f} {y:10.4f} {y - x:+10.4f}")
    lines += ["", f"chi2/N on each fit's own data by rest-frame bin (pixels); N free {len(labels)}"]
    for lo, hi in BINS:
        row = f"{lo:5d}-{hi:5d}"
        for k, d in f.items():
            s = d["good"] & (rest(d) >= lo) & (rest(d) < hi)
            row += f"  {k}: {np.nanmean(chi(d)[s] ** 2):6.2f} ({s.sum():4d})" if s.any() else ""
        lines.append(row)
    for k, d in f.items():
        c2 = float(np.nansum(chi(d) ** 2))
        n = int(d["good"].sum())
        lines.append(
            f"{k}: chi2 {c2:.1f} on {n} px, chi2/(N - {len(labels)}) {c2 / (n - len(labels)):.3f}"
        )
    lines += ["", "windows: median data, median model (1e-17 erg/s/cm^2/A), mean chi (pixels)"]
    for name, (lo, hi) in WINDOWS.items():
        row = f"{name:>8}"
        for k, d in f.items():
            s = d["good"] & (rest(d) >= lo) & (rest(d) <= hi)
            row += (f" | {k}: {np.median(flam(d, d['flux'])[s]):6.2f} "
                    f"{np.median(flam(d, d['sp_best'])[s]):6.2f} {np.nanmean(chi(d)[s]):+5.2f} ({s.sum()})")  # fmt: skip
        lines.append(row)
    lines.append("")
text = "\n".join(lines)
(OUT / "dr1_vs_recoadd.txt").write_text(text)
print(text)

# %% figures: recoadded data with both fits, then the chi of both fits in one panel
use_apj_style()
bands = index_bands()
for survey, f in fits.items():
    m = f["recoadd"]
    g = m["good"]
    data = flam(m, m["flux"])
    fig, ax = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}
    )
    ax[0].plot(rest(m), np.where(g, data, np.nan), color="0.75", lw=0.5,
               label=f"DESI {survey}, PR 2377 recoadd")  # fmt: skip
    med = np.full(data.size, np.nan)
    med[g] = medfilt(data[g], N_MEDIAN)
    ax[0].plot(rest(m), med, color="k", lw=0.8, label=f"median filter, {N_MEDIAN} pixels")
    for k, label in (("DR1", "MAP on the DR1 coadd"), ("recoadd", "MAP on the recoadd")):
        d = f[k]
        ax[0].plot(rest(d), np.where(d["good"], flam(d, d["sp_best"]), np.nan), color=COLORS[k],
                   lw=0.8, label=label)  # fmt: skip
    ax[0].set_ylim(bottom=0)
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
    ax[0].legend(frameon=False, fontsize=12, loc="upper right")
    # one residual panel: the DR1 fit on its data, with the recoadd fit on its data on top
    for k in ("DR1", "recoadd"):
        ax[1].plot(rest(f[k]), chi(f[k]), color=COLORS[k], lw=0.5)
    for h in (-2, 2):
        ax[1].axhline(h, color="0.3", ls="--", lw=1)
    ax[1].axhline(0, color="0.3", lw=0.6)
    ax[1].set_ylim(-5, 5)
    ax[1].set_ylabel(r"$\chi$", fontsize=14)
    for name, spans in bands.items():
        for s0, s1 in spans:
            for a in ax:
                a.axvspan(s0, s1, color="0.5", alpha=0.12, lw=0)
        mid = 0.5 * (spans[0][0] + spans[-1][1])
        ax[0].text(mid, 1.01, name, transform=ax[0].get_xaxis_transform(), rotation=90,
                   fontsize=9, ha="center", va="bottom")  # fmt: skip
    ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
    path = OUT / f"{survey}_dr1_vs_recoadd.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
