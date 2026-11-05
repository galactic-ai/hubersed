"""SV3 against main-survey continuum MAPs of the continuum outlier 39627757533007793 (issue #17).

QUESTION: do the C3K ER + afe + R matrix continuum MAP parameters and residuals of the SV3
spectrum repeat on the independent DR1 main-survey exposure?
HYPOTHESIS: none registered.
INPUTS: <TID>_map.npz of 2026-10-28_continuum_c3k_er_rmatrix_map (SV3, icx job 3655467) and of
this folder (main, icx job 3656341), each with the data it fitted, copied from stampede3. Index
bands from $SPS_HOME/data/allindices.dat.
SEED: none.
COMMAND: from the repository root on the Mac,
``SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-29_continuum_main_survey_map/analysis.py``.
RESULT: pending.
FIGURES: results/2026-10-29_continuum_main_survey_map/sv3_vs_main.png, and sv3_vs_main.txt.
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
OUT = PATHS["RESULTS"] / "2026-10-29_continuum_main_survey_map"
RUNS = {
    "SV3": (
        PATHS["RESULTS"] / "2026-10-28_continuum_c3k_er_rmatrix_map" / f"{TID}_map.npz",
        "#b2182b",
    ),
    "main": (OUT / f"{TID}_map.npz", "#2166ac"),
}
INDEX_NAMES = "Dn4000 HdeltaA CN1 Ca4227 G4300 C4668 Hbeta Mgb Fe5270 Fe5335 NaD TiO1 TiO2".split()
BINS = [(lo, lo + 500) for lo in range(3300, 8800, 500)]
# rest-frame windows, Angstrom: the Mgb and NaD feature bands of allindices.dat and the Ha core
WINDOWS = {"Mgb": (5160.1, 5192.6), "NaD": (5876.9, 5909.4), "Ha core": (6555.0, 6575.0)}
N_MEDIAN = 9


def flam(d, x):
    """Return maggies on the fit's wavelength grid in 1e-17 erg/s/cm^2/A."""
    return to_flambda(d["wave"] * u.AA, np.asarray(x, float) * u.mgy).value


def index_bands():
    """Return each index's shaded rest-frame spans (feature, or Dn4000's two bands).

    As index_bands of 2026-10-28_continuum_c3k_er_rmatrix_map/analysis.py, from
    $SPS_HOME/data/allindices.dat.
    """
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


# %% load and compare
fits = {k: np.load(p) for k, (p, _) in RUNS.items()}
lines = []
labels = [str(x) for x in fits["SV3"]["labels"]]
assert labels == [str(x) for x in fits["main"]["labels"]]
lines += [f"{'parameter':>16} {'SV3':>10} {'main':>10} {'main-SV3':>10}"]
for i, name in enumerate(labels):
    a, b = float(fits["SV3"]["best"][i]), float(fits["main"]["best"][i])
    lines.append(f"{name:>16} {a:10.4f} {b:10.4f} {b - a:+10.4f}")
lines += ["", f"chi2/N on each fit's own data, by rest-frame bin; N free {len(labels)}"]
lines.append(f"{'rest A':>11} " + " ".join(f"{k:>14}" for k in fits))
chi = {}
for k, d in fits.items():
    g = d["good"]
    with np.errstate(divide="ignore", invalid="ignore"):
        chi[k] = np.where(g, (d["flux"] - d["sp_best"]) / d["unc"], np.nan)
for lo, hi in BINS:
    row = f"{lo:5d}-{hi:5d} "
    for k, d in fits.items():
        s = (
            d["good"]
            & (d["wave"] / (1 + float(d["z"])) >= lo)
            & (d["wave"] / (1 + float(d["z"])) < hi)
        )
        row += f" {np.mean(chi[k][s] ** 2):7.2f} ({s.sum():4d})" if s.any() else f" {'-':>14}"
    lines.append(row)
for k, d in fits.items():
    g = d["good"]
    c2 = float(np.nansum(chi[k][g] ** 2))
    lines.append(f"{k}: chi2 {c2:.1f} on {g.sum()} px, chi2/(N - {len(labels)}) "
                 f"{c2 / (g.sum() - len(labels)):.3f}")  # fmt: skip
lines += [
    "",
    "windows: median data, median model (1e-17 erg/s/cm^2/A) and mean chi on fitted pixels",
]
for name, (lo, hi) in WINDOWS.items():
    row = f"{name:>8} {lo:.0f}-{hi:.0f}"
    for k, d in fits.items():
        rest = d["wave"] / (1 + float(d["z"]))
        s = d["good"] & (rest >= lo) & (rest <= hi)
        row += (f" | {k}: {np.median(flam(d, d['flux'])[s]):6.2f} "
                f"{np.median(flam(d, d['sp_best'])[s]):6.2f} {np.mean(chi[k][s]):+5.2f} ({s.sum()})")  # fmt: skip
    lines.append(row)
text = "\n".join(lines) + "\n"
(OUT / "sv3_vs_main.txt").write_text(text)
print(text)

# %% figure: main data and its fit, then chi of each fit on its own data
use_apj_style()
fig, ax = plt.subplots(
    3, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.06}
)
m = fits["main"]
rest_m = m["wave"] / (1 + float(m["z"]))
g = m["good"]
data = flam(m, m["flux"])
ax[0].plot(rest_m, np.where(g, data, np.nan), color="0.75", lw=0.5, label="DESI main (DR1 coadd)")
med = np.full(data.size, np.nan)
med[g] = medfilt(data[g], N_MEDIAN)
ax[0].plot(rest_m, med, color="k", lw=0.8, label=f"median filter, {N_MEDIAN} pixels")
ax[0].plot(rest_m, np.where(g, flam(m, m["sp_best"]), np.nan), color=RUNS["main"][1], lw=0.8,
           label="C3K ER + afe + R matrix (MAP), main")  # fmt: skip
ax[0].set_ylim(bottom=0)
ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
ax[0].legend(frameon=False, fontsize=12, loc="upper right")
for a, (k, (_, color)) in zip(ax[1:], RUNS.items(), strict=True):
    d = fits[k]
    a.plot(d["wave"] / (1 + float(d["z"])), chi[k], color=color, lw=0.5)
    for h in (-2, 2):
        a.axhline(h, color="0.3", ls="--", lw=1)
    a.axhline(0, color="0.3", lw=0.6)
    a.set_ylim(-5, 5)
    a.set_ylabel(r"$\chi$", fontsize=14)
    a.text(0.005, 0.88, f"{k} spectrum, its own MAP", transform=a.transAxes, fontsize=12,
           va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))  # fmt: skip
for name, spans in index_bands().items():
    for s0, s1 in spans:
        for a in ax:
            a.axvspan(s0, s1, color="0.5", alpha=0.12, lw=0)
    mid = 0.5 * (spans[0][0] + spans[-1][1])
    ax[0].text(mid, 1.01, name, transform=ax[0].get_xaxis_transform(), rotation=90, fontsize=9,
               ha="center", va="bottom")  # fmt: skip
ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
fig.savefig(OUT / "sv3_vs_main.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'sv3_vs_main.png'}")
