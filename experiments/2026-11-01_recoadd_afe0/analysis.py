"""Does free afe improve the recoadd continuum MAPs of 39627757533007793 over afe = 0 (#17)?

QUESTION: on the recoadded SV3 and main spectra, how much does the C3K ER + R matrix MAP improve
with free afe over afe fixed at 0, in total and in the Mgb, Fe5270 and Fe5335 feature bands?
HYPOTHESIS: none registered.
INPUTS: <TID>_map.npz of 2026-10-29_continuum_recoadd/<survey> (afe free) and of
results/2026-11-01_recoadd_afe0/<survey> (afe = 0, fit.py collect), copied from stampede3. Feature
bands from $SPS_HOME/data/allindices.dat.
SEED: none.
COMMAND: from the repository root on the Mac,
``SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-11-01_recoadd_afe0/analysis.py``.
RESULT: pending.
FIGURES: results/2026-11-01_recoadd_afe0/<survey>_afe_free_vs_zero.png, zoom_5100_5400.png,
afe_free_vs_zero.txt and afe_free_vs_zero_params.txt.
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
OUT = RES / "2026-11-01_recoadd_afe0"
RUNS = {
    "C3K ER + afe + R matrix": (RES / "2026-10-29_continuum_recoadd", "#1b9e77"),
    "C3K ER + R matrix (afe = 0)": (OUT, "#d95f02"),
}
INDEX_NAMES = ("Mgb", "Fe5270", "Fe5335")
N_MEDIAN = 9


def flam(d, x):
    """Return maggies on the fit's wavelength grid in 1e-17 erg/s/cm^2/A, NaN off its fit."""
    f = to_flambda(d["wave"] * u.AA, np.asarray(x, float) * u.mgy).value
    return np.where(d["good"], f, np.nan)


def chi(d):
    """Return (data - model) / sigma on the fitted pixels, NaN elsewhere."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d["good"], (d["flux"] - d["sp_best"]) / d["unc"], np.nan)


def feature_bands():
    """Return the rest-frame feature band of each index in allindices.dat."""
    table = np.genfromtxt(
        Path(os.environ["SPS_HOME"]) / "data" / "allindices.dat",
        dtype=None,
        encoding=None,
        comments="#",
    )
    return {str(r[-1]): (float(r[0]), float(r[1])) for r in table if str(r[-1]) in INDEX_NAMES}


# %% chi2 in total and per feature band, on pixels both fits used
bands = feature_bands()
fits = {}
lines = []
# a survey whose afe = 0 collect has not been copied yet is left out
for survey in [s for s in ("sv3", "main") if (OUT / s / f"{TID}_map.npz").exists()]:
    f = {k: np.load(path / survey / f"{TID}_map.npz") for k, (path, _) in RUNS.items()}
    a, b = f.values()
    assert np.array_equal(a["good"], b["good"]) and np.array_equal(a["flux"], b["flux"]), survey
    fits[survey] = f
    rest = a["wave"] / (1 + float(a["z"]))
    lines.append(f"== {survey}: {int(a['good'].sum())} fitted pixels")
    for k, d in f.items():
        lab = dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))
        lines.append(f"{k:>28}: chi2 {np.nansum(chi(d) ** 2):9.1f}, -lnp {-float(d['fun_all'].min()):9.2f}"
                     f" (best of {len(d['fun_all'])} starts), afe {lab.get('afe', 0.0):+.3f}, "
                     f"logzsol {lab['logzsol']:+.3f}")  # fmt: skip
    d_tot = np.nansum(chi(b) ** 2) - np.nansum(chi(a) ** 2)
    lines.append(f"{'chi2(afe = 0) - chi2(afe free)':>28}: total {d_tot:+.1f}")
    for name, (f0, f1) in bands.items():
        m = a["good"] & (rest >= f0) & (rest <= f1)
        dc = np.sum(chi(b)[m] ** 2) - np.sum(chi(a)[m] ** 2)
        lines.append(f"{name:>28}: {dc:+.1f} ({int(m.sum())} px)")
text = "\n".join(lines)
(OUT / "afe_free_vs_zero.txt").write_text(text + "\n")
print(text)

# %% figures: recoadd data with both MAPs, chi of both overlaid
use_apj_style()
for survey, f in fits.items():
    a = next(iter(f.values()))
    rest = a["wave"] / (1 + float(a["z"]))
    data = flam(a, a["flux"])
    med = np.full(data.size, np.nan)
    med[a["good"]] = medfilt(data[a["good"]], N_MEDIAN)
    fig, ax = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}
    )
    ax[0].plot(rest, data, color="0.6", lw=0.6, label=f"DESI spectrum, {survey.upper()} recoadd")
    ax[0].plot(rest, med, color="k", lw=0.5, label=f"median filter ({N_MEDIAN} pix)")
    for k, d in f.items():
        colour = RUNS[k][1]
        ax[0].plot(rest, flam(d, d["sp_best"]), color=colour, lw=0.9, label=k)
        ax[1].plot(rest, chi(d), color=colour, lw=0.5, alpha=0.7)
    for f0, f1 in bands.values():
        for x in ax:
            x.axvspan(f0, f1, color="0.5", alpha=0.15, lw=0)
    ax[0].set_ylim(bottom=0)
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
    ax[0].legend(frameon=False, fontsize=12, loc="lower right")
    for h in (-2, 2):
        ax[1].axhline(h, color="0.3", ls="--", lw=1)
    ax[1].axhline(0, color="0.3", lw=0.6)
    ax[1].set_ylim(-5, 5)
    ax[1].set_ylabel(r"$\chi$", fontsize=14)
    ax[1].set_xlim(rest[a["good"]].min(), rest[a["good"]].max())
    ax[1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
    path = OUT / f"{survey}_afe_free_vs_zero.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

# %% parameter table, afe free against afe = 0
lines = []
for survey, f in fits.items():
    a, b = f.values()
    pa = dict(zip([str(x) for x in a["labels"]], a["best"], strict=True))
    pb = dict(zip([str(x) for x in b["labels"]], b["best"], strict=True))
    lines += [
        f"== {survey}",
        f"{'parameter':>16} {'afe free':>10} {'afe = 0':>10} {'difference':>10}",
    ]
    for name in pa:
        vb = pb.get(name, 0.0)
        lines.append(f"{name:>16} {pa[name]:10.4f} {vb:10.4f} {vb - pa[name]:+10.4f}")
text = "\n".join(lines)
(OUT / "afe_free_vs_zero_params.txt").write_text(text + "\n")
print(text)

# %% zoom on Mgb, Fe5270 and Fe5335: data and both MAPs, rest 5100 to 5400 A
ZOOM = (5100, 5400)
fig, axes = plt.subplots(len(fits), 1, figsize=(12, 4 * len(fits)), sharex=True, squeeze=False)
for ax, (survey, f) in zip(axes[:, 0], fits.items(), strict=True):
    a = next(iter(f.values()))
    rest = a["wave"] / (1 + float(a["z"]))
    g = a["good"]
    data = flam(a, a["flux"])
    med = np.full(data.size, np.nan)
    med[g] = medfilt(data[g], N_MEDIAN)
    ax.plot(rest, data, color="0.6", lw=1.0, label=f"DESI spectrum, {survey.upper()} recoadd")
    ax.plot(rest, med, color="k", lw=0.6, label=f"median filter ({N_MEDIAN} pix)")
    s = g & (rest >= ZOOM[0]) & (rest <= ZOOM[1])
    lo, hi = np.nanmin(data[s]), np.nanmax(data[s])
    for k, d in f.items():
        model = flam(d, d["sp_best"])
        ax.plot(rest, model, color=RUNS[k][1], lw=1.2, label=k)
        lo, hi = min(lo, np.nanmin(model[s])), max(hi, np.nanmax(model[s]))
    # y range from the pixels shown, as autoscale would use the whole spectrum
    pad = 0.05 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    for name, (f0, f1) in bands.items():
        ax.axvspan(f0, f1, color="0.5", alpha=0.15, lw=0)
        ax.text(0.5 * (f0 + f1), hi + 0.5 * pad, name, ha="center", va="top", fontsize=11)
    ax.set_title(f"{survey.upper()} recoadd", fontsize=14)
    ax.set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=12)
axes[0, 0].set_xlim(*ZOOM)
axes[-1, 0].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
axes[0, 0].legend(frameon=False, fontsize=10, loc="lower left")
path = OUT / f"zoom_{ZOOM[0]}_{ZOOM[1]}.png"
fig.savefig(path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {path}")
