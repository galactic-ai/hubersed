"""Residuals of the C3K ER + afe + R matrix continuum MAP of 39633433328091196 (issue #17).

QUESTION: where in rest wavelength do the MILES, C3K HR + afe and C3K ER + afe + R matrix
continuum fits of 39633433328091196 leave residuals?
HYPOTHESIS: none registered.
INPUTS: <TID>_map.npz of fit.py collect (icx job 3657895) and hr_maxl.npz of hypers.py, both in
this experiment's results folder, copied from stampede3. The 10-08 max-L spectra
<TID>_miles_miles_neboff_nodeg_pred.npz (md5 equal to the /work copy, 2026-10-29) and
<TID>_c3k_hr_full_neboff_afefree_pred.npz (not made yet) in results/2026-10-08_continuum_outlier/.
Index bands from $SPS_HOME/data/allindices.dat.
SEED: none.
COMMAND: from the repository root on the Mac,
``SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-29_continuum_39633433328091196_map/analysis.py``.
RESULT: 2026-10-29, map_summary.txt, map_ew.txt. ER MAP (job 3657895) chi2/(N-16) 1.117 on 4561 px,
MILES max-L chi2/(N-19) 1.127 on 4070 px; 500 A bins 1.03-1.25 for both. Band EWs agree with the
data within ~1 sigma (largest: Fe5335 +0.87 +- 0.75 A, NaD ER -0.56 +- 0.51 A). The HR afefree row is
empty (its pred was never made). Assumptions: one MAP and one max-L, DR1 coadd ivar.
FIGURES: results/2026-10-29_continuum_39633433328091196_map/residuals.png, zoom_features.png,
map_summary.txt and map_ew.txt.
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

TID = 39633433328091196
RUNS10 = PATHS["RESULTS"] / "2026-10-08_continuum_outlier"
OUT = PATHS["RESULTS"] / "2026-10-29_continuum_39633433328091196_map"
ER_COLOR = "#b2182b"
N_MEDIAN = 9  # kernel size in pixels of the data's median filter
# Each row is a fit compared with the data it saw.
ROWS = {
    "miles": ("MILES, nebular off", "#4393c3", RUNS10 / f"{TID}_miles_miles_neboff_nodeg_pred.npz"),
    "c3k_hr": (
        "C3K HR + afe, nebular off (degraded)",
        "#e08214",
        RUNS10 / f"{TID}_c3k_hr_full_neboff_afefree_pred.npz",
    ),
    "c3k_er": ("C3K ER + afe + R matrix (MAP)", ER_COLOR, OUT / f"{TID}_map.npz"),
}
INDEX_NAMES = "Dn4000 HdeltaA CN1 Ca4227 G4300 C4668 Hbeta Mgb Fe5270 Fe5335 NaD TiO1 TiO2".split()
BINS = [(lo, lo + 500) for lo in range(3300, 8800, 500)]
# rest-frame windows, Angstrom: the Mgb and NaD feature bands of allindices.dat and the Ha core
WINDOWS = {"Mgb": (5160.1, 5192.6), "NaD": (5876.9, 5909.4), "Ha core": (6555.0, 6575.0)}
# prior bounds of the ER fit, from hubersed.sps.config.get_priors and build_continuum_model, and
# afe from 2026-10-28_continuum_c3k_er_rmatrix_map/fit.py make_model
BOUNDS = {
    "logmass": (7.0, 12.0),
    "logzsol": (-2.5, 0.5),
    "dust2": (0.0, 4.0),
    "dust_ratio": (0.0, 2.0),
    "dust_index": (-2.5, 0.4),
    "sigma_smooth": (10.0, 400.0),
    "afe": (-0.2, 0.6),
}


def flam(d, x):
    """Return maggies on the fit's wavelength grid in 1e-17 erg/s/cm^2/A."""
    return to_flambda(np.asarray(d["wave"], float) * u.AA, np.asarray(x, float) * u.mgy).value


def chi(d):
    """Return the chi of a fit on its own data, NaN on unfitted pixels."""
    with np.errstate(divide="ignore", invalid="ignore"):  # unfitted pixels can have unc 0
        return np.where(d["good"], (d["flux"] - d["sp_best"]) / d["unc"], np.nan)


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


# %% load the fits
fits = {k: np.load(v[2]) for k, v in ROWS.items() if v[2].exists()}
e, m = fits["c3k_er"], fits["miles"]
z = float(e["z"])
assert np.isclose(float(m["z"]), z)
# the undegraded MILES run saw the same coadd as the ER MAP, so the fitted fluxes must agree
both = e["good"] & m["good"]
assert np.allclose(m["flux"][both], e["flux"][both], rtol=1e-6, atol=0), "different data"
chis = {k: chi(d) for k, d in fits.items()}

# %% table: parameters, chi2 by rest-frame bin, windows
hr = np.load(OUT / "hr_maxl.npz")
hr_best = dict(zip([str(x) for x in hr["labels"]], hr["best"], strict=True))
labels = [str(x) for x in e["labels"]]
fun = np.asarray(e["fun_all"], float)
lines = [
    f"C3K ER + afe + R matrix MAP of {TID}, z {z:.5f}, best start {int(e['start'])} of {fun.size}",
    "-lnp of each start minus the best: " + " ".join(f"{x - fun.min():.2f}" for x in fun),
    "",
    f"{'parameter':>16} {'HR max-L':>10} {'ER MAP':>10} {'MAP-HR':>10} {'prior':>15} "
    f"{'to edge':>9}",
]
for i, name in enumerate(labels):
    a, b = float(hr_best[name]), float(e["best"][i])
    if name in BOUNDS:
        lo, hi = BOUNDS[name]
        prior, edge = f"{lo:g} to {hi:g}", f"{min(b - lo, hi - b):9.4f}"
    else:
        prior, edge = "-", "-"
    lines.append(f"{name:>16} {a:10.4f} {b:10.4f} {b - a:+10.4f} {prior:>15} {edge:>9}")
lines += ["", "chi2/N on each fit's own data by rest-frame bin, fitted pixels in brackets"]
lines.append(f"{'rest A':>11} " + " ".join(f"{k:>14}" for k in fits))
for lo, hi in BINS:
    row = f"{lo:5d}-{hi:5d} "
    for k, d in fits.items():
        rest = d["wave"] / (1 + float(d["z"]))
        s = d["good"] & (rest >= lo) & (rest < hi)
        row += f" {np.mean(chis[k][s] ** 2):7.2f} ({s.sum():4d})" if s.any() else f" {'-':>14}"
    lines.append(row)
for k, d in fits.items():
    g, n_free = d["good"], len(d["labels"])
    c2 = float(np.sum(chis[k][g] ** 2))
    lines.append(f"{k}: chi2 {c2:.1f} on {g.sum()} px, chi2/(N - {n_free}) "
                 f"{c2 / (g.sum() - n_free):.3f}")  # fmt: skip
lines += [
    "",
    "windows: median data, median model (1e-17 erg/s/cm^2/A) and mean chi on fitted pixels",
]
for name, (lo, hi) in WINDOWS.items():
    row = f"{name:>8} {lo:.0f}-{hi:.0f}"
    for k, d in fits.items():
        rest = d["wave"] / (1 + float(d["z"]))
        s = d["good"] & (rest >= lo) & (rest <= hi)
        if not s.any():
            row += f" | {k}: -"
            continue
        row += (f" | {k}: {np.median(flam(d, d['flux'])[s]):6.2f} "
                f"{np.median(flam(d, d['sp_best'])[s]):6.2f} {np.mean(chis[k][s]):+5.2f} ({s.sum()})")  # fmt: skip
    lines.append(row)
lines += ["", f"not run yet: {[k for k in ROWS if k not in fits]}"]
text = "\n".join(lines) + "\n"
(OUT / "map_summary.txt").write_text(text)
print(text)

# %% figure: data with the ER MAP on top, then chi of each fit on its own data
use_apj_style()
fig, ax = plt.subplots(
    4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1], "hspace": 0.06}
)
rest = e["wave"] / (1 + z)
g = e["good"]
data = flam(e, e["flux"])
ax[0].plot(rest, np.where(g, data, np.nan), color="0.75", lw=0.5, label="DESI SV3 (DR1 coadd)")
# median filter over the fitted pixels; medfilt pads the two ends with zeros
med = np.full(data.size, np.nan)
med[g] = medfilt(data[g], N_MEDIAN)
ax[0].plot(rest, med, color="k", lw=0.8, label=f"median filter, {N_MEDIAN} pixels")
ax[0].plot(rest, np.where(g, flam(e, e["sp_best"]), np.nan), color=ER_COLOR, lw=0.8,
           label=ROWS["c3k_er"][0])  # fmt: skip
ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
ax[0].set_ylim(bottom=0)
ax[0].legend(frameon=False, fontsize=13, loc="upper right")
for a, (key, (label, color, _)) in zip(ax[1:], ROWS.items(), strict=True):
    if key in fits:
        d = fits[key]
        a.plot(d["wave"] / (1 + float(d["z"])), chis[key], color=color, lw=0.5)
    else:
        label += ", not run yet"
    for h in (-2, 2):
        a.axhline(h, color="0.3", ls="--", lw=1)
    a.axhline(0, color="0.3", lw=0.6)
    a.set_ylim(-5, 5)
    a.set_ylabel(r"$\chi$", fontsize=14)
    a.text(0.005, 0.88, label, transform=a.transAxes, fontsize=12, va="top",
           bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))  # fmt: skip
for name, spans in index_bands().items():
    for s0, s1 in spans:
        for a in ax:
            a.axvspan(s0, s1, color="0.5", alpha=0.12, lw=0)
    mid = 0.5 * (spans[0][0] + spans[-1][1])
    ax[0].text(mid, 1.01, name, transform=ax[0].get_xaxis_transform(), rotation=90, fontsize=9,
               ha="center", va="bottom")  # fmt: skip
ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
fig.savefig(OUT / "residuals.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'residuals.png'}")


# %% equivalent widths of data and model in the allindices.dat Angstrom bands
def lick_bands(names):
    """Return {name: (feature, blue, red)} rest-frame bands from $SPS_HOME/data/allindices.dat."""
    table = np.genfromtxt(
        Path(os.environ["SPS_HOME"]) / "data" / "allindices.dat",
        dtype=None,
        encoding=None,
        comments="#",
    )
    return {
        str(r[-1]): ((r[0], r[1]), (r[2], r[3]), (r[4], r[5])) for r in table if str(r[-1]) in names
    }


def equivalent_width(rest, f, bands, good):
    """Return the equivalent width in A of ``f`` over the fitted pixels of the feature band.

    The continuum is the line through the mean flux of the fitted pixels of the blue and red
    bands at their mean wavelengths. Pixels count whole and masked pixels add nothing, so data
    and model on the same pixels compare directly.
    """
    feat, blue, red = bands
    pts = []
    for lo, hi in (blue, red):
        s = good & (rest >= lo) & (rest <= hi)
        pts.append((rest[s].mean(), f[..., s].mean(axis=-1)))
    (xb, yb), (xr, yr) = pts
    s = good & (rest >= feat[0]) & (rest <= feat[1])
    cont = yb[..., None] + (yr - yb)[..., None] * (rest[s] - xb) / (xr - xb)
    return np.sum((1 - f[..., s] / cont) * np.gradient(rest)[s], axis=-1)


EW_NAMES = ["Ca4227", "G4300", "Hbeta", "Mgb", "Fe5270", "Fe5335", "NaD"]
N_NOISE = 500  # noise draws of the data for the EW scatter
ew_bands = lick_bands(EW_NAMES)
rng = np.random.default_rng(0)
lines = [
    f"equivalent widths in A on each fit's own data and fitted pixels; data scatter from "
    f"{N_NOISE} draws of unc, seed 0; fitted/all pixels of feature, blue, red band in brackets"
]
for name in EW_NAMES:
    lines.append(f"{name}")
    for k, d in fits.items():
        rest_d = d["wave"] / (1 + float(d["z"]))
        good_d = d["good"]
        frac = []
        for lo, hi in ew_bands[name]:
            s = (rest_d >= lo) & (rest_d <= hi)
            frac.append(f"{(good_d & s).sum()}/{s.sum()}")
        if any(f.startswith("0/") for f in frac):
            lines.append(f"  {k:>8}: - ({', '.join(frac)})")
            continue
        data_d, model_d, unc_d = flam(d, d["flux"]), flam(d, d["sp_best"]), flam(d, d["unc"])
        noise = rng.normal(size=(N_NOISE, data_d.size)) * np.where(good_d, unc_d, 0.0)
        w = equivalent_width(rest_d, data_d + noise, ew_bands[name], good_d)
        w0 = equivalent_width(rest_d, data_d, ew_bands[name], good_d)
        w_model = equivalent_width(rest_d, model_d, ew_bands[name], good_d)
        lines.append(
            f"  {k:>8}: data {w0:6.2f} +- {w.std():4.2f}, model {w_model:6.2f}, "
            f"data - model {w0 - w_model:+5.2f} ({', '.join(frac)})"
        )
text = "\n".join(lines) + "\n"
(OUT / "map_ew.txt").write_text(text)
print(text)

# %% zoom on Ca H+K, Hbeta, Mgb and NaD: data and models on top, chi of each fit below
# Ca H+K has no allindices.dat band, so its window is set by hand. The others run from the blue
# to the red continuum band, 10 A wider on each side. The degraded HR model is not drawn on the
# native data; only its chi on its own data is.
ZOOMS = {"Ca H+K": (3900.0, 4010.0)} | {
    n: (ew_bands[n][1][0] - 10, ew_bands[n][2][1] + 10) for n in ("Hbeta", "Mgb", "NaD")
}
fig, ax = plt.subplots(
    2, len(ZOOMS), figsize=(16, 6.5), gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05}
)
for j, (name, (lo, hi)) in enumerate(ZOOMS.items()):
    top, bot = ax[0, j], ax[1, j]
    s = (rest >= lo) & (rest <= hi)
    top.plot(rest[s], np.where(g, data, np.nan)[s], color="0.75", lw=0.6)
    top.plot(rest[s], med[s], color="k", lw=1.0, label=f"median filter, {N_MEDIAN} pixels")
    for key in ("miles", "c3k_er"):
        d = fits[key]
        top.plot(rest[s], np.where(d["good"], flam(d, d["sp_best"]), np.nan)[s],
                 color=ROWS[key][1], lw=1.0, label=ROWS[key][0])  # fmt: skip
    for key, d in fits.items():
        bot.plot(d["wave"][s] / (1 + float(d["z"])), chis[key][s], color=ROWS[key][1], lw=0.7)
    for h in (-2, 2):
        bot.axhline(h, color="0.3", ls="--", lw=1)
    bot.axhline(0, color="0.3", lw=0.6)
    bot.set_ylim(-5, 5)
    if name in ew_bands:
        for a in (top, bot):
            a.axvspan(*ew_bands[name][0], color="0.5", alpha=0.15, lw=0)
    top.set_xlim(lo, hi)
    bot.set_xlim(lo, hi)
    top.set_xticklabels([])
    top.set_title(name, fontsize=13)
    bot.set_xlabel(r"rest wavelength [$\AA$]", fontsize=12)
ax[0, 0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=12)
ax[1, 0].set_ylabel(r"$\chi$", fontsize=13)
ax[0, -1].legend(frameon=False, fontsize=10, loc="lower right")
fig.savefig(OUT / "zoom_features.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'zoom_features.png'}")
