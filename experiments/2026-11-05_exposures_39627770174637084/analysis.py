"""Do the single SV3 and main bright exposures of 39627770174637084 agree with its main coadd (#14)?

QUESTION: do the SV3 bright and main bright exposures of 39627770174637084 agree with the main coadd in
Hbeta + [OIII], Halpha + [NII] and [SII], and do outlying exposures have bright sky, poor seeing, low
efftime or a FIBERSTATUS flag?
HYPOTHESIS: none registered. Descriptive, no threshold.
INPUTS: results/2026-11-05_exposures_39627770174637084/extract.npz (extract.py, DR1 iron healpix 26004
spectra and coadd files) and conditions.csv (conditions.py). Arms are joined ivar-weighted as spender
does. Excess is the median of exposure minus main coadd flux (1e-17 erg/s/cm2/A) over good pixels in
each rest window, at the SV3 redrock redshift.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-05_exposures_39627770174637084/analysis.py``.
RESULT: pending.
FIGURES: results/2026-11-05_exposures_39627770174637084/ exposures.txt, full.png, zoom_<window>.png.
"""

# %%
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from spender.data.desi import DESI

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

OUT = PATHS["RESULTS"] / "2026-11-05_exposures_39627770174637084"
Z = 0.0253189  # SV3 bright redrock redshift (SPARCL, 2026-11-05 handoff)
WINDOWS = {"hbeta_oiii": (4830, 5030), "halpha_nii": (6520, 6610), "sii": (6690, 6750)}  # rest A
SURVEYS = ("sv3", "main")
N_FLUX, N_RESID = 9, 51  # medfilt widths of the full-spectrum figure, as 11-02
YLABEL = r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]"


def merge_arms(d, prefix, k=None):
    """Return flux, ivar and mask on the DESI grid, arms joined as spender does.

    Copy of merge_arms in 2026-11-02_exposure_checks/analysis.py. ``prefix`` picks the exposure
    (``sv3`` or ``main``, row ``k``) or coadd (``coadd_sv3`` or ``coadd_main``) arrays.
    """
    wave = DESI._wave_obs.numpy()
    flux, ivar = np.zeros(wave.size), np.zeros(wave.size)
    mask = np.zeros(wave.size, int)
    ncam = np.zeros(wave.size)
    win = {}
    for b in "BRZ":
        imin = np.argmin(np.abs(d[f"{b}_wave"][0] - wave))
        win[b] = np.arange(imin, imin + len(d[f"{b}_wave"]))
        assert np.abs(d[f"{b}_wave"] - wave[win[b]]).max() < 1e-4, "arm grid differs from DESI grid"
        ncam[win[b]] += 1
    for b in "BRZ":
        fl, iv, mk = (d[f"{prefix}_{b}_{x}"] for x in ("flux", "ivar", "mask"))
        if k is not None:
            fl, iv, mk = fl[k], iv[k], mk[k]
        j = win[b]
        good = mk == 0
        ivar[j] += iv * good
        flux[j] += iv * good * fl
        one, many = ncam[j] == 1, ncam[j] > 1
        mask[j[one]] = mk[one]
        mask[j[many]] = mask[j[many]] & mk[many]
    ok = ivar > 0
    flux[ok] /= ivar[ok]
    return flux, ivar, mask


# %% spectra and conditions
d = np.load(OUT / "extract.npz")
cond = {int(r["expid"]): r for r in csv.DictReader(open(OUT / "conditions.csv"))}
wave = DESI._wave_obs.numpy()
rest = wave / (1 + Z)
coadds = {}
for s in SURVEYS:
    f, iv, m = merge_arms(d, f"coadd_{s}")
    coadds[s] = dict(flux=f, good=(m == 0) & (iv > 0), numexp=int(d[f"coadd_{s}_numexp"]),
                     fstat=int(d[f"coadd_{s}_fiberstatus"]))  # fmt: skip
ref = coadds["main"]
exps = []
for s in SURVEYS:
    for k, expid in enumerate(d[f"{s}_EXPID"]):
        f, iv, m = merge_arms(d, s, k)
        c = cond[int(expid)]
        exps.append(dict(
            survey=s, expid=int(expid), night=int(d[f"{s}_NIGHT"][k]), tile=int(d[f"{s}_TILEID"][k]),
            fiber=int(d[f"{s}_FIBER"][k]), fstat=str(d[f"{s}_fiberstatus_names"][k]) or "-",
            sky=float(c["sky_mag_g_spec"]), seeing=float(c["seeing_gfa"]),
            efft=float(c["efftime_spec"]), flux=f, good=(m == 0) & (iv > 0),
        ))  # fmt: skip


def excess(flux, good):
    """Return the median of flux minus the main coadd in each rest window."""
    both = good & ref["good"]
    return {
        w: float(np.median((flux - ref["flux"])[both & (rest >= lo) & (rest < hi)]))
        for w, (lo, hi) in WINDOWS.items()
    }


for e in exps:
    e["excess"] = excess(e["flux"], e["good"])

# %% table
head = "  ".join(f"{w:>10}" for w in WINDOWS)
lines = [
    f"39627770174637084, rest frame at z {Z}. Per exposure: night, tile, fibre, FIBERSTATUS, sky g "
    "(mag, brighter = smaller), seeing (arcsec), EFFTIME (s), excess = median exposure - main coadd "
    "(1e-17 erg/s/cm2/A) in rest windows "
    + ", ".join(f"{w} {lo}-{hi} A" for w, (lo, hi) in WINDOWS.items())
    + ".",
    "",
    f"{'survey':6} {'EXPID':>6} {'night':>8} {'tile':>5} {'fibre':>5} {'FIBERSTATUS':>12} "
    f"{'sky g':>5} {'seeing':>6} {'efftime':>7}  {head}",
]
for e in exps:
    lines.append(
        f"{e['survey']:6} {e['expid']:6d} {e['night']:8d} {e['tile']:5d} {e['fiber']:5d} "
        f"{e['fstat']:>12} {e['sky']:5.2f} {e['seeing']:6.2f} {e['efft']:7.0f}  "
        + "  ".join(f"{e['excess'][w]:+10.3f}" for w in WINDOWS)
    )
sv3c = excess(coadds["sv3"]["flux"], coadds["sv3"]["good"])
lines.append(f"{'SV3 coadd - main coadd':>70}  " + "  ".join(f"{sv3c[w]:+10.3f}" for w in WINDOWS))
lines.append("")
for s in SURVEYS:
    lines.append(f"{s} coadd: COADD_NUMEXP {coadds[s]['numexp']}, "
                 f"COADD_FIBERSTATUS {coadds[s]['fstat']}")  # fmt: skip
text = "\n".join(lines)
(OUT / "exposures.txt").write_text(text + "\n")
print(text)

# %% figures
use_apj_style()
n = {s: sum(e["survey"] == s for e in exps) for s in SURVEYS}
cmaps = {"sv3": plt.cm.Blues, "main": plt.cm.Oranges}
seen = {s: 0 for s in SURVEYS}
for e in exps:
    s = e["survey"]
    e["colour"] = cmaps[s](0.4 + 0.5 * seen[s] / max(n[s] - 1, 1))
    seen[s] += 1
COADD_STYLE = {"main": dict(color="k", lw=1.0), "sv3": dict(color="0.45", lw=1.0, ls="--")}


def panel(ax, sel, smooth_flux, smooth_resid):
    """Draw exposures and coadds on ax[0] and exposure minus main coadd on ax[1] for pixels sel."""
    for e in exps:
        g = e["good"] & sel
        ax[0].plot(rest[g], medfilt(e["flux"][g], smooth_flux), color=e["colour"], lw=0.7,
                   label=f"{e['survey']} {e['expid']}")  # fmt: skip
        both = g & ref["good"]
        ax[1].plot(rest[both], medfilt((e["flux"] - ref["flux"])[both], smooth_resid),
                   color=e["colour"], lw=0.7)  # fmt: skip
    for s in SURVEYS:
        g = coadds[s]["good"] & sel
        ax[0].plot(rest[g], medfilt(coadds[s]["flux"][g], smooth_flux), label=f"{s} coadd",
                   **COADD_STYLE[s])  # fmt: skip
    both = coadds["sv3"]["good"] & ref["good"] & sel
    ax[1].plot(rest[both], medfilt((coadds["sv3"]["flux"] - ref["flux"])[both], smooth_resid),
               **COADD_STYLE["sv3"])  # fmt: skip
    ax[1].axhline(0, color="0.3", lw=0.6)
    ax[0].set_ylabel(YLABEL, fontsize=13)
    ax[1].set_xlabel(r"rest wavelength [\AA]", fontsize=13)
    ax[0].legend(frameon=False, fontsize=9, loc="upper right")


def new_fig():
    """Return a two-row figure with a flux and a residual panel."""
    return plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})  # fmt: skip


fig, ax = new_fig()
panel(ax, np.ones(wave.size, bool), N_FLUX, N_RESID)
ax[1].set_ylabel(f"minus main coadd, {N_RESID} px", fontsize=12)
ax[0].set_title(f"39627770174637084   z {Z:.4f}", fontsize=13)
fig.savefig(OUT / "full.png", dpi=110, bbox_inches="tight")
plt.close(fig)
for w, (lo, hi) in WINDOWS.items():
    fig, ax = new_fig()
    panel(ax, (rest >= lo) & (rest < hi), 1, 1)
    ax[1].set_ylabel("minus main coadd", fontsize=12)
    ax[0].set_title(f"39627770174637084   {w}   unsmoothed", fontsize=13)
    fig.savefig(OUT / f"zoom_{w}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
print(f"wrote full.png and {len(WINDOWS)} zooms to {OUT}")
