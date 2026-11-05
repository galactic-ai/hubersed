"""Does Cue's strong-line O/H depend on EW(Hbeta), as the 20-galaxy subset suggests?

QUESTION: in the 20-galaxy subset, Cue O/H rises with mass for EW(Hbeta) > 20 A (Spearman +0.60, 12)
and falls for EW <= 20 A (-0.48, 8), and the three galaxies stuck at low O/H with high N/O all have
EW < 20 A. Does that hold for 80 galaxies stratified by EW and mass?

HYPOTHESIS: at low EW, diffuse or evolved-star ionized gas that Cue does not model drives the fits
to low U, high N/O and low O/H, with the held-out [OII]7325 over-predicted. Above some EW the fits
give a rising MZR and 7325 near 1.

INPUTS: lineflux_table.fits, subset_20.txt (subset.py) and subset_ew60.txt (subset_ew.py), and
fits/<TARGETID>_strong_emfit_mu0.4_zs-0.41.npz (fit.py --lines strong, flat priors) in
results/2026-10-14_cue_lineflux_calibration/. Cue O/H = gas_logz + 8.71, weighted median. Low-Z
solution: O/H < 8.2 with gas_logno > 0. Held-out ratios use lines with S/N >= 3. chi2 per line is
at the median model of the 200 posterior draws with the measurement and Cue emulation errors.
Galaxies at z <= 0.01 are listed but left out of the statistics. SEED: none.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/analysis_ew.py``
from the repository root.

RESULT: not yet run.

FIGURES: ew_compare.txt and ew_compare.png in results/2026-10-14_cue_lineflux_calibration/.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from cuejax.emulator import Emulator
from fit import LINES

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
CUE_OH_ZERO = 8.71
EW_BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, np.inf))
Z_MIN = 0.01
SNR_MIN = 3.0

emul = Emulator()
names = [str(x).strip() for x in np.atleast_1d(emul.line_names)]
frac = np.asarray(emul.line_frac_error, float)
frac_line = {k: max(frac[names.index(n)] for n in v) for k, v in LINES.items()}


def spearman(x, y):
    """Return the Spearman rank correlation of finite pairs and their number."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, int(ok.sum())
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def wmed(x, w):
    """Return the weighted median of x."""
    i = np.argsort(x)
    c = np.cumsum(w[i])
    return float(np.interp(0.5, c / c[-1], x[i]))


# %%
t = Table.read(OUT / "lineflux_table.fits")
ids = np.concatenate(
    [np.loadtxt(OUT / f, dtype=np.int64) for f in ("subset_20.txt", "subset_ew60.txt")]
)
rows = []
for tid in ids:
    f = OUT / "fits" / f"{tid}_strong_emfit_mu0.4_zs-0.41.npz"
    if not f.exists():
        continue
    d = np.load(f)
    lab = [str(x) for x in d["labels"]]
    lw = d["log_w"]
    keep = lw > lw.max() - 30
    w = np.exp(lw[keep] - lw.max())
    p = d["points"][keep]
    lines = [str(x) for x in d["lines"]]
    used = {str(x) for x in d["used"]}
    ratio, chi2 = {}, []
    for j, k in enumerate(lines):
        obs, err = float(d["obs"][j]), float(d["err"][j])
        if not (np.isfinite(obs) and err > 0):
            continue
        mod = float(np.median(d["draw_flux"][:, j]))
        if k in used:
            chi2.append(((obs - mod) ** 2) / (err**2 + (frac_line[k] * mod) ** 2))
        if obs / err >= SNR_MIN:
            ratio[k] = mod / obs
    r = t[t["TARGETID"] == tid][0]
    rows.append(
        {
            "TARGETID": int(tid),
            "Z": float(r["Z"]),
            "EW_HB": float(r["EW_HB"]),
            "LOGMSTAR": float(r["LOGMSTAR"]),
            "N2": float(np.log10(r["F_NII6584"] / r["F_HA"])) if r["F_NII6584"] > 0 else np.nan,
            "CUE_OH": wmed(p[:, lab.index("gas_logz")], w) + CUE_OH_ZERO,
            "LOGNO": wmed(p[:, lab.index("gas_logno")], w),
            "LOGU": wmed(p[:, lab.index("gas_logu")], w),
            "R7325": ratio.get("OII7325", np.nan),
            "ROI": ratio.get("OI6300", np.nan),
            "CHI2_LINE": float(np.mean(chi2)),
        }
    )
a = Table(rows=rows)
a["LOWZ_SOL"] = (a["CUE_OH"] < 8.2) & (a["LOGNO"] > 0)
s = a[a["Z"] > Z_MIN]

# %%
out = [f"{len(a)} fitted galaxies, {len(s)} at z > {Z_MIN} used below", ""]
out.append(
    f"{'EW(Hb) bin':>12} {'N':>3} | {'rho(O/H,M*)':>11} {'rho(O/H,N2)':>11} | {'low-Z sol':>9} | "
    f"{'7325 m/d':>8} {'[OI] m/d':>8} | {'chi2/line':>9} | {'U':>5} {'N/O':>5}"
)
for lo, hi in EW_BINS:
    b = s[(s["EW_HB"] >= lo) & (s["EW_HB"] < hi)]
    rm = spearman(np.asarray(b["LOGMSTAR"]), np.asarray(b["CUE_OH"]))[0]
    rn = spearman(np.asarray(b["N2"]), np.asarray(b["CUE_OH"]))[0]
    out.append(
        f"{lo:5.0f}-{hi:5.0f} {len(b):4d} | {rm:+11.2f} {rn:+11.2f} | "
        f"{int(b['LOWZ_SOL'].sum()):3d}/{len(b):<5d} | "
        f"{np.nanmedian(b['R7325']):8.2f} {np.nanmedian(b['ROI']):8.2f} | "
        f"{np.median(b['CHI2_LINE']):9.2f} | {np.median(b['LOGU']):+5.2f} {np.median(b['LOGNO']):+5.2f}"
    )
for cut in (10.0, 20.0, 40.0):
    hi_ew = s[s["EW_HB"] > cut]
    out.append(
        f"EW > {cut:4.0f}: N {len(hi_ew)}, rho(O/H, M*) "
        f"{spearman(np.asarray(hi_ew['LOGMSTAR']), np.asarray(hi_ew['CUE_OH']))[0]:+.2f}, "
        f"rho(O/H, N2) {spearman(np.asarray(hi_ew['N2']), np.asarray(hi_ew['CUE_OH']))[0]:+.2f}"
    )
out += [
    "",
    "rho(EW, 7325 m/d) over all",
    f"{spearman(np.asarray(s['EW_HB']), np.asarray(s['R7325']))[0]:+.2f}",
]
out += ["", "per galaxy, sorted by EW(Hb)"]
out.append(
    f"{'TARGETID':>18} {'z':>6} {'EW':>6} {'logM*':>5} {'N2':>6} {'O/H':>5} {'N/O':>5} {'U':>5} {'7325':>5} {'[OI]':>5} {'chi2':>5}"
)
a.sort("EW_HB")
for r in a:
    out.append(
        f"{r['TARGETID']:>18} {r['Z']:6.3f} {r['EW_HB']:6.1f} {r['LOGMSTAR']:5.2f} {r['N2']:+6.2f} "
        f"{r['CUE_OH']:5.2f} {r['LOGNO']:+5.2f} {r['LOGU']:+5.2f} {r['R7325']:5.2f} {r['ROI']:5.2f} "
        f"{r['CHI2_LINE']:5.2f}{' low-Z' if r['LOWZ_SOL'] else ''}"
    )
text = "\n".join(out)
print(text)
(OUT / "ew_compare.txt").write_text(text + "\n")

# %%
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
sc = ax[0].scatter(s["LOGMSTAR"], s["CUE_OH"], c=np.log10(s["EW_HB"]), cmap="viridis", s=20)
ax[0].set_xlabel(r"log $M_*$")
ax[0].set_ylabel("Cue 12+log(O/H), strong lines")
fig.colorbar(sc, ax=ax[0], label=r"log EW(H$\beta$)")
ax[1].scatter(s["N2"], s["CUE_OH"], c=np.log10(s["EW_HB"]), cmap="viridis", s=20)
ax[1].set_xlabel("N2 = log [NII]/Ha")
ax[2].scatter(s["EW_HB"], s["R7325"], s=20)
ax[2].set_xscale("log")
ax[2].set_yscale("log")
ax[2].axhline(1, color="0.5", lw=0.8)
ax[2].axvline(20, color="0.5", lw=0.8, ls="--")
ax[2].set_xlabel(r"EW(H$\beta$) [A]")
ax[2].set_ylabel("held-out [OII]7325 model/data")
fig.tight_layout()
fig.savefig(OUT / "ew_compare.png", dpi=150)
