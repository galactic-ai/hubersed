"""Two checks from existing outputs, no fits: is there an MZR signal in the raw line ratios, and do the
Cue posteriors trade gas_logz against N/O, C/O and U?

(a) Raw observed line ratios against FastSpecFit LOGMSTAR, with the Spearman rank correlation, for
the 20 fitted galaxies (subset_20.txt) and for all 1108 of lineflux_table.fits. N2 = log
[NII]6584/Halpha and O3 = log [OIII]5007/Hbeta are close pairs, so dust barely enters. N2O2 = log
[NII]6584/[OII]3727 and O32 = log [OIII]5007/[OII]3727 span a wide wavelength range and are shown
uncorrected, so they carry dust.
(b) Weighted correlation coefficients of gas_logz with gas_logno, gas_logco, gas_logu and gas_lognH
inside each strong-line posterior (fits/<TARGETID>_strong_emfit_mu0.4_zs-0.41.npz), and the
Spearman correlations of the posterior medians of gas_logz, gas_logno and gas_logco with N2 over
the 20. Points more than e^-30 below the heaviest are dropped. SEED: none.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/free_checks.py``
from the repository root.

OUTPUT: free_checks.txt in results/2026-10-14_cue_lineflux_calibration/.
"""

import numpy as np
from astropy.table import Table

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
RATIOS = {
    "N2": ("NII6584", "HA"),
    "O3": ("OIII5007", "HB"),
    "N2O2": ("NII6584", "OII3727"),
    "O32": ("OIII5007", "OII3727"),
}
PAIRS = ("gas_logno", "gas_logco", "gas_logu", "gas_lognH")


def spearman(x, y):
    """Return the Spearman rank correlation of finite pairs and their number."""
    ok = np.isfinite(x) & np.isfinite(y)
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def wcorr(x, y, w):
    """Return the weighted Pearson correlation of x and y."""
    mx, my = np.sum(w * x), np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    return float(cov / np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2)))


def log_ratio(t, a, b):
    """Return log10 of F_a / F_b, NaN where either is not positive."""
    num, den = np.asarray(t[f"F_{a}"], float), np.asarray(t[f"F_{b}"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((num > 0) & (den > 0), np.log10(num / den), np.nan)


t = Table.read(OUT / "lineflux_table.fits")
sub_ids = np.loadtxt(OUT / "subset_20.txt", dtype=np.int64)
sub = t[np.isin(t["TARGETID"], sub_ids)]
out = ["(a) Spearman rank correlation of raw observed ratios with LOGMSTAR (rho, N)"]
for name, (a, b) in RATIOS.items():
    r_all = spearman(np.asarray(t["LOGMSTAR"], float), log_ratio(t, a, b))
    r_sub = spearman(np.asarray(sub["LOGMSTAR"], float), log_ratio(sub, a, b))
    out.append(f"{name:>5}: all {r_all[0]:+.2f} ({r_all[1]}), subset {r_sub[0]:+.2f} ({r_sub[1]})")
mass = np.asarray(t["LOGMSTAR"], float)
n2 = log_ratio(t, "NII6584", "HA")
out += ["", "N2 median by LOGMSTAR bin, all 1108"]
for lo, hi in ((5, 8.5), (8.5, 9), (9, 9.5), (9.5, 10), (10, 12)):
    m = (mass >= lo) & (mass < hi)
    out.append(f"  {lo:4.1f}-{hi:4.1f}: N = {m.sum():4d}, N2 median {np.nanmedian(n2[m]):+.2f}")

out += ["", "(b) weighted correlation of gas_logz with each parameter inside each posterior"]
out.append(
    f"{'TARGETID':>18} {'logM*':>6} {'N2':>6} {'Z50':>6} " + " ".join(f"{p:>10}" for p in PAIRS)
)
med = {k: [] for k in ("gas_logz", "gas_logno", "gas_logco")}
n2_sub, rows = [], []
for r in sub:
    tid = int(r["TARGETID"])
    d = np.load(OUT / "fits" / f"{tid}_strong_emfit_mu0.4_zs-0.41.npz")
    lab = [str(x) for x in d["labels"]]
    lw = d["log_w"]
    keep = lw > lw.max() - 30
    w = np.exp(lw[keep] - lw.max())
    w /= w.sum()
    p = d["points"][keep]
    z = p[:, lab.index("gas_logz")]
    cors = [wcorr(z, p[:, lab.index(q)], w) for q in PAIRS]
    for k in med:
        x = p[:, lab.index(k)]
        i = np.argsort(x)
        med[k].append(float(np.interp(0.5, np.cumsum(w[i]), x[i])))
    n2v = float(np.log10(r["F_NII6584"] / r["F_HA"])) if r["F_NII6584"] > 0 else np.nan
    n2_sub.append(n2v)
    rows.append((float(r["LOGMSTAR"]), tid, n2v, med["gas_logz"][-1], cors))
for m_, tid, n2v, z50, cors in sorted(rows):
    out.append(
        f"{tid:>18} {m_:6.2f} {n2v:+6.2f} {z50:+6.2f} " + " ".join(f"{c:+10.2f}" for c in cors)
    )
n2_sub = np.array(n2_sub)
out += ["", "Spearman rho over the 20 of posterior medians against N2 and against LOGMSTAR"]
for k, v in med.items():
    v = np.array(v)
    out.append(
        f"{k:>10}: vs N2 {spearman(n2_sub, v)[0]:+.2f}, "
        f"vs LOGMSTAR {spearman(np.asarray(sub['LOGMSTAR'], float), v)[0]:+.2f}"
    )
text = "\n".join(out)
print(text)
(OUT / "free_checks.txt").write_text(text + "\n")
