"""
TID-based characterization of the latent-OOD outliers + mock prior-coverage test.
(Consolidated, corrected replacement for outliers_check.ipynb. Catalogue key = TARGETID.)
Run:  PYTHONPATH=src python tmp/outliers_analysis.py
"""

import numpy as np, torch, h5py
from astropy.table import Table
from hubersed.paths import PATHS

DATA, RES = PATHS["DATA"], PATHS["RESULTS"]
VAC = DATA / "fastspec-iron-sv3-bright.fits"
CUE = DATA / "prospector_model" / "prospector_stochastic_model_seds_cue_500000.h5"
SNR = 3.0
LINES = ["HBETA", "OIII_5007", "HALPHA", "NII_6584", "SII_6716", "SII_6731"]

# ---- outliers (TID) ----
b = torch.load(RES / "desi_outliers_cue_snr3.pt", weights_only=False)
out_tid = np.asarray(b["outlier_target_ids"]).astype(np.int64)
par_tid = np.asarray(b["desi_target_ids"]).astype(np.int64)
print(f"outliers {out_tid.size} | parent(S/N>3) {par_tid.size}")

# ---- VAC, match by TARGETID ----
cols = (
    ["TARGETID", "HALPHA_EW", "DN4000", "LOGMSTAR", "SFR"]
    + [f"{l}_FLUX" for l in LINES]
    + [f"{l}_FLUX_IVAR" for l in LINES]
)
vac = Table.read(VAC, hdu="FASTSPEC")[cols]
vtid = np.asarray(vac["TARGETID"], np.int64)
order = np.argsort(vtid)
vts = vtid[order]


def rows(t):
    pos = np.clip(np.searchsorted(vts, t), 0, len(vts) - 1)
    return order[pos], vts[pos] == t


def grab(t):
    r, ok = rows(t)
    d = {}
    for l in LINES:
        f = np.asarray(vac[f"{l}_FLUX"])[r]
        iv = np.asarray(vac[f"{l}_FLUX_IVAR"])[r]
        d[l] = np.where(ok, f, np.nan)
        d[l + "_SNR"] = np.where(ok, f * np.sqrt(np.clip(iv, 0, None)), 0.0)
    for c in ["HALPHA_EW", "DN4000", "LOGMSTAR", "SFR"]:
        d[c] = np.where(ok, np.asarray(vac[c])[r], np.nan)
    return d, ok


do, oko = grab(out_tid)
da, oka = grab(par_tid)


# ---- BPT classification (NII) ----
def k03(x):
    return 0.61 / (x - 0.05) + 1.30


def k01(x):
    return 0.61 / (x - 0.47) + 1.19


def bpt(d):
    with np.errstate(all="ignore"):
        x = np.log10(d["NII_6584"] / d["HALPHA"])
        y = np.log10(d["OIII_5007"] / d["HBETA"])
    sel = (
        (d["NII_6584_SNR"] >= SNR)
        & (d["HALPHA_SNR"] >= SNR)
        & (d["OIII_5007_SNR"] >= SNR)
        & (d["HBETA_SNR"] >= SNR)
        & np.isfinite(x)
        & np.isfinite(y)
    )
    return x, y, sel


def cls(d, nt):
    x, y, sel = bpt(d)
    xs, ys = x[sel], y[sel]
    sf = (ys < k03(xs)) & (xs < 0.05)
    agn = (ys > k01(xs)) | (xs > 0.47)
    comp = ~sf & ~agn
    return sel.sum(), sf, comp, agn, xs


print("\n=== NII-BPT ===")
for nm, d, nt in [("outliers", do, out_tid.size), ("parent", da, par_tid.size)]:
    n, sf, comp, agn, xs = cls(d, nt)
    print(
        f"  {nm:8} 4-line {n}/{nt} ({100 * n / nt:.0f}%)  SF {100 * sf.mean():.1f}%  comp {100 * comp.mean():.1f}%  AGN {100 * agn.mean():.1f}%"
    )
    if nm == "outliers" and agn.sum():
        print(
            f"     '{agn.sum()} AGN' median log[NII]/Ha = {np.median(xs[agn]):.2f}  (real AGN sit at >0; low = low-Z SF artifact)"
        )

# ---- Halpha EW enrichment ----
print("\n=== Halpha EW ===")
eo = do["HALPHA_EW"][oko]
ea = da["HALPHA_EW"][oka]
eo = eo[eo > 0]
ea = ea[ea > 0]
print(f"  median: outliers {np.median(eo):.0f} A  parent {np.median(ea):.0f} A")
for thr in [50, 100, 200, 300]:
    print(
        f"  EW>{thr:4d}: outliers {100 * np.mean(eo > thr):5.1f}%  parent {100 * np.mean(ea > thr):5.2f}%  enrich {np.mean(eo > thr) / max(np.mean(ea > thr), 1e-9):.0f}x"
    )

# ---- THE coverage test: mock PRIOR ranges vs outliers ----
print("\n=== MOCK PRIOR COVERAGE (Cue draws) vs outliers ===")
with h5py.File(CUE, "r") as hf:
    sm = hf["priors/stellar_masses"][:]
    gz = hf["priors/gas_metallicities"][:]
    gu = hf["priors/gas_ionization_parameters"][:]


def pct(a):
    return np.round(np.nanpercentile(a, [0.5, 50, 99.5]), 2)


sm_log = sm if np.nanmedian(sm) < 20 else np.log10(sm)
print(
    f"  mock log stellar_mass  p0.5/50/99.5 = {pct(sm_log)}   (min {np.nanmin(sm_log):.2f})"
)
print(f"  outlier logM* (VAC)    p0.5/50/99.5 = {pct(do['LOGMSTAR'][oko])}")
print(
    f"  -> mocks below logM*=9 : {100 * np.mean(sm_log < 9):.2f}%   below 8.5: {100 * np.mean(sm_log < 8.5):.2f}%"
)
print(f"  mock gas_metallicity   p0.5/50/99.5 = {pct(gz)}   (grid floor -2.2)")
print(f"  mock gas_logU          p0.5/50/99.5 = {pct(gu)}   (grid ceiling -1)")

# ---- [OIII]/Ha: outliers beyond mock? ----
print("\n=== log([OIII]5007/Ha): outliers vs Cue mock ===")
with h5py.File(CUE, "r") as hf:
    lw = hf["line_wave"][:]
    io = int(np.argmin(np.abs(lw - 5008.2)))
    ih = int(np.argmin(np.abs(lw - 6564.6)))
    sub = np.sort(
        np.random.default_rng(0).choice(
            hf["priors/line_lum"].shape[0], 40000, replace=False
        )
    )
    ll = hf["priors/line_lum"][sub]
with np.errstate(all="ignore"):
    mo3 = np.log10(ll[:, io] / ll[:, ih])
    mo3 = mo3[np.isfinite(mo3)]
with np.errstate(all="ignore"):
    oo3 = np.log10(do["OIII_5007"] / do["HALPHA"])
s = oko & (do["OIII_5007_SNR"] >= SNR) & (do["HALPHA_SNR"] >= SNR)
oo3 = oo3[s & np.isfinite(oo3)]
lo, hi = np.percentile(mo3, [0.5, 99.5])
print(f"  Cue mock  p0.5/50/99.5 = {np.percentile(mo3, [0.5, 50, 99.5]).round(2)}")
print(f"  outliers  p1/50/99     = {np.percentile(oo3, [1, 50, 99]).round(2)}")
print(
    f"  outliers beyond mock [{lo:.2f},{hi:.2f}] = {100 * np.mean((oo3 > hi) | (oo3 < lo)):.0f}%"
)
