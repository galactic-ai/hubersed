"""
STEP 2b: infer mu_alpha(M,z) and mu_Z(M,z) per locked bin by IMPORTANCE-REWEIGHTING the base set.

For each (M,z) bin: weight base draws by  N(alpha; mu_a, sig_a)/U(alpha) * N(logzsol; mu_Z, sig_Z)/U(logz),
build the predicted 2D (log EW, Dn4000) histogram, and score it against the observed 2D histogram with a
MULTINOMIAL likelihood.  2D histogram (not summaries) because Step 1 showed mu_alpha moves the LOCATION
while sigma_reg moves the LOW TAIL -- summaries would collapse them.  sigma_reg is marginalized (the base's
log-spaced grid == its log-uniform prior).  ESS guard: refuse any bin whose effective sample size is too low.
Observed measurement noise is added to the base features so the model carries the same broadening.
"""

import os, json

os.environ["MPLCONFIGDIR"] = "/tmp/_mplconfig"
import numpy as np
from astropy.io import fits

B = np.load("results/mualpha_base_set.npz")
bz, ba, blz, bew, bdn = [
    B[k].astype(float) for k in ["z", "alpha", "logzsol", "ew", "dn4000"]
]
A_LO, A_HI = B["alpha_range"]
Z_LO, Z_HI = B["logz_range"]
grid = json.load(open("tmp/figs/mualpha_bin_grid.json"))
h = fits.open("data/fastspec-iron-sv3-bright.fits", memmap=True)


def col(n):
    for hd in h:
        c = getattr(getattr(hd, "columns", None), "names", None)
        if c and n in c:
            return np.asarray(hd.data[n])


olm = col("LOGMSTAR").astype(float)
oz = col("Z").astype(float)
oew = col("HALPHA_EW").astype(float)
odn = col("DN4000").astype(float)
oewi = col("HALPHA_EW_IVAR")
odni = col("DN4000_IVAR")
oews = (
    1 / np.sqrt(np.clip(np.asarray(oewi, float), 1e-8, None))
    if oewi is not None
    else np.full(len(oew), 1.0)
)
odns = (
    1 / np.sqrt(np.clip(np.asarray(odni, float), 1e-8, None))
    if odni is not None
    else np.full(len(odn), 0.05)
)
# 2D histogram axes
XE = np.linspace(-1.3, 2.6, 11)
YE = np.linspace(0.9, 2.4, 11)
NC = (len(XE) - 1) * (len(YE) - 1)
tx = lambda ew: np.clip(np.log10(np.clip(ew, 0.05, None)), XE[0] + 1e-6, XE[-1] - 1e-6)
ty = lambda dn: np.clip(dn, YE[0] + 1e-6, YE[-1] - 1e-6)


def cells(ew, dn):
    ix = np.clip(np.digitize(tx(ew), XE) - 1, 0, len(XE) - 2)
    iy = np.clip(np.digitize(ty(dn), YE) - 1, 0, len(YE) - 2)
    return ix * (len(YE) - 1) + iy


npdf = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
MU_A = np.linspace(-0.5, 2.2, 28)
SIG_A = np.array([0.2, 0.4, 0.6, 0.9])
MU_Z = np.linspace(-1.3, 0.35, 18)
SIG_Z = np.array([0.15, 0.3, 0.5])
ESS_FLOOR = 200
rng = np.random.default_rng(0)
out = []
print(f"{'z':>6}{'logM':>13}{'N':>7}{'ESS':>7}{'mu_alpha':>10}{'mu_Z':>8}{'sig_a':>7}")
for g in grid:
    zc = 0.5 * (g["z0"] + g["z1"])
    zs = np.unique(bz)[np.argmin(np.abs(np.unique(bz) - zc))]
    m = bz == zs
    A, LZ, EW, DN = ba[m], blz[m], bew[m], bdn[m]
    o = (
        (olm >= g["m0"])
        & (olm < g["m1"])
        & (oz >= g["z0"])
        & (oz < g["z1"])
        & np.isfinite(oew)
        & np.isfinite(odn)
    )
    if o.sum() < 100:
        continue
    nobs = np.bincount(cells(oew[o], odn[o]), minlength=NC).astype(float)
    # add observed measurement noise to the base (one realization per draw) so widths are comparable
    se = np.clip(np.median(oews[o]), 0, 50)
    sd = np.clip(np.median(odns[o]), 0, 0.5)
    EWn = EW + rng.normal(0, se, len(EW))
    DNn = DN + rng.normal(0, sd, len(DN))
    cid = cells(EWn, DNn)
    best = (-np.inf, None)
    for sa in SIG_A:
        for ma in MU_A:
            wa = npdf(A, ma, sa) * (A_HI - A_LO)
            for sz in SIG_Z:
                for mz in MU_Z:
                    w = wa * npdf(LZ, mz, sz) * (Z_HI - Z_LO)
                    ess = w.sum() ** 2 / np.maximum((w**2).sum(), 1e-30)
                    if ess < ESS_FLOOR:
                        continue
                    p = np.bincount(cid, weights=w, minlength=NC)
                    s = p.sum()
                    if s <= 0:
                        continue
                    p = p / s
                    ll = float(np.sum(nobs * np.log(p + 1e-9)))
                    if ll > best[0]:
                        best = (ll, (ma, sa, mz, sz, ess))
    if best[1] is None:
        print(
            f"{zc:6.3f}{f'{g[chr(109) + chr(48)]}-{g[chr(109) + chr(49)]}':>13}{g['N']:7d}   ESS-FAIL"
        )
        continue
    ma, sa, mz, sz, ess = best[1]
    out.append(
        dict(
            z=zc,
            m0=g["m0"],
            m1=g["m1"],
            N=g["N"],
            mu_alpha=ma,
            sig_a=sa,
            mu_Z=mz,
            sig_Z=sz,
            ess=float(ess),
            lnL=best[0],
        )
    )
    print(
        f"{zc:6.3f}{f'{g[chr(109) + chr(48)]:.2f}-{g[chr(109) + chr(49)]:.2f}':>13}{g['N']:7d}{ess:7.0f}{ma:+10.2f}{mz:+8.2f}{sa:7.2f}"
    )
json.dump(out, open("tmp/figs/mualpha_results.json", "w"), indent=1)
print(f"\nfitted {len(out)}/{len(grid)} bins -> mualpha_results.json")
