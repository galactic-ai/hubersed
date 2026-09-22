"""sSFR vs M* : OLD (mean-zero) mocks vs NEW (alpha-tilted) mocks vs OBSERVED DESI BGS.

Uses priors/ only (no flux reads) -> all 500k of each set.

SFR convention: mass formed in the last 100 Myr / 1e8 yr, matching FastSpecFit's SFR (a ~100 Myr
SED average -- established earlier in this project). Partial-bin fractions applied, since the
stochastic age bins (make_stochastic_agebins) do not land on 100 Myr exactly.

SFH algebra (prospector convention, verified against the project log):
  logsfr_ratios[i] = log10(SFR[i]/SFR[i+1]), bin 0 = YOUNGEST
  -> SFR_i = SFR_0 / prod_{j<i} 10^r_j ;  mass_i = SFR_i*dt_i ;  M_tot = 10^logmass
Note sSFR here is independent of M_tot by construction -> any M* trend must come from a
correlation between the SFH ratios and mass. alpha is drawn INDEPENDENT of mass, so the mock
panels should be FLAT in M* -- that is the covering prior working as designed, not a bug.
"""

import os

os.environ["MPLCONFIGDIR"] = "/tmp/_mplconfig"
import numpy as np, h5py
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "data/prospector_model/"
ZLO, ZHI = 0.05, 0.15  # same slice as the coverage test; keeps t_univ comparable
TAVG = 1e8  # 100 Myr


def ssfr_from(h5):
    h = h5py.File(h5, "r")
    z = h["priors/redshifts"][:]
    lm = h["priors/stellar_masses"][:]
    s = (z >= ZLO) & (z < ZHI)
    z, lm = z[s], lm[s]
    r = h["priors/logsfr_ratios"][s, :].astype(float)
    N = len(z)
    tu = cosmo.age(z).to_value(u.Gyr)
    # make_stochastic_agebins, vectorized over galaxies
    e = np.zeros((N, 10, 2))
    e[:, 0] = [0.001, 0.005]
    e[:, 1] = [0.005, 0.01]
    k = np.arange(9)
    lt = 0.01 * (0.95 * tu[:, None] / 0.01) ** (
        k[None, :] / 8.0
    )  # geomspace(0.01, .95tu, 9)
    for i in range(2, 10):
        e[:, i, 0] = lt[:, i - 2]
        e[:, i, 1] = lt[:, i - 1]
    t0 = e[:, :, 0] * 1e9
    t1 = e[:, :, 1] * 1e9
    dt = t1 - t0  # yr
    sr = 10.0**r
    P = np.ones((N, 10))
    P[:, 1:] = np.cumprod(sr, axis=1)  # prod_{j<i} 10^r_j
    coef = dt / P
    Mtot = 10.0**lm
    SFR0 = Mtot / np.sum(coef, axis=1)
    mass = SFR0[:, None] * coef
    frac = np.clip(
        (TAVG - t0) / np.maximum(dt, 1e-30), 0.0, 1.0
    )  # bin fraction within 100 Myr
    M100 = np.sum(mass * frac, axis=1)
    ssfr = np.log10(np.maximum(M100 / TAVG, 1e-30) / Mtot)
    return lm, ssfr


lm_old, ss_old = ssfr_from(
    D + "prospector_stochastic_model_seds_cue_500000.meanzero.h5"
)
lm_new, ss_new = ssfr_from(D + "prospector_stochastic_model_seds_cue_500000.h5")

hv = fits.open("data/fastspec-iron-sv3-bright.fits", memmap=True)


def col(n):
    for hd in hv:
        c = getattr(getattr(hd, "columns", None), "names", None)
        if c and n in c:
            return np.asarray(hd.data[n])


olm = col("LOGMSTAR").astype(float)
oz = col("Z").astype(float)
osfr = col("SFR").astype(float)
o = (
    (oz >= ZLO)
    & (oz < ZHI)
    & np.isfinite(olm)
    & np.isfinite(osfr)
    & (osfr > 0)
    & (olm > 6)
)
olm_s = olm[o]
oss = np.log10(osfr[o]) - olm[o]

print(
    f"OLD mocks  N={len(lm_old):,}   sSFR med {np.median(ss_old):+.2f}  16/84 {np.percentile(ss_old, 16):+.2f}/{np.percentile(ss_old, 84):+.2f}"
)
print(
    f"NEW mocks  N={len(lm_new):,}   sSFR med {np.median(ss_new):+.2f}  16/84 {np.percentile(ss_new, 16):+.2f}/{np.percentile(ss_new, 84):+.2f}"
)
print(
    f"OBSERVED   N={len(olm_s):,}   sSFR med {np.median(oss):+.2f}  16/84 {np.percentile(oss, 16):+.2f}/{np.percentile(oss, 84):+.2f}"
)

XL, YL = (8.0, 12.0), (-13.5, -7.5)
fig, ax = plt.subplots(
    1, 3, figsize=(16.5, 5.0), sharex=True, sharey=True, constrained_layout=True
)
sets = [
    (lm_old, ss_old, f"OLD mocks (mean-zero)  N={len(lm_old):,}"),
    (lm_new, ss_new, f"NEW mocks (alpha-tilted)  N={len(lm_new):,}"),
    (olm_s, oss, f"OBSERVED DESI BGS  N={len(olm_s):,}"),
]
for i, (x, y, t) in enumerate(sets):
    g = np.isfinite(x) & np.isfinite(y)
    ax[i].hexbin(
        np.clip(x[g], *XL),
        np.clip(y[g], *YL),
        gridsize=70,
        extent=(*XL, *YL),
        bins="log",
        cmap="viridis",
        mincnt=1,
    )
    # observed running median on EVERY panel, for direct comparison
    be = np.arange(8.5, 12.01, 0.25)
    cen = 0.5 * (be[:-1] + be[1:])
    med = [
        np.median(oss[(olm_s >= be[j]) & (olm_s < be[j + 1])])
        if ((olm_s >= be[j]) & (olm_s < be[j + 1])).sum() > 20
        else np.nan
        for j in range(len(be) - 1)
    ]
    ax[i].plot(cen, med, "-", color="red", lw=2.4, label="observed median")
    ax[i].set_title(t, fontsize=10.5)
    ax[i].set_xlabel(r"$\log M_\star$")
    ax[i].axhline(-11, color="w", ls=":", lw=1.2)  # conventional SF/quenched divide
    if i == 0:
        ax[i].set_ylabel(r"$\log$ sSFR (100 Myr) [yr$^{-1}$]")
        ax[i].legend(fontsize=8, loc="lower left")
fig.savefig("tmp/figs/ssfr_mstar.png", dpi=130)
print("\nsaved ssfr_mstar.png")
for nm, x, y in [("OLD", lm_old, ss_old), ("NEW", lm_new, ss_new), ("OBS", olm_s, oss)]:
    print(
        f"\n{nm}: sSFR median vs logM (flat mock panels are BY DESIGN -- alpha drawn indep. of mass)"
    )
    for m0 in (9.0, 9.5, 10.0, 10.5, 11.0, 11.5):
        s = (x >= m0) & (x < m0 + 0.5)
        if s.sum() > 20:
            print(
                f"   logM {m0:.1f}-{m0 + 0.5:.1f}:  med {np.median(y[s]):+6.2f}   frac quenched (<-11) {100 * np.mean(y[s] < -11):5.1f}%   N={s.sum()}"
            )
