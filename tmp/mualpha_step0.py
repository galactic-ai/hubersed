"""STEP 0: does the OBSERVED DESI feature distribution carry a mass/z signal (i.e. is mu_alpha
constrainable)? Uses FastSpecFit DIRECT measurements only (Halpha EW, Dn4000) -- not its model params."""

import os

os.environ["MPLCONFIGDIR"] = "/tmp/_mplconfig"
import numpy as np
from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VAC = "data/fastspec-iron-sv3-bright.fits"
h = fits.open(VAC, memmap=True)


def col(n):
    for hd in h:
        c = getattr(getattr(hd, "columns", None), "names", None)
        if c and n in c:
            return np.asarray(hd.data[n])


lm = col("LOGMSTAR").astype(float)
zz = col("Z").astype(float)
ew = col("HALPHA_EW").astype(float)
dn = col("DN4000").astype(float)
ok = (
    np.isfinite(lm)
    & np.isfinite(zz)
    & np.isfinite(ew)
    & np.isfinite(dn)
    & (zz > 0.01)
    & (zz < 0.45)
    & (lm > 7)
    & (lm < 12)
    & (dn > 0.5)
    & (dn < 3)
)
lm, zz, ew, dn = lm[ok], zz[ok], ew[ok], dn[ok]
print(f"N usable = {len(lm):,}")
ZB = [(0.05, 0.15), (0.15, 0.25), (0.25, 0.35)]
MB = [(8.5, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0), (11.0, 11.5)]
cols = plt.cm.viridis(np.linspace(0, 0.92, len(MB)))
fig, ax = plt.subplots(2, len(ZB), figsize=(15, 7), constrained_layout=True)
print(f"\n{'zbin':12s}{'massbin':12s}{'N':>7}{'med EW':>9}{'med Dn4000':>12}")
for j, (z0, z1) in enumerate(ZB):
    for i, (m0, m1) in enumerate(MB):
        s = (zz >= z0) & (zz < z1) & (lm >= m0) & (lm < m1)
        n = int(s.sum())
        if n < 100:
            continue
        print(
            f"{f'{z0}-{z1}':12s}{f'{m0}-{m1}':12s}{n:7d}{np.median(ew[s]):9.1f}{np.median(dn[s]):12.3f}"
        )
        ax[0, j].hist(
            np.clip(np.log10(np.clip(ew[s], 0.05, None)), -1.3, 2.6),
            bins=40,
            density=True,
            histtype="step",
            lw=2,
            color=cols[i],
            label=f"{m0}-{m1} (N={n})",
        )
        ax[1, j].hist(
            np.clip(dn[s], 0.8, 2.4),
            bins=40,
            density=True,
            histtype="step",
            lw=2,
            color=cols[i],
        )
    ax[0, j].set_title(f"z = {z0}-{z1}", fontsize=11)
    ax[0, j].set_xlabel(r"$\log_{10}$ H$\alpha$ EW [$\AA$]  (<10 Myr)")
    ax[1, j].set_xlabel(r"D$_n$4000  ($\sim$1 Gyr)")
    ax[0, j].set_ylabel("density")
    ax[1, j].set_ylabel("density")
    for a in (ax[0, j], ax[1, j]):
        a.grid(alpha=0.2)
ax[0, 0].legend(fontsize=7, title=r"$\log M_\star$", title_fontsize=7)
fig.suptitle(
    "Observed DESI BGS feature distributions vs stellar mass — the signal $\\mu_\\alpha$ would carry",
    fontsize=12,
)
fig.savefig("tmp/figs/mualpha_step0.png", dpi=130)
print("\nsaved mualpha_step0.png")
