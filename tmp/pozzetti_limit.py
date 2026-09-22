"""Pozzetti+2010 (arXiv:0907.5416, eq. in Sec 3.3) 95% mass-completeness limit for DESI BGS SV3.

Verified from TeX source, line 807:   log(M_lim) = log(M_star) + 0.4 (I - I_lim)
Procedure (lines 804-821): per galaxy compute M_lim (mass it would have AT the flux limit, same M/L);
take the 20% FAINTEST galaxies in each z bin; M_bias(z) = 95th percentile of their M_lim
("upper envelope below which lie 95% of the M_lim values") = 95% completeness limit on M*/L.
Here I -> r (Legacy Survey r, FLUX_R in nanomaggies).
"""

import os

os.environ["MPLCONFIGDIR"] = "/tmp/_mplconfig"
import numpy as np
from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

h = fits.open("data/fastspec-iron-sv3-bright.fits", memmap=True)


def col(n):
    for hd in h:
        c = getattr(getattr(hd, "columns", None), "names", None)
        if c and n in c:
            return np.asarray(hd.data[n])


lm = col("LOGMSTAR").astype(float)
zz = col("Z").astype(float)
fr = np.asarray(col("FLUX_R"), float)
ok = (
    np.isfinite(lm)
    & np.isfinite(zz)
    & (zz > 0.01)
    & (zz < 0.5)
    & (lm > 6)
    & (lm < 12.5)
    & (fr > 0)
)
lm, zz, fr = lm[ok], zz[ok], fr[ok]
r = 22.5 - 2.5 * np.log10(fr)  # Legacy nanomaggies -> AB mag
ok2 = np.isfinite(r) & (r > 12) & (r < 23)
lm, zz, r = lm[ok2], zz[ok2], r[ok2]
for p in [50, 90, 95, 99, 99.5, 99.9]:
    print(f"  r {p:5.1f}th pct = {np.percentile(r, p):.3f}")
R_LIM = float(np.percentile(r, 99.5))  # empirical survey limit (SV3 BGS bright+faint)
print(f"\nadopting r_lim = {R_LIM:.3f}  (N={len(r):,})")
logMlim = lm + 0.4 * (r - R_LIM)  # Pozzetti eq: mass if dimmed to the limit, same M/L
ZB = [
    (0.05, 0.10),
    (0.10, 0.15),
    (0.15, 0.20),
    (0.20, 0.25),
    (0.25, 0.30),
    (0.30, 0.35),
    (0.35, 0.40),
    (0.40, 0.45),
]
print(
    f"\n{'z bin':13s}{'N':>8}{'r(faint20%)>':>13}{'Mbias(95%)':>12}   usable mass range"
)
out = []
for z0, z1 in ZB:
    s = (zz >= z0) & (zz < z1)
    n = int(s.sum())
    if n < 300:
        continue
    rc = np.percentile(r[s], 80)  # faintest 20%
    faint = s & (r >= rc)
    mbias = float(
        np.percentile(logMlim[faint], 95)
    )  # upper envelope: 95% of Mlim below this
    hi = float(np.percentile(lm[s], 99.5))
    out.append((0.5 * (z0 + z1), mbias, hi, n))
    print(f"{f'{z0}-{z1}':13s}{n:8d}{rc:13.2f}{mbias:12.2f}   {mbias:.2f} -> {hi:.2f}")
O = np.array([(a, b, c, d) for a, b, c, d in out])
fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
ax.fill_between(
    O[:, 0],
    O[:, 1],
    O[:, 2],
    color="tab:green",
    alpha=0.18,
    label="usable (selection-free)",
)
ax.plot(
    O[:, 0],
    O[:, 1],
    "o-",
    color="tab:red",
    lw=2,
    label=r"Pozzetti $\mathcal{M}_{\rm bias}$ (95% complete)",
)
ax.set_xlabel("z")
ax.set_ylabel(r"$\log M_\star$")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.savefig("tmp/figs/pozzetti_limit.png", dpi=130)
print("\nsaved pozzetti_limit.png")
