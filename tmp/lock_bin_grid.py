"""Lock the (M*, z) bin grid for the mu_alpha population inference: bins that are BOTH
above the Pozzetti 95% completeness limit AND have N >= 100 (Burnham+2026 App. B)."""

import os

os.environ["MPLCONFIGDIR"] = "/tmp/_mplconfig"
import numpy as np, json
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
ew = col("HALPHA_EW").astype(float)
dn = col("DN4000").astype(float)
ok = (
    np.isfinite(lm)
    & np.isfinite(zz)
    & (zz > 0.01)
    & (zz < 0.45)
    & (lm > 6)
    & (lm < 12.5)
    & (fr > 0)
    & np.isfinite(ew)
    & np.isfinite(dn)
)
lm, zz, fr, ew, dn = lm[ok], zz[ok], fr[ok], ew[ok], dn[ok]
r = 22.5 - 2.5 * np.log10(fr)
g = np.isfinite(r) & (r > 12) & (r < 23)
lm, zz, r, ew, dn = lm[g], zz[g], r[g], ew[g], dn[g]
R_LIM = float(np.percentile(r, 99.5))
logMlim = lm + 0.4 * (r - R_LIM)
ZB = [
    (0.05, 0.10),
    (0.10, 0.15),
    (0.15, 0.20),
    (0.20, 0.25),
    (0.25, 0.30),
    (0.30, 0.35),
    (0.35, 0.40),
]
MB = np.arange(9.0, 11.76, 0.25)
NMIN = 100
grid = []
print(f"{'z bin':12s}{'Mbias':>7}   usable mass bins (N)")
for z0, z1 in ZB:
    s = (zz >= z0) & (zz < z1)
    if s.sum() < 300:
        continue
    rc = np.percentile(r[s], 80)
    mbias = float(np.percentile(logMlim[s & (r >= rc)], 95))
    row = []
    for m0, m1 in zip(MB[:-1], MB[1:]):
        if m0 < mbias:
            continue  # below completeness -> selection-biased, drop
        b = s & (lm >= m0) & (lm < m1)
        n = int(b.sum())
        if n < NMIN:
            continue
        row.append((round(m0, 2), round(m1, 2), n))
        grid.append(
            dict(
                z0=z0,
                z1=z1,
                m0=float(m0),
                m1=float(m1),
                N=n,
                med_ew=float(np.median(ew[b])),
                med_dn=float(np.median(dn[b])),
            )
        )
    print(
        f"{f'{z0}-{z1}':12s}{mbias:7.2f}   "
        + " ".join(f"{a:.2f}-{b:.2f}({n})" for a, b, n in row)
    )
print(f"\nTOTAL usable bins: {len(grid)}   galaxies: {sum(d['N'] for d in grid):,}")
print(
    f"min N in a bin: {min(d['N'] for d in grid)}   median N: {int(np.median([d['N'] for d in grid]))}"
)
json.dump(grid, open("tmp/figs/mualpha_bin_grid.json", "w"), indent=1)
# visual
fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
for d in grid:
    ax.add_patch(
        plt.Rectangle(
            (d["z0"], d["m0"]),
            d["z1"] - d["z0"],
            d["m1"] - d["m0"],
            facecolor=plt.cm.viridis(min(d["N"], 6000) / 6000),
            ec="w",
            lw=0.7,
        )
    )
    ax.text(
        0.5 * (d["z0"] + d["z1"]),
        0.5 * (d["m0"] + d["m1"]),
        f"{d['N']}",
        ha="center",
        va="center",
        fontsize=5.5,
        color="w",
    )
ax.set_xlim(0.05, 0.40)
ax.set_ylim(9.0, 11.75)
ax.set_xlabel("z")
ax.set_ylabel(r"$\log M_\star$")
ax.set_title(
    r"Locked $\mu_\alpha$ grid: complete (Pozzetti 95%) $\wedge$ N$\geq$100",
    fontsize=11,
)
fig.savefig("tmp/figs/mualpha_bin_grid.png", dpi=130)
print("saved mualpha_bin_grid.png + mualpha_bin_grid.json")
