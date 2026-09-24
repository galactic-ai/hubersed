"""Where is BGS mass-complete? Decides whether the mu_alpha populations can dodge the
flux-limit selection entirely (safe bins) or must forward-model it."""

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
dn = col("DN4000").astype(float)
ew = col("HALPHA_EW").astype(float)
fr = col("FLUX_R")
print("FLUX_R present:", fr is not None)
ok = (
    np.isfinite(lm)
    & np.isfinite(zz)
    & (zz > 0.01)
    & (zz < 0.5)
    & (lm > 7)
    & (lm < 12.5)
)
if fr is not None:
    fr = np.asarray(fr, float)
    rmag = np.where(fr > 0, 22.5 - 2.5 * np.log10(np.clip(fr, 1e-3, None)), np.nan)
    ok &= np.isfinite(rmag)
lm, zz = lm[ok], zz[ok]
dn = dn[ok]
ew = ew[ok]
if fr is not None:
    rmag = rmag[ok]
    print(
        f"r mag: median {np.nanmedian(rmag):.2f}  95th {np.nanpercentile(rmag, 95):.2f}  99th {np.nanpercentile(rmag, 99):.2f}"
    )
ZB = [
    (0.05, 0.10),
    (0.10, 0.15),
    (0.15, 0.20),
    (0.20, 0.25),
    (0.25, 0.30),
    (0.30, 0.35),
    (0.35, 0.40),
]
mbins = np.arange(7.5, 12.3, 0.15)
mc = 0.5 * (mbins[1:] + mbins[:-1])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
print(
    f"\n{'z bin':14s}{'N':>8}{'peak logM':>11}{'90% lim':>10}  (turnover = incompleteness)"
)
lims = []
for i, (z0, z1) in enumerate(ZB):
    s = (zz >= z0) & (zz < z1)
    n = int(s.sum())
    if n < 200:
        continue
    cnt, _ = np.histogram(lm[s], bins=mbins)
    pk = mc[np.argmax(cnt)]
    # 90% limit: lowest mass bin (above peak-side) where counts still >= 90% of running max going down
    # simple robust proxy: mass where counts fall below 50% of peak on the LOW-mass side
    ipk = int(np.argmax(cnt))
    lo = ipk
    while lo > 0 and cnt[lo] > 0.5 * cnt[ipk]:
        lo -= 1
    lim = mc[lo]
    lims.append((0.5 * (z0 + z1), lim, pk))
    print(f"{f'{z0}-{z1}':14s}{n:8d}{pk:11.2f}{lim:10.2f}")
    ax[0].step(mc, cnt / max(cnt.max(), 1), where="mid", lw=1.8, label=f"{z0}-{z1}")
ax[0].set_xlabel(r"$\log M_\star$")
ax[0].set_ylabel("normalized counts")
ax[0].set_title(
    "BGS mass histogram per z — low-mass turnover = flux limit", fontsize=11
)
ax[0].legend(fontsize=7, title="z")
ax[0].grid(alpha=0.2)
L = np.array(lims)
ax[1].plot(
    L[:, 0],
    L[:, 1],
    "o-",
    color="tab:red",
    label="50%-of-peak limit (approx completeness)",
)
ax[1].plot(L[:, 0], L[:, 2], "s--", color="0.5", label="histogram peak")
ax[1].set_xlabel("z")
ax[1].set_ylabel(r"$\log M_\star$")
ax[1].set_title("Usable (selection-free) mass range vs z", fontsize=11)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.2)
fig.savefig("tmp/figs/bgs_completeness.png", dpi=130)
print("\nsaved bgs_completeness.png")
