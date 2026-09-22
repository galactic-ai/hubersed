"""WHY does mu_Z rail? Scan lnL(mu_alpha, mu_Z) for one massive bin and compare the predicted vs
observed MARGINALS at the best fit. If the model simply cannot make the observed (EW, Dn4000) joint,
the params rail as a symptom -- so look at the actual distributions, not the point estimate."""

import os

os.environ["MPLCONFIGDIR"] = "/tmp/_mplconfig"
import numpy as np
from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = np.load("results/mualpha_base_set.npz")
bz, ba, blz, bew, bdn = [
    B[k].astype(float) for k in ["z", "alpha", "logzsol", "ew", "dn4000"]
]
A_LO, A_HI = B["alpha_range"]
Z_LO, Z_HI = B["logz_range"]
h = fits.open("data/fastspec-iron-sv3-bright.fits", memmap=True)


def col(n):
    for hd in h:
        c = getattr(getattr(hd, "columns", None), "names", None)
        if c and n in c:
            return np.asarray(hd.data[n])


lm = col("LOGMSTAR").astype(float)
z = col("Z").astype(float)
ew = col("HALPHA_EW").astype(float)
dn = col("DN4000").astype(float)
M0, M1, Z0, Z1 = 11.0, 11.25, 0.05, 0.10
o = (lm >= M0) & (lm < M1) & (z >= Z0) & (z < Z1) & np.isfinite(ew) & np.isfinite(dn)
OEW, ODN = ew[o], dn[o]
m = bz == np.unique(bz)[0]
A, LZ, EW, DN = ba[m], blz[m], bew[m], bdn[m]
print(f"observed N={o.sum()}   base N={m.sum()}")
XE = np.linspace(-1.3, 2.6, 11)
YE = np.linspace(0.9, 2.4, 11)
NC = (len(XE) - 1) * (len(YE) - 1)


def cells(e, d):
    ix = np.clip(
        np.digitize(
            np.clip(np.log10(np.clip(e, 0.05, None)), XE[0] + 1e-6, XE[-1] - 1e-6), XE
        )
        - 1,
        0,
        len(XE) - 2,
    )
    iy = np.clip(
        np.digitize(np.clip(d, YE[0] + 1e-6, YE[-1] - 1e-6), YE) - 1, 0, len(YE) - 2
    )
    return ix * (len(YE) - 1) + iy


npdf = lambda x, mu, s: np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))
nobs = np.bincount(cells(OEW, ODN), minlength=NC).astype(float)
cid = cells(EW, DN)  # NO noise added here -- isolate the reweighting itself
MU_A = np.linspace(-0.5, 2.2, 28)
MU_Z = np.linspace(-1.3, 0.35, 18)
SA, SZ = 0.4, 0.3
L = np.full((len(MU_A), len(MU_Z)), -np.inf)
E = np.zeros_like(L)
for i, ma in enumerate(MU_A):
    wa = npdf(A, ma, SA)
    for j, mz in enumerate(MU_Z):
        w = wa * npdf(LZ, mz, SZ)
        E[i, j] = w.sum() ** 2 / max((w**2).sum(), 1e-30)
        p = np.bincount(cid, weights=w, minlength=NC)
        s = p.sum()
        if s > 0:
            L[i, j] = float(np.sum(nobs * np.log(p / s + 1e-9)))
i, j = np.unravel_index(np.argmax(L), L.shape)
ma, mz = MU_A[i], MU_Z[j]
print(f"best: mu_alpha={ma:+.2f}  mu_Z={mz:+.2f}  lnL={L[i, j]:.1f}  ESS={E[i, j]:.0f}")
print(
    f"mu_Z at grid edge? {'YES -> ' + ('rails high' if j == len(MU_Z) - 1 else 'rails low') if j in (0, len(MU_Z) - 1) else 'no'}"
)
w = npdf(A, ma, SA) * npdf(LZ, mz, SZ)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
c = ax[0].pcolormesh(
    MU_Z,
    MU_A,
    np.ma.masked_invalid(L - np.nanmax(L)),
    vmin=-200,
    vmax=0,
    cmap="viridis",
)
plt.colorbar(c, ax=ax[0], label="lnL - max")
ax[0].plot(mz, ma, "r*", ms=16)
ax[0].set_xlabel(r"$\mu_Z$")
ax[0].set_ylabel(r"$\mu_\alpha$")
ax[0].set_title("lnL surface")
bE = np.linspace(-1.3, 2.6, 40)
ax[1].hist(
    np.log10(np.clip(OEW, 0.05, None)),
    bins=bE,
    density=True,
    histtype="step",
    lw=2.3,
    color="k",
    label=f"OBSERVED N={o.sum()}",
)
ax[1].hist(
    np.log10(np.clip(EW, 0.05, None)),
    bins=bE,
    weights=w,
    density=True,
    histtype="stepfilled",
    alpha=0.45,
    color="tab:orange",
    label="model @ best fit",
)
ax[1].set_xlabel(r"log H$\alpha$ EW")
ax[1].legend(fontsize=8)
ax[1].set_title("EW marginal")
bD = np.linspace(0.9, 2.4, 40)
ax[2].hist(
    np.clip(ODN, 0.9, 2.4),
    bins=bD,
    density=True,
    histtype="step",
    lw=2.3,
    color="k",
    label="OBSERVED",
)
ax[2].hist(
    np.clip(DN, 0.9, 2.4),
    bins=bD,
    weights=w,
    density=True,
    histtype="stepfilled",
    alpha=0.45,
    color="tab:orange",
    label="model @ best fit",
)
ax[2].set_xlabel(r"D$_n$4000")
ax[2].legend(fontsize=8)
ax[2].set_title("Dn4000 marginal")
fig.savefig("tmp/figs/diag_railing.png", dpi=130)
wn = w / w.sum()
for nm, mo, ob in [
    ("log EW", np.log10(np.clip(EW, 0.05, None)), np.log10(np.clip(OEW, 0.05, None))),
    ("Dn4000", DN, ODN),
]:
    q = lambda x, ww=None: [
        float(np.percentile(x, p))
        if ww is None
        else float(np.interp(p / 100, np.cumsum(ww[np.argsort(x)]), np.sort(x)))
        for p in (16, 50, 84)
    ]
    print(f"{nm:8s} model {q(mo, wn)}   obs {q(ob)}")
print("saved diag_railing.png")
