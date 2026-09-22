"""Does the model's reachable (Halpha EW, Dn4000) region contain the 37 observed bins?
Model grid = (mu_alpha x logzsol) from mualpha_step1_zmet.pkl; observed = mualpha_bin_grid.json."""

import os

os.environ["MPLCONFIGDIR"] = "/tmp/_mplconfig"
import numpy as np, pickle, json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pickle.load(open("results/mualpha_step1_zmet.pkl", "rb"))
res = d["res"]
MU = d["mu_grid"]
ZM = d["zmet_grid"]
grid = json.load(open("tmp/figs/mualpha_bin_grid.json"))
med = lambda k, f: float(np.nanmedian(np.asarray(res[k][f], float)))
fig, ax = plt.subplots(figsize=(9.2, 6.4), constrained_layout=True)
zc = plt.cm.autumn(np.linspace(0, 0.82, len(ZM)))
# tracks of constant logzsol (mu_alpha varying) -- the AGE direction
for zi, zm in enumerate(ZM):
    xs = [med(f"{mu}_{zm}", "ew") for mu in MU]
    ys = [med(f"{mu}_{zm}", "dn") for mu in MU]
    ax.plot(
        xs,
        ys,
        "-o",
        color=zc[zi],
        lw=2,
        ms=5,
        zorder=3,
        label=rf"$\log z_\odot$={zm:+.2f}  ($\mu_\alpha$ track)",
    )
    for mi, mu in enumerate(MU):
        ax.annotate(
            f"{mu:+.1f}",
            (xs[mi], ys[mi]),
            fontsize=6,
            color=zc[zi],
            xytext=(3, 3),
            textcoords="offset points",
        )
# tracks of constant mu_alpha (logzsol varying) -- the METALLICITY direction
for mu in MU:
    xs = [med(f"{mu}_{zm}", "ew") for zm in ZM]
    ys = [med(f"{mu}_{zm}", "dn") for zm in ZM]
    ax.plot(xs, ys, "--", color="0.55", lw=1.1, zorder=2)
# observed bins
ow = np.array([g["med_ew"] for g in grid])
od = np.array([g["med_dn"] for g in grid])
om = np.array([0.5 * (g["m0"] + g["m1"]) for g in grid])
s = ax.scatter(ow, od, c=om, cmap="viridis", s=62, ec="k", lw=0.7, zorder=5, marker="s")
plt.colorbar(s, ax=ax, label=r"observed bin $\log M_\star$")
ax.set_xscale("symlog", linthresh=1)
ax.set_xlabel(
    r"median H$\alpha$ EW [$\AA$]   ($\mu_\alpha$ moves this ~200$\times$; Z only ~2$\times$)"
)
ax.set_ylabel(r"median D$_n$4000   (raised by BOTH age and Z)")
ax.set_title(
    "Model reachable region (coloured $\\mu_\\alpha$ tracks, grey Z tracks) vs 37 observed DESI bins (squares)",
    fontsize=10.5,
)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.2)
fig.savefig("tmp/figs/zmet_plane.png", dpi=130)
print("saved zmet_plane.png")
print(
    f"observed: EW {ow.min():.2f}-{ow.max():.2f}   Dn4000 {od.min():.3f}-{od.max():.3f}"
)
mn = min(med(f"{mu}_{zm}", "dn") for mu in MU for zm in ZM)
mx = max(med(f"{mu}_{zm}", "dn") for mu in MU for zm in ZM)
print(
    f"model   : Dn4000 {mn:.3f}-{mx:.3f}  -> observed inside model range: {od.min() >= mn and od.max() <= mx}"
)
