import os

os.environ["MPLCONFIGDIR"] = "/tmp/_mplconfig"
import numpy as np, pickle
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pickle.load(open("results/mualpha_step1.pkl", "rb"))
res = d["res"]
MU = d["mu_grid"]
SR = d["sigreg_grid"]
cols = plt.cm.coolwarm(np.linspace(0, 1, len(MU)))
sizes = {0.3: 55, 1.0: 120, 3.0: 240}
fig = plt.figure(figsize=(15, 4.8))
# --- panel 1: median grid: mu_alpha slides location, sigma_reg barely moves along it
a0 = fig.add_subplot(1, 3, 1)
for si, sr in enumerate(SR):
    xs = [np.nanmedian(res[f"{mu}_{sr}"]["ew"]) for mu in MU]
    ys = [np.nanmedian(res[f"{mu}_{sr}"]["dn"]) for mu in MU]
    a0.plot(xs, ys, "-", color="0.6", lw=1, zorder=1)
    for mi, mu in enumerate(MU):
        a0.scatter(
            xs[mi], ys[mi], s=sizes[sr], color=cols[mi], ec="k", lw=0.6, zorder=3
        )
a0.set_xscale("symlog", linthresh=1)
a0.set_xlabel(r"median H$\alpha$ EW [$\AA$]")
a0.set_ylabel(r"median D$_n$4000")
a0.set_title(
    r"$\mu_\alpha$ slides location (colour);  $\sigma_{\rm reg}$ (size) barely moves it",
    fontsize=10,
)
for mi, mu in enumerate(MU):
    a0.scatter([], [], color=cols[mi], ec="k", label=rf"$\mu_\alpha$={mu:+.1f}")
a0.legend(fontsize=7, loc="lower left")
a0.grid(alpha=0.2)
# --- panels 2,3: sigma_reg fattens the LOW TAIL at fixed mu_alpha
for pi, mu in enumerate([-0.5, 0.0]):
    ax = fig.add_subplot(1, 3, 2 + pi)
    for si, sr in enumerate(SR):
        ew = np.asarray(res[f"{mu}_{sr}"]["ew"], float)
        ew = ew[np.isfinite(ew)]
        ax.hist(
            np.clip(np.log10(np.clip(ew, 0.05, None)), -1.3, 3.4),
            bins=32,
            density=True,
            histtype="step",
            lw=2.2,
            color=plt.cm.viridis(si / 2.0),
            label=rf"$\sigma_{{\rm reg}}$={sr}  (16th={np.nanpercentile(ew, 16):.1f})",
        )
    ax.set_xlabel(r"$\log_{10}$ H$\alpha$ EW  (0.05 floor = absorption)")
    ax.set_ylabel("density")
    ax.set_title(
        rf"$\mu_\alpha$={mu:+.1f}: burstiness fattens the low tail", fontsize=10
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("tmp/figs/mualpha_step1_grid.png", dpi=130, bbox_inches="tight")
print("saved mualpha_step1_grid.png")
for mu in MU:
    r = [
        f"{np.nanpercentile(np.asarray(res[f'{mu}_{sr}']['ew'], float)[np.isfinite(res[f'{mu}_{sr}']['ew'])], 16):8.2f}"
        for sr in SR
    ]
    print(f"mu={mu:+.1f}  EW 16th pct vs sigma_reg{SR}: {' '.join(r)}")
