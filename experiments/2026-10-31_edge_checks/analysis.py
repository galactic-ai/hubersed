"""Training-edge checks for two groups of screened continuum outliers: z > 0.5 and dwarfs (#17).

QUESTION: (a) does the flag rate rise toward the z = 0.6 end of the mocks, and where do the z > 0.5
survivors sit? (b) how many training mocks lie near the low-mass, low-z survivors in (z, log M)?
HYPOTHESIS: none registered.
INPUTS: seed 0 flags (results/wide_flow_corrected); DESI cont10 latent file (z, TARGETID); training
mocks data/latents/noised_cue_meanzero_wide/prospector_noise_spec_cue_cont10latent_snr3.h5 (154,144
after S/N > 3) with their masses from stochastic_priors_sample_cue_500000.npz (Corral, 2026-08-25;
mock target_ids index the draws, z matches to 3e-8); FastSpecFit LOGMSTAR for DESI. Mock mass is the
Prospector mass parameter, LOGMSTAR a current stellar mass, so (b) compares them only roughly.
Neighbourhood for (b): |dz| < 0.02 and |dlogM| < 0.25.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_edge_checks/analysis.py``.
RESULT: 2026-10-31. Common-flag rate is 0.03-0.3% at z 0.08-0.44 but 3.5-10.5% at z 0.48-0.60
and 3.8% below z 0.04, with mocks plentiful in every bin up to 0.6. Around the z > 0.5 survivors
4-25% of DESI is flagged, around the dwarfs 2.5-6.3%; 54-266 mocks lie near each dwarf.
FIGURES: results/2026-10-31_edge_checks/edge.txt and edge.png.
"""

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

RES, DATA = PATHS["RESULTS"], PATHS["DATA"]
OUT = RES / "2026-10-31_edge_checks"
OUT.mkdir(parents=True, exist_ok=True)
HIGHZ = (39627763623136430, 39633322460057381, 39627854832472847)
DWARF = (39627770254327945, 39633123138340748, 39627896897143091)
ZBINS = np.round(np.arange(0.0, 0.82, 0.04), 2)

flag = {}
for tag in ("cont10", "cont15"):
    o = torch.load(RES / "wide_flow_corrected" / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt",
                   weights_only=False, map_location="cpu")  # fmt: skip
    flag[tag] = {int(t) for t in o["outlier_target_ids"]}
with h5py.File(DATA / "latents" / "spender_spec_cont10latent_snr3.h5", "r") as f:
    dtid, dz = f["target_ids"][:].astype(np.int64), f["zs"][:].astype(float)
with h5py.File(DATA / "latents" / "noised_cue_meanzero_wide" / "prospector_noise_spec_cue_cont10latent_snr3.h5",
               "r") as f:  # fmt: skip
    mtid, mz = f["target_ids"][:].astype(np.int64), f["zs"][:].astype(float)
prior = np.load(DATA / "prospector_model" / "wide" / "stochastic_priors_sample_cue_500000.npz")
assert np.abs(prior["redshifts"][mtid] - mz).max() < 1e-6, "mock ids do not index the prior draws"
mm = prior["stellar_masses"][mtid].astype(float)

h = fits.open(DATA / "fastspec-iron-sv3-bright.fits", memmap=True)
M, S = h["METADATA"].data, h["FASTSPEC"].data
vrow = {int(t): i for i, t in enumerate(M["TARGETID"])}
dm = np.array([float(S["LOGMSTAR"][vrow[t]]) if t in vrow else np.nan for t in dtid.tolist()])
f10 = np.isin(dtid, list(flag["cont10"]))
f15 = np.isin(dtid, list(flag["cont15"]))
both = f10 & f15

# %% (a) by redshift
lines = [
    f"(a) DESI {len(dz)} spectra (z {dz.min():.3f}-{dz.max():.3f}), mocks {len(mz)} "
    f"(z {mz.min():.3f}-{mz.max():.3f}). Flagged % per z bin.",  # fmt: skip
    f"{'z bin':>11} {'DESI':>6} {'mocks':>6} {'cont10':>7} {'cont15':>7} {'both':>6}",
]
idx = np.digitize(dz, ZBINS) - 1
midx = np.digitize(mz, ZBINS) - 1
for k in range(len(ZBINS) - 1):
    s = idx == k
    n, nm = int(s.sum()), int((midx == k).sum())
    if n == 0:
        continue
    lines.append(f"[{ZBINS[k]:.2f},{ZBINS[k + 1]:.2f}) {n:6d} {nm:6d} {100 * f10[s].mean():7.2f} "
                 f"{100 * f15[s].mean():7.2f} {100 * both[s].mean():6.2f}")  # fmt: skip
lines.append(f"DESI with z > 0.6 (no mocks there): {int((dz > 0.6).sum())}, of them common flags "
             f"{int((both & (dz > 0.6)).sum())}")  # fmt: skip

# %% (b) mocks near each survivor in (z, log M)
lines += ["", "(b) Neighbourhood |dz| < 0.02, |dlogM| < 0.25: mocks, DESI, DESI common-flagged %."]
for name, group in (("z > 0.5", HIGHZ), ("dwarf", DWARF)):
    for t in group:
        j = int(np.flatnonzero(dtid == t)[0])
        zt, mt = dz[j], dm[j]
        nm = int(((np.abs(mz - zt) < 0.02) & (np.abs(mm - mt) < 0.25)).sum())
        sd = (np.abs(dz - zt) < 0.02) & (np.abs(dm - mt) < 0.25)
        lines.append(f"  {name:>7} {t}: z {zt:.3f}, logM {mt:.2f}: mocks {nm}, DESI {int(sd.sum())}, "
                     f"flagged {100 * both[sd].mean():.1f}%")  # fmt: skip
low = (dz < 0.06) & (dm < 8)
mlow = (mz < 0.06) & (mm < 8)
lines.append(f"z < 0.06 and logM < 8: DESI {int(low.sum())} ({100 * both[low].mean():.1f}% common-"
             f"flagged), mocks {int(mlow.sum())}")  # fmt: skip
text = "\n".join(lines)
(OUT / "edge.txt").write_text(text + "\n")
print(text)

# %% figure
use_apj_style()
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
mid = 0.5 * (ZBINS[:-1] + ZBINS[1:])
for lab, arr, c in (("cont10", f10, "#1b9e77"), ("cont15", f15, "#7570b3"), ("both", both, "k")):
    fr = [100 * arr[idx == k].mean() if (idx == k).any() else np.nan for k in range(len(mid))]
    ax[0].plot(mid, fr, "o-", color=c, ms=4, label=lab)
ax[0].axvline(0.6, color="r", ls="--", lw=1, label="mock z max")
for t in HIGHZ:
    ax[0].axvline(dz[dtid == t][0], color="0.5", lw=0.8)
ax[0].set_xlabel("redshift", fontsize=13)
ax[0].set_ylabel(r"DESI flagged [\%]", fontsize=13)
ax[0].legend(frameon=False, fontsize=11)
ax[1].hexbin(mz, mm, gridsize=60, bins="log", cmap="Greys", mincnt=1)
ax[1].scatter(dz, dm, s=0.5, color="#d95f02", alpha=0.3, label="DESI")
ax[1].scatter(dz[both], dm[both], s=4, color="k", label="common flags")
for group, mk in ((HIGHZ, "s"), (DWARF, "*")):
    for t in group:
        j = dtid == t
        ax[1].scatter(dz[j], dm[j], s=120, marker=mk, facecolor="none", edgecolor="r", lw=1.5)
ax[1].set_xlabel("redshift", fontsize=13)
ax[1].set_ylabel(r"$\log M$ (mocks: prior mass; DESI: LOGMSTAR)", fontsize=13)
ax[1].legend(frameon=False, fontsize=11, loc="lower right", markerscale=4)
fig.savefig(OUT / "edge.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'edge.png'}")
