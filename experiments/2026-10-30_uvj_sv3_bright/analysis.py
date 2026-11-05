"""UVJ diagram of SV3 BGS bright galaxies at 0 < z < 0.5 with the Muzzin+13 quiescent box.

QUESTION: where does the continuum outlier 39633433328091196 sit on the rest-frame UVJ diagram?
HYPOTHESIS: none registered.
INPUTS: data/cigale_dr1/cigale_sv3_bright.npz of extract.py (DR1 CIGALE VAC v1.2, UV and VJ
in AB mag) and Z from the METADATA of data/fastspec-iron-sv3-bright.fits. Boxes from Muzzin+13
(arXiv:1303.4409, ms.tex:128-133, 0 < z < 1 branch) and Whitaker+12 as quoted by Fumagalli+14
(arXiv:1308.4132, ms.tex:155).
SEED: none.
COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-30_uvj_sv3_bright/analysis.py``.
RESULT: 2026-10-30, uvj.txt. 250351 galaxies; 95922 in the Muzzin+13 box, 108732 in Whitaker+12.
39633433328091196 (CIGALE row from sv3/dark) U-V 2.151, V-J 1.279, inside both. Assumptions:
CIGALE photometric SED colours; one CIGALE row per TARGETID, 19% of rows (47086) not from sv3/bright.
FIGURES: results/2026-10-30_uvj_sv3_bright/uvj.png and uvj.txt.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm

from hubersed.paths import get_paths
from hubersed.plotting.style import use_apj_style

TID = 39633433328091196
PATHS = get_paths()
OUT = PATHS["RESULTS"] / "2026-10-30_uvj_sv3_bright"
OUT.mkdir(parents=True, exist_ok=True)
ZMAX = 0.5
BIN = 0.025  # mag


BOXES = {
    "Muzzin+13": (0.88, 0.69, "-"),
    "Whitaker+12 (Fumagalli+14)": (0.8, 0.7, "--"),
}  # slope, intercept and line style of the sloped edge; both add U-V > 1.3 and V-J < 1.5


def quiescent(uv, vj, slope=0.88, icpt=0.69):
    """Return True inside a UVJ quiescent box, by default Muzzin+13 for 0 < z < 1."""
    return (uv > 1.3) & (vj < 1.5) & (uv > slope * vj + icpt)


# %% sample: SV3 bright TARGETIDs at 0 < z < 0.5 (FastSpecFit Z) with finite CIGALE colours
cat = np.load(PATHS["DATA"] / "cigale_dr1" / "cigale_sv3_bright.npz")
meta = fits.getdata(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", "METADATA")
zfs = dict(zip(np.asarray(meta["TARGETID"]), np.asarray(meta["Z"], float), strict=True))
z = np.array([zfs[t] for t in cat["TARGETID"]])
uv, vj = cat["UV"], cat["VJ"]
inz = (z > 0) & (z < ZMAX)
ok = inz & np.isfinite(uv) & np.isfinite(vj)
t = np.flatnonzero(cat["TARGETID"] == TID)
assert len(t) == 1, t
t = t[0]
lines = [
    f"SV3 bright TARGETIDs {len(meta)}, with a CIGALE row {len(cat['TARGETID'])}",
    f"0 < z < {ZMAX}: {inz.sum()}, with finite UV and VJ: {ok.sum()}",
    "CIGALE SURVEY/PROGRAM of the plotted rows: "
    + ", ".join(
        f"{s}/{p} {n}"
        for (s, p), n in zip(
            *np.unique(np.c_[cat["SURVEY"][ok], cat["PROGRAM"][ok]], axis=0, return_counts=True),
            strict=True,
        )
    ),  # fmt: skip
    *[
        f"quiescent ({name} box): {quiescent(uv[ok], vj[ok], s, c).sum()} of {ok.sum()}"
        for name, (s, c, _) in BOXES.items()
    ],  # fmt: skip
    f"{TID}: z {z[t]:.4f} (CIGALE z {cat['Z'][t]:.4f}, {cat['SURVEY'][t]}/{cat['PROGRAM'][t]}), "
    f"U-V {uv[t]:.3f} +- {cat['UV_ERR'][t]:.3f}, V-J {vj[t]:.3f} +- {cat['VJ_ERR'][t]:.3f}, "
    + ", ".join(
        f"in {name} box {bool(quiescent(uv[t], vj[t], s, c))}" for name, (s, c, _) in BOXES.items()
    ),  # fmt: skip
]
(OUT / "uvj.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# %% figure: density of the sample, the box and the target
use_apj_style()
fig, ax = plt.subplots(figsize=(7, 7))
xe = np.arange(np.floor(vj[ok].min() / BIN) * BIN, vj[ok].max() + BIN, BIN)
ye = np.arange(np.floor(uv[ok].min() / BIN) * BIN, uv[ok].max() + BIN, BIN)
h, _, _ = np.histogram2d(vj[ok], uv[ok], bins=[xe, ye])
im = ax.pcolormesh(xe, ye, h.T, cmap="Greys", norm=LogNorm(vmin=1), rasterized=True)
fig.colorbar(im, ax=ax, label="galaxies per bin", pad=0.02)
top = max(ye[-1], uv[t] + 0.2)
for name, (s, c, ls) in BOXES.items():
    x_knee = (1.3 - c) / s
    ax.plot([xe[0], x_knee, 1.5, 1.5], [1.3, 1.3, s * 1.5 + c, top], color="k", ls=ls, lw=1.5,
            label=name)  # fmt: skip
ax.errorbar(vj[t], uv[t], xerr=cat["VJ_ERR"][t], yerr=cat["UV_ERR"][t], fmt="*", ms=16,
            color="#b2182b", mec="k", label=str(TID))  # fmt: skip
ax.set_xlabel(r"$V - J$ [AB mag]", fontsize=14)
ax.set_ylabel(r"$U - V$ [AB mag]", fontsize=14)
ax.legend(frameon=False, fontsize=12, loc="lower right")
fig.savefig(OUT / "uvj.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'uvj.png'}")
