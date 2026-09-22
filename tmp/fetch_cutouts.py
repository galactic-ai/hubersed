import sys, io
import numpy as np, torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from astropy.table import Table
from spender.data import desi
from hubersed.paths import PATHS

DATA, RES = PATHS["DATA"], PATHS["RESULTS"]
VAC = DATA / "fastspec-iron-sv3-bright.fits"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8

# cutout geometry + DESI fiber aperture overlay
CUTOUT_SIZE = 256  # px (max 512)
PIXSCALE = 0.262  # arcsec/px (Legacy native)
FIBER_DIAM = 1.5  # arcsec, DESI fiber on-sky diameter
APER_R_PX = 0.5 * FIBER_DIAM / PIXSCALE  # = 2.86 px radius in the cutout

# worst-N outliers by IsoForest score
# source: 'iso' (default) = IsoForest worst by score; 'flow' = NSF flow worst by log p
src = sys.argv[2] if len(sys.argv) > 2 else "iso"
if src == "flow":
    b = torch.load(RES / "desi_outliers_flow_nsf_15latent_snr3.pt", weights_only=False)
    dtid = np.asarray(b["desi_target_ids"]).astype(np.int64)
    lp = np.asarray(b["log_p_desi"])
    worst = dtid[np.argsort(lp)][:N]  # lowest log p = most OOD
    tag = "flowNSF15D"
elif src == "file":
    worst = np.loadtxt(sys.argv[3], dtype=np.int64)[:N]
    tag = sys.argv[4] if len(sys.argv) > 4 else "file"
elif src == "sample":
    d = np.load(RES / "cont_outlier_sample20.npz")
    worst = np.asarray(d["target_ids"], np.int64)[:N]
    tag = "cont20"
else:
    b = torch.load(RES / "desi_outliers_cue_snr3.pt", weights_only=False)
    tid = np.asarray(b["outlier_target_ids"]).astype(np.int64)
    dtid = np.asarray(b["desi_target_ids"]).astype(np.int64)
    ds = np.asarray(b["scores_desi"])
    so = dict(zip(dtid.tolist(), ds.tolist()))
    worst = tid[np.argsort([so[int(t)] for t in tid])][:N]
    tag = "iso"

# RA/DEC + EW from the VAC.
meta = Table.read(VAC, hdu="METADATA")[["TARGETID", "RA", "DEC"]]
mt = np.asarray(meta["TARGETID"], np.int64)
mo = np.argsort(mt)
mts = mt[mo]
mra = np.asarray(meta["RA"])
mdec = np.asarray(meta["DEC"])
vac = Table.read(VAC, hdu="FASTSPEC")[["TARGETID", "HALPHA_EW"]]
vt = np.asarray(vac["TARGETID"], np.int64)
vo = np.argsort(vt)
vts = vt[vo]
vew = np.asarray(vac["HALPHA_EW"])


def radec(t):
    j = np.searchsorted(mts, t)
    if j < len(mts) and mts[j] == t:
        p = mo[j]
        return float(mra[p]), float(mdec[p])
    return np.nan, np.nan


def ew(t):
    j = np.searchsorted(vts, t)
    return float(vew[vo[j]]) if (j < len(vts) and vts[j] == t) else np.nan


ncol = 4
nrow = int(np.ceil(N / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.4 * nrow))
for ax, t in zip(np.ravel(axes), worst):
    ra, dec = radec(int(t))
    if not np.isfinite(ra):  # TID absent from VAC -> likely a contaminant
        ax.text(0.5, 0.5, "not in VAC\n(no RA/DEC)", ha="center", va="center")
        ax.set_title(f"{t}\nnot in FastSpecFit", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        continue
    try:
        content = desi.DESI.get_image(
            ra=ra, dec=dec, size=CUTOUT_SIZE, pixscale=PIXSCALE, bands="griz"
        )
        img = Image.open(io.BytesIO(content))
        ax.imshow(img)
        # overlay 1.5" DESI fiber aperture at image center (fiber footprint on sky)
        cx, cy = (img.width - 1) / 2.0, (img.height - 1) / 2.0
        ax.add_patch(
            Circle((cx, cy), APER_R_PX, fill=False, ec="lime", lw=1.0, alpha=0.9)
        )
    except Exception as e:
        ax.text(0.5, 0.5, f"fetch failed\n{type(e).__name__}", ha="center", va="center")
    ax.set_title(f"{t}\nEW={ew(int(t)):.0f}  ({ra:.4f},{dec:.4f})", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
for ax in np.ravel(axes)[N:]:
    ax.axis("off")
fig.tight_layout()
out = RES / f"worst{N}_{tag}_cutouts.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
