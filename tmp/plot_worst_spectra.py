"""
Plot the spectra of the worst-N latent-OOD outliers (mirror of tmp/fetch_cutouts.py,
but stored DESI spectra instead of Legacy Survey image cutouts).

Spectra are read from the same spec h5 that fed the encoder, matched by TARGETID
(never by position). They are the normalized, observed-frame DESI spectra on the
spender grid _wave_obs = linspace(3600, 9824, 7781); we display them in the REST
frame (wave_obs / (1+z)) so emission lines line up across panels.

Usage (from hubersed/):  python tmp/plot_worst_spectra.py [N] [iso|flow]
  iso  (default): IsoForest worst by score  -> desi_outliers_cue_snr3.pt   (6D, Cue)
  flow          : NSF flow worst by log p   -> desi_outliers_flow_nsf_cont10latent_snr3.pt
Saves results/worst{N}_{tag}_spectra.png
"""

import sys
import numpy as np, torch, h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from hubersed.paths import PATHS

DATA, RES = PATHS["DATA"], PATHS["RESULTS"]
VAC = DATA / "fastspec-iron-sv3-bright.fits"

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
src = sys.argv[2] if len(sys.argv) > 2 else "iso"

# spender observed-frame grid (DESI.__wave_obs); hardcoded to avoid importing torch-heavy spender
WAVE_OBS = np.linspace(3600.0, 9824.0, 7781)

# rest-frame emission (and a couple of stellar) lines to overplot
LINES = {
    "[OII]": 3727.0,
    "CaK": 3934.0,
    "CaH": 3969.0,
    "Hd": 4102.0,
    "Hg": 4340.0,
    "Hb": 4861.0,
    "[OIII]": 5007.0,
    "Mgb": 5175.0,
    "NaD": 5892.0,
    "Ha": 6563.0,
    "[NII]": 6584.0,
    "[SII]": 6725.0,
}

# ---- worst-N outliers (identical selection logic to fetch_cutouts.py) ----
if src == "flow":
    b = torch.load(RES / "desi_outliers_flow_nsf_15latent_snr3.pt", weights_only=False)
    dtid = np.asarray(b["desi_target_ids"]).astype(np.int64)
    lp = np.asarray(b["log_p_desi"])
    worst = dtid[np.argsort(lp)][:N]  # lowest log p = most OOD
    tag = "flowNSF15D"
    spec_tag = b.get("tag", "15latent")
else:
    b = torch.load(RES / "desi_outliers_cue_snr3.pt", weights_only=False)
    tid = np.asarray(b["outlier_target_ids"]).astype(np.int64)
    dtid = np.asarray(b["desi_target_ids"]).astype(np.int64)
    ds = np.asarray(b["scores_desi"])
    so = dict(zip(dtid.tolist(), ds.tolist()))
    worst = tid[np.argsort([so[int(t)] for t in tid])][
        :N
    ]  # IsoForest: lowest score = worst
    tag = "iso"
    spec_tag = "6latent"

# ---- stored spectra, matched by TARGETID ----
spec_h5 = DATA / f"spender_spec_{spec_tag}_snr3.h5"
with h5py.File(spec_h5, "r") as f:
    h5_tid = np.asarray(f["target_ids"], np.int64)
    pos = {int(t): i for i, t in enumerate(h5_tid)}
    rows = [
        pos[int(t)] for t in worst if int(t) in pos
    ]  # KeyError-safe; keep order of `worst`
    found = [int(t) for t in worst if int(t) in pos]
    specs = f["specs"][:][rows].astype(np.float32)  # (n, L) normalized, observed frame
    zs = np.asarray(f["zs"], np.float32)[rows]
missing = [int(t) for t in worst if int(t) not in pos]
if missing:
    print(f"WARNING: {len(missing)} worst TIDs not in {spec_h5.name}: {missing}")

# ---- Halpha EW from the VAC (same helper as fetch_cutouts) ----
vac = Table.read(VAC, hdu="FASTSPEC")[["TARGETID", "HALPHA_EW"]]
vt = np.asarray(vac["TARGETID"], np.int64)
vo = np.argsort(vt)
vts = vt[vo]
vew = np.asarray(vac["HALPHA_EW"])


def ew(t):
    j = np.searchsorted(vts, t)
    return float(vew[vo[j]]) if (j < len(vts) and vts[j] == t) else np.nan


# ---- montage ----
n = len(found)
ncol = 4
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow), squeeze=False)
axf = np.ravel(axes)
for ax, t, spec, z in zip(axf, found, specs, zs):
    spec = spec.copy()
    spec[spec == 0] = np.nan  # masked pixels -> gaps, not zeros
    wave_rest = WAVE_OBS / (1.0 + z)
    ax.plot(wave_rest, spec, lw=0.5, color="k")
    hi = np.nanpercentile(spec, 99.7)
    ax.set_ylim(-0.3, (hi if np.isfinite(hi) and hi > 0 else 2.0) * 1.1)
    ax.set_xlim(wave_rest.min(), wave_rest.max())
    for name, wl in LINES.items():
        if wave_rest.min() < wl < wave_rest.max():
            ax.axvline(wl, color="tab:red", ls=":", lw=0.5, alpha=0.6)
            ax.text(
                wl,
                ax.get_ylim()[1],
                name,
                rotation=90,
                va="top",
                ha="right",
                fontsize=5,
                color="tab:red",
                alpha=0.8,
            )
    ax.set_title(f"{t}\nz={z:.4f}  EW={ew(int(t)):.0f}", fontsize=7)
    ax.tick_params(labelsize=6)
for ax in axf[n:]:
    ax.axis("off")
fig.supxlabel("rest-frame wavelength [A]", fontsize=9)
fig.supylabel("flux (normalized)", fontsize=9)
fig.tight_layout()
out = RES / f"worst{N}_{tag}_spectra.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
