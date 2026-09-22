"""
Overplot the MAP best-fit model on the data for the worst-8 MAP fits.
Reads results/map_chi2_cue_outliers_worst8_lsf_full.pkl -- which already stores, per
galaxy, the MAP `model`, the `flux`/`unc`/`mask`, theta, and chi2_red on a shared
observed-frame `wave` grid. No refit needed (numpy + matplotlib only).

Usage (from hubersed/):  python tmp/plot_map_models.py [pkl]
Saves results/<pklstem>_modelvsdata.png
"""

import sys, pickle
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hubersed.paths import PATHS

RES = PATHS["RESULTS"]

fp = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else RES / "map_chi2_cue_outliers_worst8_lsf_full.pkl"
)
d = pickle.load(open(fp, "rb"))
wave = d["wave"]
res = d["results"]

LINES = {
    "[OII]": 3727.0,
    "Hb": 4861.0,
    "[OIII]": 5007.0,
    "Ha": 6563.0,
    "[NII]": 6584.0,
    "[SII]": 6725.0,
}

n = len(res)
ncol = 4
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.8 * nrow), squeeze=False)
axf = np.ravel(axes)
for ax, r in zip(axf, res):
    z = r["z"]
    m = r["mask"].astype(bool)
    wl = wave / (1.0 + z)
    flux = r["flux"].copy()
    model = r["model"].copy()
    flux[~m] = np.nan
    model[~m] = np.nan
    ax.plot(wl, flux, lw=0.5, color="0.4", label="data")
    ax.plot(wl, model, lw=0.7, color="tab:red", label="MAP model")
    hi = np.nanpercentile(flux, 99.7)
    ax.set_ylim(-0.1 * hi, hi * 1.15 if np.isfinite(hi) and hi > 0 else 1.0)
    ax.set_xlim(np.nanmin(wl), np.nanmax(wl))
    for name, w0 in LINES.items():
        if wl.min() < w0 < wl.max():
            ax.axvline(w0, color="tab:blue", ls=":", lw=0.4, alpha=0.5)
    ax.set_title(f"{r['id']}\nz={z:.4f}  chi2_red={r['chi2_red']:.1f}", fontsize=7)
    ax.tick_params(labelsize=6)
axf[0].legend(fontsize=6, loc="upper right")
for ax in axf[n:]:
    ax.axis("off")
fig.supxlabel("rest-frame wavelength [A]", fontsize=9)
fig.supylabel("flux", fontsize=9)
fig.tight_layout()
out = RES / f"{fp.stem}_modelvsdata.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
