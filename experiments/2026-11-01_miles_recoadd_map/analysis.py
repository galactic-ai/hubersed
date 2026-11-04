"""MILES against C3K ER + afe + R matrix continuum MAPs on the SV3 recoadd of 39627757533007793 (#17).

QUESTION: on the same recoadded SV3 data and errors, inside the MILES window, does the C3K ER + afe +
R matrix MAP fit better than the MILES MAP (no degrading, median LSF)?
HYPOTHESIS: none registered.
INPUTS: <TID>_map.npz of results/2026-11-01_miles_recoadd_map (fit.py collect) and of
2026-10-29_continuum_recoadd/sv3, copied from stampede3. Feature bands from
$SPS_HOME/data/allindices.dat.
SEED: none.
COMMAND: from the repository root on the Mac,
``SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-11-01_miles_recoadd_map/analysis.py``.
RESULT: pending.
FIGURES: results/2026-11-01_miles_recoadd_map/miles_vs_er.png and miles_vs_er.txt.
"""

# %%
import os
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

from hubersed.conversion import to_flambda
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39627757533007793
RES = PATHS["RESULTS"]
OUT = RES / "2026-11-01_miles_recoadd_map"
RUNS = {
    "MILES": (OUT / f"{TID}_map.npz", "#b2182b"),
    "C3K ER + afe + R matrix": (RES / "2026-10-29_continuum_recoadd" / "sv3" / f"{TID}_map.npz",
                                "#1b9e77"),
}  # fmt: skip
INDEX_NAMES = ("Hbeta", "Mgb", "Fe5270", "Fe5335", "NaD")
BINS = [(lo, lo + 500) for lo in range(3500, 7500, 500)]
N_MEDIAN = 9


def flam(d, x, good):
    """Return maggies on the fit's wavelength grid in 1e-17 erg/s/cm^2/A, NaN off ``good``."""
    f = to_flambda(d["wave"] * u.AA, np.asarray(x, float) * u.mgy).value
    return np.where(good, f, np.nan)


def chi(d, good):
    """Return (data - model) / sigma on ``good``, NaN elsewhere."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(good, (d["flux"] - d["sp_best"]) / d["unc"], np.nan)


def feature_bands():
    """Return the rest-frame feature band of each index in allindices.dat."""
    table = np.genfromtxt(
        Path(os.environ["SPS_HOME"]) / "data" / "allindices.dat",
        dtype=None,
        encoding=None,
        comments="#",
    )
    return {str(r[-1]): (float(r[0]), float(r[1])) for r in table if str(r[-1]) in INDEX_NAMES}


# %% chi2 on the pixels both fits used, in total, by rest-frame bin and per feature band
f = {k: np.load(p) for k, (p, _) in RUNS.items()}
m, e = f.values()
common = m["good"] & e["good"]
on = common
assert np.allclose(m["flux"][on], e["flux"][on]) and np.allclose(m["unc"][on], e["unc"][on])
rest = m["wave"] / (1 + float(m["z"]))
bands = feature_bands()
c = {k: chi(d, common) for k, d in f.items()}
names = list(f)
lines = [f"{int(common.sum())} pixels fitted by both ({int(m['good'].sum())} MILES, "
         f"{int(e['good'].sum())} C3K ER); same recoadd flux and uncertainty there", ""]  # fmt: skip
lines.append(f"{'':>14} " + " ".join(f"{k:>24}" for k in names) + f" {'difference':>11} {'px':>5}")
rows = [("total", common)]
rows += [(f"{lo}-{hi}", common & (rest >= lo) & (rest < hi)) for lo, hi in BINS]
rows += [(name, common & (rest >= f0) & (rest <= f1)) for name, (f0, f1) in bands.items()]
for label, s in rows:
    if not s.any():
        continue
    v = [np.sum(c[k][s] ** 2) for k in names]
    lines.append(f"{label:>14} " + " ".join(f"{x:24.1f}" for x in v) + f" {v[0] - v[1]:+11.1f} "
                 f"{int(s.sum()):5d}")  # fmt: skip
lines.append("")
lines.append("difference is chi2(MILES) - chi2(C3K ER); chi2/N for each bin is chi2 / px")
for k, d in f.items():
    p = dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))
    lines.append(f"{k}: logzsol {p['logzsol']:+.3f}, dust2 {p['dust2']:.3f}, "
                 f"sigma_smooth {p['sigma_smooth']:.1f}, afe {p.get('afe', 0.0):+.3f}")  # fmt: skip
text = "\n".join(lines)
(OUT / "miles_vs_er.txt").write_text(text + "\n")
print(text)

# %% figure: recoadd data with both MAPs on the common pixels, chi of both overlaid
use_apj_style()
data = flam(m, m["flux"], common)
med = np.full(data.size, np.nan)
med[common] = medfilt(data[common], N_MEDIAN)
fig, ax = plt.subplots(
    2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}
)
ax[0].plot(rest, data, color="0.6", lw=0.6, label="DESI spectrum, SV3 recoadd")
ax[0].plot(rest, med, color="k", lw=0.5, label=f"median filter ({N_MEDIAN} pix)")
for k, d in f.items():
    colour = RUNS[k][1]
    ax[0].plot(rest, flam(d, d["sp_best"], common), color=colour, lw=0.9, label=k)
    ax[1].plot(rest, c[k], color=colour, lw=0.5, alpha=0.7)
ax[0].set_ylim(bottom=0)
ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
ax[0].legend(frameon=False, fontsize=12, loc="lower right")
for h in (-2, 2):
    ax[1].axhline(h, color="0.3", ls="--", lw=1)
ax[1].axhline(0, color="0.3", lw=0.6)
ax[1].set_ylim(-5, 5)
ax[1].set_ylabel(r"$\chi$", fontsize=14)
ax[1].set_xlim(rest[common].min(), rest[common].max())
ax[1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
path = OUT / "miles_vs_er.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {path}")
