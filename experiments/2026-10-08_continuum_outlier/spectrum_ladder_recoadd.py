"""SV3 recoadd of 39627757533007793 with its C3K ER + afe + R matrix MAP, and the chi of all six models.

Top: the recoadd flux of 2026-10-29_continuum_recoadd/sv3 on the pixels that fit used (the file
holds no flux elsewhere), a 9 pixel median filter, and its MAP. Below, one chi panel per model,
each on the data and uncertainty that model was fitted to or predicted for: run 1d (MILES), 2b-off
(C3K HR), 3n-free (C3K HR + afe, nebular on), the C3K ER + afe MAP with the median LSF
(predict_median_lsf.py) and with its R matrix (both on the SV3 DR1 coadd), and the recoadd MAP.

COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-08_continuum_outlier/spectrum_ladder_recoadd.py``.

OUTPUT: results/2026-10-08_continuum_outlier/39627757533007793/spectrum_ladder_recoadd.png.
"""

# %%
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

from hubersed.conversion import to_flambda
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39627757533007793
RES = PATHS["RESULTS"] / "2026-10-08_continuum_outlier"
ER = PATHS["RESULTS"] / "2026-10-28_continuum_c3k_er_rmatrix_map"
RECOADD = PATHS["RESULTS"] / "2026-10-29_continuum_recoadd" / "sv3" / f"{TID}_map.npz"
N_MEDIAN = 9
# label, file, model key, colour
STEPS = (
    ("MILES", RES / f"{TID}_miles_miles_neboff_nodeg_pred.npz", "sp_best", "#b2182b"),
    ("C3K HR (degraded)", RES / f"{TID}_c3k_hr_full_neboff_pred.npz", "sp_best", "#2166ac"),
    ("C3K HR + afe (degraded, nebular on)", RES / f"{TID}_c3k_hr_full_nebon_afefree_pred.npz",
     "sp_best", "#e66101"),
    ("C3K ER + afe", ER / f"{TID}_median_lsf_pred.npz", "sp_best", "#1b7837"),
    ("C3K ER + afe + R matrix, DR1 coadd", ER / f"{TID}_median_lsf_pred.npz", "sp_best_rmatrix",
     "#762a83"),
    ("C3K ER + afe + R matrix, recoadd", RECOADD, "sp_best", "#1b9e77"),
)  # fmt: skip


def flam(d, x):
    """Return maggies on the file's wavelength grid in 1e-17 erg/s/cm^2/A, NaN off its fit."""
    f = to_flambda(d["wave"] * u.AA, np.asarray(x, float) * u.mgy).value
    return np.where(d["good"], f, np.nan)


rc = np.load(RECOADD)
z = float(rc["z"])
rest = rc["wave"] / (1 + z)
data = flam(rc, rc["flux"])
med = np.full(data.size, np.nan)
med[rc["good"]] = medfilt(data[rc["good"]], N_MEDIAN)
chis, lo, hi = [], np.inf, -np.inf
for _, path, key, _ in STEPS:
    d = np.load(path)
    assert np.allclose(d["wave"], rc["wave"]), f"{path.name} is on another grid"
    assert abs(float(d["z"]) - z) < 1e-6, f"{path.name} has another redshift"
    with np.errstate(divide="ignore", invalid="ignore"):
        chis.append(np.where(d["good"], (d["flux"] - d[key]) / d["unc"], np.nan))
    r = d["wave"][d["good"]] / (1 + z)
    lo, hi = min(lo, r.min()), max(hi, r.max())

# %% figure
use_apj_style()
fig, ax = plt.subplots(
    1 + len(STEPS),
    1,
    figsize=(14, 4 + 1.6 * len(STEPS)),
    sharex=True,
    gridspec_kw={"height_ratios": [3] + [1] * len(STEPS), "hspace": 0.06},
)
ax[0].plot(rest, data, color="0.6", lw=0.6, label="DESI spectrum, SV3 recoadd")
ax[0].plot(rest, med, color="k", lw=0.5, label=f"median filter ({N_MEDIAN} pix)")
ax[0].plot(rest, flam(rc, rc["sp_best"]), color=STEPS[-1][3], lw=0.9, label=STEPS[-1][0])
ax[0].set_ylim(bottom=0)
ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
ax[0].legend(frameon=False, fontsize=12, loc="lower right")
for a, (label, _, _, colour), chi in zip(ax[1:], STEPS, chis, strict=True):
    a.plot(rest, chi, color=colour, lw=0.5)
    for h in (-2, 2):
        a.axhline(h, color="0.3", ls="--", lw=1)
    a.axhline(0, color="0.3", lw=0.6)
    a.set_ylim(-5, 5)
    a.set_ylabel(r"$\chi$", fontsize=14)
    a.text(0.01, 0.93, label, transform=a.transAxes, ha="left", va="top", fontsize=11,
           bbox=dict(facecolor="white", edgecolor=colour, boxstyle="round,pad=0.25"))  # fmt: skip
ax[-1].set_xlim(lo, hi)
ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
path = RES / str(TID) / "spectrum_ladder_recoadd.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {path}")
