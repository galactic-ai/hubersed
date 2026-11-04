"""The C3K ER + afe MAP of 39627757533007793 with the median DESI LSF and with its R matrix.

Both models are the 2026-10-28 MAP best point (SV3 DR1 coadd, native data): sp_best is the median
LSF prediction of predict_median_lsf.py there, sp_best_rmatrix the MAP's own R matrix model. Data
are the DR1 coadd from load_spectrum on every unmasked pixel, with a 9 pixel median filter. Models
and chi are drawn on the fitted pixels.

COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-08_continuum_outlier/spectrum_er_lsf_rmatrix_full.py``.

OUTPUT: results/2026-10-08_continuum_outlier/39627757533007793/spectrum_er_lsf_rmatrix_full.png.
"""

# %%
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

from hubersed.conversion import to_flambda
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39627757533007793
RES = PATHS["RESULTS"] / "2026-10-08_continuum_outlier"
N_MEDIAN = 9

PRED = PATHS["RESULTS"] / "2026-10-28_continuum_c3k_er_rmatrix_map" / f"{TID}_median_lsf_pred.npz"
RUNS = {
    "C3K ER + afe (median LSF)": ("sp_best", "#1b7837"),
    "C3K ER + afe (R matrix)": ("sp_best_rmatrix", "#762a83"),
}

s = load_spectrum(TID)
wave = np.asarray(s.spectral_axis.value, float)
fits = {}
d = np.load(PRED)
assert np.allclose(wave, d["wave"]), "load_spectrum grid differs from the prediction"
for label, (key, colour) in RUNS.items():
    g = d["good"]
    model = np.where(g, to_flambda(wave * u.AA, d[key] * u.mgy).value, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.where(g, (d["flux"] - d[key]) / d["unc"], np.nan)
    fits[label] = {"model": model, "chi": chi, "colour": colour, "z": float(d["z"])}
z = fits["C3K ER + afe (median LSF)"]["z"]
assert all(abs(f["z"] - z) < 1e-6 for f in fits.values()), "runs differ in redshift"
rest = wave / (1 + z)
ok = ~np.asarray(s.mask, bool)
data = np.where(ok, np.asarray(s.flux.value, float), np.nan)
med = np.full(data.size, np.nan)
med[ok] = medfilt(data[ok], N_MEDIAN)

# %% figure
use_apj_style()
fig, ax = plt.subplots(
    2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}
)
ax[0].plot(rest, data, color="0.6", lw=0.6, label="DESI spectrum, DR1 coadd")
ax[0].plot(rest, med, color="k", lw=0.5, label=f"median filter ({N_MEDIAN} pix)")
for label, f in fits.items():
    ax[0].plot(rest, f["model"], color=f["colour"], lw=0.9, label=label)
    ax[1].plot(rest, f["chi"], color=f["colour"], lw=0.5, alpha=0.7)
# as spectrum_1d.png: the noisy blue end would otherwise set the flux axis
ax[0].set_ylim(0, 1.3 * np.nanpercentile(fits["C3K ER + afe (median LSF)"]["model"], 99.5))
ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
ax[0].legend(frameon=False, fontsize=12, loc="lower right")
for h in (-2, 2):
    ax[1].axhline(h, color="0.3", ls="--", lw=1)
ax[1].axhline(0, color="0.3", lw=0.6)
ax[1].set_ylim(-5, 5)
ax[1].set_ylabel(r"$\chi$", fontsize=14)
ax[1].set_xlim(rest[ok].min(), rest[ok].max())
ax[1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
path = RES / str(TID) / "spectrum_er_lsf_rmatrix_full.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {path}")
