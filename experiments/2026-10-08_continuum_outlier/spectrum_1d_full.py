"""Run 1d of 39627757533007793 (MILES, MILES window, no degrading) over the whole DESI spectrum.

Data are the DR1 coadd from load_spectrum on every unmasked pixel, with a 9 pixel median filter.
The max-L model and chi are drawn on the pixels run 1d fitted.

COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-08_continuum_outlier/spectrum_1d_full.py``.

OUTPUT: results/2026-10-08_continuum_outlier/39627757533007793/spectrum_1d_full.png.
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

d = np.load(RES / f"{TID}_miles_miles_neboff_nodeg_pred.npz")
s = load_spectrum(TID)
wave = np.asarray(s.spectral_axis.value, float)
assert np.allclose(wave, d["wave"]), "load_spectrum grid differs from the run 1d grid"
z = float(d["z"])
rest = wave / (1 + z)
ok = ~np.asarray(s.mask, bool)
data = np.where(ok, np.asarray(s.flux.value, float), np.nan)
med = np.full(data.size, np.nan)
med[ok] = medfilt(data[ok], N_MEDIAN)
g = d["good"]
model = np.where(g, to_flambda(wave * u.AA, d["sp_best"] * u.mgy).value, np.nan)
with np.errstate(divide="ignore", invalid="ignore"):
    chi = np.where(g, (d["flux"] - d["sp_best"]) / d["unc"], np.nan)

# %% figure
use_apj_style()
fig, ax = plt.subplots(
    2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}
)
ax[0].plot(rest, data, color="0.6", lw=0.6, label="DESI spectrum, DR1 coadd")
ax[0].plot(rest, med, color="k", lw=0.5, label=f"median filter ({N_MEDIAN} pix)")
ax[0].plot(rest, model, color="#b2182b", lw=0.9, label="MILES")
# as spectrum_1d.png: the noisy blue end would otherwise set the flux axis
ax[0].set_ylim(0, 1.3 * np.nanpercentile(model, 99.5))
ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
ax[0].legend(frameon=False, fontsize=12, loc="lower right")
ax[1].plot(rest, chi, color="#b2182b", lw=0.5)
for h in (-2, 2):
    ax[1].axhline(h, color="0.3", ls="--", lw=1)
ax[1].axhline(0, color="0.3", lw=0.6)
ax[1].set_ylim(-5, 5)
ax[1].set_ylabel(r"$\chi$", fontsize=14)
ax[1].set_xlim(rest[ok].min(), rest[ok].max())
ax[1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
path = RES / str(TID) / "spectrum_1d_full.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {path}")
