"""Spectrum and residuals of the continuum outlier 39627757533007793 by model (issue #17).

QUESTION: how do the residuals of MILES + Cue, C3K HR + Cue + free afe, and a C3K ER + free afe +
resolution matrix MAP compare across the DESI spectrum?
HYPOTHESIS: none registered.
INPUTS: the DESI DR1 coadd from hubersed.io.desi.load_spectrum. The 10-08 max-L spectra
miles_miles_nebon_nodeg_pred.npz and c3k_hr_full_nebon_afefree_pred.npz in
results/2026-10-08_continuum_outlier/ (md5 equal to the ls6 copies, 2026-10-28). The MAP
<TID>_map.npz of fit.py collect, icx job 3655467, copied from stampede3. Index bands from
$SPS_HOME/data/allindices.dat, as in the 10-08 analysis.
SEED: none.
COMMAND: from the repository root on the Mac,
``SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-28_continuum_c3k_er_rmatrix_map/analysis.py``.
RESULT: 2026-10-28, map.txt. All 7 Powell starts stopped on tolerance (1836-14003 calls). Best
start 6, -lnp -93625.70. Start 0 is 2.4 worse, starts 3-4 8-12 worse, starts 1, 2, 5 44-52 worse.
MAP against the HR max-L: afe 0.109 (0.128), logzsol -0.021 (0.037), dust2 0.042 (0.056),
dust_index -2.500 at its prior edge (-1.953), sigma_smooth 282.2 (281.1), logmass 11.13 (11.15).
logsfr_ratios 1-5 move by 0.02 to 0.83, 6-9 by up to 0.13.
Redward of rest 6400 A the ER and MILES chi, both on native data, scatter more than the HR chi on
degraded data, so chi amplitude does not compare across those panels.
Assumptions: SFH hyperparameters fixed at the HR values (fit with nebular on); no nebular
emission; DR1 coadd ivar; one MAP, not shown to be the global optimum.
FIGURES: results/2026-10-28_continuum_c3k_er_rmatrix_map/residuals_by_model.png.
"""

# %%
import os
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

from hubersed.conversion import to_flambda, to_maggies
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style
from hubersed.sps.parameter_file import WAVE_OBS

TID = 39627757533007793
RUNS10 = PATHS["RESULTS"] / "2026-10-08_continuum_outlier"
OUT = PATHS["RESULTS"] / "2026-10-28_continuum_c3k_er_rmatrix_map"
OUT.mkdir(parents=True, exist_ok=True)
ER_COLOR = "#b2182b"
N_MEDIAN = 9  # kernel size in pixels of the data's median filter
# Each row is a fit compared with the data it saw. The ER MAP file is written by fit.py.
ROWS = {
    "miles": ("MILES + Cue", "#4393c3", RUNS10 / f"{TID}_miles_miles_nebon_nodeg_pred.npz"),
    "c3k_hr": (
        "C3K HR + Cue + afe (degraded)",
        "#e08214",
        RUNS10 / f"{TID}_c3k_hr_full_nebon_afefree_pred.npz",
    ),
    "c3k_er": ("C3K ER + afe + R matrix (MAP)", ER_COLOR, OUT / f"{TID}_map.npz"),
}
INDEX_NAMES = "Dn4000 HdeltaA CN1 Ca4227 G4300 C4668 Hbeta Mgb Fe5270 Fe5335 NaD TiO1 TiO2".split()


def desi(maggies):
    """Return maggies on WAVE_OBS as DESI flux density, 1e-17 erg/s/cm^2/A."""
    return to_flambda(WAVE_OBS * u.AA, np.asarray(maggies, float) * u.mgy).value


def index_bands():
    """Return the index name and its shaded rest-frame spans (feature, or Dn4000's two bands)."""
    table = np.genfromtxt(
        Path(os.environ["SPS_HOME"]) / "data" / "allindices.dat",
        dtype=None,
        encoding=None,
        comments="#",
    )
    out = {}
    for r in table:
        name = str(r[-1])
        if name in INDEX_NAMES:
            f0, f1, b0, b1, r0, r1, kind = (r[0], r[1], r[2], r[3], r[4], r[5], int(r[6]))
            out[name] = [(b0, b1), (r0, r1)] if kind == 3 else [(f0, f1)]
    return out


# %% data and the saved fits
s = load_spectrum(TID)
z = float(s.redshift.value)
rest = WAVE_OBS / (1 + z)
flux = to_maggies(WAVE_OBS * u.AA, s.flux).value
ok = ~s.mask
fits = {k: np.load(v[2]) for k, v in ROWS.items() if v[2].exists()}
# the undegraded MILES run saw the same coadd, so its flux must equal load_spectrum's
m = fits["miles"]
assert np.isclose(float(m["z"]), z)
assert np.allclose(m["flux"][m["good"]], flux[m["good"]], rtol=1e-6, atol=0)

# %% figure: data with the ER MAP on top, then chi of each fit on its own data
use_apj_style()
fig, ax = plt.subplots(
    4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1], "hspace": 0.06}
)
ax[0].plot(rest, np.where(ok, desi(flux), np.nan), color="0.75", lw=0.5, label="DESI (DR1 coadd)")
# median filter over the unmasked pixels; medfilt pads the two ends with zeros
med = np.full(flux.size, np.nan)
med[ok] = medfilt(desi(flux)[ok], N_MEDIAN)
ax[0].plot(rest, med, color="k", lw=0.8, label=f"median filter, {N_MEDIAN} pixels")
if "c3k_er" in fits:
    e = fits["c3k_er"]
    ax[0].plot(
        rest, np.where(e["good"], desi(e["sp_best"]), np.nan), color=ER_COLOR, lw=0.8,
        label=ROWS["c3k_er"][0],
    )  # fmt: skip
ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
ax[0].set_ylim(bottom=0)
ax[0].legend(frameon=False, fontsize=13, loc="upper right")
for a, (key, (label, color, _)) in zip(ax[1:], ROWS.items(), strict=True):
    if key in fits:
        d = fits[key]
        with np.errstate(divide="ignore", invalid="ignore"):  # unfitted pixels can have unc 0
            c = np.where(d["good"], (d["flux"] - d["sp_best"]) / d["unc"], np.nan)
        a.plot(rest, c, color=color, lw=0.5)
    else:
        label += ", not run yet"
    for h in (-2, 2):
        a.axhline(h, color="0.3", ls="--", lw=1)
    a.axhline(0, color="0.3", lw=0.6)
    a.set_ylim(-5, 5)
    a.set_ylabel(r"$\chi$", fontsize=14)
    a.text(
        0.005, 0.88, label, transform=a.transAxes, fontsize=12, va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
    )  # fmt: skip
for name, spans in index_bands().items():
    for s0, s1 in spans:
        for a in ax:
            a.axvspan(s0, s1, color="0.5", alpha=0.12, lw=0)
    mid = 0.5 * (spans[0][0] + spans[-1][1])
    ax[0].text(
        mid, 1.01, name, transform=ax[0].get_xaxis_transform(), rotation=90, fontsize=9,
        ha="center", va="bottom",
    )  # fmt: skip
ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
fig.savefig(OUT / "residuals_by_model.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'residuals_by_model.png'}")
