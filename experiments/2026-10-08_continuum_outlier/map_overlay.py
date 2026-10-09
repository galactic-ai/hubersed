"""Plot run 1c of 39627757533007793 with the v2 MAP fit of the same spectrum.

Both use the same DESI flux and uncertainty, and chi is shown on the pixels 1c fitted. The MAP
fit is results/map_fits_outliers310_v2/<tid>.pkl, a Powell fit with Cue nebular emission and the
GP stochastic SFH prior. Run from the repository root with
``uv run python experiments/2026-10-08_continuum_outlier/map_overlay.py``.
"""

import pickle

import astropy.units as u
import numpy as np

from hubersed.conversion import to_flambda
from hubersed.paths import PATHS
from hubersed.plotting.spectra import plot_residual, residual_chi, spectrum_figure

TID = 39627757533007793
OUT = PATHS["RESULTS"] / "2026-10-08_continuum_outlier"
d = np.load(OUT / f"{TID}_miles_miles_nebon_nodeg_pred.npz")
with open(PATHS["RESULTS"] / "map_fits_outliers310_v2" / f"{TID}.pkl", "rb") as f:
    m = pickle.load(f)
assert np.allclose(m["wave"], d["wave"]) and np.allclose(m["flux"], d["flux"])
z, good = float(d["z"]), d["good"]
rest = d["wave"] / (1 + z)


def flam(maggies):
    """Convert maggies to DESI f_lambda units, NaN outside the pixels 1c fitted."""
    f = to_flambda(d["wave"] * u.AA, np.asarray(maggies, float) * u.mgy).value
    return np.where(good, f, np.nan)


chi_1c = residual_chi(d["flux"], d["sp_best"], d["unc"], good)
chi_map = residual_chi(d["flux"], m["model"], d["unc"], good)
x2 = [np.nansum(c**2) / good.sum() for c in (chi_1c, chi_map)]
fig, ax = spectrum_figure(
    d["wave"],
    z=z,
    extra_panels=1,
    figsize=(11, 7.5),
    data=flam(d["flux"]),
    unc=flam(d["unc"]),
    band_kw={},
    data_kw={"label": "DESI spectrum, as fitted in 1c"},
    models=[
        {"flux": flam(d["sp_best"]), "color": "#b2182b", "lw": 0.8, "label": "1c max-L"},
        {"flux": flam(m["model"]), "color": "#2166ac", "lw": 0.8, "label": "v2 MAP"},
    ],
)
for a, c, col, lab, x in zip(
    ax[1:], (chi_1c, chi_map), ("#b2182b", "#2166ac"), ("1c", "MAP"), x2, strict=True
):
    plot_residual(a, d["wave"], z=z, chi=c, lw=0.5, color=col)
    a.set_ylim(-6, 6)
    a.text(0.01, 0.9, rf"{lab}, $\chi^2_\nu$ {x:.2f} on 1c pixels", transform=a.transAxes, va="top")
ax[1].set_xlabel("")
for a in ax:
    a.set_xlim(rest[good].min(), rest[good].max())
ax[0].set_ylim(0, 1.3 * np.nanpercentile(flam(d["sp_best"]), 99.5))
fig.savefig(OUT / "spectrum_1c_map.png", dpi=150, bbox_inches="tight")
print("chi2_nu on 1c pixels: 1c", round(x2[0], 3), "MAP", round(x2[1], 3))
