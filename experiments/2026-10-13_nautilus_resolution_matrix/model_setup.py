"""Data, model and observation for the resolution-matrix fits of 39627770174637084.

Both runs of this experiment fit the same data with the same priors. Only the line spread differs.

- rmatrix, the target's own DESI resolution matrix through ResolutionMatrixModel and
  ResolutionMatrixSpectrum (hubersed.sps.resolution_model).
- median, the control, prospect's own path with the median Gaussian LSF of map_fits.LSF, as every
  fit so far.

The data are the row-weighted recoadd with the corrected calibration ivar,
resolution_matrix_rows_ivar.npz of experiments/2026-10-09_nautilus_c3k_er_broadline/
ivar_recompute.py (iron flux calibration rerun with desispec PRs 2496, 2827 and 2452, ivar with
PR 2422, coadd rows weighted as PR 2377). The fitted pixels are those of the 10-09 fit (spender's
mask and sky mask, rest 3320 to 9000 A) where the recoadded ivar is positive. Optionally the
emission lines that are missing from the Cue line list are masked at rest +-4 A.

The parameters are those of the 10-09 ER fit (make_params there): the Cue broadline model with a
shared forbidden broad fraction and sigma_split 45 km/s.
"""

import astropy.units as u
import numpy as np
from prospect.models.priors import LogUniform

from hubersed.conversion import DESI_FLAM, ivar_to_maggies, to_maggies
from hubersed.fitting.chi2 import WAVE_OBS
from hubersed.fitting.map_fits import LSF, get_sps
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.sps import broadline
from hubersed.sps.config import build_continuum_model, build_full_cue_model
from hubersed.sps.parameter_file import build_obs
from hubersed.sps.resolution_matrix import load_resolution_npz
from hubersed.sps.resolution_model import ResolutionMatrixModel, build_resolution_obs
from hubersed.sps.utils import universe_age_gyr

TID = 39627770174637084
FB, SPLIT = "shared", 45.0
REST_MIN, REST_MAX = 3320.0, 9000.0
ER_RESULTS = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline"
COADD = ER_RESULTS / str(TID) / "resolution_matrix_rows_ivar.npz"
DR1_COADD = ER_RESULTS / str(TID) / "resolution_matrix.npz"
LINE_HALF = 4.0  # rest-frame half width of the missing-line masks, in Angstrom
# Lines missing from the Cue line list, rest vacuum Angstrom. NIST ASD air wavelengths (H I the
# Aki-weighted mean over fine structure) converted to vacuum with Morton (2000).
MISSING_LINES = {
    "H12": 3751.21,
    "H11": 3771.70,
    "He I 5016": 5017.08,
    "He I 7281": 7283.36,
    "Pa16": 8504.83,
    "Pa15": 8547.73,
    "Pa14": 8600.75,
    "Pa13": 8667.40,
    "Pa12": 8752.86,
    "Pa11": 8865.32,
}


def load_data(masked_lines=()):
    """Return the redshift, flux and uncertainty in maggies, and the fitted pixels.

    Parameters
    ----------
    masked_lines : iterable of str
        Names in ``MISSING_LINES`` to mask at rest +-``LINE_HALF``.

    Returns
    -------
    z : float
        Redshift.
    flux, unc : np.ndarray
        Recoadded flux and its uncertainty in maggies on ``WAVE_OBS``. Pixels outside ``good``
        have infinite uncertainty.
    good : np.ndarray of bool
        Fitted pixels.
    """
    s = load_spectrum(TID)
    z = float(s.redshift.value)
    wave = WAVE_OBS * u.AA
    flux0 = to_maggies(wave, s.flux).value
    iv0 = ivar_to_maggies(wave, s.uncertainty.quantity).value
    rest = WAVE_OBS / (1 + z)
    good = (
        np.isfinite(flux0) & np.isfinite(iv0) & (iv0 > 0) & (rest >= REST_MIN) & (rest < REST_MAX)
    )
    # the spectrum the 10-09 fit used is the DR1 coadd exactly; check the conversion on it
    dr1 = np.load(DR1_COADD)
    check = to_maggies(wave, dr1["flux"] * DESI_FLAM).value
    assert np.allclose(check[good], flux0[good], rtol=1e-5, atol=0), "DR1 coadd does not match"
    new = load_resolution_npz(COADD)
    good &= new["ivar"] > 0
    flux = to_maggies(wave, new["flux"] * DESI_FLAM).value
    ivar = ivar_to_maggies(wave, np.where(good, new["ivar"], 1.0) * DESI_FLAM**-2).value
    for name in masked_lines:
        good &= np.abs(rest - MISSING_LINES[name]) >= LINE_HALF
    unc = np.where(good, 1 / np.sqrt(ivar), np.inf)
    return z, np.where(good, flux, flux0), unc, good


def make_params(z):
    """Return the parameter specification of the 10-09 ER fit."""
    cont_model, cont_tmpl = build_continuum_model(z)
    _, tmpl = build_full_cue_model(cont_tmpl, cont_model.theta, cont_model)
    tmpl["tau_in"]["isfree"] = False
    tmpl["tau_in"]["init"] = universe_age_gyr(z)
    tmpl["sigma_dyn"]["prior"] = LogUniform(mini=0.01, maxi=1.0)
    tmpl["tau_dyn"]["prior"] = LogUniform(mini=0.001, maxi=0.02)
    tmpl["sigma_reg"]["prior"] = LogUniform(mini=0.1, maxi=3.0)
    tmpl["tau_eq"]["prior"] = LogUniform(mini=0.01, maxi=1.0)
    tmpl["sigma_dyn"]["init"] = 0.05
    tmpl["tau_dyn"]["init"] = 0.005
    tmpl["tau_eq"]["init"] = 0.1
    params = broadline.add_broad_params(
        tmpl, separate_forbidden_width=True, forbidden_broad=FB, sigma_split=SPLIT
    )
    params["eline_sigma"]["init"] = 0.5 * (10.0 + SPLIT)
    return params


def make_model(params, lsf):
    """Return the model for ``lsf`` ("rmatrix" or "median")."""
    if lsf == "rmatrix":
        return ResolutionMatrixModel(params)
    if lsf == "median":
        return broadline.TwoCompLineModel(params)
    raise ValueError(f"lsf must be rmatrix or median, not {lsf!r}")


def make_obs(flux, unc, good, lsf):
    """Return the observation list for ``lsf``."""
    if lsf == "rmatrix":
        new = load_resolution_npz(COADD)
        return build_resolution_obs(
            flux, unc, good, new["res"], resolution_wave=new["wave"], wavelength=WAVE_OBS
        )
    return build_obs(spec=flux, unc=unc, mask=good, resolution=LSF, wavelength=WAVE_OBS)


def get_cue():
    """Return the Cue source with the library resolution kept, checked to be C3K_ER."""
    cue = get_sps(zero_library_resolution=False)["cue"]
    assert cue.ssp.libraries[1] in ("c3k_er", b"c3k_er"), cue.ssp.libraries
    return cue
