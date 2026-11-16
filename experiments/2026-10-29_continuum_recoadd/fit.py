"""Continuum MAP of 39627757533007793 on its recoadded SV3 or main-survey spectrum (issue #17).

Changes from 2026-10-28_continuum_c3k_er_rmatrix_map and 2026-10-29_continuum_main_survey_map are
the recoadd in resolution_matrix_rows_ivar.npz, pixels with its ivar > 0, and their MAP as start 0.
COMMAND: on a compute node through map.slurm, for example
``RECOADD_SURVEY=main <venv>/bin/python experiments/2026-10-29_continuum_recoadd/fit.py fit --start 0``.
"""

import importlib.util
import os
from pathlib import Path

import astropy.units as u
import numpy as np

from hubersed.conversion import DESI_FLAM, ivar_to_maggies, to_maggies
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.sps.parameter_file import WAVE_OBS
from hubersed.sps.resolution_matrix import load_resolution_npz

# the 10-28 fit.py, loaded by path since this file has the same name
_spec = importlib.util.spec_from_file_location(
    "fit_1028",
    Path(__file__).resolve().parents[1] / "2026-10-28_continuum_c3k_er_rmatrix_map" / "fit.py",
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

SURVEY = os.environ.get("RECOADD_SURVEY", "sv3")
assert SURVEY in ("sv3", "main"), f"RECOADD_SURVEY {SURVEY} is not sv3 or main"
OUT = PATHS["RESULTS"] / "2026-10-29_continuum_recoadd" / SURVEY
DR1_MAP = {
    "sv3": PATHS["RESULTS"] / "2026-10-28_continuum_c3k_er_rmatrix_map",
    "main": PATHS["RESULTS"] / "2026-10-29_continuum_main_survey_map",
}[SURVEY]
Z = {"sv3": None, "main": 0.1706559587111022}[SURVEY]  # main: desi_dr1.zpix, as targets.py


def spender_good():
    """Return the redshift and the pixels spender keeps (positive ivar) for this spectrum."""
    if SURVEY == "sv3":
        s = load_spectrum(base.TID)
        return float(s.redshift.value), np.asarray(s.uncertainty.array) > 0
    d = np.load(DR1_MAP / "main_spectrum.npz")
    return Z, d["ivar"] > 0


def load_data():
    """Return the redshift, the recoadded flux and uncertainty in maggies, and the fitted pixels."""
    z, keep = spender_good()
    new = load_resolution_npz(base.MATRIX)
    wave = WAVE_OBS * u.AA
    flux = to_maggies(wave, new["flux"] * DESI_FLAM).value
    iv = ivar_to_maggies(wave, new["ivar"] * DESI_FLAM**-2).value
    rest = WAVE_OBS / (1 + z)
    good = (
        keep
        & np.isfinite(flux)
        & np.isfinite(iv)
        & (iv > 0)
        & (rest >= base.REST_MIN)
        & (rest < base.REST_MAX)
    )
    unc = np.where(good, 1 / np.sqrt(np.where(good, iv, 1.0)), np.inf)
    return z, np.where(good, flux, 0.0), unc, good


def start_values(nebular):
    """Return start 0, the HR max-L overridden by this spectrum's DR1-coadd continuum MAP."""
    assert not nebular, "this experiment fits the continuum only"
    best = base.hr_best()
    m = np.load(DR1_MAP / f"{base.TID}_map.npz")
    best.update(zip([str(x) for x in m["labels"]], m["best"], strict=True))
    return best


base.OUT = OUT
base.MATRIX = OUT / "resolution_matrix_rows_ivar.npz"
base.load_data = load_data
base.start_values = start_values

if __name__ == "__main__":
    base.main()
