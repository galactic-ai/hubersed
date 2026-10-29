"""Continuum MAP of 39627757533007793 on its DR1 main-survey spectrum (issue #17).

Changes from 2026-10-28_continuum_c3k_er_rmatrix_map (continuum, job 3655467): the data are the
DR1 main-survey coadd (tile 23598, 1 exposure) with spender's mask (main_spectrum.npz of
prepare.py) instead of the SV3 spender chunk, the resolution matrix is the main coadd's, and start 0
is the SV3 continuum MAP best point. Model, priors, fixed SFH hyperparameters, rest 3320-9000 A,
starts and the steps check, fit and collect are those of the 10-28 fit.py, which this file runs with
those three things swapped.

Run on a stampede3 compute node through check.slurm and
map.slurm, for example ``<venv>/bin/python experiments/2026-10-29_continuum_main_survey_map/fit.py
fit --start 0``.
"""

import importlib.util
from pathlib import Path

import astropy.units as u
import numpy as np

from hubersed.conversion import DESI_FLAM, ivar_to_maggies, to_maggies
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

OUT = PATHS["RESULTS"] / "2026-10-29_continuum_main_survey_map"
SV3_MAP = PATHS["RESULTS"] / "2026-10-28_continuum_c3k_er_rmatrix_map" / f"{base.TID}_map.npz"


def load_data():
    """Return the redshift, the flux and uncertainty in maggies, and the fitted pixels.

    The rules are those of the 10-28 load_data, on the main spectrum: spender's positive ivar,
    positive ivar in the matrix file, finite values and rest REST_MIN to REST_MAX.
    """
    d = np.load(OUT / "main_spectrum.npz")
    z = float(d["z"])
    wave = WAVE_OBS * u.AA
    flux = to_maggies(wave, d["flux"] * DESI_FLAM).value
    iv = ivar_to_maggies(wave, d["ivar"] * DESI_FLAM**-2).value
    rest = WAVE_OBS / (1 + z)
    new = load_resolution_npz(base.MATRIX)
    good = (
        np.isfinite(flux)
        & np.isfinite(iv)
        & (iv > 0)
        & (new["ivar"] > 0)
        & (rest >= base.REST_MIN)
        & (rest < base.REST_MAX)
    )
    check = to_maggies(wave, new["flux"] * DESI_FLAM).value
    assert np.allclose(check[good], flux[good], rtol=1e-4, atol=0), "matrix file differs"
    unc = np.where(good, 1 / np.sqrt(np.where(good, iv, 1.0)), np.inf)
    return z, np.where(good, flux, 0.0), unc, good


def start_values(nebular):
    """Return start 0: the HR max-L, overridden by the SV3 continuum MAP best point."""
    assert not nebular, "this experiment fits the continuum only"
    best = base.hr_best()
    m = np.load(SV3_MAP)
    best.update(zip([str(x) for x in m["labels"]], m["best"], strict=True))
    return best


base.OUT = OUT
base.MATRIX = OUT / "resolution_matrix.npz"
base.load_data = load_data
base.start_values = start_values

if __name__ == "__main__":
    base.main()
