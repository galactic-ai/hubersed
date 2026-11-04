"""MILES continuum MAP of 39627757533007793 on its recoadded SV3 spectrum (#17).

Changes from 2026-10-08_continuum_outlier run 1d (MILES, MILES window, nebular off, no degrading,
nautilus on the SV3 DR1 coadd): the data are the SV3 recoadd flux and ivar of
2026-10-29_continuum_recoadd (fitted pixels as there, cut to the MILES window, rest 3601.8 to
7400.8 A), and the fit is a 7-start Powell MAP as 2026-10-28_continuum_c3k_er_rmatrix_map, with the
four SFH hyperparameters fixed at the C3K HR afefree max-L instead of free. As in run 1d the MILES
library resolution is set to zero and prospect smooths the model to a Gaussian of the median DESI
resolution (hubersed.sps.lsf.desi_resolution). Start 0 is the run 1d max-L. Steps check, fit and
collect are those of the 10-28 fit.py, which this file runs with model, observation, source, data
and output folder swapped.

FSPS must be the MILES build first on PYTHONPATH, with SPS_HOME holding the MILES spectra. Run on a
stampede3 compute node through map.slurm.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
from prospect.models.sedmodel import HyperSpecModel

from hubersed.paths import PATHS
from hubersed.sps import rebin
from hubersed.sps.config import build_continuum_model
from hubersed.sps.lsf import desi_resolution
from hubersed.sps.parameter_file import WAVE_OBS, build_obs, build_sps
from hubersed.sps.utils import universe_age_gyr

os.environ.setdefault("RECOADD_SURVEY", "sv3")
assert os.environ["RECOADD_SURVEY"] == "sv3", "this experiment fits the SV3 recoadd only"
# the 10-29 recoadd fit.py, loaded by path since this file has the same name; it sets up the 10-28
# fit.py (``recoadd.base``) with the recoadded data and matrix
_spec = importlib.util.spec_from_file_location(
    "fit_recoadd",
    Path(__file__).resolve().parents[1] / "2026-10-29_continuum_recoadd" / "fit.py",
)
recoadd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recoadd)
base = recoadd.base

OUT = PATHS["RESULTS"] / "2026-11-01_miles_recoadd_map"
RUN_1D = (
    PATHS["RESULTS"]
    / "2026-10-08_continuum_outlier"
    / f"{base.TID}_miles_miles_neboff_nodeg_pred.npz"
)
MILES_REST = (3601.8, 7400.8)  # the MILES window of 2026-10-08 fit.py
LSF = rebin.C_KMS / (2.355 * desi_resolution(WAVE_OBS.astype(np.float64)))


def make_model(z, nebular=False):
    """Return the continuum model (no afe), SFH hyperparameters fixed at the HR max-L."""
    assert not nebular, "this experiment fits the continuum only"
    best = base.hr_best()
    _, t = build_continuum_model(z)
    t["add_neb_emission"] = {"N": 1, "isfree": False, "init": False}
    t["tau_in"]["isfree"] = False
    t["tau_in"]["init"] = universe_age_gyr(z)
    for name in base.HYPERS:
        t[name]["isfree"] = False
        t[name]["init"] = best[name]
    return HyperSpecModel(t)


def make_obs(flux, unc, good):
    """Return the observation with the median DESI LSF, at least 10 km/s in the padding."""
    obs = build_obs(spec=flux, unc=unc, mask=good, resolution=LSF, wavelength=WAVE_OBS)
    for spec in obs:
        wave = spec.padded_wavelength
        pad = (wave < WAVE_OBS[0]) | (wave > WAVE_OBS[-1])
        spec.padded_resolution[pad] = np.maximum(spec.padded_resolution[pad], 10.0)
    return obs


def get_sps(nebular=False):
    """Return the plain FSPS source, checked to be MILES, with its library resolution zeroed."""
    assert not nebular, "this experiment fits the continuum only"
    sps = build_sps(zcontinuous=1, zero_library_resolution=True)
    assert sps.ssp.libraries[1] in ("miles", b"miles"), sps.ssp.libraries
    assert not np.any(sps.spectral_resolution), "library resolution is not zero"
    return sps


def start_values(nebular):
    """Return start 0: the HR max-L, overridden by the run 1d max-L for the parameters it has."""
    assert not nebular, "this experiment fits the continuum only"
    best = base.hr_best()
    d = np.load(RUN_1D)
    run_1d = dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))
    best.update({k: v for k, v in run_1d.items() if k not in base.HYPERS})
    return best


OUT.mkdir(parents=True, exist_ok=True)
base.OUT = OUT
base.REST_MIN, base.REST_MAX = MILES_REST
base.make_model = make_model
base.make_obs = make_obs
base.get_sps = get_sps
base.start_values = start_values

if __name__ == "__main__":
    base.main()
