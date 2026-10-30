"""Continuum MAP of the SV3 dark spectrum of the continuum outlier 39633433328091196 (#17).

Changes from 2026-10-29_continuum_39633433328091196_map/fit.py (SV3 bright, icx job 3657895): the
data are the SV3 dark DR1 coadd (1 exposure, EXPID 88050) and its DR1 resolution matrix, both from
resolution_matrix.py, since the SV3 bright coadd carries sky residual in the blue
(2026-10-30_recoadd_39633433328091196). Start 0 is the SV3 bright ER MAP best point. Model, priors,
the SFH hyperparameters fixed at the C3K_HR max-L (hr_maxl.npz of the 10-29 folder), rest window
3320 to 9000 A, starts and the steps check, fit and collect are those of the 10-28 fit.py.

Run on a stampede3 compute node through map.slurm, for example
``<venv>/bin/python experiments/2026-10-30_continuum_39633433328091196_dark_map/fit.py fit --start 0``.
"""

import importlib.util
from pathlib import Path

import numpy as np

from hubersed.io.desi import desi_spectrum
from hubersed.paths import PATHS

EXPS = Path(__file__).resolve().parents[1]
BRIGHT = PATHS["RESULTS"] / "2026-10-29_continuum_39633433328091196_map"
OUT = PATHS["RESULTS"] / "2026-10-30_continuum_39633433328091196_dark_map"

# the 10-29 fit.py, loaded by path since this file has the same name; it loads the 10-28 one
_spec = importlib.util.spec_from_file_location(
    "fit_1029", EXPS / "2026-10-29_continuum_39633433328091196_map" / "fit.py"
)
target = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(target)
base = target.base


def dark_spectrum(tid):
    """Return the SV3 dark spectrum of resolution_matrix.py in the form of load_spectrum."""
    d = np.load(OUT / "dark_spectrum.npz")
    assert int(d["targetid"]) == tid, (int(d["targetid"]), tid)
    return desi_spectrum(d["flux"], d["ivar"], float(d["z"]), tid)


def start_values(nebular):
    """Return start 0, the SV3 bright ER MAP best point."""
    assert not nebular, "this experiment fits the continuum only"
    d = np.load(BRIGHT / f"{base.TID}_map.npz")
    return dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))


base.load_spectrum = dark_spectrum
base.OUT = OUT
base.MATRIX = OUT / "resolution_matrix.npz"
base.start_values = start_values

if __name__ == "__main__":
    base.main()
