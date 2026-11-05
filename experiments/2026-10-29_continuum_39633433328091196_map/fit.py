"""Continuum MAP of the continuum outlier 39633433328091196 with C3K_ER, free afe and its R matrix.

Changes from 2026-10-28_continuum_c3k_er_rmatrix_map (39627757533007793): the target is
39633433328091196 (z 0.392, rank 4 of 2026-10-21_continuum_outlier_snr), its DR1 SV3 coadd and
its DR1 resolution matrix (resolution_matrix.py, healpix 10231). The fixed SFH hyperparameters
and start 0 are the max-L point of its own C3K_HR afefree nautilus run, which had nebular
emission off (hypers.py, hr_maxl.npz), where 10-28 used the nebular-on run. Model, priors, rest
window 3320 to 9000 A, starts and the steps check, fit and collect are those of the 10-28
fit.py, which this file runs with target, matrix, output folder and start swapped. No Cue.

Run on a stampede3 compute node through test.slurm or map.slurm, for example
``<venv>/bin/python experiments/2026-10-29_continuum_39633433328091196_map/fit.py fit --start 0``.
"""

import importlib.util
from pathlib import Path

import numpy as np

from hubersed.paths import PATHS

# the 10-28 fit.py, loaded by path since this file has the same name
_spec = importlib.util.spec_from_file_location(
    "fit_1028",
    Path(__file__).resolve().parents[1] / "2026-10-28_continuum_c3k_er_rmatrix_map" / "fit.py",
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

OUT = PATHS["RESULTS"] / "2026-10-29_continuum_39633433328091196_map"


def hr_best():
    """Return the max-L point of the C3K_HR afefree run as a dict of label to value."""
    d = np.load(OUT / "hr_maxl.npz")
    return dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))


def start_values(nebular):
    """Return start 0, the C3K_HR max-L."""
    assert not nebular, "this experiment fits the continuum only"
    return hr_best()


base.TID = 39633433328091196
base.OUT = OUT
base.MATRIX = OUT / "resolution_matrix.npz"
base.hr_best = hr_best
base.start_values = start_values

if __name__ == "__main__":
    base.main()
