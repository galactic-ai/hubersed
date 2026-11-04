"""Continuum MAP of 39627757533007793 on its recoadded SV3 or main spectrum with afe fixed at 0 (#17).

Changes from 2026-10-29_continuum_recoadd/fit.py, which this file runs: afe is fixed at 0 instead of
free over -0.2 to 0.6, so logzsol is [Fe/H] and the model has 15 free parameters, not 16. Start 0 is
that spectrum's afe-free recoadd MAP without afe. Data, resolution matrix, fitted pixels, priors,
fixed SFH hyperparameters and the steps check, fit and collect are unchanged. Outputs go to
results/2026-11-01_recoadd_afe0/<survey>/.

RECOADD_SURVEY (sv3 or main) picks the spectrum. Run on a stampede3 compute node through map.slurm.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
from prospect.models.priors import TopHat

from hubersed.paths import PATHS
from hubersed.sps.config import build_continuum_model
from hubersed.sps.resolution_model import ResolutionMatrixModel
from hubersed.sps.utils import universe_age_gyr

# the 10-29 recoadd fit.py, loaded by path since this file has the same name; it sets up the 10-28
# fit.py (``recoadd.base``) with the recoadded data and matrix
_spec = importlib.util.spec_from_file_location(
    "fit_recoadd",
    Path(__file__).resolve().parents[1] / "2026-10-29_continuum_recoadd" / "fit.py",
)
recoadd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recoadd)
base = recoadd.base

SURVEY = recoadd.SURVEY
OUT = PATHS["RESULTS"] / "2026-11-01_recoadd_afe0" / SURVEY
AFE_FREE_MAP = recoadd.OUT / f"{base.TID}_map.npz"


def make_model(z, nebular=False):
    """Return the 10-28 continuum model with afe fixed at 0."""
    assert not nebular, "this experiment fits the continuum only"
    best = base.hr_best()
    _, t = build_continuum_model(z)
    t["add_neb_emission"] = {"N": 1, "isfree": False, "init": False}
    t["tau_in"]["isfree"] = False
    t["tau_in"]["init"] = universe_age_gyr(z)
    for name in base.HYPERS:
        t[name]["isfree"] = False
        t[name]["init"] = best[name]
    t["afe"] = {"N": 1, "isfree": False, "init": 0.0, "prior": TopHat(mini=-0.2, maxi=0.6)}
    return ResolutionMatrixModel(t)


def start_values(nebular):
    """Return start 0: the HR max-L, overridden by this spectrum's afe-free recoadd MAP."""
    assert not nebular, "this experiment fits the continuum only"
    best = base.hr_best()
    m = np.load(AFE_FREE_MAP)
    best.update(zip([str(x) for x in m["labels"]], m["best"], strict=True))
    best.pop("afe")
    return best


OUT.mkdir(parents=True, exist_ok=True)
base.OUT = OUT
base.make_model = make_model
base.start_values = start_values
assert os.environ.get("RECOADD_SURVEY") in ("sv3", "main"), "set RECOADD_SURVEY to sv3 or main"

if __name__ == "__main__":
    base.main()
