"""Importing a pipeline script changes no process wide state until its main runs."""

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.fsps, pytest.mark.slow]

MODULES = [
    "hubersed.fitting.run_map_fits_outliers",
    "hubersed.fitting.run_dynesty_outliers",
    "hubersed.mocks.make_model_seds",
]
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "XLA_FLAGS",
)

# astropy, scipy and pkg_resources add their own warning filters on import, and chi2 sets
# the BLAS thread variables before numpy loads, so these are imported before the first
# snapshot. One process checks every module, because each start of Python with FSPS
# takes several seconds.
CHECK = """
import importlib, os, warnings, numpy as np, matplotlib
import hubersed.fitting.chi2, prospect.fitting, prospect.models.sedmodel, scipy.signal
import dynesty, dynesty.utils
state = lambda: (np.geterr(), list(warnings.filters), matplotlib.get_backend(), dict(os.environ))
for name in {modules!r}:
    before = state()
    importlib.import_module(name)
    print(name, "same" if state() == before else "changed")
"""


def test_import_changes_nothing_process_wide():
    """No warning filter, numpy error setting, plot backend or env variable changes on import."""
    pytest.importorskip("fsps")
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}
    code = CHECK.format(modules=MODULES)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    # The modules print progress while they load, so keep only the result lines.
    lines = [ln for ln in out.stdout.splitlines() if ln.split(" ")[0] in MODULES]
    assert lines == [f"{m} same" for m in MODULES]
