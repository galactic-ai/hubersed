"""Importing a fitting driver leaves numpy errors, warning filters and the plot backend alone."""

import subprocess
import sys

import pytest

pytestmark = [pytest.mark.fsps, pytest.mark.slow]

DRIVERS = ["run_map_fits_outliers", "run_dynesty_outliers"]

# astropy, scipy and pkg_resources add their own warning filters on import, so the
# drivers' dependencies are imported before the first snapshot. One process checks
# every driver, because each start of Python with FSPS takes several seconds.
CHECK = """
import importlib, warnings, numpy as np, matplotlib
import hubersed.fitting.chi2, prospect.fitting, prospect.models.sedmodel, scipy.signal
import dynesty, dynesty.utils
state = lambda: (np.geterr(), list(warnings.filters), matplotlib.get_backend())
for name in {drivers!r}:
    before = state()
    importlib.import_module("hubersed.fitting." + name)
    print(name, "same" if state() == before else "changed")
"""


def test_import_changes_nothing_process_wide():
    """A driver only quiets warnings, and the MAP driver only switches to Agg, when main runs."""
    pytest.importorskip("fsps")
    code = CHECK.format(drivers=DRIVERS)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    # The drivers print progress while they load, so keep only the result lines.
    lines = [ln for ln in out.stdout.splitlines() if ln.split(" ")[0] in DRIVERS]
    assert lines == [f"{d} same" for d in DRIVERS]
