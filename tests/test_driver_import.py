"""Importing the MAP driver leaves numpy errors, warning filters and the plot backend alone."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.fsps

# astropy, scipy and pkg_resources add their own warning filters on import, so the
# driver's dependencies are imported before the snapshot.
CHECK = """
import warnings, numpy as np, matplotlib
import hubersed.fitting.chi2, prospect.fitting, prospect.models.sedmodel, scipy.signal
before = (np.geterr(), list(warnings.filters), matplotlib.get_backend())
import hubersed.fitting.run_map_fits_outliers
after = (np.geterr(), list(warnings.filters), matplotlib.get_backend())
print("same" if before == after else f"changed {before} {after}")
"""


def test_import_changes_nothing_process_wide():
    """The driver only quiets warnings and switches to Agg when main or a worker runs."""
    pytest.importorskip("fsps")
    out = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "same"
