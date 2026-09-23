"""Importing a fitting driver leaves numpy errors, warning filters and the plot backend alone."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.fsps

# astropy, scipy and pkg_resources add their own warning filters on import, so the
# drivers' dependencies are imported before the snapshot.
CHECK = """
import warnings, numpy as np, matplotlib
import hubersed.fitting.chi2, prospect.fitting, prospect.models.sedmodel, scipy.signal
import dynesty, dynesty.utils
before = (np.geterr(), list(warnings.filters), matplotlib.get_backend())
import hubersed.fitting.{}
after = (np.geterr(), list(warnings.filters), matplotlib.get_backend())
print("same" if before == after else f"changed {{before}} {{after}}")
"""


@pytest.mark.parametrize("driver", ["run_map_fits_outliers", "run_dynesty_outliers"])
def test_import_changes_nothing_process_wide(driver):
    """A driver only quiets warnings, and the MAP driver only switches to Agg, when main runs."""
    pytest.importorskip("fsps")
    code = CHECK.format(driver)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "same"
