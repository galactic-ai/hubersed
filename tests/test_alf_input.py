"""Check the default wavelength intervals written into alf input files."""

import pytest

pytestmark = pytest.mark.fsps  # make_alf_input imports chi2, which loads FSPS


def test_default_intervals_skip_tio():
    """No default interval overlaps 6400-8000 A, where the polynomial over-fits TiO."""
    from hubersed.alf.make_alf_input import DEFAULT_INTERVALS

    edges = [float(v) * 1e4 for v in DEFAULT_INTERVALS.split(",")]
    for lo, hi in zip(edges[::2], edges[1::2], strict=True):
        assert hi <= 6400 or lo >= 8000
