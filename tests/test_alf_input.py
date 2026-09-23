"""Check what make_alf_input writes into alf input files."""

import numpy as np
import pytest

pytestmark = pytest.mark.fsps  # make_alf_input imports chi2, which loads FSPS


def test_default_intervals_skip_tio():
    """No default interval overlaps 6400-8000 A, where the polynomial over-fits TiO."""
    from hubersed.alf.make_alf_input import DEFAULT_INTERVALS

    edges = [float(v) * 1e4 for v in DEFAULT_INTERVALS.split(",")]
    for lo, hi in zip(edges[::2], edges[1::2], strict=True):
        assert hi <= 6400 or lo >= 8000


def test_flux_is_written_as_f_lambda(tmp_path, monkeypatch):
    """The flx and err columns are the DESI f_lambda and 1/sqrt(ivar), not maggies."""
    from hubersed.alf import make_alf_input as mai

    tid = 39633140817331167
    spec = 2.0 + 1e-4 * np.arange(mai.WAVE_OBS.size)
    ivar = np.full(mai.WAVE_OBS.size, 4.0)
    monkeypatch.setattr(mai, "tids_to_indices", lambda tids: np.array([0]))
    monkeypatch.setattr(mai, "load_by_index", lambda idx: (spec, ivar, 0.05, tid))

    out = tmp_path / "desi.dat"
    mai.main(["--tid", str(tid), "-o", str(out), "--mask", ""])
    lam, flx, err, wgt, _ = np.loadtxt(out, comments="#", unpack=True)

    expected = np.interp(lam * 1.05, mai.WAVE_OBS, spec)
    assert np.all(wgt == 1.0)
    np.testing.assert_allclose(flx, expected, rtol=1e-6)
    np.testing.assert_allclose(err, 0.5, rtol=1e-6)
