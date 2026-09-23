"""Check what make_alf_input writes into alf input files."""

import numpy as np
import pytest

pytestmark = pytest.mark.fsps  # make_alf_input imports chi2, which loads FSPS

TID = 39633140817331167
Z = 0.05


def test_default_intervals_skip_tio():
    """No default interval overlaps 6400-8000 A, where the polynomial over-fits TiO."""
    from hubersed.alf.make_alf_input import DEFAULT_INTERVALS

    edges = [float(v) * 1e4 for v in DEFAULT_INTERVALS.split(",")]
    for lo, hi in zip(edges[::2], edges[1::2], strict=True):
        assert hi <= 6400 or lo >= 8000


@pytest.fixture
def written(tmp_path, monkeypatch):
    """Run make_alf_input on a fake DESI spectrum and return the file and the spectrum."""
    from hubersed.alf import make_alf_input as mai

    spec = 2.0 + 1e-4 * np.arange(mai.WAVE_OBS.size)
    ivar = np.full(mai.WAVE_OBS.size, 4.0)
    monkeypatch.setattr(mai, "tids_to_indices", lambda tids: np.array([0]))
    monkeypatch.setattr(mai, "load_by_index", lambda idx: (spec, ivar, Z, TID))

    out = tmp_path / "desi.dat"
    mai.main(["--tid", str(TID), "-o", str(out), "--mask", ""])
    return out, spec, mai.WAVE_OBS


def test_flux_is_written_as_f_lambda(written):
    """The flx and err columns are the DESI f_lambda and 1/sqrt(ivar), not maggies."""
    out, spec, wave_obs = written
    lam, flx, err, wgt, _ = np.loadtxt(out, comments="#", unpack=True)

    expected = np.interp(lam * (1 + Z), wave_obs, spec)
    assert np.all(wgt == 1.0)
    np.testing.assert_allclose(flx, expected, rtol=1e-6)
    np.testing.assert_allclose(err, 0.5, rtol=1e-6)


def test_wavelengths_are_rest_frame(written):
    """Written wavelengths are the DESI grid divided by 1+z, the frame alf reads intervals in."""
    out, _, wave_obs = written
    lam = np.loadtxt(out, comments="#", usecols=0)
    rest = wave_obs / (1 + Z)
    i = np.clip(np.searchsorted(rest, lam), 1, rest.size - 1)
    gap = np.minimum(np.abs(lam - rest[i - 1]), np.abs(lam - rest[i]))
    assert gap.max() < 1e-3  # the file stores 4 decimals


def test_observed_flag_is_gone():
    """--observed wrote observed-frame edges that alf then shifted again, so it was removed."""
    from hubersed.alf import make_alf_input as mai

    with pytest.raises(SystemExit):
        mai.main(["-o", "unused.dat", "--observed"])
