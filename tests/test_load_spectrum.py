"""load_spectrum returns a DESI spectrum with units, mask and the TARGETID it was asked for."""

import astropy.units as u
import numpy as np
import pytest
from spender.data.dataset import get_features, write_dataset

from hubersed.conversion import DESI_FLAM
from hubersed.io import desi

NPIX = len(desi.DESI_WAV)
TIDS = np.array([39_627_000_000_000_001, 39_627_000_000_000_002, 39_627_000_000_000_003])
Z = 0.1


@pytest.fixture
def dataset(tmp_path):
    """Write three normalized spectra, one with a NaN flux pixel and one zero ivar pixel.

    Spectrum 2 has norm 2, so it is stored as flux 1 and ivar 16.
    """
    spec = np.full((3, NPIX), 2.0, np.float32)
    spec[1, 5] = np.nan
    ivar = np.full((3, NPIX), 4.0, np.float32)
    ivar[1, 7] = 0.0
    norm = np.array([1.0, 1.0, 2.0], np.float32)
    batch = dict(
        spec=spec / norm[:, None],
        w=ivar * norm[:, None] ** 2,
        z=np.full(3, Z, np.float32),
        zerr=np.zeros(3, np.float32),
        norm=norm,
        target_id=TIDS,
    )
    path = tmp_path / "ds"
    write_dataset(str(path), [batch], get_features(NPIX, ["target_id"]), ["target_id"])
    return path


def test_units_mask_and_redshift(dataset):
    """Flux, wavelength and inverse variance carry units, and bad pixels are masked."""
    spec = desi.load_spectrum(int(TIDS[1]), dataset)
    assert spec.meta["targetid"] == TIDS[1]
    assert spec.flux.unit == DESI_FLAM
    assert spec.spectral_axis.unit == u.AA
    np.testing.assert_array_equal(spec.spectral_axis.value, desi.DESI_WAV)
    assert spec.uncertainty.unit == DESI_FLAM**-2
    assert spec.uncertainty.array[0] == 4.0
    assert spec.redshift.value == pytest.approx(Z)
    assert np.flatnonzero(spec.mask).tolist() == [5, 7]


def test_flux_converts_to_cgs(dataset):
    """2 DESI flux units are 2e-17 erg/s/cm^2/A, to float32 precision."""
    spec = desi.load_spectrum(int(TIDS[0]), dataset)
    assert spec.flux.dtype == np.float32
    cgs = spec.flux.to(u.erg / u.s / u.cm**2 / u.AA)
    np.testing.assert_allclose(cgs.value, 2e-17, rtol=1e-7)


def test_normalization_is_undone(dataset):
    """A spectrum stored with norm 2 comes back as flux 2 and ivar 4."""
    flux, ivar, _, tid = desi.load_by_targetid(int(TIDS[2]), dataset)
    assert tid == TIDS[2]
    np.testing.assert_array_equal(flux, 2.0)
    np.testing.assert_array_equal(ivar, 4.0)


def test_nan_ivar_is_masked():
    """A NaN inverse variance is masked, as the fitting drivers' own mask always did."""
    flux = np.full(NPIX, 2.0, np.float32)
    ivar = np.full(NPIX, 4.0, np.float32)
    ivar[3] = np.nan
    spec = desi.desi_spectrum(flux, ivar, Z, int(TIDS[0]))
    assert np.flatnonzero(spec.mask).tolist() == [3]


def test_unknown_tid_raises(dataset):
    """A TARGETID missing from the dataset fails rather than returning another spectrum."""
    with pytest.raises(ValueError, match="appears 0 times"):
        desi.load_spectrum(int(TIDS.max() + 1), dataset)


def test_wrong_row_raises(dataset, monkeypatch):
    """If the row found holds another TARGETID, loading fails."""
    wrong = desi.load_by_targetid(int(TIDS[2]), dataset)
    monkeypatch.setattr(desi, "load_by_targetid", lambda tid, ds: wrong)
    with pytest.raises(desi.TargetIDMismatchError, match="holds"):
        desi.load_spectrum(int(TIDS[0]), dataset)
