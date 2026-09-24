"""Mock loaders return DESI-unit spectra, like load_spectrum does for data."""

import pickle

import astropy.units as u
import h5py
import numpy as np
import pytest

from hubersed.conversion import DESI_FLAM, to_flambda
from hubersed.io.mocks import load_mock, load_noisy_mock

WAVE = np.linspace(3600.0, 9824.0, 50)


@pytest.fixture
def mock_h5(tmp_path):
    """Write a mock h5 with three spectra in maggies and two per-mock priors."""
    path = tmp_path / "mocks.h5"
    with h5py.File(path, "w") as f:
        f["wavelength"] = WAVE
        f["fluxes"] = np.full((3, WAVE.size), 1e-9, np.float32) * np.arange(1, 4)[:, None]
        f["priors/redshifts"] = np.array([0.05, 0.10, 0.15])
        f["priors/stellar_masses"] = np.array([9.0, 10.0, 11.0])
        f["priors/line_lum"] = np.ones((3, 4))
        f.attrs["priors_seed"] = 42
    return path


def test_noiseless_mock_in_desi_units(mock_h5):
    """Flux is f_lambda in DESI units, with the row's redshift and priors."""
    spec = load_mock(mock_h5, 1)
    assert spec.flux.unit == DESI_FLAM
    expected = to_flambda(WAVE * u.AA, np.full(WAVE.size, 2e-9) * u.mgy).value
    np.testing.assert_allclose(spec.flux.value, expected, rtol=1e-6)
    assert spec.redshift.value == pytest.approx(0.10)
    assert spec.meta["row"] == 1
    assert float(spec.meta["priors"]["stellar_masses"]) == 10.0
    assert spec.meta["priors"]["line_lum"].shape == (4,)
    assert not spec.mask.any()


def write_noisy(path, norm):
    """Write one noisy chunk with two spectra normalized by ``norm``.

    A zero norm leaves the spectrum unscaled, as normalize_spectra does.
    """
    flux = np.full((2, WAVE.size - 1), 4.0)
    ivar = np.full((2, WAVE.size - 1), 9.0)
    ivar[0, 2] = 0.0
    ivar[0, 3] = np.nan
    scale = np.where(norm == 0, 1.0, norm)[:, None]
    batch = [
        flux / scale,
        ivar * scale**2,
        np.array([0.1, 0.2]),
        np.array([7, 8]),
        norm,
        np.zeros(2),
    ]
    with open(path, "wb") as f:
        pickle.dump(batch, f)


def test_noisy_mock_undoes_the_norm(tmp_path):
    """Flux and ivar are back in DESI units, bad pixels are masked, the mock row is kept."""
    path = tmp_path / "noisy.pkl"
    write_noisy(path, np.array([2.0, 0.5]))
    spec = load_noisy_mock(path, 0, WAVE)
    assert spec.flux.unit == DESI_FLAM
    np.testing.assert_allclose(spec.flux.value, 4.0)
    np.testing.assert_allclose(spec.uncertainty.array[:2], 9.0)
    assert spec.spectral_axis.size == WAVE.size - 1
    assert np.flatnonzero(spec.mask).tolist() == [2, 3]
    assert spec.meta["mock_row"] == 7
    assert spec.redshift.value == pytest.approx(0.1)


def test_noisy_mock_with_zero_norm_raises(tmp_path):
    """A spectrum the noisy script could not normalize is refused."""
    path = tmp_path / "noisy.pkl"
    write_noisy(path, np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="norm 0"):
        load_noisy_mock(path, 0, WAVE)


def test_noisy_mock_on_the_wrong_grid_raises(tmp_path):
    """A wavelength grid that does not fit the pixel count is refused."""
    path = tmp_path / "noisy.pkl"
    write_noisy(path, np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="pixels"):
        load_noisy_mock(path, 0, WAVE[:-5])
