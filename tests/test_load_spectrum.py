"""load_spectrum returns a DESI spectrum with units, mask and the TARGETID it was asked for."""

import astropy.units as u
import numpy as np
import pytest
import torch
from spender.data.desi import DESI

from hubersed.conversion import DESI_FLAM
from hubersed.io import desi

NPIX = len(desi.DESI_WAV)
TIDS = np.array([39_627_000_000_000_001, 39_627_000_000_000_002, 39_627_000_000_000_003])
Z = 0.1


@pytest.fixture
def one_chunk(tmp_path, monkeypatch):
    """Write one chunk of three spectra with one NaN flux pixel and one zero ivar pixel."""
    chunk_dir = tmp_path / "desi_spectra"
    chunk_dir.mkdir()
    spec = torch.full((3, NPIX), 2.0)
    spec[1, 5] = float("nan")
    ivar = torch.full((3, NPIX), 4.0)
    ivar[1, 7] = 0.0
    one = torch.ones(3)
    batch = [spec, ivar, torch.full((3,), Z), torch.from_numpy(TIDS), one, one]
    DESI.save_batch(str(chunk_dir), batch, tag="chunk1024", counter=0)
    np.save(tmp_path / "all_target_ids.npy", TIDS)
    monkeypatch.setattr(desi, "DATA_PATH", tmp_path)


@pytest.mark.usefixtures("one_chunk")
def test_units_mask_and_redshift():
    """Flux, wavelength and inverse variance carry units, and bad pixels are masked."""
    spec = desi.load_spectrum(int(TIDS[1]))
    assert spec.meta["targetid"] == TIDS[1]
    assert spec.flux.unit == DESI_FLAM
    assert spec.spectral_axis.unit == u.AA
    np.testing.assert_array_equal(spec.spectral_axis.value, desi.DESI_WAV)
    assert spec.uncertainty.unit == DESI_FLAM**-2
    assert spec.uncertainty.array[0] == 4.0
    assert spec.redshift.value == pytest.approx(Z)
    assert np.flatnonzero(spec.mask).tolist() == [5, 7]


@pytest.mark.usefixtures("one_chunk")
def test_flux_converts_to_cgs():
    """2 DESI flux units are 2e-17 erg/s/cm^2/A, to float32 precision like the chunk files."""
    spec = desi.load_spectrum(int(TIDS[0]))
    assert spec.flux.dtype == np.float32
    cgs = spec.flux.to(u.erg / u.s / u.cm**2 / u.AA)
    np.testing.assert_allclose(cgs.value, 2e-17, rtol=1e-7)


def test_nan_ivar_is_masked():
    """A NaN inverse variance is masked, as the fitting drivers' own mask always did."""
    flux = np.full(NPIX, 2.0, np.float32)
    ivar = np.full(NPIX, 4.0, np.float32)
    ivar[3] = np.nan
    spec = desi.desi_spectrum(flux, ivar, Z, int(TIDS[0]))
    assert np.flatnonzero(spec.mask).tolist() == [3]


@pytest.mark.usefixtures("one_chunk")
def test_wrong_row_raises(monkeypatch):
    """If all_target_ids.npy points at a row holding another TARGETID, loading fails."""
    monkeypatch.setattr(desi, "tids_to_indices", lambda tids: np.array([2]))
    with pytest.raises(desi.TargetIDMismatchError, match="holds"):
        desi.load_spectrum(int(TIDS[0]))
