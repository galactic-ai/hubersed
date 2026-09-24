"""Load single mock spectra as specutils Spectrum objects in the same units as DESI data.

Noiseless mocks come from the h5 written by ``scripts/make_model_seds.py`` (flux in maggies on
the DESI grid). Noisy mocks come from the chunk pickles written by
``scripts/make_prospector_noisy_sed.py``, which use the DESI chunk layout (flux divided by a
per-spectrum norm, and the DESI grid without its last pixel). Both come back as f_lambda in
``DESI_FLAM``, like ``hubersed.io.desi.load_spectrum``, so mocks and data can be compared
directly. The encoder reads the chunk files in bulk as arrays and does not use this module.
"""

import pickle
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
from astropy.nddata import InverseVariance
from specutils import Spectrum

from hubersed.conversion import DESI_FLAM, to_flambda


def load_mock(path, row):
    """Load one noiseless mock from a mock h5 file.

    Parameters
    ----------
    path : str or pathlib.Path
        h5 file with ``fluxes`` in maggies, ``wavelength`` in Angstrom and ``priors/*``.
    row : int
        Row of the mock in the file.

    Returns
    -------
    specutils.Spectrum
        Flux in ``DESI_FLAM`` on the file's observed-frame grid, the prior redshift, no
        uncertainty and no masked pixels. ``meta`` holds ``row``, ``file`` and ``priors``,
        the value of every per-mock prior for this row.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        wave = np.asarray(f["wavelength"], np.float64)
        maggies = np.asarray(f["fluxes"][row], np.float64)
        n = f["fluxes"].shape[0]
        priors = {
            k: np.asarray(v[row])
            for k, v in f["priors"].items()
            if isinstance(v, h5py.Dataset) and v.shape[:1] == (n,)
        }
    flux = to_flambda(wave * u.AA, maggies * u.mgy)
    return Spectrum(
        flux=flux,
        spectral_axis=wave * u.AA,
        mask=np.zeros(wave.size, bool),
        redshift=float(priors["redshifts"]),
        meta={"row": int(row), "file": path.name, "priors": priors},
    )


def load_noisy_mock(path, row, wave):
    """Load one noisy mock from a noisy mock chunk pickle, undoing its normalization.

    Parameters
    ----------
    path : str or pathlib.Path
        Pickle holding flux, ivar, z, mock row, norm and z error, in that order.
    row : int
        Row within the pickle.
    wave : numpy.ndarray
        Observed wavelength in Angstrom of the noiseless file the mocks were made from. The
        noisy flux has one pixel fewer, so ``wave[:-1]`` is used.

    Returns
    -------
    specutils.Spectrum
        Flux in ``DESI_FLAM``, inverse variance in ``DESI_FLAM**-2``, the redshift and a mask
        of pixels whose inverse variance is not positive or not finite, or whose flux is not
        finite. ``meta`` holds ``mock_row``, the row of this mock in the noiseless file.

    Raises
    ------
    ValueError
        If the spectrum's norm is zero. The noisy script leaves such spectra unscaled, so
        multiplying by the norm would not give physical flux.
    """
    with open(path, "rb") as f:
        s, w, z, mock_row, norm, _ = pickle.load(f)

    def to(x):
        return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

    s, w, norm = to(s)[row], to(w)[row], float(to(norm)[row])
    if norm == 0:
        raise ValueError(f"row {row} of {path} has norm 0 and was never normalized")
    flux, ivar = s * norm, w / norm**2
    grid = np.asarray(wave, np.float64)[:-1]
    if flux.size != grid.size:
        raise ValueError(f"{path} has {flux.size} pixels, expected {grid.size}")
    return Spectrum(
        flux=flux * DESI_FLAM,
        spectral_axis=grid * u.AA,
        uncertainty=InverseVariance(ivar * DESI_FLAM**-2),
        mask=(ivar <= 0) | ~np.isfinite(ivar) | ~np.isfinite(flux),
        redshift=float(to(z)[row]),
        meta={"mock_row": int(to(mock_row)[row])},
    )
