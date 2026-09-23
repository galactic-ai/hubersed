"""Build prospector observations and stellar population sources for DESI fits."""

import numpy as np
from prospect.observation import Spectrum
from prospect.sources import FastStepBasis, SSPBasis

from hubersed.sps.lsf import DESI_WAV
from hubersed.sps.utils import load_lines

WAVE_OBS = DESI_WAV.astype(np.float32)


EM_LINES_A = load_lines()["emission"]["wave_vac"]


def build_obs(
    spec: np.ndarray,
    unc: np.ndarray,
    mask: np.ndarray,
    resolution: np.ndarray | None = None,
    wavelength: np.ndarray = WAVE_OBS,
) -> list:
    """Wrap a DESI spectrum as a prospect observation.

    Parameters
    ----------
    spec : np.ndarray
        Observed flux in maggies. Convert from f_lambda first.
    unc : np.ndarray
        One sigma uncertainty of ``spec`` in maggies, not the variance.
    mask : np.ndarray
        True for pixels used in the likelihood. Use it to drop bad pixels and sky lines.
    resolution : np.ndarray, optional
        Instrumental resolution as a Gaussian sigma in km/s at each wavelength. When
        given, prospect blurs the model to match. Pass ``C_KMS / (2.355 * R)``.
    wavelength : np.ndarray
        Observed wavelength in Angstrom. The default is the DESI grid.

    Returns
    -------
    list of prospect.observation.Spectrum
        A one-element list, already passed through ``rectify``.

    Notes
    -----
    Passing ``resolution`` works only because ``build_sps`` sets the template resolution to
    zero. Otherwise prospect refuses data that is sharper than the templates.
    """
    spec_obs = Spectrum(
        wavelength=wavelength,
        flux=spec,
        uncertainty=unc,
        mask=mask,
        resolution=resolution,
    )
    spec_obs.rectify()
    return [spec_obs]


def build_sps(zcontinuous=1):
    """Build the FSPS source with a non-parametric star formation history.

    Parameters
    ----------
    zcontinuous : int
        FSPS metallicity interpolation mode, passed to ``FastStepBasis``.

    Returns
    -------
    prospect.sources.FastStepBasis
        The stellar population source.

    Notes
    -----
    The MILES templates have lower resolution than DESI spectra, so prospect would refuse to
    blur the model to the DESI line spread function. Setting the template resolution to zero
    lets it through. This replaces ``SSPBasis.spectral_resolution`` with zeros on the class
    itself, so the change applies to every SSPBasis in the process, not only the one returned.
    """
    sps = FastStepBasis(zcontinuous=zcontinuous)
    SSPBasis.spectral_resolution = property(lambda self: np.zeros_like(self.ssp.wavelengths))
    return sps


def build_cue_sps():
    """Build the source that takes nebular emission from the Cue emulator instead of FSPS.

    Returns
    -------
    prospect.sources.NebStepBasis
        The stellar population source with Cue nebular emission.
    """
    from prospect.sources import NebStepBasis

    return NebStepBasis()


def mask_spectral_lines(wave_obs, mask, z, line_waves=EM_LINES_A, halfwidth_kms=500.0):
    """Mask pixels within a velocity window around each emission line.

    Parameters
    ----------
    wave_obs : np.ndarray
        Observed wavelength in Angstrom.
    mask : np.ndarray
        True for good pixels. It is copied, not changed.
    z : float
        Redshift.
    line_waves : np.ndarray
        Rest-frame vacuum wavelengths of the lines in Angstrom. The default is the emission
        line list from ``load_lines``.
    halfwidth_kms : float
        Half width of each masked window in km/s.

    Returns
    -------
    np.ndarray
        The mask with the line windows set to False.
    """
    c_kms = 299792.458
    mask = mask.copy()
    wave_rest = wave_obs / (1.0 + z)

    for line in line_waves:
        dwave = line * halfwidth_kms / c_kms
        mask &= np.abs(wave_rest - line) > dwave

    return mask
