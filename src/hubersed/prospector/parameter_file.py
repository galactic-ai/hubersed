from prospect.observation import Spectrum
from prospect.sources import FastStepBasis, SSPBasis
import numpy as np

from hubersed.prospector.utils import load_lines
from hubersed.prospector.lsf import DESI_WAV


WAVE_OBS = DESI_WAV.astype(np.float32)


EM_LINES_A = load_lines()["emission"]["wave_vac"]


# build obs
def build_obs(
    spec: np.ndarray, 
    unc: np.ndarray, 
    mask: np.ndarray, 
    resolution: np.ndarray | None = None, 
    wavelength: np.ndarray = WAVE_OBS
) -> list:
    """
    Build the observation dictionary for the fit.
    This should include at least the spectrum and uncertainty, for DESI data.

    Parameters
    ----------
    spec : np.ndarray
        The observed spectrum. In maggies, make sure to convert it before hand.
    unc : np.ndarray
        The uncertainty on the observed spectrum. In maggies, make sure to convert it before hand.
        Should be sigma, not variance.
    mask: np.ndarray
        A boolean array indicating which pixels to use in the fit. False elements will be ignored
        in the likelihood calculation. This can be used to mask out bad pixels, sky lines, etc.
    resolution: np.ndarray, optional
        Instrumental resolution (sigma) at each wavelength, in km/s (prospect
        Spectrum convention). When given, prospect smooths the model to the DESI
        LSF. Pass C_KMS/(2.355*R(lambda)). Safe because build_sps zeroes the
        library resolution (no `data higher resolution than library` assert).
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


# build sps
def build_sps():
    sps = FastStepBasis()
    # SSPBasis.spectral_resolution = property(
    #     lambda self: np.zeros_like(self.ssp.wavelengths)
    # )
    return sps


# use Cue instead
def build_cue_sps():
    """
    SPS with Cue (Li+24) nebular emulator instead of FSPS+Cloudy lines.
    """
    from prospect.sources import NebStepBasis

    return NebStepBasis()



def mask_spectral_lines(wave_obs, mask, z, line_waves=EM_LINES_A, halfwidth_kms=500.0):
    """Mask spectral lines using a velocity-based window.

    Parameters
    ----------
    wave_obs : np.ndarray
        Observed wavelength array [Å].
    mask : np.ndarray
        Boolean array; True = good pixel, False = masked.
    z : float
        Redshift of the object.
    line_waves : np.ndarray
        Rest-frame vacuum wavelengths of lines to mask [Å].
    halfwidth_kms : float, optional
        Half-width of mask window in km/s. Default is 500 km/s.

    Returns
    -------
    np.ndarray
        Updated boolean mask.
    """
    c_kms = 299792.458
    mask = mask.copy()
    wave_rest = wave_obs / (1.0 + z)

    for line in line_waves:
        dwave = line * halfwidth_kms / c_kms
        mask &= np.abs(wave_rest - line) > dwave

    return mask
