"""Convert spectra between maggies and f_lambda.

The two directions use different f_lambda scales, see the Notes of each function.
"""

import astropy.units as u
from astropy.constants import c

C_AA_PER_S = c.to(u.AA / u.s).value

C_CGS = 2.99792458e10  # cm/s
FNU_PER_MAGGIE = 3631e-23  # erg/s/cm^2/Hz


def maggies_to_flambda(wave, maggies):
    """Convert flux in maggies to f_lambda in plain cgs units.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength in Angstrom.
    maggies : np.ndarray
        Flux density in maggies, where 1 maggie is 3631 Jy.

    Returns
    -------
    np.ndarray
        f_lambda in erg/s/cm^2/A, without the DESI factor of 1e-17.

    Notes
    -----
    This is not the inverse of ``flambda_to_maggies``, which expects DESI units of
    1e-17 erg/s/cm^2/A. A round trip scales the flux by 1e-17. The only caller,
    ``make_prospector_noisy_sed.py``, divides by 1e-17 right after the call.
    ``tests/test_conversion.py`` pins this.
    """
    fnu_cgs = (maggies * 3631.0) * 1e-23
    flam = fnu_cgs * C_AA_PER_S / (wave**2)
    return flam


def flambda_to_maggies(wave_A, flambda):
    """Convert DESI f_lambda to flux in maggies.

    Parameters
    ----------
    wave_A : np.ndarray
        Wavelength in Angstrom.
    flambda : np.ndarray
        f_lambda in DESI units of 1e-17 erg/s/cm^2/A.

    Returns
    -------
    np.ndarray
        Flux density in maggies, where 1 maggie is 3631 Jy.
    """
    flambda_cgs = flambda * 1e-17
    return flambda_cgs * (wave_A**2) * 1e-8 / C_CGS / FNU_PER_MAGGIE


def ivar_flambda_to_ivar_maggies(wave_A, ivar_flambda):
    """Convert the inverse variance of DESI f_lambda to inverse variance in maggies.

    Parameters
    ----------
    wave_A : np.ndarray
        Wavelength in Angstrom.
    ivar_flambda : np.ndarray
        Inverse variance in (1e-17 erg/s/cm^2/A)^-2.

    Returns
    -------
    np.ndarray
        Inverse variance in maggies^-2.

    Notes
    -----
    The flux conversion is a multiplication by a factor K per pixel, so the inverse
    variance is divided by K squared.
    """
    K = (wave_A**2) * 1e-8 * 1e-17 / C_CGS / FNU_PER_MAGGIE
    return ivar_flambda / (K**2)
