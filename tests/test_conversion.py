"""Check hubersed.conversion against astropy units.

Maggies use 3631 Jy directly, because astropy's u.ABflux is 3630.78 Jy and differs by 6e-5.
"""

import astropy.units as u
import numpy as np

from hubersed.conversion import (
    flambda_to_maggies,
    ivar_flambda_to_ivar_maggies,
    maggies_to_flambda,
)

WAVE = np.array([3600.0, 5000.0, 9800.0])  # Angstrom, DESI range
MAGGIES = np.array([1e-9, 3e-8, 2e-7])
FLAM = u.erg / u.s / u.cm**2 / u.AA
TO_JY = u.zero_point_flux(3631 * u.Jy)


def _astropy_flambda(maggies):
    """Convert maggies to f_lambda in erg/s/cm^2/A with astropy."""
    fnu = (maggies * u.mgy).to(u.Jy, TO_JY)
    return fnu.to(FLAM, u.spectral_density(WAVE * u.AA)).value


def test_maggies_to_flambda_is_cgs():
    """maggies_to_flambda returns f_lambda in plain cgs units, without the 1e-17 factor."""
    np.testing.assert_allclose(maggies_to_flambda(WAVE, MAGGIES), _astropy_flambda(MAGGIES))


def test_flambda_to_maggies_takes_desi_units():
    """flambda_to_maggies expects f_lambda in DESI units of 1e-17 erg/s/cm^2/A."""
    desi = _astropy_flambda(MAGGIES) / 1e-17
    np.testing.assert_allclose(flambda_to_maggies(WAVE, desi), MAGGIES)


def test_round_trip_is_off_by_1e_minus_17():
    """The two functions are not inverses. A round trip scales the flux by 1e-17."""
    back = flambda_to_maggies(WAVE, maggies_to_flambda(WAVE, MAGGIES))
    np.testing.assert_allclose(back, 1e-17 * MAGGIES)


def test_ivar_scales_like_flux():
    """Inverse variance in maggies equals 1 / sigma**2 with sigma converted like a flux."""
    sigma = np.array([0.5, 1.0, 2.0])  # 1e-17 erg/s/cm^2/A
    expected = 1.0 / flambda_to_maggies(WAVE, sigma) ** 2
    np.testing.assert_allclose(ivar_flambda_to_ivar_maggies(WAVE, 1.0 / sigma**2), expected)
