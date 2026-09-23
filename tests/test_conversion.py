"""Check hubersed.conversion against f_nu = f_lambda * lambda**2 / c written out by hand.

One maggie is 3631 Jy. astropy's u.ABflux is 3630.78 Jy and differs by 6e-5, so it is not used.
"""

import astropy.units as u
import numpy as np
import pytest

from hubersed.conversion import DESI_FLAM, ivar_to_maggies, to_flambda, to_maggies

WAVE = np.array([3600.0, 5000.0, 9800.0])  # Angstrom, DESI range
DESI = np.array([0.5, 3.0, 20.0])  # f_lambda in 1e-17 erg/s/cm^2/A
C_CM_S = 2.99792458e10
MAGGIE_CGS = 3631e-23  # erg/s/cm^2/Hz


def _hand_maggies(desi):
    """Convert DESI f_lambda to maggies with plain numbers."""
    flam_per_cm = desi * 1e-17 * 1e8  # erg/s/cm^2/cm
    fnu = flam_per_cm * (WAVE * 1e-8) ** 2 / C_CM_S
    return fnu / MAGGIE_CGS


def test_to_maggies_matches_hand_formula():
    """DESI f_lambda converts to the maggies given by f_nu = f_lambda * lambda**2 / c."""
    got = to_maggies(WAVE * u.AA, DESI * DESI_FLAM)
    assert got.unit == u.mgy
    np.testing.assert_allclose(got.value, _hand_maggies(DESI), rtol=1e-12)


def test_to_flambda_matches_hand_formula():
    """Maggies convert back to DESI units by default."""
    got = to_flambda(WAVE * u.AA, _hand_maggies(DESI) * u.mgy)
    assert got.unit == DESI_FLAM
    np.testing.assert_allclose(got.value, DESI, rtol=1e-12)


def test_round_trip_is_identity():
    """to_flambda undoes to_maggies."""
    back = to_flambda(WAVE * u.AA, to_maggies(WAVE * u.AA, DESI * DESI_FLAM))
    np.testing.assert_allclose(back.value, DESI, rtol=1e-14)


@pytest.mark.parametrize("unit", [DESI_FLAM, u.erg / u.s / u.cm**2 / u.AA])
def test_ivar_scales_like_flux(unit):
    """Inverse variance in maggies equals 1 / sigma**2 with sigma converted like a flux."""
    sigma = DESI * DESI_FLAM
    expected = 1.0 / to_maggies(WAVE * u.AA, sigma) ** 2
    got = ivar_to_maggies(WAVE * u.AA, (1.0 / sigma**2).to(unit**-2))
    assert got.unit == u.mgy**-2
    np.testing.assert_allclose(got.value, expected.value, rtol=1e-12)


def test_wrong_units_raise():
    """f_nu passed as f_lambda, or a bare array, is refused."""
    with pytest.raises(u.UnitsError):
        to_maggies(WAVE * u.AA, DESI * u.Jy)
    with pytest.raises(TypeError):
        to_maggies(WAVE * u.AA, DESI)
