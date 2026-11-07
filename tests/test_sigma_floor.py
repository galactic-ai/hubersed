"""The sigma_smooth prior floor: library_sigma_floor and the sigma_floor argument."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.fsps  # hubersed.sps.config imports prospect, which reads SPS_HOME

C_KMS = 2.998e5


def miles_like():
    """Return an SPS stand-in with a constant 2.54 A FWHM, flagged negative outside 3525-7500 A."""
    wave = np.arange(3000.0, 9000.0, 0.9)
    res = C_KMS * (2.54 / 2.355) / wave
    res[(wave < 3525.0) | (wave > 7500.0)] *= -1
    return SimpleNamespace(wavelengths=wave, spectral_resolution=res)


def test_floor_is_the_bluest_resolution_for_constant_fwhm():
    """For a constant FWHM the floor is the resolution at the bluest fitted wavelength."""
    from hubersed.sps.config import library_sigma_floor

    floor = library_sigma_floor(miles_like(), 3601.8, 7400.8)
    assert np.isclose(floor, C_KMS * (2.54 / 2.355) / 3601.8, rtol=1e-3)
    assert 89.0 < floor < 90.5


def test_floor_for_constant_resolving_power():
    """For a constant R the floor is c / (2.355 R), here C3K_HR at R = 3000."""
    from hubersed.sps.config import library_sigma_floor

    wave = np.exp(np.arange(np.log(3001.0), np.log(10000.0), 50.0 / C_KMS))
    sps = SimpleNamespace(wavelengths=wave, spectral_resolution=C_KMS / (2.355 * 3000.0))
    assert np.isclose(library_sigma_floor(sps, 3320.0, 9000.0), 42.43, atol=0.01)


def test_floor_outside_the_library_raises():
    """A range with no library wavelengths is refused."""
    from hubersed.sps.config import library_sigma_floor

    with pytest.raises(ValueError, match="no library wavelengths"):
        library_sigma_floor(miles_like(), 20000.0, 21000.0)


def test_sigma_floor_sets_the_prior_and_default_is_unchanged():
    """The default keeps TopHat(10, 400); sigma_floor moves only the lower bound."""
    from hubersed.sps.config import build_continuum_model

    _, template = build_continuum_model(0.0253)
    assert template["sigma_smooth"]["prior"].params["mini"] == 10.0
    _, template = build_continuum_model(0.0253, sigma_floor=12.7)
    assert template["sigma_smooth"]["prior"].params["mini"] == 12.7
    assert template["sigma_smooth"]["prior"].params["maxi"] == 400.0
