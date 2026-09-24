"""trapz_rebin conserves flux, keeps a flat input flat, and refuses bins outside its input."""

import numpy as np
import pytest

from hubersed.sps import rebin

# DESI-like pixels and a coarser constant-velocity grid that does not line up with them
X = np.linspace(3600.0, 9824.0, 7781)
Y = 1.0 + np.random.default_rng(0).random(X.size)
EDGES = np.concatenate(([X[0]], rebin.common_obs_edges(3700.0, 9700.0)[1:-1], [X[-1]]))


def test_trapz_rebin_conserves_flux():
    """The rebinned flux summed over bin widths equals the integral of the input."""
    out = rebin.trapz_rebin(X, Y, edges=EDGES)
    np.testing.assert_allclose(np.sum(out * np.diff(EDGES)), np.trapezoid(Y, X), rtol=1e-12)


def test_constant_density_stays_constant():
    """A flat input comes out flat, because the output is a density and not a sum."""
    out = rebin.trapz_rebin(X, np.ones_like(X), edges=EDGES)
    np.testing.assert_allclose(out, 1.0, rtol=1e-12)


def test_edges_outside_input_raise():
    """Bins that reach past the input wavelengths are refused."""
    with pytest.raises(ValueError, match="within input x range"):
        rebin.trapz_rebin(X, Y, edges=[X[0] - 1.0, X[1]])
