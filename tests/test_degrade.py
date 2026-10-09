"""Resolution matching: MILES path unchanged, C3K_HR path smooths only where DESI is sharper."""

import numpy as np

from hubersed.sps import rebin

C = rebin.C_KMS
W = np.linspace(3600.0, 9824.0, 7781)
Z = 0.025


def _desi_sigma():
    """Return a DESI-like LSF sigma in km/s: 56 at the blue end, falling to 24 at the red end."""
    return np.interp(W, [3600, 5000, 6000, 9824], [56.0, 42.0, 32.0, 24.0])


def test_miles_matches_legacy_wrapper():
    """Check degrade_to_miles returns the same arrays as degrade_to_library for MILES."""
    rng = np.random.default_rng(0)
    f, iv = 1 + rng.normal(0, 0.1, W.size), np.full(W.size, 100.0)
    a = rebin.degrade_to_miles(W, f, iv, Z)
    b = rebin.degrade_to_library(W, f, iv, Z, library="miles")
    for x, y in zip(a, b, strict=False):
        np.testing.assert_array_equal(x, y)


def test_c3k_leaves_blue_alone_and_hits_target_width():
    """Check C3K_HR skips pixels where DESI is coarser and brings a line to the library width."""
    sig = _desi_sigma()
    k, good = rebin.match_kernel_sigma_A(W, Z, "c3k_hr", desi_sigma_kms=sig)
    lib = C / (2.355 * 3000)
    assert np.all(k[sig >= lib] == 0) and good[W / (1 + Z) < 9000].all()
    # a narrow line at 6729 A must come out at the library width
    lam0 = 6729.0
    s_in = np.interp(lam0, W, sig) / C * lam0
    f = np.exp(-0.5 * ((W - lam0) / s_in) ** 2)
    fd, ivd, _ = rebin.degrade_to_library(W, f, np.ones_like(W), Z, "c3k_hr", sig, keep_ivar=True)
    sel = np.abs(W - lam0) < 10
    width = np.sqrt(np.sum((W[sel] - lam0) ** 2 * fd[sel]) / fd[sel].sum()) / lam0 * C
    assert abs(width - lib) < 1.5
    np.testing.assert_array_equal(ivd[good], 1.0)  # keep_ivar keeps the input errors


def test_keep_ivar_gives_calibrated_errors():
    """Check keep_ivar gives the right error on the weighted mean of a flat noisy spectrum."""
    rng = np.random.default_rng(1)
    sel = (W > 6000) & (W < 7400)
    w, sig = W[sel], _desi_sigma()[sel]
    est, stated = [], []
    for _ in range(200):
        y = 1 + rng.normal(0, 1, w.size)
        fd, ivd, g = rebin.degrade_to_library(
            w, y, np.ones(w.size), Z, "c3k_hr", sig, keep_ivar=True
        )
        est.append(np.sum(fd * ivd) / ivd.sum())
        stated.append(1 / np.sqrt(ivd.sum()))
    assert 0.85 < np.std(est) / np.mean(stated) < 1.15
