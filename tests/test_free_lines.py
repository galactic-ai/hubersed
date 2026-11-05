"""Analytic amplitude marginalization and ion lookup of the free-line model."""

import numpy as np
import pytest

pytestmark = pytest.mark.fsps  # free_lines imports prospect, which reads SPS_HOME files on import


def _toy(seed=0):
    """Return pixels, two overlapping templates, noise and data with known amplitudes."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-5, 5, 40)
    T = np.column_stack([np.exp(-0.5 * x**2), np.exp(-0.5 * (x / 2.5) ** 2) / 2.5])
    ivar = np.full(x.size, 4.0)
    resid = T @ np.array([1.3, 0.7]) + rng.normal(0, 0.5, x.size)
    return T, ivar, resid


def _lnlike(resid, ivar, T, a):
    """Return the Gaussian log likelihood, without the constant normalization."""
    r = resid - T @ a
    return -0.5 * np.sum(ivar * r**2)


def test_marginal_matches_numeric_integral():
    """One flat and one Gaussian amplitude: the analytic marginal equals a grid integral."""
    from hubersed.sps.free_lines import marginalize_amplitudes

    T, ivar, resid = _toy()
    mean, prec = np.array([0.0, 1.0]), np.array([0.0, 1 / 0.3**2])
    a, cov, ln_extra = marginalize_amplitudes(resid, ivar, T, mean, prec)
    analytic = _lnlike(resid, ivar, T, a) + ln_extra

    g0 = np.linspace(a[0] - 8 * np.sqrt(cov[0, 0]), a[0] + 8 * np.sqrt(cov[0, 0]), 401)
    g1 = np.linspace(a[1] - 8 * np.sqrt(cov[1, 1]), a[1] + 8 * np.sqrt(cov[1, 1]), 401)
    A0, A1 = np.meshgrid(g0, g1, indexing="ij")
    amps = np.stack([A0.ravel(), A1.ravel()], axis=1)
    lnl = -0.5 * np.sum(ivar * (resid[None, :] - amps @ T.T) ** 2, axis=1)
    lnprior = -0.5 * prec[1] * (amps[:, 1] - mean[1]) ** 2 + 0.5 * np.log(prec[1] / (2 * np.pi))
    f = np.exp(lnl + lnprior - analytic).reshape(A0.shape)
    numeric = analytic + np.log(np.trapezoid(np.trapezoid(f, g1, axis=1), g0))
    assert numeric == pytest.approx(analytic, abs=1e-6)


def test_tight_prior_gives_fixed_amplitude():
    """A very narrow Gaussian prior holds the amplitude at its mean, with no extra term."""
    from hubersed.sps.free_lines import marginalize_amplitudes

    T, ivar, resid = _toy()
    mean, prec = np.array([0.4, 1.0]), np.array([1e14, 1e14])
    a, _, ln_extra = marginalize_amplitudes(resid, ivar, T, mean, prec)
    assert a == pytest.approx(mean, abs=1e-6)
    total = _lnlike(resid, ivar, T, a) + ln_extra
    assert total == pytest.approx(_lnlike(resid, ivar, T, mean), abs=1e-4)


def test_flat_amplitudes_are_least_squares():
    """With flat priors the amplitudes are the weighted least-squares solution."""
    from hubersed.sps.free_lines import marginalize_amplitudes

    T, ivar, resid = _toy()
    a, _, _ = marginalize_amplitudes(resid, ivar, T, np.zeros(2), np.zeros(2))
    w = np.sqrt(ivar)
    expect = np.linalg.lstsq(T * w[:, None], resid * w, rcond=None)[0]
    assert a == pytest.approx(expect, rel=1e-10)


def test_no_data_and_no_prior_raises():
    """A flat amplitude whose template is zero on every fitted pixel cannot be marginalized."""
    from hubersed.sps.free_lines import marginalize_amplitudes

    T, ivar, resid = _toy()
    T[:, 1] = 0.0
    with pytest.raises(np.linalg.LinAlgError):
        marginalize_amplitudes(resid, ivar, T, np.zeros(2), np.zeros(2))


def test_ion_lines():
    """Balmer lines are found by prefix, and a missing ion line raises."""
    from hubersed.sps.free_lines import IONS, ion_lines

    names = ["Ba-beta 4861", "Ba-alpha 6563"]
    names += [n for spec in IONS.values() if spec["lines"] for n in spec["lines"]]
    idx = ion_lines(names)
    assert idx["H"].tolist() == [0, 1]
    assert [names[i] for i in idx["[OIII]"]] == ["[O III] 4959", "[O III] 5007"]
    with pytest.raises(ValueError, match=r"\[O I\] 6363"):
        ion_lines([n for n in names if n != "[O I] 6363"])
