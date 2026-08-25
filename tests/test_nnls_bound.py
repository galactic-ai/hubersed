"""Unit tests for the NNLS solve that backs the 'best possible SFH' bound.

The bound in ``bin/prospector/fsps_tabular_sfh.py --nnls`` rests on two things:

  1. the map SFH -> observed spectrum is non-negative linear, and
  2. ``nnls_chi2`` really returns the GLOBAL minimum of the inverse-variance weighted
     chi2 over the non-negative orthant.

(1) can only be checked with FSPS in the loop and is tested at runtime by CHECK 1/2/3
inside ``nnls_bound``. (2) is pure linear algebra and is tested here, so that a
regression in the weighting or in the rnorm -> chi2 conversion fails loudly instead of
silently turning a bound into a number.

Run with ``uv run pytest tests/test_nnls_bound.py`` (never ``uvx``).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# repo-root-relative via __file__, NOT cwd-relative. Deliberately not the
# sys.path.insert(0, "bin/prospector") pattern that CLAUDE.md forbids adding to.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin" / "prospector"))

from fsps_tabular_sfh import nnls_chi2  # noqa: E402


def _problem(seed=0, n=400, k=20):
    """A well-posed non-negative linear problem: A >= 0, heteroscedastic sigma."""
    rng = np.random.default_rng(seed)
    A = np.abs(rng.normal(size=(n, k))) + 0.1
    sigma = rng.uniform(0.5, 2.0, n)
    x_true = np.abs(rng.normal(size=k))
    return rng, A, sigma, x_true


def test_exact_when_truth_is_in_the_cone():
    """Noiseless data generated from a non-negative x must be recovered with chi2 = 0.

    This is the property that makes a large chi2 meaningful: if the solver could not
    reach zero here, a large chi2 on real data would say nothing about the model.
    """
    _, A, sigma, x_true = _problem()
    x, chi2 = nnls_chi2(A, A @ x_true, sigma)
    assert chi2 < 1e-18, f"chi2 = {chi2:.3e}, expected ~0"
    assert np.allclose(x, x_true, atol=1e-10)


def test_chi2_equals_the_explicit_weighted_sum():
    """rnorm**2 from the whitened system must equal sum(((b - A x)/sigma)**2) exactly.

    Guards the 1/sigma row scaling. Forgetting it, or applying 1/sigma**2, still returns
    a plausible-looking number, so only this identity catches it.
    """
    rng, A, sigma, x_true = _problem()
    b = A @ x_true + rng.normal(0, sigma)
    x, chi2 = nnls_chi2(A, b, sigma)
    explicit = float(np.sum(((b - A @ x) / sigma) ** 2))
    assert chi2 == pytest.approx(explicit, rel=1e-12)


def test_it_is_the_global_optimum_not_a_local_one():
    """No random non-negative perturbation may beat the NNLS solution.

    NNLS is convex with a unique minimum value, so this must hold. It is the single
    assumption the word 'bound' depends on.
    """
    rng, A, sigma, x_true = _problem()
    b = A @ x_true + rng.normal(0, sigma)
    x, chi2 = nnls_chi2(A, b, sigma)
    for _ in range(2000):
        xp = np.abs(x + rng.normal(0, 0.05, x.size))
        assert float(np.sum(((b - A @ xp) / sigma) ** 2)) >= chi2 - 1e-9


def test_negativity_is_actually_clipped():
    """A truth with a negative component is unreachable, so chi2 must be large and x >= 0.

    An unconstrained lstsq would sail through this with chi2 = 0.
    """
    _, A, sigma, x_true = _problem()
    x_neg = x_true.copy()
    x_neg[3] = -5.0
    x, chi2 = nnls_chi2(A, A @ x_neg, sigma)
    assert np.all(x >= 0)
    assert chi2 > 1.0, f"chi2 = {chi2:.3e}; a negative-truth target should not be fittable"


def test_downweighting_the_worst_pixels_lowers_chi2():
    """sigma has to be doing something. Inflating it on the worst-fitting rows must help."""
    rng, A, sigma, x_true = _problem()
    b = A @ x_true + rng.normal(0, sigma)
    x, chi2 = nnls_chi2(A, b, sigma)
    worst = np.argsort(np.abs(b - A @ x))[-40:]
    sigma2 = sigma.copy()
    sigma2[worst] *= 100
    _, chi2_dw = nnls_chi2(A, b, sigma2)
    assert chi2_dw < chi2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(A=np.ones((4, 2)), b=np.ones(3), sigma=np.ones(4)), "row count"),
        (dict(A=np.ones((4, 2)), b=np.ones(4), sigma=np.zeros(4)), "positive"),
        (dict(A=np.ones((4, 2)), b=np.array([1.0, np.nan, 1, 1]), sigma=np.ones(4)), "finite"),
        (dict(A=np.ones(4), b=np.ones(4), sigma=np.ones(4)), "2-D"),
    ],
)
def test_bad_input_raises(kwargs, match):
    """Silent garbage-in is how a wrong bound gets published. Fail at the door."""
    with pytest.raises(AssertionError, match=match):
        nnls_chi2(**kwargs)
