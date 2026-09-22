"""_map_optimize gives every galaxy the same Powell budget: n_seeds + 1 runs, same options."""

import numpy as np
import pytest
from scipy.optimize import minimize

from hubersed.fitting import chi2

INIT = np.zeros(3)
MAXFEV = 200


def quadratic(theta):
    """Smooth objective, valid everywhere, minimum at theta = 1."""
    return float(np.sum((theta - 1.0) ** 2))


def valid_only_at_init(theta):
    """Objective whose every jittered start is invalid (the 1e18 sentinel)."""
    return quadratic(theta) if np.array_equal(theta, INIT) else 1e18


@pytest.fixture
def powell_starts(monkeypatch):
    """Record the start point and options of every Powell run."""
    calls = []

    def spy(fun, x0, method=None, options=None):
        calls.append((np.array(x0), options))
        return minimize(fun, x0, method=method, options=options)

    monkeypatch.setattr(chi2, "minimize", spy)
    return calls


@pytest.mark.parametrize("n_seeds", [1, 3, 5])
def test_one_run_per_start_same_options(powell_starts, n_seeds):
    """Init plus n_seeds jittered starts, each with the same maxfev and maxiter."""
    chi2._map_optimize(quadratic, INIT, n_seeds=n_seeds, maxfev=MAXFEV)
    assert len(powell_starts) == n_seeds + 1
    np.testing.assert_array_equal(powell_starts[0][0], INIT)
    for _, options in powell_starts:
        assert options == {"maxiter": MAXFEV // 10, "maxfev": MAXFEV, "ftol": 1e-6}


def test_invalid_jitter_reruns_init(powell_starts):
    """Pin how the code works today, which we plan to change.

    If a jittered start is invalid, Powell runs again from the initial point.
    """
    chi2._map_optimize(valid_only_at_init, INIT, n_seeds=3, maxfev=MAXFEV)
    assert len(powell_starts) == 4
    for start, _ in powell_starts:
        np.testing.assert_array_equal(start, INIT)
