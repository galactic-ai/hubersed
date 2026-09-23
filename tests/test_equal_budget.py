"""Every galaxy gets the same Powell budget, n_seeds + 1 runs with the same options."""

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
    """Objective that returns the invalid value 1e18 everywhere except the initial point."""
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


def valid_if_first_positive(theta):
    """Objective that is invalid, 1e18, wherever theta[0] is negative."""
    return quadratic(theta) if theta[0] >= 0 else 1e18


def test_invalid_jitter_is_redrawn(powell_starts):
    """An invalid jittered start is drawn again, so every run starts from a new valid point."""
    chi2._map_optimize(valid_if_first_positive, INIT, n_seeds=5, maxfev=MAXFEV)
    starts = [s for s, _ in powell_starts]
    assert len(starts) == 6
    assert all(s[0] >= 0 for s in starts)
    assert len({tuple(s) for s in starts}) == 6


def test_start_without_valid_draw_is_dropped(powell_starts):
    """If no jittered draw is valid, only the initial point is run."""
    chi2._map_optimize(valid_only_at_init, INIT, n_seeds=3, maxfev=MAXFEV, max_tries=20)
    assert len(powell_starts) == 1
    np.testing.assert_array_equal(powell_starts[0][0], INIT)


class StartsChosen(Exception):
    """Raised in place of the first Powell run, to stop once the starts are chosen."""


def test_each_start_is_evaluated_once(monkeypatch):
    """Choosing the starts costs one objective call per candidate when all are valid."""
    calls = []

    def counted(theta):
        calls.append(1)
        return quadratic(theta)

    def stop(*args, **kwargs):
        raise StartsChosen

    monkeypatch.setattr(chi2, "minimize", stop)
    with pytest.raises(StartsChosen):
        chi2._map_optimize(counted, INIT, n_seeds=3, maxfev=MAXFEV)
    assert len(calls) == 4
