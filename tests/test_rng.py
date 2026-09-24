"""Prior samplers take a numpy Generator by keyword, so every mock draw can be re-seeded."""

import numpy as np
import pytest

from hubersed.mocks.distributions import (
    sample_clipped_normal,
    sample_log_uniform,
    sample_truncated_normal,
    sample_uniform,
)

# sampler and positional args, as mocks/priors.py calls them
SAMPLERS = [
    (sample_uniform, (0.01, 0.6)),
    (sample_log_uniform, (0.1, 5.0)),
    (sample_clipped_normal, (0.3, 1.0, 0.0, 4.0)),
    (sample_truncated_normal, (0.3, 1.0, 0.0, 4.0)),
]
IDS = [s.__name__ for s, _ in SAMPLERS]


@pytest.mark.parametrize(("sampler", "args"), SAMPLERS, ids=IDS)
def test_rng_is_required(sampler, args):
    """Calling without rng raises, so no draw can fall back to unseeded global state."""
    with pytest.raises(TypeError, match="rng"):
        sampler(*args, size=5)


@pytest.mark.parametrize(("sampler", "args"), SAMPLERS, ids=IDS)
def test_same_seed_same_draws(sampler, args):
    """Two Generators with the same seed give identical draws."""
    a = sampler(*args, size=100, rng=np.random.default_rng(7))
    b = sampler(*args, size=100, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
