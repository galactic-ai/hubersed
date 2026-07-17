import numpy as np

# uniform sampling
def sample_uniform(low, high, size=1, *, rng):
    """Sample from a uniform distribution between [low, high)."""
    return rng.uniform(low, high, size)


# log uniform sampling
def sample_log_uniform(low, high, size=1, *, rng):
    log_low = np.log10(low)
    log_high = np.log10(high)
    return 10 ** sample_uniform(log_low, log_high, size, rng=rng)


# clipped normal sampling
def sample_clipped_normal(mean, std, low, high, size=1, *, rng):
    samples = rng.normal(mean, std, size)
    samples = np.clip(samples, low, high)
    return samples
