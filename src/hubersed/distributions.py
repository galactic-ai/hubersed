import numpy as np
import scipy.stats


# uniform sampling
def sample_uniform(low, high, size=1, *, rng):
    """
    Sample from a uniform distribution between [low, high).
    """
    return rng.uniform(low, high, size)


# log uniform sampling
def sample_log_uniform(low, high, size=1, *, rng):
    """
    Log-uniform on [low, high). Both bounds must be > 0.
    """
    log_low = np.log10(low)
    log_high = np.log10(high)
    return 10 ** sample_uniform(log_low, log_high, size, rng=rng)


# clipped normal sampling
def sample_clipped_normal(mean, std, low, high, size=1, *, rng):
    """
    Normal(mean, std) hard-clipped to [low, high].
    """
    samples = rng.normal(mean, std, size)
    samples = np.clip(samples, low, high)
    return samples


# truncated normal sampling -- NOT WIRED IN, see docstring
def sample_truncated_normal(mean, std, low, high, size=1, *, rng):
    """
    Truncated normal on [low, high] with mean and std. Uses scipy.stats.truncnorm.rvs.
    """
    a = (low - mean) / std
    b = (high - mean) / std
    return scipy.stats.truncnorm.rvs(
        a, b, loc=mean, scale=std, size=size, random_state=rng
    )
