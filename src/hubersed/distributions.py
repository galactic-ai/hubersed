"""Draw prior samples for mock galaxies. Every sampler needs a numpy Generator."""

import numpy as np
import scipy.stats


def sample_uniform(low, high, size=1, *, rng):
    """Draw from a uniform distribution on [low, high).

    Parameters
    ----------
    low, high : float or np.ndarray
        Lower and upper bounds.
    size : int or tuple of int
        Output shape.
    rng : np.random.Generator
        Source of randomness. Required, so every draw can be repeated from its seed.

    Returns
    -------
    np.ndarray
        The samples.
    """
    return rng.uniform(low, high, size)


def sample_log_uniform(low, high, size=1, *, rng):
    """Draw values whose log10 is uniform between log10(low) and log10(high).

    Parameters
    ----------
    low, high : float
        Bounds, both greater than zero.
    size : int or tuple of int
        Output shape.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    np.ndarray
        The samples.
    """
    log_low = np.log10(low)
    log_high = np.log10(high)
    return 10 ** sample_uniform(log_low, log_high, size, rng=rng)


def sample_clipped_normal(mean, std, low, high, size=1, *, rng):
    """Draw from a normal distribution and clip the samples to [low, high].

    Samples beyond a bound are set to that bound, so the bounds get extra weight.
    Use ``sample_truncated_normal`` to redraw them instead.

    Parameters
    ----------
    mean, std : float
        Mean and standard deviation of the normal distribution.
    low, high : float
        Clipping bounds.
    size : int or tuple of int
        Output shape.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    np.ndarray
        The samples.
    """
    samples = rng.normal(mean, std, size)
    samples = np.clip(samples, low, high)
    return samples


def sample_truncated_normal(mean, std, low, high, size=1, *, rng):
    """Draw from a normal distribution truncated to [low, high].

    Parameters
    ----------
    mean, std : float
        Mean and standard deviation of the normal distribution before truncation.
    low, high : float
        Truncation bounds.
    size : int or tuple of int
        Output shape.
    rng : np.random.Generator
        Source of randomness, passed to ``scipy.stats.truncnorm.rvs``.

    Returns
    -------
    np.ndarray
        The samples.
    """
    a = (low - mean) / std
    b = (high - mean) / std
    return scipy.stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size, random_state=rng)
