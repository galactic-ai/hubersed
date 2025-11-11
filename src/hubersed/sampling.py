import numpy as np

rng = np.random.default_rng(42)

# uniform sampling
def sample_uniform(low, high, size=1):
    return rng.uniform(low, high, size)

# log uniform sampling
def sample_log_uniform(low, high, size=1):
    log_low = np.log10(low)
    log_high = np.log10(high)
    return 10 ** sample_uniform(log_low, log_high, size)

# clipped normal sampling
def sample_clipped_normal(mean, std, low, high, size=1):
    samples = rng.normal(mean, std, size)
    samples = np.clip(samples, low, high)
    return samples