from astropy.cosmology import Planck18 as cosmo
import numpy as np
import astropy.units as u

# From Leja et al. 2019 (Non parameteric models paper)
BASE_EDGES_GYR = np.array([0, 0.03, 0.10, 0.33, 1.10, 3.60, 11.70, 13.80])

def universe_age_gyr(z):
    """Get the age of the universe in Gyr at redshift z.

    Parameters
    ----------
    z : float
        Redshift

    Returns
    -------
    np.array
        Age of universe.
    """
    return cosmo.age(z).to_value(u.Gyr)

def make_agebins_for_z(z):
    """Make age bins for continuity SFH model at redshift z.

    Parameters
    ----------
    z : float
        Redshift
    
    Returns
    -------
    edges : np.array
        Array of bin edges in Gyr.
    mids : np.array
        Array of bin midpoints in Gyr.
    dt_yr : np.array
        Array of bin widths in years.
    """
    # Universe age at this z (in Gyr)
    Tuz = universe_age_gyr(z)

    # Keep lookback bins but cap the max lookback at Tuz
    edges = BASE_EDGES_GYR.copy()
    edges[-1] = min(edges[-1], Tuz)

    # Ensure monotonic & at least 2 edges
    edges = np.unique(edges)

    if len(edges) < 2:
        raise ValueError("Universe age at this z is smaller than first bin edge.")
    
    dt_gyr = np.diff(edges)

    # Drop any zero-width tail bins (can happen if Tuz cuts through first/last edge)
    keep = dt_gyr > 0
    edges = edges[np.concatenate([keep, [True]])]

    dt_gyr = np.diff(edges)
    mids = 0.5*(edges[1:] + edges[:-1])
    dt_yr = dt_gyr * 1e9

    return edges, mids, dt_yr

def make_stochastic_agebins(z):
    """ready to use for fsps or be in the dictionary.
    Make age bins for stochastic SFH model at redshift z.
    
    Parameters
    ----------
    z : float
        Redshift
    
    Returns
    -------
    age_bins_log : np.array
        Array of shape (10, 2) with log10(yr) bin edges.
    """
    t_univ = universe_age_gyr(z)
    # each bin should be in Gyr, shape (n, 2) (start, end)
    age_bins = np.zeros((10, 2))

    age_bins[0] = [0.001, 0.005]
    age_bins[1] = [0.005, 0.01]
    log_t_edges = np.log10(np.linspace(0.01, 0.95*t_univ, 9))
    for i in range(2, 10):
        age_bins[i] = [10**log_t_edges[i-2], 10**log_t_edges[i-1]]

    # convert age bins to log(yr)
    age_bins_log = np.log10(age_bins * 1e9)

    return age_bins_log