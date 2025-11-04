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
