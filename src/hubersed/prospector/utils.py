from astropy.cosmology import Planck18 as cosmo
import numpy as np
import astropy.units as u

__all__ = [
    "universe_age_gyr",
    "make_agebins_for_z",
    "make_stochastic_agebins",
    "load_lines",
]

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
    mids = 0.5 * (edges[1:] + edges[:-1])
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
    log_t_edges = np.geomspace(0.01, 0.95 * t_univ, 9)  # 9 edges to make 8 bins
    for i in range(2, 10):
        age_bins[i] = [log_t_edges[i - 2], log_t_edges[i - 1]]

    # convert age bins to log(yr)
    age_bins_log = np.log10(age_bins * 1e9)

    return age_bins_log


def _airtovac(w):
    # From https://github.com/desihub/prospect/blob/1694e3f2eb35e33778f9ab73dc535719f45a959b/py/prospect/viewer/cds.py#L34
    """Convert air wavelengths to vacuum wavelengths. Don't convert less than 2000 Å.

    Parameters
    ----------
    w : :class:`float`
        Wavelength [Å] of the line in air.

    Returns
    -------
    :class:`float`
        Wavelength [Å] of the line in vacuum.
    """
    if w < 2000.0:
        return w
    vac = w
    for iter in range(2):
        sigma2 = (1.0e4 / vac) * (1.0e4 / vac)
        fact = 1.0 + 5.792105e-2 / (238.0185 - sigma2) + 1.67917e-3 / (57.362 - sigma2)
        vac = w * fact
    return vac


def _parse_line_file(filepath):
    """Parse a single prospect-format spectral line CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    np.ndarray
        Structured array with fields: name (U20), longname (U40),
        wave_vac (float64), major (bool).
    """
    names = []
    longnames = []
    wavelengths = []
    majors = []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue

            name = parts[0].strip()
            longname = parts[1].strip()
            wave = float(parts[2].strip())
            is_vacuum = parts[3].strip() == "True"
            is_major = parts[4].strip() == "True"

            # Convert air -> vacuum if needed
            if not is_vacuum:
                wave = float(_airtovac(np.array([wave]))[0])

            names.append(name)
            longnames.append(longname)
            wavelengths.append(wave)
            majors.append(is_major)

    n = len(names)
    dtype = np.dtype(
        [
            ("name", "U20"),
            ("longname", "U40"),
            ("wave_vac", "f8"),
            ("major", "?"),
        ]
    )
    result = np.empty(n, dtype=dtype)
    result["name"] = names
    result["longname"] = longnames
    result["wave_vac"] = wavelengths
    result["major"] = majors

    return result


def load_lines():
    from pathlib import Path
    import numpy as np

    ab_lines_path = Path(__file__).parent / "data" / "absorption_lines.txt"
    em_lines_path = Path(__file__).parent / "data" / "emission_lines.txt"

    ab_lines = _parse_line_file(ab_lines_path)
    em_lines = _parse_line_file(em_lines_path)

    all_waves = np.unique(np.concatenate([ab_lines["wave_vac"], em_lines["wave_vac"]]))

    major_ab = ab_lines["wave_vac"][ab_lines["major"]]
    major_em = em_lines["wave_vac"][em_lines["major"]]
    major_waves = np.unique(np.concatenate([major_em, major_ab]))

    return {
        "emission": em_lines,
        "absorption": ab_lines,
        "all_waves": all_waves,
        "major_waves": major_waves,
    }
