"""Star formation history age bins, the age of the universe, and spectral line lists."""

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18 as cosmo

__all__ = [
    "universe_age_gyr",
    "make_stochastic_agebins",
    "load_lines",
]


def universe_age_gyr(z):
    """Return the age of the universe at redshift z in the Planck18 cosmology.

    Parameters
    ----------
    z : float or np.ndarray
        Redshift.

    Returns
    -------
    float or np.ndarray
        Age in Gyr.
    """
    return cosmo.age(z).to_value(u.Gyr)


def make_stochastic_agebins(z):
    """Return the 10 lookback-time bins of the stochastic star formation history at redshift z.

    The first two bins are 1 to 5 Myr and 5 to 10 Myr. The other eight are evenly spaced
    in log time from 10 Myr to 0.95 times the age of the universe.

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    np.ndarray
        Bin start and end as log10 of lookback time in years, shape ``(10, 2)``. This is the
        ``agebins`` format prospect expects.
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
    """Convert an air wavelength to vacuum. Wavelengths below 2000 Angstrom are returned as is.

    Copied from desihub/prospect, py/prospect/viewer/cds.py at commit 1694e3f.

    Parameters
    ----------
    w : float
        Wavelength in air, in Angstrom.

    Returns
    -------
    float
        Wavelength in vacuum, in Angstrom.
    """
    if w < 2000.0:
        return w
    vac = w
    for _ in range(2):
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
        Structured array with the fields name, longname, wave_vac in Angstrom, and major.
        Air wavelengths in the file are converted to vacuum.
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
    """Load the emission and absorption line lists shipped in ``sps/data``.

    Returns
    -------
    dict
        ``emission`` and ``absorption`` are the structured arrays from ``_parse_line_file``.
        ``all_waves`` holds every vacuum wavelength and ``major_waves`` only the lines
        marked major, both sorted and in Angstrom.
    """
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
