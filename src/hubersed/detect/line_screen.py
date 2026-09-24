"""Classify emission lines on the BPT diagram and read the Gaia on-source screen.

The screen itself is run by ``scripts/agn_star_screen.py``.
"""

import csv

import numpy as np

SNMIN_BPT = 5.0  # sigma, per line, to attempt a BPT classification
SNMIN_FORB = 3.0  # sigma, above which a forbidden line counts as present
SEP_STAR = 0.5  # arcsec, the Gaia source must be inside the fibre core
NSPECIES_MIN = 2  # independent forbidden species needed to call it an emission-line galaxy

SINGLETS = ["OIII_5007", "NII_6584", "OI_6300"]
DOUBLETS = [("SII_6716", "SII_6731"), ("OII_3726", "OII_3729")]


def snr(d, i, line):
    """Return flux times sqrt(ivar) for one line of row ``i``, or 0 if the ivar is not positive."""
    f = float(d[f"{line}_FLUX"][i])
    iv = float(d[f"{line}_FLUX_IVAR"][i])
    return f * np.sqrt(iv) if iv > 0 else 0.0


def forbidden(d, i):
    """Return the highest forbidden-line S/N and the number of species above SNMIN_FORB.

    Each doublet counts as one species and scores with the lower S/N of its two lines.

    Parameters
    ----------
    d : astropy.io.fits.FITS_rec
        Fastspec table with ``<line>_FLUX`` and ``<line>_FLUX_IVAR`` columns.
    i : int
        Row index into ``d``.

    Returns
    -------
    max_snr : float
        Highest S/N over the singlets and doublets.
    n_species : int
        Number of singlets and doublets with S/N above SNMIN_FORB.
    """
    s = [snr(d, i, ln) for ln in SINGLETS]
    s += [min(snr(d, i, a), snr(d, i, b)) for a, b in DOUBLETS]
    return max(s), sum(x > SNMIN_FORB for x in s)


def bpt(d, i):
    """Classify row ``i`` on the BPT diagram using the demarcation curves in the code.

    A row is AGN if log [OIII]/Hb lies above the first [NII] curve or above the [SII]
    curve, composite if it lies above the second [NII] curve, and star-forming otherwise.

    Parameters
    ----------
    d : astropy.io.fits.FITS_rec
        Fastspec table with line flux and ivar columns.
    i : int
        Row index into ``d``.

    Returns
    -------
    cls : str or None
        "AGN", "composite" or "star-forming". None if Halpha, Hbeta, [OIII] 5007 or
        [NII] 6584 has S/N at or below SNMIN_BPT or a flux at or below 0.
    log_n2_ha : float
        log10 of [NII] 6584 over Halpha, or NaN when ``cls`` is None.
    log_o3_hb : float
        log10 of [OIII] 5007 over Hbeta, or NaN when ``cls`` is None.
    log_s2_ha : float
        log10 of the summed [SII] doublet over Halpha, or NaN if that sum is not positive.
    """
    need = ["HALPHA", "HBETA", "OIII_5007", "NII_6584"]
    if any(snr(d, i, ln) <= SNMIN_BPT for ln in need):
        return None, np.nan, np.nan, np.nan
    ha, hb = float(d["HALPHA_FLUX"][i]), float(d["HBETA_FLUX"][i])
    o3, n2f = float(d["OIII_5007_FLUX"][i]), float(d["NII_6584_FLUX"][i])
    s2f = float(d["SII_6716_FLUX"][i]) + float(d["SII_6731_FLUX"][i])
    if min(ha, hb, o3, n2f) <= 0:
        return None, np.nan, np.nan, np.nan
    n2, o3r = np.log10(n2f / ha), np.log10(o3 / hb)
    s2 = np.log10(s2f / ha) if s2f > 0 else np.nan

    ke_n = 0.61 / (n2 - 0.47) + 1.19 if n2 < 0.47 else -np.inf
    ke_s = 0.72 / (s2 - 0.32) + 1.30 if np.isfinite(s2) and s2 < 0.32 else -np.inf
    ka_n = 0.61 / (n2 - 0.05) + 1.30 if n2 < 0.05 else -np.inf

    if o3r > ke_n or (np.isfinite(s2) and o3r > ke_s):
        return "AGN", n2, o3r, s2
    if o3r > ka_n:
        return "composite", n2, o3r, s2
    return "star-forming", n2, o3r, s2


def read_gaia_screen(path):
    """Read a Gaia screen CSV into rows keyed by target id.

    The on-source star rule needs the onsource_star and sep_arcsec columns that
    contam_screens writes. A file without them would make every target look off-source,
    so it is refused.

    Parameters
    ----------
    path : str or pathlib.Path
        Gaia screen CSV with a target_id column.

    Returns
    -------
    dict
        One CSV row, as a dict of strings, per integer target id.

    Raises
    ------
    ValueError
        If the file lacks onsource_star or sep_arcsec.
    """
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in ("onsource_star", "sep_arcsec") if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{path} has no {', '.join(missing)} column. "
                "Write it with hubersed.detect.sky_screens first."
            )
        return {int(r["target_id"]): r for r in reader}
