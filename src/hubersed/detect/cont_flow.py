"""Load continuum-flow scores, classify sources and read screen CSVs.

The sample itself is built by ``scripts/build_cont_outlier_sample.py``.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

TAGS = ("cont10latent", "cont15latent")


def flow_scores(tag, flow_dir):
    """Load the DESI flow scores for one tag.

    Parameters
    ----------
    tag : str
        Latent tag in the file name ``desi_outliers_flow_nsf_<tag>_snr3.pt``.
    flow_dir : str or Path
        Directory holding that file.

    Returns
    -------
    lp : dict
        Log probability keyed by TARGETID.
    pct : dict
        Rank percentile of the log probability among all DESI galaxies, keyed by TARGETID.
        Low values are the least likely galaxies.
    outliers : set of int
        TARGETIDs the flow flagged as outliers.
    threshold : float
        Log probability threshold used for the outliers.
    """
    d = torch.load(Path(flow_dir) / f"desi_outliers_flow_nsf_{tag}_snr3.pt", weights_only=False)
    tid = np.asarray(d["desi_target_ids"], np.int64)
    lp = np.asarray(d["log_p_desi"], np.float64)
    pct = rankdata(lp, "average") / len(lp)
    return (
        dict(zip(tid.tolist(), lp, strict=True)),
        dict(zip(tid.tolist(), pct, strict=True)),
        set(int(x) for x in d["outlier_target_ids"]),
        float(d["threshold"]),
    )


LAM_MAX = 9824.0  # DESI red-arm cutoff; Halpha 6563 leaves it at z = 0.497


def source_class(S, i, z):
    """Classify one galaxy as emission, weak emission or continuum from its line fits.

    When the observed Halpha wavelength is below LAM_MAX the class uses the Halpha S/N and
    equivalent width. Otherwise it uses the strongest of Hbeta, [OIII]5007 and the weaker
    [OII] line, together with how many of five line groups have S/N above 3.

    Parameters
    ----------
    S : astropy.io.fits.FITS_rec
        The FASTSPEC table.
    i : int
        Row of the galaxy in S.
    z : float
        Redshift used to place Halpha.

    Returns
    -------
    str
        ``"emission"``, ``"weak-em"`` or ``"continuum"``.
    """

    def snr(ln):
        """Return the S/N of line ln in row i, or 0 when its inverse variance is not positive."""
        f, iv = float(S[f"{ln}_FLUX"][i]), float(S[f"{ln}_FLUX_IVAR"][i])
        return f * np.sqrt(iv) if iv > 0 else 0.0

    if 6563.0 * (1 + z) < LAM_MAX:
        ew, s_ha = float(S["HALPHA_EW"][i]), snr("HALPHA")
        if s_ha > 5 and ew > 10:
            return "emission"
        return "weak-em" if s_ha > 5 and ew > 3 else "continuum"

    strong = max(snr("HBETA"), snr("OIII_5007"), min(snr("OII_3726"), snr("OII_3729")))
    nspec = sum(
        x > 3
        for x in [
            snr("OIII_5007"),
            snr("NII_6584"),
            snr("OI_6300"),
            min(snr("SII_6716"), snr("SII_6731")),
            min(snr("OII_3726"), snr("OII_3729")),
        ]
    )
    if strong > 5 and nspec >= 2:
        return "emission"
    return "weak-em" if strong > 3 else "continuum"


def read_screen(path, value_col=None):
    """Read a screen CSV with target_id and flagged columns.

    A missing file raises on purpose, so a screen cannot be skipped silently.

    Parameters
    ----------
    path : str or Path
        CSV written by contam_screens.py or agn_star_screen.py.
    value_col : str, optional
        Column to return as floats for every row.

    Returns
    -------
    flag : set of int
        TARGETIDs whose flagged column is ``"True"``.
    vals : dict
        value_col keyed by TARGETID, or empty when value_col is None.
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    flag = {int(r["target_id"]) for r in rows if r["flagged"] == "True"}
    vals = {int(r["target_id"]): float(r[value_col]) for r in rows} if value_col else {}
    return flag, vals
