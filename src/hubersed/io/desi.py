"""Load DESI spectra from a spender Parquet dataset and from coadd npz files."""

from functools import lru_cache
from pathlib import Path

import astropy.units as u
import numpy as np
import pyarrow.parquet as pq
from astropy.nddata import InverseVariance
from specutils import Spectrum

from hubersed.conversion import DESI_FLAM
from hubersed.paths import PATHS
from hubersed.sps.lsf import DESI_WAV

DATA_PATH = PATHS["DATA"]
DATASET = DATA_PATH / "datasets" / "desi_sv3_bright_v1"


class TargetIDMismatchError(ValueError):
    """The spectrum on disk has a different TARGETID from the one asked for."""


@lru_cache(maxsize=4)
def _index(dataset):
    """Return the Parquet files of a dataset and each TARGETID's file and row.

    Only the ``target_id`` column is read. The arrays are sorted by TARGETID.
    """
    files = tuple(sorted((Path(dataset) / "data").glob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"no Parquet files in {Path(dataset) / 'data'}")
    tids, fidx, rows = [], [], []
    for i, f in enumerate(files):
        t = pq.read_table(f, columns=["target_id"])["target_id"].to_numpy()
        tids.append(t)
        fidx.append(np.full(t.size, i))
        rows.append(np.arange(t.size))
    tids, fidx, rows = (np.concatenate(x) for x in (tids, fidx, rows))
    order = np.argsort(tids, kind="stable")
    return files, tids[order], fidx[order], rows[order]


def load_by_targetid(targetid, dataset=DATASET):
    """Load one DESI spectrum by TARGETID and undo the spender normalization.

    Parameters
    ----------
    targetid : int
        DESI TARGETID.
    dataset : str or pathlib.Path, optional
        Spender Parquet dataset. The default is ``DATASET``.

    Returns
    -------
    spec : np.ndarray
        Flux density f_lambda in units of 1e-17 erg/s/cm^2/A, on ``DESI_WAV``.
    ivar : np.ndarray
        Inverse variance of ``spec``, in (1e-17 erg/s/cm^2/A)^-2.
    z : float
        Redshift.
    tid : int
        TARGETID read from the file.

    Raises
    ------
    ValueError
        If the TARGETID is missing from the dataset or appears more than once.
    """
    files, tids, fidx, rows = _index(str(dataset))
    lo, hi = np.searchsorted(tids, [targetid, targetid + 1])
    if hi - lo != 1:
        raise ValueError(f"TARGETID {targetid} appears {hi - lo} times in {dataset}")
    pf = pq.ParquetFile(files[fidx[lo]])
    row = int(rows[lo])
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if row < n:
            break
        row -= n
    t = pf.read_row_group(g, columns=["spec", "w", "z", "norm", "target_id"]).slice(row, 1)
    t = {c: t[c].combine_chunks() for c in t.column_names}
    norm = t["norm"].to_numpy()[0]
    s = t["spec"].values.to_numpy() * norm
    w = t["w"].values.to_numpy() / norm**2  # un-normalize
    return s, w, float(t["z"].to_numpy()[0]), int(t["target_id"].to_numpy()[0])


def load_spectrum(targetid, dataset=DATASET):
    """Load one DESI spectrum by TARGETID, with units, mask and redshift attached.

    Parameters
    ----------
    targetid : int
        DESI TARGETID.
    dataset : str or pathlib.Path, optional
        Spender Parquet dataset. The default is ``DATASET``.

    Returns
    -------
    specutils.Spectrum
        Flux in ``DESI_FLAM`` on the observed frame ``DESI_WAV`` grid in Angstrom, the
        inverse variance as uncertainty, and the redshift. A pixel is masked when its
        inverse variance is not positive or its flux is not finite. ``meta["targetid"]``
        holds the TARGETID read from the file. Flux and inverse variance are float32.

    Raises
    ------
    ValueError
        If the TARGETID is missing from the dataset or appears more than once.
    TargetIDMismatchError
        If the row found for the TARGETID holds a different TARGETID.
    """
    flux, ivar, z, tid = load_by_targetid(targetid, dataset)
    if tid != targetid:
        raise TargetIDMismatchError(f"asked for TARGETID {targetid}, the row holds {tid}")
    return desi_spectrum(flux, ivar, z, tid)


def desi_spectrum(flux, ivar, z, targetid):
    """Wrap one DESI spectrum on the ``DESI_WAV`` grid as a specutils Spectrum.

    Parameters
    ----------
    flux : np.ndarray
        Flux in ``DESI_FLAM`` units, one value per ``DESI_WAV`` pixel.
    ivar : np.ndarray
        Inverse variance of ``flux`` in ``DESI_FLAM**-2``.
    z : float
        Redshift.
    targetid : int
        DESI TARGETID, stored in ``meta["targetid"]``.

    Returns
    -------
    specutils.Spectrum
        A pixel is masked when its inverse variance is not positive or not finite, or its
        flux is not finite.
    """
    return Spectrum(
        flux=flux * DESI_FLAM,
        spectral_axis=DESI_WAV * u.AA,
        uncertainty=InverseVariance(ivar * DESI_FLAM**-2),
        mask=(ivar <= 0) | ~np.isfinite(ivar) | ~np.isfinite(flux),
        redshift=z,
        meta={"targetid": targetid},
    )


def load_resolution_npz(path, arm=None):
    """Load a coadded spectrum and its banded resolution matrix from an npz file.

    The format is that of resolution_matrix_rows.npz and resolution_matrix_rows_ivar.npz in
    experiments/2026-10-09_nautilus_c3k_er_broadline (coadd_rows.py, ivar_recompute.py). The
    keys wave, res, flux and ivar hold the arms merged onto the DESI grid, and the same keys with
    a b_, r_ or z_ prefix hold each arm. The flux is in the DESI units of 1e-17 erg/s/cm^2/A.

    Parameters
    ----------
    path : str or pathlib.Path
        The npz file.
    arm : {None, "b", "r", "z"}
        Arm to load. None loads the merged spectrum.

    Returns
    -------
    dict
        wave (uniform, float64), res (shape (ndiag, n), float64), flux and ivar (float64).

    Raises
    ------
    ValueError
        If the grid is not uniform or the matrix shape does not match it.
    """
    # prospect reads SPS_HOME on import
    from prospect.observation.resolution import banded_offsets, uniform_step

    if arm not in (None, "b", "r", "z"):
        raise ValueError(f"arm must be None, 'b', 'r' or 'z', not {arm!r}")
    pre = "" if arm is None else f"{arm}_"
    with np.load(path) as d:
        out = {k: np.asarray(d[pre + k], dtype=np.float64) for k in ("wave", "res", "flux", "ivar")}
    uniform_step(out["wave"])
    ndiag, n = out["res"].shape
    banded_offsets(ndiag)
    if n != out["wave"].size or out["flux"].shape != (n,) or out["ivar"].shape != (n,):
        raise ValueError(f"{path}: the matrix, flux and ivar do not match the {n} wavelengths")
    return out
