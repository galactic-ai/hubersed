"""Load DESI spectra from the spender chunk files by global index or TARGETID."""

import pickle

import astropy.units as u
import numpy as np
from astropy.nddata import InverseVariance
from specutils import Spectrum

from hubersed.conversion import DESI_FLAM
from hubersed.paths import PATHS
from hubersed.sps.lsf import DESI_WAV

DATA_PATH = PATHS["DATA"]
CHUNK = 1024


class TargetIDMismatchError(ValueError):
    """The spectrum on disk has a different TARGETID from the one asked for."""


def load_by_index(gidx):
    """Load one DESI spectrum by its global index and undo the spender normalization.

    Parameters
    ----------
    gidx : int
        Global index. The spectrum is row ``gidx % 1024`` of the file
        ``DESIchunk1024_{gidx // 1024}.pkl``, counting chunks in numeric order.

    Returns
    -------
    spec : np.ndarray
        Flux density f_lambda in units of 1e-17 erg/s/cm^2/A, on ``WAVE_OBS``.
    ivar : np.ndarray
        Inverse variance of ``spec``, in (1e-17 erg/s/cm^2/A)^-2.
    z : float
        Redshift.
    tid : int
        DESI TARGETID. Callers should check it matches the galaxy they asked for.

    Notes
    -----
    spender reads the same files in string order (0, 1, 10, 100, ...), so a row in an
    encoder output file is not a global index. Go from TARGETID to index with
    ``tids_to_indices``. ``tests/test_targetid.py`` pins both orders.
    """
    chunk, row = gidx // CHUNK, gidx % CHUNK
    with open(DATA_PATH / "desi_spectra" / f"DESIchunk1024_{chunk}.pkl", "rb") as f:
        s, w, z, tid, norm, *_ = pickle.load(f)
    s = s * norm[:, None]
    w = w / norm[:, None] ** 2  # un-normalize

    def to(x):
        return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

    s, w, z, tid = to(s)[row], to(w)[row], float(to(z)[row]), int(to(tid)[row])
    return s, w, z, tid


def tids_to_indices(tids):
    """Find the global index of each TARGETID.

    Parameters
    ----------
    tids : np.ndarray
        TARGETIDs as int64.

    Returns
    -------
    np.ndarray
        Global indices for ``load_by_index``, in the same order as ``tids``.

    Raises
    ------
    ValueError
        If any TARGETID is missing from ``all_target_ids.npy``. Nothing is dropped.

    Notes
    -----
    ``all_target_ids.npy`` lists TARGETIDs in numeric chunk order, the order
    ``load_by_index`` uses. This was checked against all 249 chunk files on 2026-09-22.
    """
    all_tids = np.load(DATA_PATH / "all_target_ids.npy").astype(np.int64)
    order = np.argsort(all_tids)
    sa = all_tids[order]
    pos = np.clip(np.searchsorted(sa, tids), 0, len(sa) - 1)
    ok = sa[pos] == tids
    if not ok.all():
        raise ValueError(
            f"{int((~ok).sum())}/{len(tids)} TARGETIDs not found in all_target_ids.npy"
        )
    return order[pos].astype(int)


def load_spectrum(targetid):
    """Load one DESI spectrum by TARGETID, with units, mask and redshift attached.

    Parameters
    ----------
    targetid : int
        DESI TARGETID.

    Returns
    -------
    specutils.Spectrum
        Flux in ``DESI_FLAM`` on the observed frame ``DESI_WAV`` grid in Angstrom, the
        inverse variance as uncertainty, and the redshift. A pixel is masked when its
        inverse variance is not positive or its flux is not finite. ``meta["targetid"]``
        holds the TARGETID read from the file. Flux and inverse variance keep the float32
        type of the chunk files.

    Raises
    ------
    ValueError
        If the TARGETID is missing from ``all_target_ids.npy``.
    TargetIDMismatchError
        If the file row found for the TARGETID holds a different TARGETID.
    """
    gidx = tids_to_indices(np.array([targetid], dtype=np.int64))[0]
    flux, ivar, z, tid = load_by_index(gidx)
    if tid != targetid:
        raise TargetIDMismatchError(f"asked for TARGETID {targetid}, row {gidx} holds {tid}")
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
