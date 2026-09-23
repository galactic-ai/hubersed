"""Load DESI spectra from the spender chunk files by global index or TARGETID."""

import pickle

import numpy as np

from hubersed.paths import PATHS

DATA_PATH = PATHS["DATA"]
CHUNK = 1024


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
    to = lambda x: x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
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
