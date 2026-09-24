"""Read spender latent files written by ``scripts/get_latent_space.py``."""

from pathlib import Path

import h5py
import numpy as np


def load_latents(path):
    """Read latents, their TARGETIDs and the file attributes from a latent h5 file.

    Match rows to a catalogue by TARGETID, never by row position.

    Parameters
    ----------
    path : str or pathlib.Path
        h5 file with ``latents`` and ``target_ids`` datasets.

    Returns
    -------
    lat : numpy.ndarray
        Latents as float32, one row per spectrum.
    tid : numpy.ndarray
        TARGETID of each row as int64.
    attrs : dict
        The file attributes, for example ``snr_min`` and ``checkpoint``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If the file has no ``target_ids`` dataset, which means it predates the TARGETID fix.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no latent file at {path}")
    with h5py.File(path, "r") as f:
        if "target_ids" not in f:
            raise KeyError(
                f"{path} has no 'target_ids'. Re-encode it with scripts/get_latent_space.py."
            )
        lat = np.asarray(f["latents"], np.float32)
        tid = np.asarray(f["target_ids"], np.int64)
        attrs = dict(f.attrs)
    return lat, tid, attrs
