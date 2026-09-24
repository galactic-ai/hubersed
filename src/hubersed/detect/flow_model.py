"""Build normalizing flows over latent vectors and load latent files.

Training, DESI scoring and the outlier cut are run by ``scripts/get_outliers_flow.py``.
"""

from pathlib import Path

import h5py
import numpy as np
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms import (
    CompositeTransform,
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    RandomPermutation,
)


def build_flow(method, dim, hidden, n_transforms, num_bins=8, tail_bound=10.0):
    """Build a masked autoregressive flow with a standard normal base.

    Each block is one autoregressive transform followed by a random permutation of the
    features.

    Parameters
    ----------
    method : str
        ``"nsf"`` for rational quadratic spline transforms. Any other value gives affine
        transforms.
    dim : int
        Number of features.
    hidden : int
        Hidden layer width of each autoregressive network.
    n_transforms : int
        Number of blocks.
    num_bins : int
        Spline bins per transform, used only for ``"nsf"``.
    tail_bound : float
        Bound of the spline interval with linear tails, used only for ``"nsf"``.

    Returns
    -------
    nflows.flows.Flow
        The untrained flow.
    """
    ts = []
    for _ in range(n_transforms):
        if method == "nsf":
            ts.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim,
                    hidden_features=hidden,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=tail_bound,
                )
            )
        else:
            ts.append(MaskedAffineAutoregressiveTransform(features=dim, hidden_features=hidden))
        ts.append(RandomPermutation(features=dim))
    return Flow(CompositeTransform(ts), StandardNormal([dim]))


def load_h5(path):
    """Read latents, their TARGETIDs and the encoder checkpoint name from a latent h5 file.

    Rows are matched to galaxies by TARGETID, never by position. The caller passes a full
    path rather than a name resolved against one data directory.

    Parameters
    ----------
    path : str or Path
        Full path to the h5 file.

    Returns
    -------
    lat : numpy.ndarray
        Latents as float32.
    tid : numpy.ndarray
        TARGETID of each latent row as int64.
    ckpt : str
        The file's ``checkpoint`` attribute, or ``"unknown"`` if it is missing.

    Raises
    ------
    SystemExit
        If the file does not exist.
    KeyError
        If the file has no ``target_ids`` dataset.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"no latent file at {path}")
    with h5py.File(path, "r") as f:
        lat = np.asarray(f["latents"], np.float32)
        if "target_ids" not in f:
            raise KeyError(
                f"{path.name} has no 'target_ids' -- re-encode with the fixed get_latent_space.py"
            )
        tid = np.asarray(f["target_ids"], np.int64)
        ckpt = f.attrs.get("checkpoint", "unknown")
    return lat, tid, str(ckpt)
