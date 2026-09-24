"""Build normalizing flows over latent vectors.

Training, DESI scoring and the outlier cut are run by ``scripts/get_outliers_flow.py``.
"""

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
