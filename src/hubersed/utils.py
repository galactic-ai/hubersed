"""Small helpers shared by the spender scripts."""

import torch


def nanstd(x, dim=None, keepdim=False, eps=0.0):
    """Return the standard deviation of a tensor, ignoring NaNs.

    Parameters
    ----------
    x : torch.Tensor
        Input values. NaNs are left out of both the mean and the variance.
    dim : int, optional
        Dimension to reduce. By default the whole tensor is reduced.
    keepdim : bool
        Keep the reduced dimension with length one.
    eps : float
        Added to the variance before the square root.

    Returns
    -------
    torch.Tensor
        The standard deviation. It divides by the number of values, not that number
        minus one.
    """
    # mean over non-NaNs
    mean = torch.nanmean(x, dim=dim, keepdim=True)  # shape keeps `dim`
    # squared deviations
    sq = (x - mean) ** 2
    # variance over non-NaNs
    var = torch.nanmean(sq, dim=dim, keepdim=True)
    std = torch.sqrt(var + eps)
    if not keepdim and dim is not None:
        std = std.squeeze(dim)
    return std
