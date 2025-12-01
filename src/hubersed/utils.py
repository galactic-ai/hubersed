import torch

def nanstd(x, dim=None, keepdim=False, eps=0.0):
    # mean over non-NaNs
    mean = torch.nanmean(x, dim=dim, keepdim=True)        # shape keeps `dim`
    # squared deviations
    sq = (x - mean) ** 2
    # variance over non-NaNs
    var = torch.nanmean(sq, dim=dim, keepdim=True)
    std = torch.sqrt(var + eps)
    if not keepdim and dim is not None:
        std = std.squeeze(dim)
    return std
