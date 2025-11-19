import torch

def normalize_spectra(
    flambda: torch.Tensor,
    redshifts: torch.Tensor,
    wavelength: torch.Tensor,
    wave_min: float = 5300.0,
    wave_max: float = 5850.0,
    inplace: bool = True,
):
    """
    Vectorized equivalent of:

        for redshift, spec in zip(redshifts, flambda):
            norm = 0
            wave_rest = wavelength / (1 + redshift)
            sel = (wave_rest > 5300) & (wave_rest < 5850)
            if sel.count_nonzero() > 0:
                norm = torch.median(spec[sel])
            if not torch.isfinite(norm):
                norm = 0
            else:
                spec /= norm

    Parameters
    ----------
    flambda : torch.Tensor
        (N, n_wave) spectra
    redshifts : torch.Tensor
        (N,) redshifts
    wavelength: torch.Tensor
        (n_wave,) observed-frame wavelengths
    wave_min, wave_max: float
        rest-frame window to require
    inplace   : bool
        if False, returns a normalized copy

    Returns
    -------
    flambda_out : torch.Tensor
        normalized spectra (same tensor if inplace=True)
    norms       : torch.Tensor
        (N,) normalization factors (0 for spectra not normalized)
    good_mask   : torch.Tensor
        (N,) boolean, True where normalization was applied
    """

    if not inplace:
        flambda = flambda.clone()

    # Make sure everything lives on the same device
    device = flambda.device
    wavelength = wavelength.to(device)
    redshifts = redshifts.to(device)

    # Rest-frame wavelength grid per spectrum: (N, n_wave)
    wave_rest = wavelength.unsqueeze(0) / (1 + redshifts.unsqueeze(1))

    # Mask for the rest-frame window
    sel = (wave_rest > wave_min) & (wave_rest < wave_max)  # (N, n_wave)

    # At least one valid pixel in the window?
    has_valid = sel.any(dim=1)  # (N,)

    # Copy and set non-selected entries to NaN, then use nanmedian along dim=1.
    masked = flambda.clone()
    masked[~sel] = float("nan")
    row_medians = torch.nanmedian(masked, dim=1).values

    # Start with norms = 0, fill only where we have valid window
    norms = torch.zeros_like(row_medians)
    norms[has_valid] = row_medians[has_valid]

    # Remove non-finite norms
    norms[~torch.isfinite(norms)] = 0

    # Normalize only spectra with non-zero norm
    good_mask = norms != 0
    flambda[good_mask] /= norms[good_mask].unsqueeze(1)

    return flambda, norms, good_mask

def compute_ivar(flux, snr):
    """Compute inverse variance from flux and SNR.
    sigma = flux / snr
    ivar = 1 / sigma^2

    Parameters
    ----------
    flux : torch.Tensor
        Flux values.
    snr : torch.Tensor
        Signal-to-noise ratio values.

    Returns
    -------
    ivar : torch.Tensor
        Inverse variance values.
    """

    sigma = flux / snr
    ivar = 1.0 / (sigma ** 2)
    return ivar