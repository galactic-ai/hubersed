"""Normalize spectra the way spender expects."""

import torch


def normalize_spectra(
    flambda: torch.Tensor,
    redshifts: torch.Tensor,
    wavelength: torch.Tensor,
    wave_min: float = 5300.0,
    wave_max: float = 5850.0,
    inplace: bool = True,
):
    """Divide each spectrum by its median flux in a rest-frame window.

    Parameters
    ----------
    flambda : torch.Tensor
        Spectra, shape ``(N, n_wave)``.
    redshifts : torch.Tensor
        Redshifts, shape ``(N,)``.
    wavelength : torch.Tensor
        Observed wavelength in Angstrom, shape ``(n_wave,)``.
    wave_min, wave_max : float
        Rest-frame window in Angstrom. Pixels strictly inside it set the median.
    inplace : bool
        Change ``flambda`` itself. If False, a normalized copy is returned.

    Returns
    -------
    flambda : torch.Tensor
        The normalized spectra.
    norms : torch.Tensor
        The median of each spectrum, shape ``(N,)``. Zero when the window has no pixels
        or the median is not finite.
    good_mask : torch.Tensor
        True for spectra that were normalized, shape ``(N,)``.

    Notes
    -----
    NaNs inside the window are ignored by the median. Spectra with a zero norm are left
    unchanged.
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
    """Return the inverse variance implied by a flux and a signal-to-noise ratio.

    The noise is ``flux / snr`` and the inverse variance is one over its square.

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
    ivar = 1.0 / (sigma**2)
    return ivar
