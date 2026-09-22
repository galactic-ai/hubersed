"""Degrade DESI spectra to MILES resolution and rebin them onto a common grid."""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from hubersed.prospector.lsf import desi_resolution  # calibrated median R(lambda)

C_KMS = 299792.458
MILES_FWHM_A = 2.5  # MILES restframe resolution [A FWHM]
MILES_LAM_MIN, MILES_LAM_MAX = (
    3750.0,
    7200.0,
)  # MILES rest-frame range, FSPS uses BaSeL (R~200) outside


def _miles_sigma_obs_A(wave_obs, z):
    """Return the MILES resolution as an observed-frame Gaussian sigma.

    Parameters
    ----------
    wave_obs : np.ndarray
        Observed wavelength in Angstrom.
    z : float
        Redshift.

    Returns
    -------
    sigma : np.ndarray
        Sigma in observed Angstrom. It is inf outside the MILES rest-frame window.
    in_window : np.ndarray
        True where the rest-frame wavelength is inside the MILES window.
    """
    lam_rest = wave_obs / (1.0 + z)
    sig = np.full_like(wave_obs, np.inf, dtype=float)
    inwin = (lam_rest >= MILES_LAM_MIN) & (lam_rest <= MILES_LAM_MAX)
    sig[inwin] = (MILES_FWHM_A * (1.0 + z)) / 2.355  # restframe FWHM stretched by (1+z)
    return sig, inwin


def _desi_sigma_obs_A(wave_obs):
    """Return the DESI line spread function as a Gaussian sigma in observed Angstrom.

    It uses the median resolution R(lambda) from ``lsf.desi_resolution``.
    """
    return (wave_obs / desi_resolution(wave_obs)) / 2.355


def match_kernel_sigma_A(wave_obs, z):
    """Return the Gaussian sigma that takes DESI data down to MILES resolution.

    Parameters
    ----------
    wave_obs : np.ndarray
        Observed wavelength in Angstrom.
    z : float
        Redshift.

    Returns
    -------
    sigma_conv : np.ndarray
        Sigma in observed Angstrom, the quadrature difference of MILES and DESI.
    good : np.ndarray
        False outside the MILES window, where FSPS uses the coarser BaSeL library, and
        where DESI is already coarser than MILES.
    """
    sig_m, inwin = _miles_sigma_obs_A(wave_obs, z)
    sig_d = _desi_sigma_obs_A(wave_obs)
    diff2 = sig_m**2 - sig_d**2
    good = inwin & (diff2 > 0.0)
    sig_conv = np.zeros_like(wave_obs, dtype=float)
    sig_conv[good] = np.sqrt(diff2[good])
    return sig_conv, good


def _variable_gaussian_matrix(wave, sigma_A):
    """Build a sparse matrix that convolves with a Gaussian whose width varies with wavelength.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength in Angstrom, length n.
    sigma_A : np.ndarray
        Gaussian sigma in Angstrom at each pixel. Pixels with sigma of zero or less are
        passed through unchanged.

    Returns
    -------
    scipy.sparse.csr_matrix
        An n by n matrix whose rows each sum to one, built the same way as
        ``lsf.build_desi_resolution_matrix``.
    """
    n = len(wave)
    sig_pix = np.zeros(n)
    dwave = np.gradient(wave)
    m = sigma_A > 0
    sig_pix[m] = sigma_A[m] / dwave[m]
    M = lil_matrix((n, n), dtype=np.float64)
    for i in range(n):
        s = sig_pix[i]
        if s <= 0:
            M[i, i] = 1.0
            continue
        hw = int(4 * s) + 1
        lo, hi = max(0, i - hw), min(n, i + hw + 1)
        j = np.arange(lo, hi)
        k = np.exp(-0.5 * ((j - i) / s) ** 2)
        M[i, lo:hi] = k / k.sum()
    return csr_matrix(M)


def degrade_to_miles(wave_obs, flux, ivar, z):
    """Convolve an observed-frame DESI spectrum down to MILES resolution.

    Parameters
    ----------
    wave_obs : np.ndarray
        Observed wavelength in Angstrom.
    flux : np.ndarray
        Flux density. Bad pixels must already be set to zero.
    ivar : np.ndarray
        Inverse variance of ``flux``. Zero marks a bad pixel.
    z : float
        Redshift.

    Returns
    -------
    flux_deg : np.ndarray
        Convolved flux, weighted so that bad pixels do not pull their neighbours down.
    ivar_deg : np.ndarray
        Inverse variance of ``flux_deg``.
    good : np.ndarray
        True inside the MILES window.

    Notes
    -----
    Convolution correlates the noise between pixels. ``ivar_deg`` keeps only the diagonal
    of the new covariance, which is exact only for independent input noise. Rebinning to
    bins about as wide as the kernel removes most of the correlation. For calibrated
    posteriors, rebin coarsely enough or model the correlation.
    """
    sig_conv, good = match_kernel_sigma_A(wave_obs, z)
    M = _variable_gaussian_matrix(wave_obs, sig_conv)
    w = (ivar > 0).astype(float)  # good-pixel weight
    den = M.dot(w)
    num = M.dot(flux * w)
    # renormalize by the convolved good-pixel weight so zeroed bad pixels don't bias neighbours
    flux_deg = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    var = np.zeros_like(ivar, dtype=float)
    gi = ivar > 0
    var[gi] = 1.0 / ivar[gi]
    var_num = M.multiply(M).dot(var)  # Var(num) = sum_j M_ij^2 var_j
    # Var(flux_deg) is Var(num) / den^2, so ivar_deg is den^2 / Var(num). Without the den^2,
    # ivar_deg blows up next to bad pixels, where den is close to zero.
    ivar_deg = np.zeros_like(var_num)
    m2 = (var_num > 0) & good & (den > 0)
    ivar_deg[m2] = den[m2] ** 2 / var_num[m2]
    return flux_deg, ivar_deg, good


def common_obs_edges(lam_min=3600.0, lam_max=9824.0, dv_kms=60.0):
    """Return bin edges of constant velocity width, shared by every spectrum.

    Parameters
    ----------
    lam_min, lam_max : float
        Observed wavelength range in Angstrom.
    dv_kms : float
        Bin width in km/s. Each edge is ``1 + dv_kms / c`` times the one before.

    Returns
    -------
    np.ndarray
        Bin edges in Angstrom. They are evenly spaced in log wavelength, so the bins get
        wider in Angstrom toward the red.
    """
    step = 1.0 + dv_kms / C_KMS
    n = int(np.log(lam_max / lam_min) / np.log(step))
    return lam_min * step ** np.arange(n + 1)


def centers2edges(centers):
    """Turn bin centers into bin edges.

    Inner edges sit halfway between centers. The two outer edges extend the first and last
    bins by half their neighbour spacing. This matches ``provabgs.util.centers2edges``.
    """
    c = np.asarray(centers, float)
    e = np.empty(c.size + 1)
    e[1:-1] = 0.5 * (c[1:] + c[:-1])
    e[0] = c[0] - 0.5 * (c[1] - c[0])
    e[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return e


def trapz_rebin(x, y, xnew=None, edges=None):
    """Rebin a density onto new bins while conserving its integral.

    The input is treated as straight lines between samples. Each output value is the
    integral over its bin divided by the bin width.

    Parameters
    ----------
    x : np.ndarray
        Input sample positions, increasing.
    y : np.ndarray
        Input density at ``x``, for example flux density.
    xnew : np.ndarray, optional
        Output bin centers, used when ``edges`` is not given.
    edges : np.ndarray, optional
        Output bin edges. They must lie within ``x``.

    Returns
    -------
    np.ndarray
        Mean density in each output bin, one fewer value than ``edges``.

    Raises
    ------
    ValueError
        If the edges reach outside the range of ``x``.

    Notes
    -----
    A pure numpy version of ``provabgs.util.trapz_rebin``. It is meant for noiseless model
    spectra. Use ``rebin`` for noisy data.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if edges is None:
        edges = centers2edges(xnew)
    edges = np.asarray(edges, float)
    if edges[0] < x[0] or x[-1] < edges[-1]:
        raise ValueError("edges must be within input x range")
    Ix = np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))))
    k = np.clip(np.searchsorted(x, edges, side="right") - 1, 0, x.size - 2)
    t = edges - x[k]
    slope = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
    I_edges = Ix[k] + y[k] * t + 0.5 * slope * t * t  # cum. integral at the edges
    return np.diff(I_edges) / np.diff(edges)  # mean density per bin


def rebin(wave_obs, flux, ivar, edges):
    """Put a noisy spectrum on coarser bins by an inverse-variance weighted mean.

    Each bin's flux is ``sum(ivar * flux) / sum(ivar)`` over the pixels inside it, and its
    inverse variance is ``sum(ivar)``.

    Parameters
    ----------
    wave_obs : np.ndarray
        Pixel wavelengths in Angstrom.
    flux : np.ndarray
        Flux density per pixel.
    ivar : np.ndarray
        Inverse variance per pixel. Pixels with zero are skipped.
    edges : np.ndarray
        Bin edges in Angstrom.

    Returns
    -------
    centers : np.ndarray
        Bin centers in Angstrom.
    flux_reb : np.ndarray
        Weighted mean flux, zero in empty bins.
    ivar_reb : np.ndarray
        Summed inverse variance.
    mask_reb : np.ndarray
        True for bins with at least one good pixel.

    Notes
    -----
    Rebinning the variance with ``trapz_rebin`` can give very large inverse variances on
    real data, which then dominate chi-squared. This weighted mean cannot. After a
    convolution the noise is correlated, so ``ivar_reb`` is somewhat too large, less so
    when bins are about as wide as the kernel.
    """
    centers = 0.5 * (edges[1:] + edges[:-1])
    nb = centers.size
    b = np.searchsorted(edges, wave_obs, side="right") - 1  # bin index of each native pixel
    keep = (b >= 0) & (b < nb) & (ivar > 0) & np.isfinite(flux)
    bb, w, f = b[keep], ivar[keep], flux[keep]
    ivar_reb = np.zeros(nb)
    np.add.at(ivar_reb, bb, w)
    fsum = np.zeros(nb)
    np.add.at(fsum, bb, w * f)
    flux_reb = np.zeros(nb)
    mask_reb = ivar_reb > 0
    flux_reb[mask_reb] = fsum[mask_reb] / ivar_reb[mask_reb]
    return centers, flux_reb, ivar_reb, mask_reb


def prep_spectrum(wave_obs, flux, ivar, z, edges=None):
    """Convolve a DESI spectrum to MILES resolution and rebin it onto the common grid.

    Parameters
    ----------
    wave_obs : np.ndarray
        Observed wavelength in Angstrom.
    flux : np.ndarray
        Flux density. NaNs are allowed and are treated as bad pixels.
    ivar : np.ndarray
        Inverse variance. Values of zero or less are treated as bad pixels.
    z : float
        Redshift.
    edges : np.ndarray, optional
        Bin edges in Angstrom. The default is ``common_obs_edges()``.

    Returns
    -------
    tuple
        ``(centers, flux_reb, ivar_reb, mask_reb)`` from ``rebin``. Bins outside the MILES
        window get zero inverse variance.

    Notes
    -----
    Training mocks should go through this same call, so data and mocks match.
    """
    if edges is None:
        edges = common_obs_edges()
    # zero bad pixels first, or a single NaN spreads through the convolution to every bin
    flux = np.asarray(flux, dtype=float).copy()
    ivar = np.asarray(ivar, dtype=float).copy()
    bad = ~np.isfinite(flux) | ~np.isfinite(ivar) | (ivar <= 0)
    flux[bad] = 0.0
    ivar[bad] = 0.0
    flux_d, ivar_d, good = degrade_to_miles(wave_obs, flux, ivar, z)
    ivar_d = np.where(good, ivar_d, 0.0)  # drop BaSeL red/blue (outside MILES window)
    return rebin(wave_obs, flux_d, ivar_d, edges)
