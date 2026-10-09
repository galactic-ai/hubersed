"""Degrade DESI spectra to an SPS library's resolution and rebin them onto a common grid."""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from hubersed.sps.lsf import desi_resolution  # calibrated median R(lambda)

C_KMS = 299792.458

# Rest-frame resolution and usable rest-frame window of each FSPS stellar library.
# Give "fwhm_A" for a constant width in Angstrom, or "R" for a constant resolving power.
LIBRARIES = {
    # MILES; FSPS uses BaSeL (R~200) outside this window
    "miles": dict(fwhm_A=2.5, window=(3750.0, 7200.0)),
    # C3K_HR (FSPS v4.0, SPECTRA/C3K/c3k_hr/readme.md): R = lambda/FWHM = 3000 for 3001-10000 A
    "c3k_hr": dict(R=3000.0, window=(3001.0, 10000.0)),
}

# kept for backward compatibility
MILES_FWHM_A = LIBRARIES["miles"]["fwhm_A"]
MILES_LAM_MIN, MILES_LAM_MAX = LIBRARIES["miles"]["window"]


def _library_sigma_obs_A(wave_obs, z, library="miles"):
    """Return a library's resolution as an observed-frame Gaussian sigma.

    Parameters
    ----------
    wave_obs : np.ndarray
        Observed wavelength in Angstrom.
    z : float
        Redshift.
    library : str or dict
        A key of ``LIBRARIES``, or a dict with ``window`` and either ``fwhm_A`` or ``R``.

    Returns
    -------
    sigma : np.ndarray
        Sigma in observed Angstrom. It is inf outside the library's rest-frame window.
    in_window : np.ndarray
        True where the rest-frame wavelength is inside the window.
    """
    lib = LIBRARIES[library] if isinstance(library, str) else library
    lam_rest = wave_obs / (1.0 + z)
    lo, hi = lib["window"]
    sig = np.full_like(wave_obs, np.inf, dtype=float)
    inwin = (lam_rest >= lo) & (lam_rest <= hi)
    if "fwhm_A" in lib:  # constant rest-frame FWHM, stretched by (1+z)
        sig[inwin] = lib["fwhm_A"] * (1.0 + z) / 2.355
    else:  # constant resolving power: same in the rest and observed frames
        sig[inwin] = wave_obs[inwin] / (2.355 * lib["R"])
    return sig, inwin


def _miles_sigma_obs_A(wave_obs, z):
    """MILES resolution as an observed-frame sigma (kept for backward compatibility)."""
    return _library_sigma_obs_A(wave_obs, z, "miles")


def _desi_sigma_obs_A(wave_obs, desi_sigma_kms=None):
    """Return the DESI line spread function as a Gaussian sigma in observed Angstrom.

    Parameters
    ----------
    wave_obs : np.ndarray
        Observed wavelength in Angstrom.
    desi_sigma_kms : np.ndarray, optional
        This target's LSF sigma in km/s on ``wave_obs``, e.g. from its resolution matrix
        (``lsf.resolution_to_sigma_kms``) or SPARCL ``wave_sigma``. The default uses the
        median resolution R(lambda) from ``lsf.desi_resolution``.
    """
    if desi_sigma_kms is None:
        return (wave_obs / desi_resolution(wave_obs)) / 2.355
    return np.asarray(desi_sigma_kms, dtype=float) / C_KMS * wave_obs


def match_kernel_sigma_A(wave_obs, z, library="miles", desi_sigma_kms=None):
    """Return the Gaussian sigma that takes DESI data down to a library's resolution.

    Parameters
    ----------
    wave_obs : np.ndarray
        Observed wavelength in Angstrom.
    z : float
        Redshift.
    library : str or dict
        See ``_library_sigma_obs_A``. Default "miles".
    desi_sigma_kms : np.ndarray, optional
        Per-target DESI LSF sigma in km/s. Default: calibrated median.

    Returns
    -------
    sigma_conv : np.ndarray
        Sigma in observed Angstrom, the quadrature difference of library and DESI. Zero
        where DESI is already as coarse as the library or coarser (no smoothing needed).
    good : np.ndarray
        True inside the library window, where the spectrum can be fitted. Pixels where DESI
        is coarser than the library stay good: Prospector smooths the model there.
    """
    sig_l, inwin = _library_sigma_obs_A(wave_obs, z, library)
    sig_d = _desi_sigma_obs_A(wave_obs, desi_sigma_kms)
    diff2 = sig_l**2 - sig_d**2
    conv = inwin & np.isfinite(diff2) & (diff2 > 0.0)
    sig_conv = np.zeros_like(wave_obs, dtype=float)
    sig_conv[conv] = np.sqrt(diff2[conv])
    return sig_conv, inwin


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


def degrade_to_library(
    wave_obs, flux, ivar, z, library="miles", desi_sigma_kms=None, keep_ivar=False
):
    """Convolve an observed-frame DESI spectrum down to a library's resolution.

    Only pixels where DESI is sharper than the library are smoothed; the others pass through.

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
    library : str or dict
        See ``_library_sigma_obs_A``. Default "miles".
    desi_sigma_kms : np.ndarray, optional
        Per-target DESI LSF sigma in km/s. Default: calibrated median.
    keep_ivar : bool
        If True, keep the input inverse variance for good pixels instead of propagating it.

    Returns
    -------
    flux_deg : np.ndarray
        Convolved flux, weighted so that bad pixels do not pull their neighbours down.
    ivar_deg : np.ndarray
        Inverse variance of ``flux_deg``; zero outside the library window.
    good : np.ndarray
        True inside the library window.

    Notes
    -----
    Convolution correlates the noise between pixels. The propagated ``ivar_deg`` keeps only
    the diagonal of the new covariance; a likelihood that treats pixels as independent then
    over-counts the information (errors on fitted quantities ~1.7x too small for a
    ~30 km/s kernel on the DESI grid, and rebinning to 60 km/s bins does not fix it).
    ``keep_ivar=True`` keeps the input ivar, which roughly cancels the two effects and gives
    correctly sized errors in that test. Use it when fitting on the native grid.
    """
    sig_conv, good = match_kernel_sigma_A(wave_obs, z, library, desi_sigma_kms)
    M = _variable_gaussian_matrix(wave_obs, sig_conv)
    w = (ivar > 0).astype(float)  # good-pixel weight
    den = M.dot(w)
    num = M.dot(flux * w)
    # renormalize by the convolved good-pixel weight so zeroed bad pixels don't bias neighbours
    flux_deg = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    ivar_deg = np.zeros_like(ivar, dtype=float)
    if keep_ivar:
        m2 = good & (ivar > 0)
        ivar_deg[m2] = ivar[m2]
        return flux_deg, ivar_deg, good
    var = np.zeros_like(ivar, dtype=float)
    gi = ivar > 0
    var[gi] = 1.0 / ivar[gi]
    var_num = M.multiply(M).dot(var)  # Var(num) = sum_j M_ij^2 var_j
    # Var(flux_deg) is Var(num) / den^2, so ivar_deg is den^2 / Var(num). Without the den^2,
    # ivar_deg blows up next to bad pixels, where den is close to zero.
    m2 = (var_num > 0) & good & (den > 0)
    ivar_deg[m2] = den[m2] ** 2 / var_num[m2]
    return flux_deg, ivar_deg, good


def degrade_to_miles(wave_obs, flux, ivar, z):
    """Convolve an observed-frame DESI spectrum down to MILES resolution (backward compatible)."""
    return degrade_to_library(wave_obs, flux, ivar, z, library="miles")


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


def prep_spectrum(
    wave_obs,
    flux,
    ivar,
    z,
    edges=None,
    library="miles",
    desi_sigma_kms=None,
    keep_ivar=False,
    do_rebin=True,
):
    """Convolve a DESI spectrum to a library's resolution and (optionally) rebin it.

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
    library : str or dict
        See ``_library_sigma_obs_A``. Default "miles".
    desi_sigma_kms : np.ndarray, optional
        Per-target DESI LSF sigma in km/s on ``wave_obs``. Default: calibrated median.
    keep_ivar : bool
        See ``degrade_to_library``.
    do_rebin : bool
        If False, return the degraded spectrum on the native grid. Use this for Prospector,
        which evaluates the model at the pixel centres rather than averaging over bins.

    Returns
    -------
    tuple
        ``(centers, flux_reb, ivar_reb, mask_reb)``. Bins or pixels outside the library
        window get zero inverse variance.

    Notes
    -----
    Training mocks should go through this same call, so data and mocks match.
    """
    # zero bad pixels first, or a single NaN spreads through the convolution to every bin
    flux = np.asarray(flux, dtype=float).copy()
    ivar = np.asarray(ivar, dtype=float).copy()
    bad = ~np.isfinite(flux) | ~np.isfinite(ivar) | (ivar <= 0)
    flux[bad] = 0.0
    ivar[bad] = 0.0
    flux_d, ivar_d, good = degrade_to_library(
        wave_obs, flux, ivar, z, library, desi_sigma_kms, keep_ivar=keep_ivar
    )
    ivar_d = np.where(good, ivar_d, 0.0)  # drop pixels outside the library window
    if not do_rebin:
        return np.asarray(wave_obs, float), flux_d, ivar_d, ivar_d > 0
    if edges is None:
        edges = common_obs_edges()
    return rebin(wave_obs, flux_d, ivar_d, edges)
