import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from hubersed.prospector.lsf import desi_resolution   # calibrated median R(lambda)

C_KMS = 299792.458
MILES_FWHM_A = 2.5                     # MILES restframe resolution [A FWHM]
MILES_LAM_MIN, MILES_LAM_MAX = 3750.0, 7200.0   # MILES restframe coverage; outside = BaSeL R~200

# The median kernel: sigma to convolve DESI data -> MILES resolution
def _miles_sigma_obs_A(wave_obs, z):
    """MILES resolution as observed-frame Gaussian sigma [A]; inf (=> mask) outside window."""
    lam_rest = wave_obs / (1.0 + z)
    sig = np.full_like(wave_obs, np.inf, dtype=float)
    inwin = (lam_rest >= MILES_LAM_MIN) & (lam_rest <= MILES_LAM_MAX)
    sig[inwin] = (MILES_FWHM_A * (1.0 + z)) / 2.355   # restframe FWHM stretched by (1+z)
    return sig, inwin


def _desi_sigma_obs_A(wave_obs):
    """DESI instrumental LSF sigma [A, observed] from the calibrated median R(lambda)."""
    return (wave_obs / desi_resolution(wave_obs)) / 2.355


def match_kernel_sigma_A(wave_obs, z):
    """Gaussian sigma [A] to convolve the data by so it reaches MILES resolution.

    Returns (sigma_conv, good) where good=False outside the MILES window (BaSeL
    red/blue) or where DESI is already >= MILES (shouldn't happen in-window).
    """
    sig_m, inwin = _miles_sigma_obs_A(wave_obs, z)
    sig_d = _desi_sigma_obs_A(wave_obs)
    diff2 = sig_m**2 - sig_d**2
    good = inwin & (diff2 > 0.0)
    sig_conv = np.zeros_like(wave_obs, dtype=float)
    sig_conv[good] = np.sqrt(diff2[good])
    return sig_conv, good


def _variable_gaussian_matrix(wave, sigma_A):
    """Sparse (n,n) flux-conserving matrix that convolves by a wavelength-dependent
    Gaussian of sigma `sigma_A` [A]. Same construction as lsf.build_desi_resolution_matrix.
    sigma<=0 -> identity row (pass-through)."""
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

    Returns (flux_deg, ivar_deg, good_mask).
    NOTE on noise: convolution correlates the noise. var_out = (M**2) @ var_in is
    the DIAGONAL of the convolved covariance (exact if input noise is independent);
    it ignores off-diagonal correlation. Rebinning to ~kernel-width bins (step 2)
    largely re-decorrelates it. For calibrated posteriors, either rebin coarse
    enough or carry the off-diagonals / a GP noise term.
    """
    sig_conv, good = match_kernel_sigma_A(wave_obs, z)
    M = _variable_gaussian_matrix(wave_obs, sig_conv)
    w = (ivar > 0).astype(float)                          # good-pixel weight
    den = M.dot(w)
    num = M.dot(flux * w)
    # renormalize by the convolved good-pixel weight so zeroed bad pixels don't bias neighbours
    flux_deg = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    var = np.zeros_like(ivar, dtype=float)
    gi = ivar > 0
    var[gi] = 1.0 / ivar[gi]
    var_num = M.multiply(M).dot(var)                      # Var(num) = sum_j M_ij^2 var_j
    # flux_deg = num/den  ->  Var(flux_deg) = Var(num)/den^2  ->  ivar_deg = den^2 / Var(num).
    # The den^2 is essential: near bad pixels den -> 0, and without it ivar_deg blows up.
    ivar_deg = np.zeros_like(var_num)
    m2 = (var_num > 0) & good & (den > 0)
    ivar_deg[m2] = den[m2] ** 2 / var_num[m2]
    return flux_deg, ivar_deg, good

# trapz_rebin edges: one common, non-uniform (constant-velocity) grid
def common_obs_edges(lam_min=3600.0, lam_max=9824.0, dv_kms=60.0):
    """Constant-velocity (log-lambda) observed-frame bin EDGES, applied to ALL spectra. 
    dv ~ 30 km/s Nyquist-samples MILES (~64 km/s sigma at 5000 A) while
    still being coarser than DESI native (~15 km/s). Uniform in velocity => non-uniform
    in Angstrom, which is exactly the 'doesn't have to be uniform' scheme."""
    step = 1.0 + dv_kms / C_KMS
    n = int(np.log(lam_max / lam_min) / np.log(step))
    return lam_min * step ** np.arange(n + 1)

def centers2edges(centers):
    """Bin centers -> edges (matches provabgs.util.centers2edges)."""
    c = np.asarray(centers, float)
    e = np.empty(c.size + 1)
    e[1:-1] = 0.5 * (c[1:] + c[:-1])
    e[0]  = c[0]  - 0.5 * (c[1]  - c[0])
    e[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return e


def trapz_rebin(x, y, xnew=None, edges=None):
    """Flux-conserving trapezoidal rebin of density y(x) onto `edges`.
    Standalone equivalent of provabgs.util.trapz_rebin (pure numpy)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if edges is None:
        edges = centers2edges(xnew)
    edges = np.asarray(edges, float)
    if edges[0] < x[0] or x[-1] < edges[-1]:
        raise ValueError("edges must be within input x range")
    Ix = np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))))
    k = np.clip(np.searchsorted(x, edges, side="right") - 1, 0, x.size - 2)
    t = edges - x[k]
    slope = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
    I_edges = Ix[k] + y[k] * t + 0.5 * slope * t * t   # cum. integral at the edges
    return np.diff(I_edges) / np.diff(edges)           # mean density per bin


def rebin(wave_obs, flux, ivar, edges):
    """Inverse-variance-weighted coadd onto `edges` -- the standard, robust way to
    put a NOISY spectrum on a coarser grid:
        flux_reb = sum(ivar*flux) / sum(ivar)     (ivar-weighted mean)
        ivar_reb = sum(ivar)                      (bounded; can't blow up)
    Rebinning the *variance* through a trapz density operator instead can produce
    spurious huge ivar on real data, which then dominates chi^2 -- that was the bug.
    (Correlated post-convolution noise makes ivar_reb a mild overestimate; bins
    ~kernel-width keep it close. trapz_rebin above is kept for noiseless model/mock
    resampling.) Returns (centers, flux_reb, ivar_reb, mask_reb)."""
    centers = 0.5 * (edges[1:] + edges[:-1])
    nb = centers.size
    b = np.searchsorted(edges, wave_obs, side="right") - 1          # native pixel -> bin index
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
    """Full data path: convolve to MILES resolution, then rebin onto the common grid.
    Use the IDENTICAL call when generating the training mocks."""
    if edges is None:
        edges = common_obs_edges()
    # real DESI spectra have bad pixels (NaN / ivar<=0); zero them BEFORE the convolution
    # and the trapz cumsum, or a single NaN poisons every rebinned bin downstream.
    flux = np.asarray(flux, dtype=float).copy()
    ivar = np.asarray(ivar, dtype=float).copy()
    bad = ~np.isfinite(flux) | ~np.isfinite(ivar) | (ivar <= 0)
    flux[bad] = 0.0
    ivar[bad] = 0.0
    flux_d, ivar_d, good = degrade_to_miles(wave_obs, flux, ivar, z)
    ivar_d = np.where(good, ivar_d, 0.0)     # drop BaSeL red/blue (outside MILES window)
    return rebin(wave_obs, flux_d, ivar_d, edges)