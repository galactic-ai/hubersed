"""Apply a DESI resolution matrix to a model sampled on a fine wavelength grid.

DESI gives each spectrum a banded resolution matrix R on its 0.8 A extraction grid. The rows of R
sum to one and the columns do not, so R is not flux conserving and the model is multiplied by R
itself, never by a column-normalised copy (Guy et al. 2023, arXiv:2209.14482, section 4.5).

Rebinning a model with unresolved features onto the extraction grid before applying R biases it.
Instead each row of R is interpolated with a cubic spline onto a finer model grid (Guy et al. 2023,
appendix D), and the spline values are multiplied by the fine step over the extraction step. This
gives a non-square matrix R' that applies the resolution and rebins in one step. R' keeps the row
sums of R, equals R on a smooth model, and maps an unresolved line that sits on an extraction pixel
to the column of R there. The figure code of that appendix (desihub/spectropaper-code,
work/extraction/plot_interpresolution.py) uses the same normalisation. The spline is tied to zero,
with zero slope, one pixel beyond the outermost diagonals.

The banded storage follows desispec.resolution.Resolution and scipy.sparse.dia_matrix. A matrix
with ``ndiag`` diagonals has offsets from ndiag//2 down to -(ndiag//2), and R[j - offset_k, j] =
data[k, j], so data[:, j] holds column j.

R' is applied to the model in maggies, a flux density per unit frequency. DESI extracts photons
per unit wavelength, so this tilts each kernel by the change of lambda^2 across it, at most about
0.2% over the 11 pixels at 3600 A.

This module needs only numpy, scipy and sedpy. The prospect model that uses it is in
``hubersed.sps.resolution_model``.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix, dia_matrix
from sedpy.smoothing import smoothspec

C_KMS = 299792.458
# Speed of light in Angstrom/s, as prospect.sources.constants.lightspeed, so the line profiles
# have the frequency normalisation of prospect's own Gaussians.
LIGHTSPEED = 2.998e18
FINE_STEP = 0.05  # Angstrom
# Velocity padding of the continuum smoothing window beyond the requested range, in km/s, on top
# of 20 kernel widths.
SMOOTH_PAD_KMS = 1000.0


def banded_offsets(ndiag):
    """Return the diagonal offsets of a banded DESI matrix.

    Parameters
    ----------
    ndiag : int
        Number of stored diagonals. It must be odd.

    Returns
    -------
    np.ndarray
        Offsets from ndiag//2 down to -(ndiag//2).
    """
    if ndiag % 2 != 1:
        raise ValueError(f"a banded DESI matrix has an odd number of diagonals, not {ndiag}")
    return ndiag // 2 - np.arange(ndiag)


def banded_to_csr(data):
    """Return a banded DESI matrix as a square sparse matrix.

    Parameters
    ----------
    data : np.ndarray
        Banded matrix of shape (ndiag, n), with R[j - offset_k, j] = data[k, j].

    Returns
    -------
    scipy.sparse.csr_matrix
        The matrix R, of shape (n, n).
    """
    data = np.asarray(data, dtype=np.float64)
    ndiag, n = data.shape
    return dia_matrix((data, banded_offsets(ndiag)), shape=(n, n)).tocsr()


def uniform_step(wave, rtol=1e-6):
    """Return the step of a uniform wavelength grid.

    Parameters
    ----------
    wave : np.ndarray
        Increasing wavelengths in Angstrom.
    rtol : float
        Largest allowed relative deviation of any step from the mean step.

    Returns
    -------
    float
        The step in Angstrom.

    Raises
    ------
    ValueError
        If the grid is not uniform to ``rtol``. A float32 copy of the DESI grid fails at the
        default tolerance, so pass it through ``extraction_grid`` first.
    """
    wave = np.asarray(wave)
    if wave.ndim != 1 or wave.size < 2:
        raise ValueError("a wavelength grid must be one dimensional with two or more points")
    step = (float(wave[-1]) - float(wave[0])) / (wave.size - 1)
    if step <= 0 or not np.allclose(np.diff(wave.astype(np.float64)), step, rtol=rtol, atol=0):
        raise ValueError(f"the wavelength grid is not uniform to a relative {rtol:g}")
    return step


def extraction_grid(wavelength, atol_pix=0.01):
    """Return the float64 uniform grid that matches an observed wavelength array.

    The observed grid may be stored in float32, whose rounding (about 5e-4 A at 9800 A) fails a
    strict uniformity test. The grid is rebuilt in float64 from its end points.

    Parameters
    ----------
    wavelength : np.ndarray
        Observed wavelengths in Angstrom, uniform up to rounding.
    atol_pix : float
        Largest allowed distance of any input wavelength from the rebuilt grid, in pixels.

    Returns
    -------
    np.ndarray
        The uniform grid in float64.

    Raises
    ------
    ValueError
        If the input is further than ``atol_pix`` pixels from a uniform grid.
    """
    w = np.asarray(wavelength)
    grid = np.linspace(float(w[0]), float(w[-1]), w.size)
    step = grid[1] - grid[0]
    if step <= 0 or np.max(np.abs(grid - w.astype(np.float64))) > atol_pix * step:
        raise ValueError(f"the wavelength grid is not uniform to {atol_pix:g} of a pixel")
    return grid


def fine_grid(wave, ndiag, step=FINE_STEP):
    """Return a uniform fine grid that covers ``wave`` and the reach of the matrix rows.

    Parameters
    ----------
    wave : np.ndarray
        The uniform extraction grid in Angstrom.
    ndiag : int
        Number of diagonals of the matrix.
    step : float
        Step of the fine grid in Angstrom.

    Returns
    -------
    np.ndarray
        The fine grid, from ndiag//2 + 1 pixels below ``wave[0]`` to as far above ``wave[-1]``.
    """
    dpix = uniform_step(wave)
    pad = (ndiag // 2 + 1) * dpix
    return np.arange(wave[0] - pad, wave[-1] + pad + step / 2, step)


def resampled_matrix(wave, data, fine):
    """Return R', the rows of a banded DESI matrix resampled onto a fine grid.

    Parameters
    ----------
    wave : np.ndarray
        The uniform extraction grid in Angstrom, in float64.
    data : np.ndarray
        Banded matrix of shape (ndiag, n), with R[j - offset_k, j] = data[k, j].
    fine : np.ndarray
        The uniform fine model grid, from ``fine_grid``.

    Returns
    -------
    scipy.sparse.csr_matrix
        R' of shape (n, fine.size). R' @ m is the extracted model for a model m on ``fine``.

    Raises
    ------
    ValueError
        If a grid is not uniform, the shapes do not match, or ``fine`` does not cover the rows.
    """
    data = np.asarray(data, dtype=np.float64)
    ndiag, n = data.shape
    offs = -banded_offsets(ndiag)
    c = ndiag // 2
    if wave.shape != (n,):
        raise ValueError(f"wave has shape {wave.shape}, the matrix has {n} columns")
    dpix = uniform_step(wave)
    step = uniform_step(fine)
    reach = (c + 1) * dpix
    if fine[0] > wave[0] - reach + step / 2 or fine[-1] < wave[-1] + reach - step / 2:
        raise ValueError("the fine grid does not cover the reach of the matrix rows")
    knots = np.arange(-c - 1, c + 2) * dpix
    rows, cols, vals = [], [], []
    for i in range(n):
        # row i holds R[i, j] at j = i + offs, stored at k = c - (j - i)
        j = i + offs
        ok = (j >= 0) & (j < n)
        y = np.zeros(ndiag)
        y[ok] = data[(c - offs)[ok], j[ok]]
        lam = wave[i] + knots
        cs = CubicSpline(lam, np.concatenate([[0.0], y, [0.0]]), bc_type=((1, 0.0), (1, 0.0)))
        f0, f1 = np.searchsorted(fine, [lam[0], lam[-1]])
        rows.append(np.full(f1 - f0, i))
        cols.append(np.arange(f0, f1))
        vals.append(cs(fine[f0:f1]) * step / dpix)
    return csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, fine.size)
    )


def line_profile(fine, mu, sigma_kms, nsig=8.0):
    """Return a unit-flux Gaussian line in f_nu sampled on the fine grid.

    The profile is that of prospect's get_eline_gaussians, a Gaussian in velocity about ``mu``
    times d(v)/d(nu). It is evaluated within ``nsig`` widths of ``mu`` and scaled so that its sum
    over frequency on the fine grid is one. The scale is taken over the whole profile, so a line
    cut by the end of the grid keeps the values it would have inside.

    Parameters
    ----------
    fine : np.ndarray
        The uniform fine grid in Angstrom.
    mu : float
        Line centre in Angstrom.
    sigma_kms : float
        Gaussian width in km/s.
    nsig : float
        Half width of the evaluated window in units of ``sigma_kms``.

    Returns
    -------
    sl : slice
        The fine grid points the line covers. Empty if it misses the grid.
    values : np.ndarray
        The profile at those points, per Hz.

    Raises
    ------
    ValueError
        If the line reaches the grid and is narrower than the fine step, which the grid cannot
        represent.
    """
    step = (fine[-1] - fine[0]) / (fine.size - 1)
    sigma_aa = mu * sigma_kms / C_KMS
    half = nsig * sigma_aa
    k0 = int(np.ceil((mu - half - fine[0]) / step))
    k1 = int(np.floor((mu + half - fine[0]) / step)) + 1
    a, b = max(k0, 0), min(k1, fine.size)
    if b <= a:
        return slice(0, 0), np.zeros(0)
    if not sigma_aa >= step:
        raise ValueError(
            f"a line of {sigma_kms:g} km/s at {mu:.1f} A is narrower than the fine step {step:g} A"
        )
    w = fine[0] + np.arange(k0, k1) * step
    dv = C_KMS * (w / mu - 1)
    g = np.exp(-0.5 * (dv / sigma_kms) ** 2) / (sigma_kms * np.sqrt(2 * np.pi))
    g *= C_KMS * w**2 / (LIGHTSPEED * mu)
    g /= np.sum(g * LIGHTSPEED / w**2) * step
    return slice(a, b), g[a - k0 : b - k0]


def losvd_kernel(sigma_smooth, library_resolution):
    """Return the velocity kernel that takes a library spectrum to a stellar LOSVD.

    The library already has its own resolution, so it is removed in quadrature. A LOSVD at or
    below the library resolution gives no smoothing.

    Parameters
    ----------
    sigma_smooth : float
        Stellar velocity dispersion in km/s.
    library_resolution : float
        Gaussian width of the library spectra in km/s.

    Returns
    -------
    float
        sqrt(sigma_smooth^2 - library_resolution^2) in km/s, or zero.
    """
    d2 = float(sigma_smooth) ** 2 - float(library_resolution) ** 2
    return float(np.sqrt(d2)) if d2 > 0 else 0.0


def smooth_log_uniform(wave, spec, sigma_kms, upsample=4):
    """Smooth a spectrum on a grid uniform in log wavelength and resample it finer.

    The Gaussian kernel is applied in Fourier space on the input samples themselves, so the
    spectrum is not interpolated before smoothing. The result is returned on a grid ``upsample``
    times finer by zero padding in Fourier space, which leaves the input samples unchanged when
    ``sigma_kms`` is zero. A linear ramp between the end points is removed before the transform
    and added back after, so the jump between the two ends does not ring. The transform is
    periodic, so the first and last few kernel widths are unreliable and the caller pads.

    Parameters
    ----------
    wave : np.ndarray
        Wavelengths in Angstrom, uniform in log wavelength.
    spec : np.ndarray
        Flux density at ``wave``.
    sigma_kms : float
        Gaussian width of the kernel in km/s. Zero only resamples.
    upsample : int
        Number of output points per input step.

    Returns
    -------
    wave_out, spec_out : np.ndarray
        The finer grid, from ``wave[0]`` to ``wave[-1]``, and the smoothed spectrum on it.

    Raises
    ------
    ValueError
        If ``wave`` is not uniform in log wavelength to 1% of a step.
    """
    lnw = np.log(np.asarray(wave, dtype=np.float64))
    n = lnw.size
    dln = (lnw[-1] - lnw[0]) / (n - 1)
    if np.max(np.abs(lnw - (lnw[0] + np.arange(n) * dln))) > 0.01 * dln:
        raise ValueError("the wavelength grid is not uniform in log wavelength")
    spec = np.asarray(spec, dtype=np.float64)
    ramp = spec[0] + (spec[-1] - spec[0]) * np.arange(n) / (n - 1)
    ff = np.fft.rfft(spec - ramp)
    if n % 2 == 0:
        # split the Nyquist term between its two frequencies before zero padding
        ff[-1] *= 0.5
    freq = np.fft.rfftfreq(n, d=C_KMS * dln)
    ff *= np.exp(-2 * np.pi**2 * sigma_kms**2 * freq**2)
    m = upsample * (n - 1) + 1
    out = np.fft.irfft(ff, n=upsample * n)[:m] * upsample
    k = np.arange(m) / upsample
    out += spec[0] + (spec[-1] - spec[0]) * k / (n - 1)
    return np.exp(lnw[0] + k * dln), out


def smooth_continuum(wave, spec, sigma_kms, wlo, whi, upsample=4):
    """Smooth a library spectrum by a velocity kernel over a wavelength range.

    The window is ``wlo`` to ``whi`` padded by 20 kernel widths plus ``SMOOTH_PAD_KMS``. If the
    library samples the window uniformly in log wavelength (C3K_ER does at rest 3300.5-10000 A)
    it is smoothed on its own samples with ``smooth_log_uniform``. Otherwise sedpy's smoothspec
    resamples and smooths it, as prospect's losvd_smoothing does.

    Parameters
    ----------
    wave : np.ndarray
        Library wavelengths in Angstrom.
    spec : np.ndarray
        Library flux density at ``wave``.
    sigma_kms : float
        Gaussian width of the kernel in km/s, for example from ``losvd_kernel``.
    wlo, whi : float
        Range to return, in the same frame as ``wave``.
    upsample : int
        Output points per library step on the log-uniform path.

    Returns
    -------
    wave_out, spec_out : np.ndarray
        Wavelengths that cover ``wlo`` to ``whi`` and the smoothed spectrum there.
    """
    pad = (20 * sigma_kms + SMOOTH_PAD_KMS) / C_KMS
    i0 = max(int(np.searchsorted(wave, wlo * (1 - pad))) - 1, 0)
    i1 = min(int(np.searchsorted(wave, whi * (1 + pad))) + 1, wave.size)
    w, s = wave[i0:i1], spec[i0:i1]
    try:
        return smooth_log_uniform(w, s, sigma_kms, upsample=upsample)
    except ValueError:
        pass
    if sigma_kms <= 0:
        return w, np.asarray(s, dtype=np.float64)
    out = smoothspec(wave, spec, sigma_kms, outwave=w, smoothtype="vel", fftsmooth=True)
    return w, out


def load_resolution_npz(path, arm=None):
    """Load a coadded spectrum and its banded resolution matrix from an npz file.

    The format is that of resolution_matrix_rows.npz and resolution_matrix_rows_ivar.npz in
    experiments/2026-10-09_nautilus_c3k_er_broadline (coadd_rows.py, ivar_recompute.py). The
    keys wave, res, flux and ivar hold the arms merged onto the DESI grid, and the same keys with
    a b_, r_ or z_ prefix hold each arm. The flux is in the DESI units of 1e-17 erg/s/cm^2/A.

    Parameters
    ----------
    path : str or pathlib.Path
        The npz file.
    arm : {None, "b", "r", "z"}
        Arm to load. None loads the merged spectrum.

    Returns
    -------
    dict
        wave (uniform, float64), res (shape (ndiag, n), float64), flux and ivar (float64).

    Raises
    ------
    ValueError
        If the grid is not uniform or the matrix shape does not match it.
    """
    if arm not in (None, "b", "r", "z"):
        raise ValueError(f"arm must be None, 'b', 'r' or 'z', not {arm!r}")
    pre = "" if arm is None else f"{arm}_"
    with np.load(path) as d:
        out = {k: np.asarray(d[pre + k], dtype=np.float64) for k in ("wave", "res", "flux", "ivar")}
    uniform_step(out["wave"])
    ndiag, n = out["res"].shape
    banded_offsets(ndiag)
    if n != out["wave"].size or out["flux"].shape != (n,) or out["ivar"].shape != (n,):
        raise ValueError(f"{path}: the matrix, flux and ivar do not match the {n} wavelengths")
    return out
