"""Apply a DESI resolution matrix to a high-resolution model, resampled as in Guy et al. 2023.

The DESI matrix R is square on the 0.8 A extraction grid. Rebinning a model with unresolved
features onto that grid before applying R biases it (Guy et al. 2023, arXiv:2209.14482, appendix
on resampling the resolution matrix). There each row of R is interpolated with a cubic spline onto
the model's finer grid, which gives a non-square matrix R' that applies the resolution and rebins
in one step. The paper does not give the normalisation. Here each spline value is multiplied by
the fine step over the 0.8 A step, so R' reproduces R on a smooth model and an unresolved line keeps
the normalisation of R. The spline is tied to zero one pixel beyond the outermost diagonals, with
zero slope there.

This is a prototype for the evaluation and MAP checks in this folder, not the prospect feature.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix

from hubersed.sps.broadline import TwoCompLineModel

C_KMS = 299792.458
C_AA = 2.998e18  # speed of light in Angstrom/s, as prospect's lightspeed
FINE_STEP = 0.05  # Angstrom


def fine_grid(wave, ndiag, step=FINE_STEP):
    """Return a uniform fine grid covering ``wave`` and the reach of the matrix rows."""
    dpix = np.median(np.diff(wave))
    pad = (ndiag // 2 + 1) * dpix
    return np.arange(wave[0] - pad, wave[-1] + pad + step / 2, step)


def resampled_matrix(wave, data, fine):
    """Return R', the rows of a banded DESI matrix resampled onto ``fine``.

    Parameters
    ----------
    wave : np.ndarray
        The uniform extraction grid in Angstrom.
    data : np.ndarray
        Banded matrix, shape ``(ndiag, n)``, with R[j - offset_k, j] = data[k, j] and offsets
        running from ndiag//2 down to -(ndiag//2), as desispec.resolution.Resolution.
    fine : np.ndarray
        The uniform fine model grid, from ``fine_grid``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(n, fine.size)``. R' @ m is the extracted model for a model m on ``fine``.
    """
    ndiag, n = data.shape
    c = ndiag // 2
    dpix = np.median(np.diff(wave))
    assert np.allclose(np.diff(wave), dpix, rtol=1e-6), "the extraction grid must be uniform"
    step = fine[1] - fine[0]
    offs = np.arange(-c, c + 1)
    rows, cols, vals = [], [], []
    for i in range(n):
        # row i holds R[i, j] at j = i + offs, stored at k = c - (j - i)
        j = i + offs
        ok = (j >= 0) & (j < n)
        y = np.zeros(ndiag)
        y[ok] = data[(c - offs)[ok], j[ok]]
        lam = wave[i] + np.arange(-c - 1, c + 2) * dpix
        cs = CubicSpline(lam, np.concatenate([[0.0], y, [0.0]]), bc_type=((1, 0.0), (1, 0.0)))
        f0, f1 = np.searchsorted(fine, [lam[0], lam[-1]])
        rows.append(np.full(f1 - f0, i))
        cols.append(np.arange(f0, f1))
        vals.append(cs(fine[f0:f1]) * step / dpix)
    return csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, fine.size)
    )


def line_segment(fine, mu, sigma_kms, nsig=8.0):
    """Return the fine-grid slice and values of a unit-flux Gaussian line in f_nu (per Hz).

    The profile is prospect's get_eline_gaussians, a Gaussian in velocity about ``mu`` times
    d(v)/d(nu), sampled at the fine grid points. Its integral over frequency is one.
    """
    half = nsig * mu * sigma_kms / C_KMS
    f0, f1 = np.searchsorted(fine, [mu - half, mu + half])
    w = fine[f0:f1]
    dv = C_KMS * (w / mu - 1)
    g = np.exp(-0.5 * (dv / sigma_kms) ** 2) / (sigma_kms * np.sqrt(2 * np.pi))
    return slice(f0, f1), g * C_KMS * w**2 / (C_AA * mu)


def intrinsic_line_segments(model, fine):
    """Return each added line of a TwoCompLineModel at its intrinsic widths on ``fine``.

    Must run after the model has predicted the observation, since it reads the cached line
    parameters. The instrumental part prospect adds to the widths is removed.

    Returns
    -------
    inds : np.ndarray
        Indices of the added lines in the model's line list.
    segments : list of list of (slice, np.ndarray)
        For each line, its components in maggies.
    """
    inds = np.where(model._fix_eline & model._valid_eline)[0]
    prof, _, _, inst2 = model._profile_widths()
    narrow = np.sqrt(np.clip(prof**2 - inst2, 0, None))
    fb = model._broad_fraction()
    s_b = model._p("eline_sigma_broad", 150.0)
    v_b = model._p("eline_vbroad", 0.0)
    elum = model._eline_lum * model.flux_norm() / (1 + model._zred)
    segments = []
    for i in inds:
        mu = model._ewave_obs[i]
        sl, g = line_segment(fine, mu, narrow[i])
        comps = [(sl, (1 - fb[i]) * elum[i] * g)]
        if fb[i] > 0:
            sl, g = line_segment(fine, mu * (1 + v_b / C_KMS), s_b)
            comps.append((sl, fb[i] * elum[i] * g))
        segments.append(comps)
    return inds, segments


def apply_segments(rp, comps):
    """Return R' applied to a line given as fine-grid components."""
    return sum(rp[:, sl] @ g for sl, g in comps)


def continuum(model, sigma):
    """Return the model continuum smoothed by a velocity dispersion ``sigma``, on the library grid.

    Below 1 km/s the continuum is left unsmoothed.
    """
    keep = model.params["sigma_smooth"]
    try:
        if sigma < 1.0:
            spec = model._norm_spec.copy()
        else:
            model.params["sigma_smooth"] = np.atleast_1d(sigma)
            spec = model.losvd_smoothing(model._wave, model._norm_spec)
        spec = model.add_dla(model._wave, spec)
        spec = model.add_damping_wing(model._wave, spec)
    finally:
        model.params["sigma_smooth"] = keep
    return spec


class ResampledRModel(TwoCompLineModel):
    """TwoCompLineModel that applies a resampled resolution matrix R' instead of prospect's LSF.

    The continuum is smoothed by sqrt(sigma_smooth^2 - lib^2), with lib the median library
    resolution over the fine grid, so the library resolution is not counted twice (no smoothing if
    sigma_smooth is below lib). The lines have their intrinsic widths only. Both are put on the fine
    grid and R' is applied once. Call ``set_resolution`` before predicting.
    """

    def set_resolution(self, rp, fine):
        """Store R' and its fine grid, from ``resampled_matrix`` and ``fine_grid``."""
        self._rp, self._fine = rp, fine

    def predict_spec(self, obs):
        """Return the model spectrum on the observation's grid, through R'."""
        obs_wave = self.observed_wave(self._wave, do_wavecal=False)
        self._outwave = obs.wavelength
        self.cache_eline_parameters(obs)
        sel = (obs_wave > self._fine[0]) & (obs_wave < self._fine[-1])
        lib = float(np.median(np.broadcast_to(self._library_resolution, obs_wave.shape)[sel]))
        sig = float(np.atleast_1d(self.params["sigma_smooth"])[0])
        cont = continuum(self, np.sqrt(max(sig**2 - lib**2, 0.0)))
        m = np.interp(self._fine, obs_wave, cont)
        _, segments = intrinsic_line_segments(self, self._fine)
        for comps in segments:
            for sl, g in comps:
                m[sl] += g
        spec = self._rp @ m
        response = obs.compute_response(spec=spec, **self.params)
        self._speccal = response
        self._sed = spec
        return spec * response
