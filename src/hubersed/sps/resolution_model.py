"""A prospect observation and model that use a target's own DESI resolution matrix.

``ResolutionMatrixSpectrum`` is a prospect Spectrum that carries the banded resolution matrix R of
its pixels and builds the resampled matrix R' of ``hubersed.sps.resolution_matrix`` once, on first
use. ``ResolutionMatrixModel`` is a TwoCompLineModel whose spectrum prediction, for such an
observation, replaces prospect's Gaussian line spread function by R'.

The model is built in four steps.

1. The library continuum is smoothed by the stellar LOSVD with the library resolution removed in
   quadrature, sqrt(sigma_smooth^2 - lib^2), and not at all when sigma_smooth <= lib. Prospect
   instead removes lib from the instrumental kernel. The total width is the same when
   sigma_smooth > lib.
2. The emission lines are built at their intrinsic widths only, with the groups, broad component
   and shift of TwoCompLineModel. Prospect adds the instrumental width to each line, and here R'
   does that.
3. R' is applied once to the continuum plus lines on the fine grid.
4. The response (calibration vector) is applied as in prospect's predict_spec.

Every emission line whose profile reaches the fine grid is added, whatever the observation mask.
R spreads light by ndiag//2 pixels, so a line centred on a masked pixel (a sky line or a line
mask) can still reach fitted pixels. Prospect's own line selection uses the mask and would drop
it. Line marginalisation (prospect's fitted lines) is not supported.

Both classes hold only numpy arrays and a scipy sparse matrix, so they pickle. The cached R' is
left out of the pickle and rebuilt on first use, in about half a second for a DESI spectrum.
"""

import numpy as np
from prospect.observation import Spectrum

from hubersed.sps.broadline import C_KMS, TwoCompLineModel
from hubersed.sps.parameter_file import WAVE_OBS
from hubersed.sps.resolution_matrix import (
    FINE_STEP,
    banded_offsets,
    banded_to_csr,
    extraction_grid,
    fine_grid,
    line_profile,
    losvd_kernel,
    resampled_matrix,
    smooth_continuum,
    uniform_step,
)

# Largest spread of the library resolution, in km/s, allowed over the wavelengths that reach
# fitted pixels. One value of lib is removed from the LOSVD, so it must be constant there.
LIBRARY_RESOLUTION_TOL = 0.5
# Smallest row sum of R allowed at a fitted pixel. A zero row would give a zero model there.
MIN_ROW_SUM = 0.5


class ResolutionMatrixSpectrum(Spectrum):
    """A prospect Spectrum with the banded DESI resolution matrix of its pixels.

    Parameters
    ----------
    wavelength : np.ndarray
        Observed wavelengths in Angstrom, uniform. A float32 grid is accepted, and R' is built
        on its float64 equivalent.
    resolution_matrix : np.ndarray
        Banded matrix of shape (ndiag, n), already coadded and merged over the arms, with
        R[j - offset_k, j] = data[k, j] as desispec.resolution.Resolution.
    resolution_wave : np.ndarray, optional
        The float64 grid the matrix was made on. It must match ``wavelength`` to 0.01 pixel.
        If not given, it is rebuilt from ``wavelength``.
    fine_step : float
        Step of the fine model grid in Angstrom.
    **kwargs
        Passed to prospect's Spectrum, for example flux, uncertainty and mask. ``resolution``
        must not be given, since R replaces it.
    """

    def __init__(
        self,
        wavelength=None,
        resolution_matrix=None,
        resolution_wave=None,
        fine_step=FINE_STEP,
        **kwargs,
    ):
        if kwargs.pop("resolution", None) is not None:
            raise ValueError("ResolutionMatrixSpectrum takes R instead of a Gaussian resolution")
        if wavelength is None or resolution_matrix is None:
            raise ValueError("ResolutionMatrixSpectrum needs wavelength and resolution_matrix")
        super().__init__(wavelength=wavelength, resolution=None, **kwargs)
        data = np.asarray(resolution_matrix, dtype=np.float64)
        banded_offsets(data.shape[0])
        if data.shape[1] != self.wavelength.size:
            raise ValueError(
                f"the matrix has {data.shape[1]} columns for {self.wavelength.size} wavelengths"
            )
        grid = extraction_grid(self.wavelength)
        if resolution_wave is not None:
            rw = np.asarray(resolution_wave, dtype=np.float64)
            step = uniform_step(rw)
            if rw.shape != grid.shape or np.max(np.abs(rw - grid)) > 0.01 * step:
                raise ValueError("resolution_wave does not match wavelength")
            grid = rw
        self.resolution_matrix = data
        self.resolution_wave = grid
        self.fine_step = float(fine_step)
        self._rprime = None

    def __getstate__(self):
        """Return the state for pickling, without the cached R'."""
        state = self.__dict__.copy()
        state["_rprime"] = None
        return state

    def resampled(self):
        """Return R', its fine grid and the row sums of R, building them on the first call.

        Returns
        -------
        rprime : scipy.sparse.csr_matrix
            R' of shape (n, fine.size).
        fine : np.ndarray
            The fine model grid in Angstrom.
        row_sums : np.ndarray
            Row sums of R.
        """
        if self._rprime is None:
            data, wave = self.resolution_matrix, self.resolution_wave
            fine = fine_grid(wave, data.shape[0], step=self.fine_step)
            rows = np.asarray(banded_to_csr(data).sum(axis=1)).ravel()
            self._rprime = (resampled_matrix(wave, data, fine), fine, rows)
        return self._rprime

    def mask_array(self):
        """Return the observation mask as a boolean array over the pixels."""
        m = np.zeros(self.wavelength.size, dtype=bool)
        m[self.mask] = True
        return m

    def reach(self):
        """Return the observed wavelength range that R' takes to the fitted pixels.

        Returns
        -------
        tuple of float
            The lowest and highest wavelength, in Angstrom, where a fitted pixel's row of R' can
            be nonzero, plus one fine step.
        """
        mask = self.mask_array()
        w = self.resolution_wave[mask]
        c = self.resolution_matrix.shape[0] // 2
        pad = (c + 1) * (self.resolution_wave[1] - self.resolution_wave[0]) + self.fine_step
        return float(w.min() - pad), float(w.max() + pad)

    def check_rows(self):
        """Raise ValueError if a fitted pixel has a row of R that sums to less than MIN_ROW_SUM."""
        rows = self.resampled()[2]
        bad = self.mask_array() & ~(rows >= MIN_ROW_SUM)
        if bad.any():
            raise ValueError(
                f"{bad.sum()} fitted pixels have R row sums below {MIN_ROW_SUM}, "
                f"first at {self.resolution_wave[bad][0]:.1f} A; mask them"
            )

    def instrumental_smoothing(self, *args, **kwargs):
        """Refuse prospect's Gaussian smoothing, which would leave out R.

        Raises
        ------
        NotImplementedError
            Always. Predict this observation with ResolutionMatrixModel.
        """
        raise NotImplementedError("predict a ResolutionMatrixSpectrum with ResolutionMatrixModel")


def build_resolution_obs(
    spec, unc, mask, resolution_matrix, resolution_wave=None, wavelength=WAVE_OBS
):
    """Wrap a DESI spectrum and its resolution matrix as a prospect observation.

    Parameters
    ----------
    spec : np.ndarray
        Observed flux in maggies.
    unc : np.ndarray
        One sigma uncertainty of ``spec`` in maggies.
    mask : np.ndarray
        True for pixels used in the likelihood.
    resolution_matrix : np.ndarray
        Banded matrix of shape (ndiag, n), see ``ResolutionMatrixSpectrum``.
    resolution_wave : np.ndarray, optional
        The float64 grid of the matrix, for example the wave key of ``load_resolution_npz``.
    wavelength : np.ndarray
        Observed wavelength in Angstrom. The default is the DESI grid.

    Returns
    -------
    list of ResolutionMatrixSpectrum
        A one-element list, already passed through ``rectify``.
    """
    obs = ResolutionMatrixSpectrum(
        wavelength=wavelength,
        resolution_matrix=resolution_matrix,
        resolution_wave=resolution_wave,
        flux=spec,
        uncertainty=unc,
        mask=mask,
    )
    obs.rectify()
    return [obs]


class ResolutionMatrixModel(TwoCompLineModel):
    """TwoCompLineModel that applies an observation's resampled resolution matrix R'.

    For a ``ResolutionMatrixSpectrum`` the prediction follows the steps in the module docstring.
    Any other observation is predicted as by TwoCompLineModel. After a prediction through R' the
    model also holds ``_resolved_continuum`` and ``_resolved_lines``, the continuum and the line
    spectrum after R' and before the response. ``_fix_eline_pixelmask`` marks the pixels where
    the line spectrum is nonzero, and ``_eline_sigma_kms`` holds ``eline_sigma`` with no
    instrumental part, so ``get_eline_gaussians`` gives intrinsic profiles.
    """

    def library_resolution(self, wlo, whi):
        """Return the library resolution over a rest-frame range, which must be constant.

        Parameters
        ----------
        wlo, whi : float
            Rest-frame range in Angstrom.

        Returns
        -------
        float
            The median library resolution in km/s over the range.

        Raises
        ------
        ValueError
            If it varies by more than LIBRARY_RESOLUTION_TOL over the range, for example where
            C3K_ER drops to R = 200 below rest 3300.5 A.
        """
        lib = np.abs(np.broadcast_to(self._library_resolution, self._wave.shape))
        sel = (self._wave >= wlo) & (self._wave <= whi)
        if not sel.any():
            return float(np.interp(0.5 * (wlo + whi), self._wave, lib))
        lo, hi = lib[sel].min(), lib[sel].max()
        if hi - lo > LIBRARY_RESOLUTION_TOL:
            raise ValueError(
                f"the library resolution varies from {lo:.1f} to {hi:.1f} km/s at rest "
                f"{wlo:.0f}-{whi:.0f} A; mask the pixels outside the library's high-resolution range"
            )
        return float(np.median(lib[sel]))

    def fine_continuum(self, fine, reach):
        """Return the stellar continuum on the fine grid, smoothed by the stellar LOSVD only.

        Within ``reach`` the library spectrum is smoothed by ``losvd_kernel(sigma_smooth, lib)``
        and Lyman alpha absorption is applied as in prospect's predict_init. Outside it, where no
        fitted pixel looks, prospect's own LOSVD-smoothed spectrum from predict_init is used.

        Parameters
        ----------
        fine : np.ndarray
            The fine grid in observed Angstrom.
        reach : tuple of float
            Observed range that reaches fitted pixels, from ``ResolutionMatrixSpectrum.reach``.

        Returns
        -------
        np.ndarray
            The continuum in maggies at ``fine``.
        """
        a = 1 + float(np.atleast_1d(self._zred)[0])
        lo, hi = reach[0] / a, reach[1] / a
        lib = self.library_resolution(lo, hi)
        kernel = losvd_kernel(self._p("sigma_smooth", 300.0), lib)
        w, s = smooth_continuum(self._wave, self._norm_spec, kernel, lo, hi)
        s = self.add_damping_wing(w, self.add_dla(w, s))
        m = np.interp(fine, self._wave * a, self._smooth_spec)
        sel = (fine >= reach[0]) & (fine <= reach[1])
        m[sel] = np.interp(fine[sel], w * a, s)
        return m

    def fine_lines(self, fine):
        """Return the emission lines on the fine grid at their intrinsic widths.

        The narrow, forbidden and high-ionization widths and the broad component (fraction, width
        and shift) are those of TwoCompLineModel. Every fixed line whose profile reaches the fine
        grid is added, whatever the observation mask.

        Parameters
        ----------
        fine : np.ndarray
            The fine grid in observed Angstrom.

        Returns
        -------
        spec : np.ndarray
            The line spectrum in maggies at ``fine``.
        added : np.ndarray of bool
            The lines that were added.

        Raises
        ------
        NotImplementedError
            If any line is set to be fitted (prospect's line marginalisation).
        """
        nline = self._ewave_obs.shape[0]
        spec = np.zeros(fine.size)
        added = np.zeros(nline, dtype=bool)
        if not (self._want_lines and self._need_lines):
            return spec, added
        if np.any(self._fit_eline & self._use_eline):
            raise NotImplementedError("line marginalisation is not supported with R'")
        if "eline_sigma" not in self.params:
            raise ValueError("ResolutionMatrixModel needs the eline_sigma parameter")
        # with no instrumental part in the cached widths, _profile_widths gives intrinsic widths
        self._eline_sigma_kms = np.full(nline, self._p("eline_sigma", 0.0))
        prof, _, _, inst2 = self._profile_widths()
        assert np.all(inst2 == 0)
        fb = self._broad_fraction()
        s_b = self._p("eline_sigma_broad", 150.0)
        v_b = self._p("eline_vbroad", 0.0)
        elum = self._eline_lum * self.flux_norm() / (1 + self._zred)
        for i in np.where(self._use_eline & self._fix_eline)[0]:
            mu = self._ewave_obs[i]
            comps = [(mu, prof[i], 1 - fb[i])]
            if fb[i] > 0:
                comps.append((mu * (1 + v_b / C_KMS), s_b, fb[i]))
            for mu_c, sigma, frac in comps:
                sl, g = line_profile(fine, mu_c, sigma)
                if g.size:
                    spec[sl] += frac * elum[i] * g
                    added[i] = True
        return spec, added

    def predict_spec(self, obs):
        """Return the model spectrum on the observation's pixels, in maggies.

        Parameters
        ----------
        obs : prospect.observation.Spectrum
            The observation. A ResolutionMatrixSpectrum goes through R'. Any other is passed to
            TwoCompLineModel.

        Returns
        -------
        np.ndarray
            The predicted spectrum, including the response.
        """
        if not isinstance(obs, ResolutionMatrixSpectrum):
            return super().predict_spec(obs)
        rprime, fine, _ = obs.resampled()
        obs.check_rows()
        self._outwave = obs.wavelength
        cont = self.fine_continuum(fine, obs.reach())
        lines, added = self.fine_lines(fine)
        self._resolved_continuum = rprime @ cont
        self._resolved_lines = rprime @ lines
        spec = self._resolved_continuum + self._resolved_lines
        n = spec.size
        self._valid_eline = added & self._use_eline
        self._elines_to_fit = np.zeros_like(added)
        self._fit_eline_pixelmask = np.zeros(n, dtype=bool)
        self._fix_eline_pixelmask = self._resolved_lines != 0
        response = obs.compute_response(spec=spec, extra_mask=True, **self.params)
        self._speccal = response
        self._sed = spec
        return spec * response
