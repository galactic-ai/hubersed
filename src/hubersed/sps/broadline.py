"""Two-component Balmer-line profiles, and an optional separate forbidden-line width.

Each line's total luminosity still comes from Cue and the star formation history, so the
Balmer flux stays tied to the young stars. Only the line profile changes::

    Balmer lines:     (1 - f_b) * G(sigma_n) + f_b * G(sigma_b, shifted by v_b)
    all other lines:  G(sigma_forb), with sigma_forb = sigma_n unless eline_sigma_forb is set
"""

import numpy as np
from prospect.models import priors
from prospect.models.sedmodel import HyperSpecModel

C_KMS = 299792.458

BROAD_PARAMS = {
    "eline_fbroad": dict(N=1, isfree=True, init=0.2, prior=priors.TopHat(mini=0.0, maxi=0.9)),
    "eline_sigma_broad": dict(
        N=1, isfree=True, init=150.0, units="km/s", prior=priors.TopHat(mini=100.0, maxi=800.0)
    ),
    "eline_vbroad": dict(
        N=1, isfree=True, init=0.0, units="km/s", prior=priors.TopHat(mini=-300.0, maxi=300.0)
    ),
}
FORB_PARAM = {
    "eline_sigma_forb": dict(
        N=1, isfree=True, init=30.0, units="km/s", prior=priors.TopHat(mini=10.0, maxi=100.0)
    )
}
# Balmer lines are named "Ba-..." in the FSPS and Cue line lists.
BALMER_PREFIX = "Ba-"
# Cue's cue_emlines_info.dat lists "He I 3888.63A" twice at 3889.7419 A and has no H8. FSPS's
# emlines_info.dat has He I 3888.63A followed by "Ba-6 3889" (H8) at the same place in the list,
# so the second duplicate is taken to be H8. This is inferred from the ordering, not documented.
CUE_H8_DUPLICATE = "He I 3888.63A"
# The narrow width must stay below the broad minimum so the components cannot swap.
NARROW_PRIOR = priors.TopHat(mini=10.0, maxi=100.0)


class TwoCompLineModel(HyperSpecModel):
    """HyperSpecModel with a narrow plus broad profile for the Balmer lines.

    The broad component takes a fraction ``eline_fbroad`` of each Balmer line's flux, so the
    line fluxes are unchanged. With ``eline_fbroad = 0`` and ``eline_sigma_forb`` equal to
    ``eline_sigma`` (or absent) the model is identical to HyperSpecModel.
    """

    def _p(self, name, default):
        """Return the first element of a model parameter, or ``default`` if it is not set."""
        return float(np.atleast_1d(self.params.get(name, default))[0])

    def _balmer_mask(self):
        """Return a boolean mask of the Balmer lines in ``emline_info``, cached on first use."""
        if not hasattr(self, "_is_balmer"):
            names = np.char.strip(np.asarray(self.emline_info["name"]).astype(str))
            is_balmer = np.char.startswith(names, BALMER_PREFIX)
            dup = np.flatnonzero(names == CUE_H8_DUPLICATE)
            if len(dup) == 2:  # Cue list: the second copy is H8, see CUE_H8_DUPLICATE
                is_balmer[dup[1]] = True
            self._is_balmer = is_balmer
        return self._is_balmer

    def _profile_widths(self):
        """Return per-line total widths of the main component and the widest component.

        Returns
        -------
        prof : np.ndarray
            Total width in km/s of the narrow (Balmer) or forbidden (other lines) component,
            including the instrumental and library part cached by prospect.
        wide : np.ndarray
            Total width in km/s of the widest component of each line.
        shift : np.ndarray
            Absolute velocity offset in km/s of the broad component (zero for other lines).
        inst2 : np.ndarray
            Squared instrumental (and library) width in km/s.
        """
        sig0 = np.atleast_1d(self._eline_sigma_kms) * np.ones_like(self._ewave_obs, dtype=float)
        bal = self._balmer_mask()
        # the cached width is hypot(eline_sigma, instrument); keep the instrumental part
        inst2 = np.clip(sig0**2 - self._p("eline_sigma", 0) ** 2, 0, None)
        s_forb = self._p("eline_sigma_forb", self._p("eline_sigma", 0))
        prof = np.where(bal, sig0, np.sqrt(s_forb**2 + inst2))
        wide, shift = prof.copy(), np.zeros_like(prof)
        if self._p("eline_fbroad", 0.0) > 0:
            broad = np.sqrt(self._p("eline_sigma_broad", 150.0) ** 2 + inst2)
            wide = np.where(bal, np.maximum(prof, broad), prof)
            shift = np.where(bal, abs(self._p("eline_vbroad", 0.0)), 0.0)
        return prof, wide, shift, inst2

    def cache_eline_parameters(self, obs, nsigma=5, forcelines=False):
        """Cache the line parameters, with line windows set by each line's own profile.

        A line is kept if a fitted pixel lies within ``nsigma`` widths of its main component
        (narrow for Balmer lines, forbidden width otherwise), as in prospect. The pixels where
        lines are added extend to ``nsigma`` widths of the widest component, so broad wings are
        not cut. Widening the validity window itself would keep lines whose main component is
        zero on every pixel, and prospect's unit-flux renormalization then divides by zero.

        Parameters
        ----------
        obs : prospect.observation.Spectrum
            The observation.
        nsigma : float
            Half-width of the line windows in units of the component width.
        forcelines : bool
            Passed to ``HyperSpecModel.cache_eline_parameters``.
        """
        super().cache_eline_parameters(obs, nsigma=nsigma, forcelines=forcelines)
        hasspec = obs.get("spectrum", None) is not None
        if not (self._want_lines & self._need_lines & hasspec):
            return
        prof, wide, shift, _ = self._profile_widths()
        dist = np.abs(self._outwave - self._ewave_obs[:, None])
        omask = obs.get("mask", None)

        def window(half_kms):
            pm = dist < (self._ewave_obs / C_KMS * half_kms)[:, None]
            return pm if omask is None else pm & omask

        self._valid_eline = window(nsigma * prof).any(axis=1) & self._use_eline
        pm = window(nsigma * wide + shift)
        self._fit_eline_pixelmask = pm[self._valid_eline & self._fit_eline, :].any(axis=0)
        self._fix_eline_pixelmask = pm[self._valid_eline & self._fix_eline, :].any(axis=0)
        self._elines_to_fit = self._fit_eline & self._valid_eline

    def get_eline_gaussians(self, lineidx=slice(None), wave=None):  # noqa: B008, same as parent
        """Return unit-flux line profiles, two-component for the Balmer lines.

        Parameters
        ----------
        lineidx : slice, np.ndarray of bool or int
            The cached lines to build profiles for.
        wave : np.ndarray, optional
            Observed wavelengths in Angstrom. Defaults to the cached output grid.

        Returns
        -------
        np.ndarray
            Profiles of shape (n_wave, n_line), each normalized to unit flux.
        """
        sig0, mu0 = self._eline_sigma_kms, self._ewave_obs
        bal = self._balmer_mask()
        prof, _, _, inst2 = self._profile_widths()
        fb = self._p("eline_fbroad", 0.0)
        try:
            self._eline_sigma_kms = prof
            g = super().get_eline_gaussians(lineidx=lineidx, wave=wave)
            if fb > 0:
                self._eline_sigma_kms = np.sqrt(self._p("eline_sigma_broad", 150.0) ** 2 + inst2)
                self._ewave_obs = mu0 * (1 + self._p("eline_vbroad", 0.0) / C_KMS)
                gb = super().get_eline_gaussians(lineidx=lineidx, wave=wave)
                b = bal[lineidx]
                g[:, b] = (1 - fb) * g[:, b] + fb * gb[:, b]
        finally:
            self._eline_sigma_kms, self._ewave_obs = sig0, mu0
        return g


def add_broad_params(params, separate_forbidden_width=True):
    """Add the broad-line parameters to a model parameter dict.

    Parameters
    ----------
    params : dict
        Prospect model parameter specification. It is changed in place; the ``eline_sigma``
        entry is replaced by a copy with the narrow prior, so templates it came from are not
        changed.
    separate_forbidden_width : bool
        Also add ``eline_sigma_forb``, a separate width for the non-Balmer lines.

    Returns
    -------
    dict
        ``params``, with the new parameters.
    """
    params.update({k: dict(v) for k, v in BROAD_PARAMS.items()})
    if separate_forbidden_width:
        params.update({k: dict(v) for k, v in FORB_PARAM.items()})
    params["eline_sigma"] = {**params["eline_sigma"], "prior": NARROW_PRIOR}
    return params


def _trapezoid(y, x):
    """Integrate ``y`` over ``x`` with the trapezoid rule."""
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))


def self_check(model_params, obs, sps, theta, n_prior=100, seed=0):
    """Check TwoCompLineModel against HyperSpecModel before sampling.

    With ``eline_fbroad = 0`` and ``eline_sigma_forb = eline_sigma`` the two-component model
    must reproduce the one-component model. With ``eline_fbroad = 0.5`` the Halpha + [NII] flux
    should barely change, because the broad component redistributes flux instead of adding it.

    Parameters
    ----------
    model_params : dict
        Parameter specification of the two-component model (after ``add_broad_params``).
    obs : prospect.observation.Spectrum or list
        The observation, or a one-element list of it as ``build_obs`` returns.
    sps : prospect.sources.SSPBasis
        The stellar population source.
    theta : np.ndarray
        Any valid parameter vector of the two-component model.
    n_prior : int
        Number of prior draws whose predicted spectrum must be finite.
    seed : int
        Seed for the prior draws.

    Raises
    ------
    AssertionError
        If ``eline_fbroad = 0`` does not reproduce the one-component model, or a prior draw gives
        a non-finite spectrum.
    """
    if isinstance(obs, list | tuple):
        (obs,) = obs
    m = TwoCompLineModel(model_params)
    ib = m.theta_index["eline_fbroad"].start
    new = [k for k in {**BROAD_PARAMS, **FORB_PARAM} if k in m.theta_index]
    drop = [m.theta_index[k].start for k in new]
    base = HyperSpecModel({k: v for k, v in model_params.items() if k not in new})
    t0 = np.array(theta, dtype=float)
    t0[ib] = 0.0
    if "eline_sigma_forb" in m.theta_index:  # same width for all lines, so equal to the base
        t0[m.theta_index["eline_sigma_forb"].start] = t0[m.theta_index["eline_sigma"].start]
    s_base = base.predict(np.delete(t0, drop), observations=[obs], sps=sps)[0][0]
    s0 = m.predict(t0, observations=[obs], sps=sps)[0][0]
    assert np.allclose(s0, s_base, rtol=1e-6), "f_b = 0 must reproduce the one-component model"

    t1 = t0.copy()
    t1[ib] = 0.5
    s1 = m.predict(t1, observations=[obs], sps=sps)[0][0]
    wr = obs.wavelength / (1 + float(np.atleast_1d(m.params["zred"])[0]))
    sel = (wr > 6480) & (wr < 6650)
    cb = np.median(s0[(wr > 6480) & (wr < 6500)])
    cr = np.median(s0[(wr > 6630) & (wr < 6650)])
    cont = np.interp(wr[sel], [6490, 6640], [cb, cr])
    frac = _trapezoid(s1[sel] - s0[sel], wr[sel]) / _trapezoid(s0[sel] - cont, wr[sel])
    rng = np.random.default_rng(seed)
    bad = 0
    with np.errstate(all="ignore"):
        for _ in range(n_prior):
            t = m.prior_transform(rng.uniform(size=m.ndim))
            bad += not np.all(np.isfinite(m.predict(t, observations=[obs], sps=sps)[0][0]))
    assert bad == 0, f"{bad} of {n_prior} prior draws give a non-finite spectrum"
    print(
        f"OK: f_b=0 reproduces the one-component model. Halpha+[NII] flux change at f_b=0.5: "
        f"{frac:+.2%} (should be ~0: broad flux is redistributed, not added). "
        f"{n_prior} prior draws give finite spectra."
    )
