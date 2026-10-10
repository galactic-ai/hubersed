"""Two-component Balmer-line profiles, and an optional separate forbidden-line width.

Each line's total luminosity still comes from Cue and the star formation history, so the
Balmer flux stays tied to the young stars. Only the line profile changes::

    Balmer lines:     (1 - f_b) * G(sigma_n) + f_b * G(sigma_b, shifted by v_b)
    all other lines:  G(sigma_forb), with sigma_forb = sigma_n unless eline_sigma_forb is set

Two options of ``add_broad_params``, both off by default, change the line groups. With
``he_balmer_profile`` the He I lines take the Balmer profile. With ``split_forbidden`` the
forbidden lines of ions made above the O+ to O++ edge (35.12 eV: [O III], [Ne III], [Ar IV],
[Ne IV], He II) get their own width ``eline_sigma_forb_hi``. Every other non-Balmer line,
including [S III], [Ar III] and [Cl III], keeps ``eline_sigma_forb``. The groups and their sources
are in knowledge/line_groups_ionization_zones_2026-10-10.md (UberSED notes); they cover the Cue
lines at rest 3500-9000 A, and lines outside that range keep the low group.
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
FORB_HI_PARAM = {
    "eline_sigma_forb_hi": dict(
        N=1, isfree=True, init=30.0, units="km/s", prior=priors.TopHat(mini=10.0, maxi=100.0)
    )
}
# Balmer lines are named "Ba-..." in the FSPS and Cue line lists, He I lines "He I ...".
BALMER_PREFIX = "Ba-"
HEI_PREFIX = "He I "
# Cue names of the lines whose ion is made above 35.12 eV, for the high-ionization width.
HIGH_ION_PREFIXES = ("[O III]", "[Ne III]", "[Ar IV]", "[Ne IV]", "He II ")


def line_groups(names, he_balmer_profile=False):
    """Split line names into the Balmer-profile and high-ionization groups.

    Parameters
    ----------
    names : array_like of str
        Line names as in the Cue or FSPS line list, for example "Ba-alpha 6563" or "[O III] 5007".
    he_balmer_profile : bool
        Put the He I lines in the Balmer-profile group.

    Returns
    -------
    balmer, high : np.ndarray of bool
        Lines with the Balmer profile, and forbidden lines with the high-ionization width. Lines
        in neither group use the low-ionization width.
    """
    names = np.char.strip(np.asarray(names).astype(str))
    balmer = np.char.startswith(names, BALMER_PREFIX)
    if he_balmer_profile:
        balmer |= np.char.startswith(names, HEI_PREFIX)
    high = np.zeros(names.shape, dtype=bool)
    for prefix in HIGH_ION_PREFIXES:
        high |= np.char.startswith(names, prefix)
    return balmer, high & ~balmer


class TwoCompLineModel(HyperSpecModel):
    """HyperSpecModel with a narrow plus broad profile for the Balmer lines.

    The broad component takes a fraction ``eline_fbroad`` of each Balmer line's flux, so the
    line fluxes are unchanged. With ``eline_fbroad = 0`` and ``eline_sigma_forb`` equal to
    ``eline_sigma`` (or absent) the model is identical to HyperSpecModel.
    """

    def _p(self, name, default):
        """Return the first element of a model parameter, or ``default`` if it is not set."""
        return float(np.atleast_1d(self.params.get(name, default))[0])

    def _line_names(self):
        """Return the line names of ``emline_info``."""
        return np.asarray(self.emline_info["name"]).astype(str)

    def _groups(self):
        """Return the cached ``line_groups`` masks of ``emline_info``."""
        if not hasattr(self, "_line_groups"):
            he = bool(self._p("eline_he_balmer_profile", 0.0))
            self._line_groups = line_groups(self._line_names(), he_balmer_profile=he)
        return self._line_groups

    def _balmer_mask(self):
        """Return a mask of the lines with the Balmer profile, see ``line_groups``."""
        return self._groups()[0]

    def _high_ion_mask(self):
        """Return a mask of the high-ionization lines, see ``line_groups``."""
        return self._groups()[1]

    def _broad_fraction(self):
        """Return the per-line broad fraction of flux in the broad component.

        Lines with the Balmer profile use ``eline_fbroad``; all other lines use
        ``eline_fbroad_forb``, which is absent (zero) unless ``add_broad_params`` was called with
        ``forbidden_broad``.
        """
        return np.where(
            self._balmer_mask(), self._p("eline_fbroad", 0.0), self._p("eline_fbroad_forb", 0.0)
        )

    def _profile_widths(self):
        """Return per-line total widths of the main component and the widest component.

        Returns
        -------
        prof : np.ndarray
            Total width in km/s of the narrow (Balmer profile) or forbidden (other lines)
            component, including the instrumental and library part cached by prospect.
            High-ionization lines use ``eline_sigma_forb_hi`` if it is set.
        wide : np.ndarray
            Total width in km/s of the widest component of each line.
        shift : np.ndarray
            Absolute velocity offset in km/s of the broad component (zero for lines without one).
        inst2 : np.ndarray
            Squared instrumental (and library) width in km/s.
        """
        sig0 = np.atleast_1d(self._eline_sigma_kms) * np.ones_like(self._ewave_obs, dtype=float)
        bal = self._balmer_mask()

        # the cached width is hypot(eline_sigma, instrument); keep the instrumental part
        inst2 = np.clip(sig0**2 - self._p("eline_sigma", 0) ** 2, 0, None)
        s_forb = self._p("eline_sigma_forb", self._p("eline_sigma", 0))
        s_hi = self._p("eline_sigma_forb_hi", s_forb)
        s_line = np.where(self._high_ion_mask(), s_hi, s_forb)
        prof = np.where(bal, sig0, np.sqrt(s_line**2 + inst2))

        has = self._broad_fraction() > 0
        broad = np.sqrt(self._p("eline_sigma_broad", 150.0) ** 2 + inst2)
        wide = np.where(has, np.maximum(prof, broad), prof)
        shift = np.where(has, abs(self._p("eline_vbroad", 0.0)), 0.0)

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
        """Return unit-flux line profiles, two-component for lines with a broad component.

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
        prof, _, _, inst2 = self._profile_widths()
        fb = self._broad_fraction()[lineidx]
        try:
            self._eline_sigma_kms = prof
            g = super().get_eline_gaussians(lineidx=lineidx, wave=wave)
            if np.any(fb > 0):
                self._eline_sigma_kms = np.sqrt(self._p("eline_sigma_broad", 150.0) ** 2 + inst2)
                self._ewave_obs = mu0 * (1 + self._p("eline_vbroad", 0.0) / C_KMS)
                gb = super().get_eline_gaussians(lineidx=lineidx, wave=wave)
                g = (1 - fb) * g + fb * gb
        finally:
            self._eline_sigma_kms, self._ewave_obs = sig0, mu0
        return g


def same_fbroad(eline_fbroad=0.0, **extras):
    """``depends_on`` function giving the forbidden lines the Balmer broad fraction."""
    return eline_fbroad


def add_broad_params(
    params,
    separate_forbidden_width=True,
    forbidden_broad=None,
    sigma_split=100.0,
    split_forbidden=False,
    he_balmer_profile=False,
):
    """Add the broad-line parameters to a model parameter dict.

    Parameters
    ----------
    params : dict
        Prospect model parameter specification. It is changed in place; the ``eline_sigma``
        entry is replaced by a copy with the narrow prior, so templates it came from are not
        changed.
    separate_forbidden_width : bool
        Also add ``eline_sigma_forb``, a separate width for the non-Balmer lines.
    forbidden_broad : {None, "shared", "free"}
        Give the non-Balmer lines the broad component too, with same width and shift.
        ``shared'' uses the Balmer fraction (``eline_fbroad_forb`` tied to ``eline_fbroad``).
        ``free'' allows the forbidden broad fraction to vary independently.
        None keeps it only on balmer lines.
    sigma_split : float
        Upper bound of the narrow and forbidden widths and lower bound of the broad width, in km/s.
    split_forbidden : bool
        Also add ``eline_sigma_forb_hi``, the width of the high-ionization forbidden lines
        (``HIGH_ION_PREFIXES``). ``eline_sigma_forb`` then covers the other non-Balmer lines.
        Needs ``separate_forbidden_width``.
    he_balmer_profile : bool
        Give the He I lines the Balmer profile (narrow ``eline_sigma`` and the Balmer broad
        component), through the fixed parameter ``eline_he_balmer_profile``.

    Returns
    -------
    dict
        ``params``, with the new parameters.
    """
    params.update({k: dict(v) for k, v in BROAD_PARAMS.items()})

    narrow = priors.TopHat(mini=10.0, maxi=sigma_split)
    params["eline_sigma_broad"]["prior"] = priors.TopHat(mini=sigma_split, maxi=800.0)
    params["eline_sigma_broad"]["init"] = max(params["eline_sigma_broad"]["init"], sigma_split)

    if separate_forbidden_width:
        params.update({k: dict(v, prior=narrow) for k, v in FORB_PARAM.items()})
    if split_forbidden:
        if not separate_forbidden_width:
            raise ValueError("split_forbidden needs separate_forbidden_width")
        params.update({k: dict(v, prior=narrow) for k, v in FORB_HI_PARAM.items()})
    if he_balmer_profile:
        params["eline_he_balmer_profile"] = dict(N=1, isfree=False, init=1.0)
    if forbidden_broad == "shared":
        params["eline_fbroad_forb"] = dict(N=1, isfree=False, init=0.0, depends_on=same_fbroad)
    elif forbidden_broad == "free":
        params["eline_fbroad_forb"] = dict(
            N=1, isfree=True, init=0.1, prior=priors.TopHat(mini=0.0, maxi=0.9)
        )
    elif forbidden_broad is not None:
        raise ValueError(
            f"forbidden_broad must be None, 'shared' or 'free', not {forbidden_broad!r}"
        )
    params["eline_sigma"] = {**params["eline_sigma"], "prior": narrow}
    return params


def _trapezoid(y, x):
    """Integrate ``y`` over ``x`` with the trapezoid rule."""
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))


def self_check(model_params, obs, sps, theta, n_prior=100, seed=0):
    """Check TwoCompLineModel against HyperSpecModel before sampling.

    With ``eline_fbroad = 0`` and every line width (``eline_sigma_forb`` and, if set,
    ``eline_sigma_forb_hi``) equal to ``eline_sigma`` the two-component model must reproduce the
    one-component model, whatever the line groups. With ``eline_fbroad = 0.5`` the Halpha + [NII] flux
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
    widths = [*FORB_PARAM, *FORB_HI_PARAM]
    new = [k for k in [*BROAD_PARAMS, *widths, "eline_fbroad_forb"] if k in m.theta_index]
    drop = [m.theta_index[k].start for k in new]
    fixed = [*new, "eline_fbroad_forb", "eline_he_balmer_profile"]
    base = HyperSpecModel({k: v for k, v in model_params.items() if k not in fixed})
    t0 = np.array(theta, dtype=float)
    t0[ib] = 0.0
    if "eline_fbroad_forb" in m.theta_index:
        t0[m.theta_index["eline_fbroad_forb"].start] = 0.0
    # with one width for all lines the line groups do not matter, so the model equals the base
    for k in widths:
        if k in m.theta_index:
            t0[m.theta_index[k].start] = t0[m.theta_index["eline_sigma"].start]

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
