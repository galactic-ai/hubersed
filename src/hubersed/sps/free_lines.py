"""Narrow lines from Cue with per-ion amplitude priors, plus broad lines with free amplitudes.

The narrow component is Cue: every narrow line has Cue's luminosity after dust. The Balmer lines
and the weak lines are fixed there. Each other ion has one narrow amplitude, a Gaussian prior on
it centred on Cue, and a width that sets how far the data may move it::

    width 0         fixed at Cue (a hard tie)
    finite width    soft tie, q ~ N(1, width^2), q = narrow flux / Cue flux
    infinite width  free (flat in flux), so the line does not constrain Cue

The width of each ion comes from its class in ``IONS``: "strong" ([OII] 3726,29, [OIII] 4959,5007,
[NII] 6548,84), "auroral" ([OIII]4363, [OII]7323,32, [NII]5755, [NeIII], [SIII]) or "free"
([SII], [OI], [ArIII]). Lines of one ion share the amplitude, so ratios inside an ion (doublet
ratios, the [SII] density ratio) stay Cue's.

The broad component has one free amplitude per ion (H, [OII], [OIII], [NII], [SII], [OI]), flat in
flux. Its template is the ion's Cue pattern after dust on a Gaussian of width ``eline_sigma_broad``
(Balmer) or ``eline_sigma_broad_forb`` (forbidden), shifted by ``eline_vbroad``. So broad and
narrow share the ratios inside an ion and the dust. The broad flux is added to Cue's: it is not
counted in Q or the SFR.

The amplitudes enter the spectrum linearly and are marginalized analytically
(``marginalize_amplitudes``). The prediction uses the posterior mean amplitudes, and the rest of
the marginal likelihood goes into ``_ln_eline_penalty``, which prospect's lnprobfn adds. Only
observations with a resolution matrix are supported (``ResolutionMatrixModel``), with a constant
response.
"""

import numpy as np
from prospect.models import priors

from hubersed.sps.broadline import C_KMS
from hubersed.sps.resolution_matrix import line_profile
from hubersed.sps.resolution_model import ResolutionMatrixModel, ResolutionMatrixSpectrum

BALMER_PREFIX = "Ba-"
# Smallest share of a column's flux on fitted pixels for its amplitude to be fitted. A line seen
# only in a wing gives a near-degenerate amplitude whose marginal term swings with the widths.
MIN_COVERAGE = 0.5
# Each ion: Cue line names, narrow class, broad width ("balmer", "forb" or None for no broad part).
IONS = {
    "H": dict(lines=None, narrow="tied", broad="balmer"),
    "[OII]": dict(lines=("[O II] 3726", "[O II] 3729"), narrow="strong", broad="forb"),
    "[OIII]": dict(lines=("[O III] 4959", "[O III] 5007"), narrow="strong", broad="forb"),
    "[NII]": dict(lines=("[N II] 6548", "[N II] 6584"), narrow="strong", broad="forb"),
    "[SII]": dict(lines=("[S II] 6716", "[S II] 6731"), narrow="free", broad="forb"),
    "[OI]": dict(lines=("[O I] 6300", "[O I] 6363"), narrow="free", broad="forb"),
    "[ArIII]7135": dict(lines=("[Ar III] 7135",), narrow="free", broad=None),
    "[OIII]4363": dict(lines=("[O III] 4363",), narrow="auroral", broad=None),
    "[OII]7325": dict(lines=("[O II] 7323", "[O II] 7332"), narrow="auroral", broad=None),
    "[NII]5755": dict(lines=("[N II] 5755",), narrow="auroral", broad=None),
    "[NeIII]": dict(lines=("[Ne III] 3869", "[Ne III] 3968"), narrow="auroral", broad=None),
    "[SIII]6312": dict(lines=("[S III] 6312",), narrow="auroral", broad=None),
    "[SIII]9069": dict(lines=("[S III] 9069",), narrow="auroral", broad=None),
}
# Fixed parameter holding the prior width of each narrow class.
WIDTH_PARAMS = {
    "strong": "eline_strong_width",
    "auroral": "eline_auroral_width",
    "free": "eline_free_width",
}


def marginalize_amplitudes(resid, ivar, templates, mean, prec):
    """Marginalize linear amplitudes with flat or Gaussian priors.

    The data are ``resid = templates @ a + noise`` with independent Gaussian noise of inverse
    variance ``ivar``. Each amplitude has a Gaussian prior of mean ``mean`` and precision ``prec``,
    or a flat prior of unit density where ``prec`` is zero.

    Parameters
    ----------
    resid : np.ndarray
        Data minus the fixed part of the model, shape (n,).
    ivar : np.ndarray
        Inverse variance of each pixel, shape (n,). Zero for pixels not fitted.
    templates : np.ndarray
        One column per amplitude, shape (n, k).
    mean, prec : np.ndarray
        Prior mean and precision of each amplitude, shape (k,).

    Returns
    -------
    a : np.ndarray
        Posterior mean amplitudes.
    cov : np.ndarray
        Posterior covariance of the amplitudes.
    ln_extra : float
        ln of the marginal likelihood minus the Gaussian log likelihood at ``a``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the posterior precision is not positive definite.
    """
    w = templates * ivar[:, None]
    A = templates.T @ w + np.diag(prec)
    b = w.T @ resid + prec * mean
    # scale to unit diagonal before the Cholesky factor, since columns differ by many decades
    d = np.sqrt(np.diag(A))
    if not np.all(d > 0):
        raise np.linalg.LinAlgError("an amplitude has no data and no prior")
    L = np.linalg.cholesky(A / np.outer(d, d))
    Linv = np.linalg.inv(L)
    cov = (Linv.T @ Linv) / np.outer(d, d)
    a = cov @ b
    gauss = prec > 0
    dev = a - mean
    ln_det = 2 * np.sum(np.log(np.diag(L))) + 2 * np.sum(np.log(d))
    ln_extra = (
        -0.5 * np.sum(prec * dev**2)
        - 0.5 * ln_det
        + 0.5 * np.sum(np.log(prec[gauss]))
        + 0.5 * np.sum(~gauss) * np.log(2 * np.pi)
    )
    return a, cov, float(ln_extra)


def ion_lines(names):
    """Return the line indices of each ion of ``IONS``.

    Parameters
    ----------
    names : array_like of str
        Line names of the model, as in the Cue line list.

    Returns
    -------
    dict
        Ion name to an integer array of line indices.

    Raises
    ------
    ValueError
        If a line named in ``IONS`` is not in ``names``.
    """
    names = np.char.strip(np.asarray(names).astype(str))
    out = {}
    for ion, spec in IONS.items():
        if spec["lines"] is None:
            out[ion] = np.where(np.char.startswith(names, BALMER_PREFIX))[0]
            continue
        idx = [np.where(names == n)[0] for n in spec["lines"]]
        missing = [n for n, i in zip(spec["lines"], idx, strict=True) if i.size == 0]
        if missing:
            raise ValueError(f"lines {missing} of {ion} are not in the model's line list")
        out[ion] = np.concatenate(idx)
    return out


class FreeLineModel(ResolutionMatrixModel):
    """Cue narrow lines with per-ion amplitude priors and free broad lines, through R'.

    After a prediction ``free_amplitudes`` holds each column's name, kind ("narrow" or "broad"),
    posterior mean and standard deviation, the Cue flux of the ion, the prior width, and the share
    of its flux on fitted pixels (``MIN_COVERAGE``). Columns below it stay at their prior mean.
    Narrow amplitudes are in units of the Cue flux when their prior width is finite, broad and
    free amplitudes in flux units (the model's line luminosity units).
    """

    def _ions(self):
        """Return the cached ``ion_lines`` of the model's line list."""
        if not hasattr(self, "_ion_index"):
            self._ion_index = ion_lines(self._line_names())
        return self._ion_index

    def _width(self, cls):
        """Return the prior width of a narrow class, 0 for a hard tie."""
        if cls == "tied":
            return 0.0
        return self._p(WIDTH_PARAMS[cls], np.inf)

    def free_columns(self, fine, elum, prof):
        """Return the fixed line spectrum and the free columns on the fine grid.

        Parameters
        ----------
        fine : np.ndarray
            The fine grid in observed Angstrom.
        elum : np.ndarray
            Line fluxes after dust, in the units of the spectrum times Hz.
        prof : np.ndarray
            Intrinsic narrow width of each line in km/s.

        Returns
        -------
        fixed : np.ndarray
            Narrow lines held at Cue, on the fine grid.
        cols : list of dict
            One per free amplitude: name, kind, template on the fine grid, prior mean and
            precision, the ion's Cue flux, and the prior width.
        """
        use = self._use_eline & self._fix_eline
        mu = self._ewave_obs
        v_b = self._p("eline_vbroad", 0.0)
        broad_on = self._p("eline_broad", 1.0) > 0
        sig_b = {
            "balmer": self._p("eline_sigma_broad", 150.0),
            "forb": self._p("eline_sigma_broad_forb", 150.0),
        }

        def add(spec, centre, sigma, scale):
            sl, g = line_profile(fine, centre, sigma)
            if g.size:
                spec[sl] += scale * g

        fixed = np.zeros(fine.size)
        in_column = np.zeros(mu.size, dtype=bool)
        cols = []
        for ion, idx in self._ions().items():
            spec = IONS[ion]
            idx = idx[use[idx]]
            if idx.size == 0:
                continue
            total = float(np.sum(elum[idx]))
            width = self._width(spec["narrow"])
            if width > 0 and total > 0:
                in_column[idx] = True
                t = np.zeros(fine.size)
                for i in idx:
                    add(t, mu[i], prof[i], elum[i])
                if np.isfinite(width):
                    cols.append(_column(ion, "narrow", t, 1.0, width**-2, total, width))
                else:
                    cols.append(_column(ion, "narrow", t / total, 0.0, 0.0, total, width))
            if spec["broad"] is not None and broad_on and total > 0:
                t = np.zeros(fine.size)
                for i in idx:
                    add(t, mu[i] * (1 + v_b / C_KMS), sig_b[spec["broad"]], elum[i] / total)
                cols.append(_column(ion, "broad", t, 0.0, 0.0, total, np.inf))
        for i in np.where(use & ~in_column)[0]:
            add(fixed, mu[i], prof[i], elum[i])
        return fixed, cols

    def predict_spec(self, obs):
        """Return the model spectrum on the observation's pixels, in maggies.

        Parameters
        ----------
        obs : ResolutionMatrixSpectrum
            The observation.

        Returns
        -------
        np.ndarray
            The predicted spectrum at the posterior mean amplitudes.
        """
        if not isinstance(obs, ResolutionMatrixSpectrum):
            raise NotImplementedError("FreeLineModel needs a ResolutionMatrixSpectrum")
        rprime, fine, _ = obs.resampled()
        obs.check_rows()
        self._outwave = obs.wavelength
        cont = self.fine_continuum(fine, obs.reach())

        nline = self._ewave_obs.shape[0]
        self._eline_sigma_kms = np.full(nline, self._p("eline_sigma", 0.0))
        prof, _, _, _ = self._profile_widths()
        elum = self._eline_lum * self.flux_norm() / (1 + self._zred)
        fixed, cols = self.free_columns(fine, elum, prof)

        base = rprime @ (cont + fixed)
        response = obs.compute_response(spec=base, extra_mask=True, **self.params)
        base = base * response
        T = np.zeros((base.size, len(cols)))
        for j, c in enumerate(cols):
            T[:, j] = (rprime @ c["template"]) * response

        m = obs.mask & np.isfinite(obs.uncertainty) & (obs.uncertainty > 0)
        ivar = np.where(m, 1.0 / np.where(m, obs.uncertainty, 1.0) ** 2, 0.0)
        mean = np.array([c["mean"] for c in cols], dtype=float)
        prec = np.array([c["prec"] for c in cols], dtype=float)
        total = T.sum(axis=0)
        cover = np.divide(T[m].sum(axis=0), total, out=np.zeros(len(cols)), where=total > 0)
        seen = cover >= MIN_COVERAGE
        a = mean.copy()
        sd = np.full(len(cols), np.inf)
        sd[prec > 0] = prec[prec > 0] ** -0.5
        ln_extra = 0.0
        if seen.any():
            a_s, cov, ln_extra = marginalize_amplitudes(
                np.where(m, obs.flux - base, 0.0), ivar, T[:, seen], mean[seen], prec[seen]
            )
            a[seen] = a_s
            sd[seen] = np.sqrt(np.diag(cov))
        # unseen columns stay at their prior mean: Cue for a soft tie, zero otherwise

        spec = base + T @ a
        self._ln_eline_penalty = getattr(self, "_ln_eline_penalty", 0.0) + ln_extra
        self.free_amplitudes = [
            dict(
                name=c["name"],
                kind=c["kind"],
                amp=float(a[j]),
                sd=float(sd[j]),
                cue=c["cue"],
                width=c["width"],
                seen=bool(seen[j]),
                coverage=float(cover[j]),
            )
            for j, c in enumerate(cols)
        ]
        self._resolved_continuum = (rprime @ cont) * response
        self._resolved_lines = spec - self._resolved_continuum
        self._valid_eline = self._use_eline.copy()
        self._elines_to_fit = np.zeros(nline, dtype=bool)
        self._fit_eline_pixelmask = np.zeros(spec.size, dtype=bool)
        self._fix_eline_pixelmask = self._resolved_lines != 0
        self._speccal = response
        self._sed = spec / response
        return spec


def _column(name, kind, template, mean, prec, cue, width):
    """Return one free column as a dict."""
    return dict(name=name, kind=kind, template=template, mean=mean, prec=prec, cue=cue, width=width)


def add_free_line_params(params, sigma_split=45.0, strong_width=0.15, auroral_width=np.inf):
    """Change a broadline parameter dict to the free-line model.

    The Balmer and forbidden broad fractions are removed, a broad forbidden width
    ``eline_sigma_broad_forb`` is added with the prior of ``eline_sigma_broad``, and the prior
    widths of the narrow classes are added as fixed parameters.

    Parameters
    ----------
    params : dict
        Parameter specification after ``broadline.add_broad_params``. Changed in place.
    sigma_split : float
        Lower bound of both broad widths in km/s.
    strong_width : float
        Prior width of the strong forbidden lines, as a fraction of Cue's flux.
    auroral_width : float
        Prior width of the auroral lines, [NeIII] and [SIII]. Infinite means free.

    Returns
    -------
    dict
        ``params``.
    """
    for k in ("eline_fbroad", "eline_fbroad_forb", "eline_fbroad_forb_hi"):
        params.pop(k, None)
    broad = priors.TopHat(mini=sigma_split, maxi=800.0)
    params["eline_sigma_broad"] = {**params["eline_sigma_broad"], "prior": broad}
    params["eline_sigma_broad_forb"] = {**params["eline_sigma_broad"], "prior": broad}
    for cls, w in (("strong", strong_width), ("auroral", auroral_width), ("free", np.inf)):
        params[WIDTH_PARAMS[cls]] = dict(N=1, isfree=False, init=float(w))
    params["eline_broad"] = dict(N=1, isfree=False, init=1.0)
    return params
