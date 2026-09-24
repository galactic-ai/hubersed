"""Fit DESI spectra with prospector at the MAP and report the reduced chi-squared.

Each galaxy gets a continuum-only fit first, which seeds a full fit with nebular emission.
"""

import warnings
from functools import cache

import astropy.units as u
import numpy as np
from prospect.fitting import lnprobfn
from scipy.optimize import minimize

from hubersed.conversion import DESI_FLAM, ivar_to_maggies, to_maggies
from hubersed.fitting.result import MapFitResult
from hubersed.io.desi import load_by_index
from hubersed.sps import parameter_file as P
from hubersed.sps.config import build_continuum_model, build_full_cue_model, build_full_model

WAVE_OBS = P.WAVE_OBS
Z_FLOOR = 0.01


def _map_optimize(neg, theta_init, n_seeds=3, jitter=0.03, maxfev=20_000, max_tries=100):
    """Minimize an objective with Powell from several starting points and keep the best.

    Parameters
    ----------
    neg : callable
        Negative log probability of a parameter vector. It returns 1e18 for invalid points,
        for example when Cue is asked for parameters outside its training range.
    theta_init : np.ndarray
        Starting parameter vector. It is always the first start.
    n_seeds : int
        Number of extra starts, each ``theta_init`` plus Gaussian jitter.
    jitter : float
        Standard deviation of the jitter, in the units of each parameter.
    maxfev : int
        Maximum number of objective calls for each Powell run.
    max_tries : int
        Most jitter draws for one extra start before that start is dropped.

    Returns
    -------
    scipy.optimize.OptimizeResult or None
        The run with the lowest objective, or None if no run finished below 1e10.

    Notes
    -----
    The jitter for start ``s`` is drawn with ``np.random.default_rng(s)``, so every galaxy
    gets the same offsets. The value 1e18 is finite, so a plain ``np.isfinite`` check would
    not catch it, which is why starts at or above 1e17 are treated as invalid.

    An invalid jittered start is drawn again from the same generator, up to ``max_tries``
    times, so every run starts from a different valid point. Each candidate is evaluated
    once. A start with no valid draw is dropped, and an invalid ``theta_init`` is not run.
    ``tests/test_equal_budget.py`` checks this.
    """

    def valid(theta):
        v = neg(theta)
        return np.isfinite(v) and v < 1e17

    starts = [theta_init] if valid(theta_init) else []
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        for _ in range(max_tries):
            st = theta_init + rng.normal(0, jitter, theta_init.shape)
            if valid(st):
                starts.append(st)
                break
    best = None
    for st in starts:
        r = minimize(
            neg,
            st,
            method="Powell",
            options={"maxiter": maxfev // 10, "maxfev": maxfev, "ftol": 1e-6},
        )
        if np.isfinite(r.fun) and r.fun < 1e10:
            best = r if (best is None or r.fun < best.fun) else best
    return best


@cache
def _fsps():
    """Build the FSPS stellar population source once per process."""
    return P.build_sps()


@cache
def _cue():
    """Build the Cue nebular emission source once per process."""
    return P.build_cue_sps()


@cache
def _lsf_sigma_kms():
    """Return the DESI instrumental resolution as a Gaussian sigma for prospect.

    Returns
    -------
    np.ndarray
        Sigma in km/s on ``WAVE_OBS``, computed as ``C_KMS / (2.355 * R)``.

    Notes
    -----
    Passing this to prospect is safe because ``build_sps`` sets the library resolution to
    zero, so prospect does not refuse data that is sharper than the templates.
    """
    from hubersed.sps.lsf import C_KMS, desi_resolution

    R = desi_resolution(WAVE_OBS)
    return (C_KMS / (2.355 * R)).astype(np.float64)


def map_chi2_one(gidx, use_cue=False, cont_nseeds=1, full_nseeds=1, maxfev=3_000):
    """Fit one galaxy at the MAP and return its reduced chi-squared.

    The continuum fit uses a mask that hides emission lines. Its best values seed the full
    fit, which uses every good pixel.

    Parameters
    ----------
    gidx : int
        Global index of the galaxy, as used by ``load_by_index``.
    use_cue : bool
        Model nebular emission with Cue instead of FSPS.
    cont_nseeds, full_nseeds : int
        Extra jittered starts for the continuum and full fits.
    maxfev : int
        Maximum objective calls for each Powell run.

    Returns
    -------
    dict
        Always has ``gidx`` and ``status``. When ``status`` is ``"ok"`` it also has
        ``id`` (TARGETID), ``z``, ``chi2``, ``ndof``, ``chi2_red``, ``npix``, ``theta``,
        ``theta_labels``, ``theta_dict``, and the spectra ``model``, ``flux``, ``unc`` and
        ``mask`` in maggies. Other statuses are ``load_fail:<error>``, ``below_zfloor``,
        ``too_masked``, ``cont_fail`` and ``full_fail``.

    Notes
    -----
    ``theta_labels`` is ``model.theta_labels()``, one label per entry of ``theta``, with
    vector parameters named ``logsfr_ratios_1`` and so on. ``MapFitResult`` checks it
    against ``theta_dict``.
    """
    try:
        spec, ivar, redshift, tid = load_by_index(gidx)
    except Exception as e:
        return dict(gidx=gidx, status=f"load_fail:{type(e).__name__}")
    if redshift < Z_FLOOR:
        return dict(gidx=gidx, id=tid, z=redshift, status="below_zfloor")

    spec_maggies = to_maggies(WAVE_OBS * u.AA, spec * DESI_FLAM).value
    ivar_maggies = ivar_to_maggies(WAVE_OBS * u.AA, ivar * DESI_FLAM**-2).value
    sigma = 1 / np.sqrt(np.where(ivar_maggies > 0, ivar_maggies, np.inf))
    mask = (sigma > 0) & np.isfinite(sigma) & np.isfinite(spec_maggies)
    if mask.sum() < 100:
        return dict(gidx=gidx, id=tid, z=redshift, status="too_masked")

    sps = _fsps()
    fw = sps.ssp.emline_wavelengths
    fopt = fw[(fw > 3600) & (fw < 9824)]
    mask_em = P.mask_spectral_lines(WAVE_OBS, mask, redshift, halfwidth_kms=1500.0, line_waves=fopt)
    res = _lsf_sigma_kms()

    obs_em = P.build_obs(
        spec=spec_maggies, unc=sigma, mask=mask_em, resolution=res, wavelength=WAVE_OBS
    )
    obs_full = P.build_obs(
        spec=spec_maggies, unc=sigma, mask=mask, resolution=res, wavelength=WAVE_OBS
    )

    # continuum MAP (seeds logmass/logzsol/sigma_smooth for the full model)
    cmodel, ctemplate = build_continuum_model(redshift)

    def neg_cont(th):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lp = lnprobfn(th, model=cmodel, observations=obs_em, sps=sps, nested=False)
                return -lp if np.isfinite(lp) else 1e18
            except Exception:
                return 1e18

    bc = _map_optimize(neg_cont, cmodel.theta.copy(), n_seeds=cont_nseeds, maxfev=maxfev)
    if bc is None:
        return dict(gidx=gidx, id=tid, z=redshift, status="cont_fail")
    theta_cont = bc.x

    # full MAP (continuum + nebular); Cue or FSPS
    if use_cue:
        sps = _cue()
        fmodel, ftemplate = build_full_cue_model(ctemplate, theta_cont, cmodel)
    else:
        fmodel, ftemplate = build_full_model(ctemplate, theta_cont, cmodel)

    def neg_full(th):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lp = lnprobfn(th, model=fmodel, observations=obs_full, sps=sps, nested=False)
                return -lp if np.isfinite(lp) else 1e18
            except Exception:
                return 1e18

    bf = _map_optimize(neg_full, fmodel.theta.copy(), n_seeds=full_nseeds, maxfev=maxfev)
    if bf is None:
        return dict(gidx=gidx, id=tid, z=redshift, status="full_fail")
    theta_map = bf.x

    preds, _ = fmodel.predict(theta_map, observations=obs_full, sps=sps)
    sp = preds[0]
    m = obs_full[0].mask
    resid = (obs_full[0].flux[m] - sp[m]) / obs_full[0].uncertainty[m]
    chi2 = float(np.nansum(resid**2))
    ndof = int(m.sum()) - len(theta_map)
    res = MapFitResult(
        int(tid),
        float(redshift),
        {k: np.asarray(theta_map[v], dtype=np.float32) for k, v in fmodel.theta_index.items()},
        tuple(fmodel.theta_labels()),
    )
    return dict(
        gidx=gidx,
        id=tid,
        z=redshift,
        status="ok",
        chi2=chi2,
        ndof=ndof,
        chi2_red=chi2 / ndof,
        npix=int(m.sum()),
        theta=res.vector().astype(np.float32),
        theta_labels=list(res.labels),
        theta_dict=res.theta,
        model=np.asarray(sp, dtype=np.float32),  # MAP model spectrum (full grid)
        flux=np.asarray(obs_full[0].flux, dtype=np.float32),
        unc=np.asarray(obs_full[0].uncertainty, dtype=np.float32),
        mask=np.asarray(m, dtype=bool),
    )
