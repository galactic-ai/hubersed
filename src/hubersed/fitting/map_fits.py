"""MAP fits of DESI spectra with prospector and Cue nebular emission, one TARGETID at a time.

Each fit writes a pickle with the best fit and its chi2 split into line and continuum pixels,
plus spectrum and SFH figures. The command line entry point is
``scripts/run_map_fits_outliers.py``.
"""

import os
import pickle
import time
import warnings
from pathlib import Path

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prospect.fitting import lnprobfn
from prospect.models import priors
from prospect.models.sedmodel import HyperSpecModel, SpecModel
from prospect.models.transforms import logsfr_ratios_to_masses
from scipy.optimize import minimize
from scipy.signal import medfilt

from hubersed.conversion import ivar_to_maggies, to_maggies
from hubersed.fitting.chi2 import WAVE_OBS
from hubersed.fitting.result import MapFitResult
from hubersed.io.desi import desi_spectrum, load_spectrum
from hubersed.plotting.sfh import sfh_figure
from hubersed.plotting.spectra import plot_residual, residual_chi, spectrum_figure
from hubersed.sps.config import build_continuum_model, build_full_cue_model
from hubersed.sps.lsf import C_KMS, desi_resolution
from hubersed.sps.parameter_file import (
    build_cue_sps,
    build_obs,
    build_sps,
    mask_spectral_lines,
)
from hubersed.sps.utils import universe_age_gyr

LSF = (C_KMS / (2.355 * desi_resolution(WAVE_OBS))).astype(np.float64)


def _quiet_process():
    """Hide warnings and floating point errors, and draw figures without a display.

    Called by ``main`` and by each worker process, so importing this module changes
    nothing process wide.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    np.seterr(divide="ignore", invalid="ignore", over="ignore", under="ignore")
    matplotlib.use("Agg")


AIR_LINES = {
    "[OII]": 3727.4,
    "[NeIII]": 3868.8,
    "Hd": 4101.7,
    "Hg": 4340.5,
    "Hb": 4861.3,
    "[OIII]4959": 4958.9,
    "[OIII]5007": 5006.8,
    "Ha": 6562.8,
    "[NII]6584": 6583.5,
    "[SII]6716": 6716.4,
    "[SII]6731": 6730.8,
}
BALMER = ("Hd", "Hg", "Hb", "Ha")


def vacuum_lines(sps):
    """Find the FSPS wavelength of each line in ``AIR_LINES``.

    Parameters
    ----------
    sps : prospect.sources.SSPBasis
        Source whose ``ssp.emline_wavelengths`` lists the FSPS lines.

    Returns
    -------
    dict of str to float
        FSPS line wavelength in Angstrom by line name.

    Raises
    ------
    AssertionError
        If no FSPS line lies within 200 km/s of a line in ``AIR_LINES``.
    """
    fw = sps.ssp.emline_wavelengths
    out = {}
    for k, lam in AIR_LINES.items():
        j = int(np.argmin(np.abs(fw - lam)))
        assert abs(fw[j] - lam) / lam * C_KMS < 200, f"{k}: no FSPS line within 200 km/s"
        out[k] = float(fw[j])
    return out


def safe_lnprior(model, theta):
    """Return the log prior of ``theta``, or -inf if prospector fails or it is not finite."""
    try:
        v = float(np.squeeze(model.prior_product(np.asarray(theta, float), nested=False)))
        return v if np.isfinite(v) else -np.inf
    except Exception:
        return -np.inf


def jitter_scale(model, theta, frac):
    """Return the jitter width of each theta entry, ``frac`` times its prior range.

    Entries whose prior has no finite range get ``frac`` itself.
    """
    scale = np.full_like(theta, frac, dtype=float)
    for k, inds in model.theta_index.items():
        try:
            lo, hi = model.config_dict[k]["prior"].range
            w = np.atleast_1d(np.asarray(hi, float) - np.asarray(lo, float))
        except Exception:
            continue
        if np.all(np.isfinite(w)) and np.all(w > 0):
            scale[inds] = frac * w
    return scale


def map_fit(
    model,
    obs,
    sps,
    n_seeds,
    maxfev,
    theta0=None,
    jitter_frac=0.02,
    seed=0,
    max_tries=400,
    tag="",
    method="Powell",
):
    """Minimize the negative log posterior from several starts and keep the best.

    Parameters
    ----------
    model : prospect.models.SpecModel
        Model to fit.
    obs : list
        Prospector observations from ``build_obs``.
    sps : prospect.sources.SSPBasis
        Source used for the model spectrum.
    n_seeds : int
        Number of jittered starts added to the start at ``theta0``.
    maxfev : int
        Function evaluation budget of each start.
    theta0 : np.ndarray, optional
        First start. The default is ``model.theta``.
    jitter_frac : float, optional
        Jitter width as a fraction of each prior range, see ``jitter_scale``.
    seed : int, optional
        Seed of the jitter draws.
    max_tries : int, optional
        Most jitter draws to try. Draws outside the prior or with a failed model are skipped.
    tag : str, optional
        Prefix for progress lines.
    method : {"Powell", "Nelder-Mead"}, optional
        scipy ``minimize`` method.

    Returns
    -------
    best : scipy.optimize.OptimizeResult or None
        Lowest result, or None if every start failed.
    info : dict
        Start counts, every final value, success flags, and the gap between the two
        best converged starts.
    """

    def neg(th):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lp = lnprobfn(th, model=model, observations=obs, sps=sps, nested=False)
                return -lp if np.isfinite(lp) else 1e18
            except Exception:
                return 1e18

    theta_init = model.theta.copy() if theta0 is None else np.asarray(theta0, float)
    rng = np.random.default_rng(seed)
    scale = jitter_scale(model, theta_init, jitter_frac)

    starts, tries = [theta_init.copy()], 0
    while len(starts) < n_seeds + 1 and tries < max_tries:
        tries += 1
        st = theta_init + rng.normal(0.0, scale)
        if np.isfinite(safe_lnprior(model, st)) and neg(st) < 1e17:
            starts.append(st)

    res = []
    for k, st in enumerate(starts):
        t0 = time.time()
        # Nelder-Mead has no line search, so it is a check on Powell at metallicity kinks.
        opts = (
            {"maxiter": maxfev // 10, "maxfev": maxfev, "ftol": 1e-6}
            if method == "Powell"
            else {"maxiter": maxfev, "maxfev": maxfev, "fatol": 1e-6, "xatol": 1e-4}
        )
        r = minimize(neg, st, method=method, options=opts)
        print(
            f"{tag} start {k}/{len(starts) - 1}  fun {r.fun:.1f}  nfev {r.nfev}  "
            f"success {r.success}  {time.time() - t0:.0f}s",
            flush=True,
        )
        if np.isfinite(r.fun) and r.fun < 1e10:
            res.append(r)
    res.sort(key=lambda r: r.fun)
    if not res:
        return None, {"n_starts": len(starts), "n_ok": 0}

    funs = np.array([r.fun for r in res], float)
    f_ok = np.array([r.fun for r in res if r.success], float)
    info = {
        "n_starts": len(starts),
        "n_ok": len(res),
        "n_converged": int(f_ok.size),
        "fun_all": funs.tolist(),
        "gap_best_second": float(f_ok[1] - f_ok[0]) if f_ok.size > 1 else np.nan,
        "success": [bool(r.success) for r in res],
        "nfev": [int(r.nfev) for r in res],
        "jitter_frac": jitter_frac,
        "seed": seed,
        "maxfev": maxfev,
    }
    return res[0], info


def theta_dict(model, theta):
    """Split a theta vector into arrays by parameter name, using ``model.theta_index``."""
    return {
        k: np.atleast_1d(np.asarray(theta, float)[v]).copy() for k, v in model.theta_index.items()
    }


def chi2_parts(model, theta, obs, sps, line_pix):
    """Compute the model spectrum and its chi2, split into line and continuum pixels.

    Parameters
    ----------
    model, theta, obs, sps
        As for ``map_fit``.
    line_pix : np.ndarray of bool
        Pixels near emission lines.

    Returns
    -------
    sp : np.ndarray
        Model flux in maggies.
    stats : dict
        ``chi2``, ``chi2_red`` (over pixels minus free parameters), pixel counts, the
        fraction of chi2 and of pixels on lines, and the mean chi2 per line and per
        continuum pixel.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds, _ = model.predict(np.asarray(theta, float), observations=obs, sps=sps)
    sp = np.asarray(preds[0], float)
    o = obs[0]
    flux, unc = np.asarray(o.flux, float), np.asarray(o.uncertainty, float)
    m = np.asarray(o.mask, bool) & np.isfinite(sp) & np.isfinite(unc) & (unc > 0)
    r2 = np.zeros_like(flux)
    r2[m] = ((flux[m] - sp[m]) / unc[m]) ** 2
    lp, cp = m & line_pix, m & ~line_pix

    def mean(x):
        return float(x.mean()) if x.size else np.nan

    return sp, {
        "chi2_red": float(r2[m].sum() / max(int(m.sum()) - len(theta), 1)),
        "chi2": float(r2[m].sum()),
        "npix": int(m.sum()),
        "ntheta": len(theta),
        "chi2_line_frac": float(r2[lp].sum() / r2[m].sum()) if r2[m].sum() else np.nan,
        "npix_line_frac": float(lp.sum() / m.sum()),
        "chi2_per_pix_line": mean(r2[lp]),
        "chi2_per_pix_cont": mean(r2[cp]),
    }


def line_ratios(wave, flux, model_sp, z, mask, lines, halfwidth_kms=400.0):
    """Return data over model flux, integrated within ``halfwidth_kms`` of each line.

    Lines with fewer than 3 good pixels, or zero model flux, get NaN.
    """
    out = {}
    for name, lam in lines.items():
        lo = lam * (1 + z)
        sel = mask & (np.abs(wave - lo) < lo * halfwidth_kms / C_KMS)
        if sel.sum() < 3:
            out[name] = np.nan
            continue
        mm = np.trapezoid(model_sp[sel], wave[sel])
        out[name] = float(np.trapezoid(flux[sel], wave[sel]) / mm) if mm else np.nan
    return out


def sfh_from_theta(model, theta):
    """Return the star formation history of ``theta`` in the model's age bins.

    Returns
    -------
    dict
        ``edges_gyr`` (lookback time bin edges in Gyr, first edge at least 1e-4),
        ``ssfr`` and ``ssfr_inplace`` (SFR over current mass, and over the mass formed
        by the end of each bin, that bin included), ``cmf`` (cumulative mass fraction),
        ``cmf_flat_null`` (the same for all logsfr_ratios zero) and ``agebins``.

    Notes
    -----
    Current mass is taken as 0.6 times the mass formed, a fixed value, as in
    ``derived_quantities.compute_logssfr``.
    """
    model.set_parameters(np.asarray(theta, float))
    ab = np.asarray(model.params["agebins"], float)
    logmass = float(np.atleast_1d(model.params["logmass"])[0])
    ratios = np.atleast_1d(model.params["logsfr_ratios"])
    masses = logsfr_ratios_to_masses(logmass=logmass, logsfr_ratios=ratios, agebins=ab)
    flat = logsfr_ratios_to_masses(logmass=logmass, logsfr_ratios=np.zeros_like(ratios), agebins=ab)
    dt = np.diff(10**ab, axis=1)[:, 0]
    sfr = masses / dt
    m_cur = masses.sum() * 0.6
    m_ge = np.cumsum(masses[::-1])[::-1]
    edges = np.append(10 ** ab[:, 0], 10 ** ab[-1, 1]) / 1e9
    edges[0] = max(edges[0], 1e-4)
    return dict(
        edges_gyr=edges,
        ssfr=sfr / m_cur,
        ssfr_inplace=sfr / np.maximum(m_ge, 1.0),
        cmf=np.cumsum(masses) / masses.sum(),
        cmf_flat_null=np.cumsum(flat) / flat.sum(),
        agebins=ab,
    )


def lnp_split(model, theta, obs, sps):
    """Return the log posterior, log prior and their difference, the log likelihood."""
    lnprior = safe_lnprior(model, theta)
    lnpost = float(
        lnprobfn(np.asarray(theta, float), model=model, observations=obs, sps=sps, nested=False)
    )
    return {"lnpost": lnpost, "lnprior": lnprior, "lnlike": lnpost - lnprior}


def plot_fit(tid, z, wave, flux, unc, mask, model_sp, lines, chi2_red, out):
    """Save the data, MAP model and residuals to ``out/<tid>_spectrum.pdf``."""
    fig, ax = spectrum_figure(
        wave,
        z=z,
        figsize=(11, 6),
        data=np.where(mask, flux, np.nan),
        unc=np.where(mask, unc, np.nan),
        band_kw={},
        medfilt=np.where(mask, medfilt(np.where(mask, flux, 0.0), 9), np.nan),
        models=[
            {
                "flux": np.where(mask, model_sp, np.nan),
                "label": f"Cue MAP  $\\chi^2_\\nu$={chi2_red:.2f}",
                "color": "#b2182b",
                "lw": 1.0,
            }
        ],
    )
    ax[0].set_ylabel("flux [maggies]")
    ax[0].set_ylim(np.nanmin(flux[mask]) * 1.1, np.nanpercentile(flux[mask], 99.8) * 1.2)
    plot_residual(ax[1], wave, z=z, chi=residual_chi(flux, model_sp, unc, mask))
    ax[1].set_ylim(-8, 8)
    fig.savefig(out / f"{tid}_spectrum.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sfh(tid, sfh, out):
    """Save the SFH, with the flat SFH mass fraction dashed, to ``out/<tid>_sfh.pdf``."""
    fig, ax = sfh_figure(
        sfh["edges_gyr"], sfh["ssfr"], sfh["cmf"], ssfr_inplace=sfh["ssfr_inplace"]
    )
    ax[1].stairs(sfh["cmf_flat_null"], sfh["edges_gyr"], color="0.6", ls="--", lw=1.2)
    fig.savefig(out / f"{tid}_sfh.pdf", bbox_inches="tight")
    plt.close(fig)


_SPS = {}


def get_sps(zcontinuous=1, zero_library_resolution=True):
    """Build the FSPS and Cue sources once per process and return them with the line lists.

    Raises
    ------
    AssertionError
        If the cache was built with a different ``zcontinuous``, which would otherwise
        be ignored.
    """
    key = int(zcontinuous)
    assert not (_SPS and _SPS.get("key") != key), (
        f"get_sps() cached with {_SPS.get('key')}; {key} would be ignored"
    )
    if not _SPS:
        _SPS["key"] = key
        _SPS["sps"] = build_sps(
            zcontinuous=zcontinuous, zero_library_resolution=zero_library_resolution
        )
        _SPS["zcontinuous"] = zcontinuous
        _SPS["cue"] = build_cue_sps()
        fw = _SPS["sps"].ssp.emline_wavelengths
        _SPS["lines"] = vacuum_lines(_SPS["sps"])
        _SPS["line_waves"] = fw[(fw > 3600) & (fw < 9824)]
    return _SPS


FROZEN_HYPERS = {"sigma_reg": 1.5, "sigma_dyn": 0.1, "tau_eq": 2.5, "tau_dyn": 0.025}

FLAT_SFH_RANGE = 5.0  # dex, symmetric


def _build_model(tmpl, flat_sfh=False):
    """Return a HyperSpecModel, or with ``flat_sfh`` a SpecModel with a TopHat SFH prior.

    Notes
    -----
    A flat prior needs both the model class and the prior changed. HyperSpecModel builds
    the logsfr_ratios prior from the hyperparameters and skips the configured one
    (prospect ``hyperparameters.py:53-85``). SpecModel calls every configured prior, and
    the stochastic ``MultiVariateNormal`` has no ``__call__`` (``priors.py:269``).
    """
    if not flat_sfh:
        return HyperSpecModel(tmpl)
    n = len(np.asarray(tmpl["agebins"]["init"], float)) - 1
    tmpl["logsfr_ratios"]["prior"] = priors.TopHat(
        mini=np.full(n, -FLAT_SFH_RANGE), maxi=np.full(n, FLAT_SFH_RANGE)
    )
    for k in ("sigma_reg", "tau_eq", "tau_in", "sigma_dyn", "tau_dyn"):
        if k in tmpl:
            tmpl[k]["isfree"] = False  # inert under SpecModel, but do not let them into theta
    return SpecModel(tmpl)


def freeze_hypers(template, z):
    """Fix the SFH hyperparameters at ``FROZEN_HYPERS``, with ``tau_in`` the age at ``z``."""
    vals = dict(FROZEN_HYPERS, tau_in=universe_age_gyr(z))
    for k, v in vals.items():
        template[k]["isfree"] = False
        template[k]["init"] = float(v)
    return template


def fit_one(
    tid,
    sps,
    cue_sps,
    lines,
    line_waves,
    n_seeds,
    maxfev,
    out,
    seeds=None,
    frozen=False,
    fixed=None,
    flat_sfh=False,
    cont_only=False,
    method="Powell",
    zcontinuous=1,
    spectra_npz=None,
    free_dust1=False,
):
    """Fit one galaxy by MAP, save its record and figures, and return the record.

    Parameters
    ----------
    tid : int
        DESI TARGETID.
    sps, cue_sps : prospect.sources.SSPBasis
        FSPS source for the continuum fit and Cue source for the full fit.
    lines : dict of str to float
        Line wavelengths for ``line_ratios``, from ``vacuum_lines``.
    line_waves : np.ndarray
        FSPS line wavelengths in Angstrom, masked for the continuum pixels.
    n_seeds, maxfev : int
        Passed to ``map_fit``.
    out : pathlib.Path
        Output directory.
    seeds : dict of str to float, optional
        Start values by parameter name, clipped to the prior.
    frozen : bool, optional
        Fix the SFH hyperparameters, see ``freeze_hypers``.
    fixed : dict of str to float, optional
        Parameters fixed at the given values.
    flat_sfh : bool, optional
        Use a flat SFH prior, see ``_build_model``.
    cont_only : bool, optional
        Fit ``build_continuum_model`` with FSPS on line-masked pixels only. Its SFH
        hyperparameters are fixed.
    method : {"Powell", "Nelder-Mead"}, optional
        Passed to ``map_fit``.
    zcontinuous : int, optional
        FSPS metallicity interpolation mode of ``sps``, stored in the record.
    spectra_npz : str, optional
        npz file with ``target_ids``, ``spec``, ``ivar`` and ``z`` used instead of the
        chunk files, on the ``WAVE_OBS`` grid in 1e-17 erg/s/cm^2/A.
    free_dust1 : bool, optional
        Fit dust1 on its own in the full Cue model.

    Returns
    -------
    dict
        The record written to ``out/<tid>.pkl``, or a short record with status
        ``map_failed``.
    """
    if spectra_npz:
        # For a different observation of a galaxy the chunk files already hold.
        ov = np.load(spectra_npz)
        hit = np.where(ov["target_ids"].astype(np.int64) == tid)[0]
        if not len(hit):
            raise SystemExit(f"--spectra-npz {spectra_npz}: no row for TARGETID {tid}")
        k = int(hit[0])
        assert len(ov["spec"][k]) == len(WAVE_OBS), "override spectrum is off the WAVE_OBS grid"
        s = desi_spectrum(ov["spec"][k], ov["ivar"][k], float(ov["z"][k]), tid)
    else:
        s = load_spectrum(tid)
    z = float(s.redshift.value)

    # Converted on the float32 WAVE_OBS grid the fit uses, not s.spectral_axis (float64).
    flux = to_maggies(WAVE_OBS * u.AA, s.flux).value
    iv = ivar_to_maggies(WAVE_OBS * u.AA, s.uncertainty.quantity).value
    mask = ~s.mask
    iv = np.where(mask, iv, 0.0)
    unc = 1.0 / np.sqrt(np.where(iv > 0, iv, np.inf))

    m_cont = mask_spectral_lines(WAVE_OBS, mask, z, halfwidth_kms=1500.0, line_waves=line_waves)
    line_pix = mask & ~m_cont
    tight = mask & ~mask_spectral_lines(
        WAVE_OBS, mask, z, halfwidth_kms=300.0, line_waves=line_waves
    )

    # cont_only fits the line-masked pixels only; the Cue arm fits everything and
    # accounts for the lines with free nebular parameters.
    fit_mask = m_cont if cont_only else mask
    obs = build_obs(spec=flux, unc=unc, mask=fit_mask, resolution=LSF, wavelength=WAVE_OBS)
    cont_model, cont_tmpl = build_continuum_model(z)
    if cont_only:
        model, tmpl, sps_use = cont_model, cont_tmpl, sps
    else:
        model, tmpl = build_full_cue_model(
            cont_tmpl, cont_model.theta, cont_model, free_dust1=free_dust1
        )
        sps_use = cue_sps
    if frozen or fixed or flat_sfh:
        if frozen:
            tmpl = freeze_hypers(tmpl, z)
        for k, v in (fixed or {}).items():
            tmpl[k]["isfree"] = False
            tmpl[k]["init"] = float(v)
        model = _build_model(tmpl, flat_sfh)
    if not cont_only:
        assert model._need_lines, "analytic eline path off; set nebemlineinspec=False"

    th0 = model.theta.copy()
    for k, v in (seeds or {}).items():
        if k in model.theta_index and np.isfinite(v):
            lo, hi = model.config_dict[k]["prior"].range
            th0[model.theta_index[k]] = np.clip(
                v, float(np.atleast_1d(lo)[0]), float(np.atleast_1d(hi)[0])
            )

    t0 = time.time()
    best, info = map_fit(
        model,
        obs,
        sps_use,
        n_seeds=n_seeds,
        maxfev=maxfev,
        theta0=th0,
        tag=f"  {tid}",
        method=method,
    )
    if best is None:
        return {"target_id": tid, "z": float(z), "status": "map_failed", "optim": info}
    info["seconds"] = time.time() - t0
    info["theta0_seeds"] = dict(seeds or {})

    sp, stats = chi2_parts(model, best.x, obs, sps_use, line_pix)
    _, stats_tight = chi2_parts(model, best.x, obs, sps_use, tight)
    td = MapFitResult(tid, float(z), theta_dict(model, best.x), tuple(model.theta_labels())).theta
    lr = line_ratios(WAVE_OBS, flux, sp, z, mask, lines)
    sfh = sfh_from_theta(model, best.x)

    rec = {
        "target_id": tid,
        "z": float(z),
        "status": "ok",
        "wave": WAVE_OBS,
        "flux": flux,
        "unc": unc,
        "mask": mask,
        "line_pix": line_pix,
        "tight_pix": tight,
        "model": sp,
        "theta": best.x,
        "theta_dict": td,
        "labels": model.theta_labels(),
        "ndim": len(best.x),
        "stats": stats,
        "stats_tight": stats_tight,
        "line_ratios": lr,
        "lnp": lnp_split(model, best.x, obs, sps_use),
        "optim": info,
        "sfh": sfh,
        "lines": lines,
        "nlines": 0 if cont_only else len(model.emline_info),
        "nebemlineinspec": bool(np.any(model.params.get("nebemlineinspec"))),
        "lsf": "median desi_resolution",
        "nebular": "none (continuum only)" if cont_only else "cue_stellar_nebular",
        "cont_only": bool(cont_only),
        "fit_mask_npix": int(fit_mask.sum()),
        "optimizer": str(method),
        "zcontinuous": int(zcontinuous),
        # build_continuum_model and --freeze-hypers fix different values, so store them.
        "hypers": "frozen" if (frozen or cont_only) else "free",
        "hyper_values": {
            k: float(np.atleast_1d(model.params[k])[0])
            for k in ("sigma_reg", "tau_eq", "tau_in", "sigma_dyn", "tau_dyn")
            if k in model.params
        },
        "fixed": dict(fixed or {}),
        "sfh_prior": f"tophat+/-{FLAT_SFH_RANGE}" if flat_sfh else "gp_stochastic",
        "eline_waves": None if cont_only else np.asarray(cue_sps.emline_wavelengths, float).copy(),
    }
    # Write the record BEFORE plotting.
    with open(out / f"{tid}.pkl", "wb") as f:
        pickle.dump(rec, f)

    try:
        plot_fit(tid, z, WAVE_OBS, flux, unc, mask, sp, lines, stats["chi2_red"], out)
        plot_sfh(tid, sfh, out)
    except Exception as e:
        print(f"    {tid}: plotting failed, fit kept -- {type(e).__name__}: {e}", flush=True)
    return rec


def _worker(
    tid,
    n_seeds,
    maxfev,
    outdir,
    seeds,
    frozen,
    fixed,
    flat_sfh=False,
    cont_only=False,
    method="Powell",
    zcontinuous=1,
    spectra_npz=None,
    free_dust1=False,
):
    """Run ``fit_one`` in a worker process with that process's cached sources."""
    _quiet_process()
    S = get_sps(zcontinuous=zcontinuous)
    return fit_one(
        tid,
        S["sps"],
        S["cue"],
        S["lines"],
        S["line_waves"],
        n_seeds,
        maxfev,
        Path(outdir),
        seeds=seeds,
        frozen=frozen,
        fixed=fixed,
        flat_sfh=flat_sfh,
        cont_only=cont_only,
        method=method,
        zcontinuous=zcontinuous,
        spectra_npz=spectra_npz,
        free_dust1=free_dust1,
    )
