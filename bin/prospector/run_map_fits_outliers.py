import argparse
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore")
np_err = dict(divide="ignore", invalid="ignore", over="ignore", under="ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

np.seterr(**np_err)

from scipy.optimize import minimize
from scipy.signal import medfilt

from prospect.fitting import lnprobfn
from prospect.models import priors
from prospect.models.sedmodel import HyperSpecModel, SpecModel
from prospect.models.transforms import logsfr_ratios_to_masses
from prospect.sources import SSPBasis

from hubersed.prospector.utils import universe_age_gyr

from hubersed.conversion import flambda_to_maggies, ivar_flambda_to_ivar_maggies
from hubersed.fitting.chi2 import WAVE_OBS, load_by_index, tids_to_indices
from hubersed.fitting.config import build_continuum_model, build_full_cue_model
from hubersed.paths import PATHS
from hubersed.plotting.sfh import sfh_figure
from hubersed.plotting.spectra import plot_residual, residual_chi, spectrum_figure
from hubersed.prospector.lsf import C_KMS, desi_resolution
from hubersed.prospector.parameter_file import build_cue_sps, build_obs, build_sps, mask_spectral_lines

LSF = (C_KMS / (2.355 * desi_resolution(WAVE_OBS))).astype(np.float64)

AIR_LINES = {
    "[OII]": 3727.4, "[NeIII]": 3868.8, "Hd": 4101.7, "Hg": 4340.5, "Hb": 4861.3,
    "[OIII]4959": 4958.9, "[OIII]5007": 5006.8, "Ha": 6562.8,
    "[NII]6584": 6583.5, "[SII]6716": 6716.4, "[SII]6731": 6730.8,
}
BALMER = ("Hd", "Hg", "Hb", "Ha")


def vacuum_lines(sps):
    fw = sps.ssp.emline_wavelengths
    out = {}
    for k, lam in AIR_LINES.items():
        j = int(np.argmin(np.abs(fw - lam)))
        assert abs(fw[j] - lam) / lam * C_KMS < 200, f"{k}: no FSPS line within 200 km/s"
        out[k] = float(fw[j])
    return out


def zero_library():
    SSPBasis.spectral_resolution = property(lambda self: np.zeros_like(self.ssp.wavelengths))


def safe_lnprior(model, theta):
    try:
        v = float(np.squeeze(model.prior_product(np.asarray(theta, float), nested=False)))
        return v if np.isfinite(v) else -np.inf
    except Exception:
        return -np.inf


def jitter_scale(model, theta, frac):
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


def map_fit(model, obs, sps, n_seeds, maxfev, theta0=None, jitter_frac=0.02, seed=0,
            max_tries=400, tag="", method="Powell"):
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
        # Powell does 1-D Brent line searches. ztinterp is piecewise LINEAR in log Z
        # (fsps.f90:238-244), so the objective has a kink at every MIST node; Brent
        # converges onto a V-shaped minimum and reports success. That is the suspected
        # cause of MAPs landing at logzsol = +0.2500 to four decimal places. Nelder-Mead
        # does no line search and is the control.
        opts = ({"maxiter": maxfev // 10, "maxfev": maxfev, "ftol": 1e-6}
                if method == "Powell" else
                {"maxiter": maxfev, "maxfev": maxfev, "fatol": 1e-6, "xatol": 1e-4})
        r = minimize(neg, st, method=method, options=opts)
        print(f"{tag} start {k}/{len(starts) - 1}  fun {r.fun:.1f}  nfev {r.nfev}  "
              f"success {r.success}  {time.time() - t0:.0f}s", flush=True)
        if np.isfinite(r.fun) and r.fun < 1e10:
            res.append(r)
    res.sort(key=lambda r: r.fun)
    if not res:
        return None, {"n_starts": len(starts), "n_ok": 0}

    funs = np.array([r.fun for r in res], float)
    f_ok = np.array([r.fun for r in res if r.success], float)
    info = {
        "n_starts": len(starts), "n_ok": len(res), "n_converged": int(f_ok.size),
        "fun_all": funs.tolist(),
        "gap_best_second": float(f_ok[1] - f_ok[0]) if f_ok.size > 1 else np.nan,
        "success": [bool(r.success) for r in res], "nfev": [int(r.nfev) for r in res],
        "jitter_frac": jitter_frac, "seed": seed, "maxfev": maxfev,
    }
    return res[0], info


def theta_dict(model, theta):
    return {k: np.atleast_1d(np.asarray(theta, float)[v]).copy()
            for k, v in model.theta_index.items()}


def chi2_parts(model, theta, obs, sps, line_pix):
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
    mean = lambda x: float(x.mean()) if x.size else np.nan
    return sp, {
        "chi2_red": float(r2[m].sum() / max(int(m.sum()) - len(theta), 1)),
        "chi2": float(r2[m].sum()), "npix": int(m.sum()), "ntheta": len(theta),
        "chi2_line_frac": float(r2[lp].sum() / r2[m].sum()) if r2[m].sum() else np.nan,
        "npix_line_frac": float(lp.sum() / m.sum()),
        "chi2_per_pix_line": mean(r2[lp]), "chi2_per_pix_cont": mean(r2[cp]),
    }


def line_ratios(wave, flux, model_sp, z, mask, lines, halfwidth_kms=400.0):
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
    model.set_parameters(np.asarray(theta, float))
    ab = np.asarray(model.params["agebins"], float)
    logmass = float(np.atleast_1d(model.params["logmass"])[0])
    ratios = np.atleast_1d(model.params["logsfr_ratios"])
    masses = logsfr_ratios_to_masses(logmass=logmass, logsfr_ratios=ratios, agebins=ab)
    flat = logsfr_ratios_to_masses(logmass=logmass, logsfr_ratios=np.zeros_like(ratios), agebins=ab)
    dt = np.diff(10 ** ab, axis=1)[:, 0]
    sfr = masses / dt
    m_cur = masses.sum() * 0.6
    m_ge = np.cumsum(masses[::-1])[::-1]
    edges = np.append(10 ** ab[:, 0], 10 ** ab[-1, 1]) / 1e9
    edges[0] = max(edges[0], 1e-4)
    return dict(edges_gyr=edges, ssfr=sfr / m_cur, ssfr_inplace=sfr / np.maximum(m_ge, 1.0),
                cmf=np.cumsum(masses) / masses.sum(),
                cmf_flat_null=np.cumsum(flat) / flat.sum(), agebins=ab)


def lnp_split(model, theta, obs, sps):
    lnprior = safe_lnprior(model, theta)
    lnpost = float(lnprobfn(np.asarray(theta, float), model=model, observations=obs,
                            sps=sps, nested=False))
    return {"lnpost": lnpost, "lnprior": lnprior, "lnlike": lnpost - lnprior}


def plot_fit(tid, z, wave, flux, unc, mask, model_sp, lines, chi2_red, out):
    fig, ax = spectrum_figure(
        wave, z=z, figsize=(11, 6),
        data=np.where(mask, flux, np.nan), unc=np.where(mask, unc, np.nan), band_kw={},
        medfilt=np.where(mask, medfilt(np.where(mask, flux, 0.0), 9), np.nan),
        models=[{"flux": np.where(mask, model_sp, np.nan),
                 "label": f"Cue MAP  $\\chi^2_\\nu$={chi2_red:.2f}", "color": "#b2182b", "lw": 1.0}],
    )
    ax[0].set_ylabel("flux [maggies]")
    ax[0].set_ylim(np.nanmin(flux[mask]) * 1.1, np.nanpercentile(flux[mask], 99.8) * 1.2)
    # for lam in lines.values():
    #     ax[0].axvline(lam, color="0.85", lw=0.7, zorder=0)
    plot_residual(ax[1], wave, z=z, chi=residual_chi(flux, model_sp, unc, mask))
    ax[1].set_ylim(-8, 8)
    fig.savefig(out / f"{tid}_spectrum.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sfh(tid, sfh, out):
    fig, ax = sfh_figure(sfh["edges_gyr"], sfh["ssfr"], sfh["cmf"],
                         ssfr_inplace=sfh["ssfr_inplace"])
    ax[1].stairs(sfh["cmf_flat_null"], sfh["edges_gyr"], color="0.6", ls="--", lw=1.2)
    fig.savefig(out / f"{tid}_sfh.pdf", bbox_inches="tight")
    plt.close(fig)


_SPS = {}

# cuejax/data/cue_emlines_info.dat ships the [O II] doublet with a splitting of
# 3.0008 A against the true 2.7907 A (+7.53%, +16.9 km/s). See the 2026-08-15
# entries in knowledge/outlier_investigation_log.md. Stage 1 (fixed theta) showed
# this alone costs 42% of the [O II] window chi2, 20/20 galaxies, with every other
# line bitwise unchanged.
# targets are lambda_true * (1 + 2.1e-5/(1+z)) so that eline_delta_zred = -2.1e-5
# lands them on the truth -- NOT lambda_true itself, which would leave [O II]
# offset by -6.1 km/s relative to every other line. See the 2026-08-15 log entry.
OII_FIX = [(3727.1180, 3727.1655), (3730.1188, 3729.9562)]


def patch_oii(sps):
    w = np.asarray(sps.emline_wavelengths, float)
    for bad, good in OII_FIX:
        j = np.where(np.abs(w - bad) < 1e-3)[0]
        assert j.size == 1, f"expected one entry near {bad}, found {j.size}"
        w[j[0]] = good
    sps.emline_wavelengths = w
    return sps


def load_waves(path, sps):
    """Replace the whole Cue wavelength array from a corrected .dat file."""
    w = np.genfromtxt(path, dtype=[("wave", "f8"), ("name", "<U40")], delimiter=",")["wave"]
    old = np.asarray(sps.emline_wavelengths, float)
    assert w.size == old.size, f"{path}: {w.size} lines vs {old.size} expected"
    dv = (w - old) / old * C_KMS
    sps.emline_wavelengths = w
    return float(np.median(dv)), float(np.abs(dv).max())


def get_sps(fix_oii=False, wave_file=None, zcontinuous=1):
    # the cache is per-process; if it is already built, a wavelength override would be
    # SILENTLY ignored -- fail loudly instead.
    assert not (_SPS and (wave_file or fix_oii or zcontinuous != 1)), \
        "get_sps() cache already built; wave_file/fix_oii/zcontinuous would be ignored"
    if not _SPS:
        zero_library()
        _SPS["sps"] = build_sps(zcontinuous=zcontinuous)
        _SPS["zcontinuous"] = zcontinuous
        _SPS["cue"] = build_cue_sps()
        if wave_file:
            med, mx = load_waves(wave_file, _SPS["cue"])
            print(f"  wavelengths from {wave_file}: median {med:+.3f} km/s, "
                  f"max |shift| {mx:.2f} km/s", flush=True)
        elif fix_oii:
            patch_oii(_SPS["cue"])
            print("  [O II] wavelengths corrected", flush=True)
        fw = _SPS["sps"].ssp.emline_wavelengths
        _SPS["lines"] = vacuum_lines(_SPS["sps"])
        _SPS["line_waves"] = fw[(fw > 3600) & (fw < 9824)]
    return _SPS


FROZEN_HYPERS = {"sigma_reg": 1.5, "sigma_dyn": 0.1, "tau_eq": 2.5, "tau_dyn": 0.025}
# NOTE: no source is recorded for these four values. outlier_investigation_log.md:493 calls
# them "moderate FIXED hypers". Everything in knowledge/stochastic_prior_young_bin_clamp.md
# follows from them, so their provenance is an open question.

FLAT_SFH_RANGE = 5.0   # dex, symmetric


def _build_model(tmpl, flat_sfh=False):
    """HyperSpecModel (GP prior on the SFH) or SpecModel with a flat TopHat prior.

    These two changes are COUPLED and neither works alone.

    * Under ``HyperSpecModel``, ``ProspectorHyperParams._prior_product``
      (``hyperparameters.py:53-62``) rebuilds the MVN from the PSD hypers itself and then
      ``continue``s past ``logsfr_ratios`` in the generic loop. So editing
      ``tmpl['logsfr_ratios']['prior']`` is dead code -- the GP prior is applied regardless.
    * Under plain ``SpecModel``, ``ProspectorParams._prior_product``
      (``parameters.py:196-197``) calls ``config_dict[k]['prior'](theta[inds])`` for every
      parameter including ``logsfr_ratios``. But the prior that
      ``templates.adjust_stochastic_params`` attached is ``MultiVariateNormal``, which
      defines no ``__call__`` (``priors.py:269``); it inherits the scalar version and
      returns a 9x9 NaN matrix. Verified: calling it on the 9 MAP ratios gives shape (9,9),
      all non-finite.

    So to escape the GP prior you must switch the model class AND replace the prior.

    What you give up (see knowledge/stochastic_prior_young_bin_clamp.md):
    the GP prior exists to make inference insensitive to bin count
    (outlier_investigation_log.md, "Parameterization note"). With a flat prior, more bins
    means more genuine freedom, so chi2 falls monotonically with nbins for reasons that
    have nothing to do with the data. Fix nbins and never compare across it.
    """
    if not flat_sfh:
        return HyperSpecModel(tmpl)
    n = len(np.asarray(tmpl["agebins"]["init"], float)) - 1
    tmpl["logsfr_ratios"]["prior"] = priors.TopHat(
        mini=np.full(n, -FLAT_SFH_RANGE), maxi=np.full(n, FLAT_SFH_RANGE))
    for k in ("sigma_reg", "tau_eq", "tau_in", "sigma_dyn", "tau_dyn"):
        if k in tmpl:
            tmpl[k]["isfree"] = False   # inert under SpecModel, but do not let them into theta
    return SpecModel(tmpl)


def freeze_hypers(template, z):
    vals = dict(FROZEN_HYPERS, tau_in=universe_age_gyr(z))
    for k, v in vals.items():
        template[k]["isfree"] = False
        template[k]["init"] = float(v)
    return template


def warm_theta(model, path, z, th0):
    src = dict(zip(*(lambda r: (r["labels"], np.asarray(r["theta"], float)))(
        pickle.load(open(path, "rb")))))
    src.setdefault("tau_in", universe_age_gyr(z) * (1 - 1e-6))
    for k, v in FROZEN_HYPERS.items():
        src.setdefault(k, v)
    th = np.array([src.get(k, d) for k, d in zip(model.theta_labels(), th0)])
    assert np.isfinite(safe_lnprior(model, th)), f"warm start outside prior: {path}"
    return th


def fit_one(tid, sps, cue_sps, lines, line_waves, n_seeds, maxfev, out, seeds=None,
            frozen=False, fixed=None, warm_from=None, flat_sfh=False, cont_only=False,
            error_floor=0.0, method="Powell", zcontinuous=1):
    """cont_only: fit build_continuum_model on line-masked pixels -- 15 free parameters
    (logzsol, dust2, logmass, 9x logsfr_ratios, dust_ratio, dust_index, sigma_smooth),
    no Cue nebular, plain FSPS sps.

    The 5 PSD hyperparameters are already isfree=False in build_continuum_model
    (fitting/config.py:85-91). That is not a convenience -- with them free the MAP
    objective is UNBOUNDED. hyperparameters.py:53-62 scores logsfr_ratios with the
    NORMALISED multivariate_normal pdf, and Sigma is linear in sigma_reg**2 and
    sigma_dyn**2 (hyperparam_transforms.py:120-136), so shrinking both sigmas and the
    ratios together holds the Mahalanobis term exactly constant while -0.5*ln|Sigma|
    gains 9*ln(10) = 20.72 nats per decade, without limit. Measured on a saved fit:
    lnP_sfh +37.55 -> +141.16 over five decades, Mahalanobis fixed at -1.914. Only the
    LogUniform floors stop it, which is why sigma_reg sits at exactly 0.1 in 10 of the
    20 emline_map_fits and sigma_dyn at exactly 0.001 in 8.
    """
    idx = int(tids_to_indices(np.array([tid], np.int64))[0])
    spec, ivar, z, tid_chk = load_by_index(idx)
    assert int(tid_chk) == tid, f"TARGETID mismatch: asked {tid}, got {tid_chk}"

    flux = flambda_to_maggies(WAVE_OBS, spec)
    iv = ivar_flambda_to_ivar_maggies(WAVE_OBS, ivar)
    mask = (iv > 0) & np.isfinite(flux)
    iv = np.where(mask, iv, 0.0)
    unc = 1.0 / np.sqrt(np.where(iv > 0, iv, np.inf))

    m_cont = mask_spectral_lines(WAVE_OBS, mask, z, halfwidth_kms=1500.0, line_waves=line_waves)
    line_pix = mask & ~m_cont
    tight = mask & ~mask_spectral_lines(WAVE_OBS, mask, z, halfwidth_kms=300.0, line_waves=line_waves)

    # Fractional error floor, added IN QUADRATURE: sigma_eff^2 = sigma^2 + (f*flux)^2.
    #
    # This is not cosmetic. A pure multiplicative rescale of the errors divides chi2 by a
    # constant and moves nothing -- same minimum, same relative depth of every local one.
    # A floor proportional to flux changes the WEIGHTING: it downweights high-S/N pixels
    # relative to low-S/N ones, which for this data means downweighting the red end
    # (S/N ~80/A) against the blue (S/N ~22/A). That is exactly the artefact behind the
    # "blue fits better than red" result, where the fractional residual is 13.81% blue vs
    # 2.28% red and a 3.68% floor takes the whole spectrum to chi2_red = 1.
    #
    # Independent support for a floor of this size: alf fits jitter = 1.391 +- 0.022 on
    # the same galaxy, i.e. 39% error inflation, as a free parameter.
    #
    # Applied to the DATA rather than the model so the noise model does not depend on
    # theta. At S/N ~60 the resulting noise bias is negligible.
    if error_floor and error_floor > 0:
        unc = np.sqrt(unc ** 2 + (float(error_floor) * np.abs(flux)) ** 2)

    # cont_only fits the line-masked pixels only; the Cue arm fits everything and
    # accounts for the lines with free nebular parameters.
    fit_mask = m_cont if cont_only else mask
    obs = build_obs(spec=flux, unc=unc, mask=fit_mask, resolution=LSF, wavelength=WAVE_OBS)
    cont_model, cont_tmpl = build_continuum_model(z)
    if cont_only:
        model, tmpl, sps_use = cont_model, cont_tmpl, sps
    else:
        model, tmpl = build_full_cue_model(cont_tmpl, cont_model.theta, cont_model, z)
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
            th0[model.theta_index[k]] = np.clip(v, float(np.atleast_1d(lo)[0]),
                                                float(np.atleast_1d(hi)[0]))

    warm = Path(warm_from) / f"{tid}.pkl" if warm_from else None
    if warm and warm.exists():
        th0 = warm_theta(model, warm, z, th0)

    t0 = time.time()
    best, info = map_fit(model, obs, sps_use, n_seeds=n_seeds, maxfev=maxfev,
                         theta0=th0, tag=f"  {tid}", method=method)
    if best is None:
        return {"target_id": tid, "z": float(z), "status": "map_failed", "optim": info}
    info["seconds"] = time.time() - t0
    info["theta0_seeds"] = dict(seeds or {})
    info["warm_from"] = str(warm) if warm and warm.exists() else None

    sp, stats = chi2_parts(model, best.x, obs, sps_use, line_pix)
    _, stats_tight = chi2_parts(model, best.x, obs, sps_use, tight)
    td = theta_dict(model, best.x)
    lr = line_ratios(WAVE_OBS, flux, sp, z, mask, lines)
    sfh = sfh_from_theta(model, best.x)

    plot_fit(tid, z, WAVE_OBS, flux, unc, mask, sp, lines, stats["chi2_red"], out)
    plot_sfh(tid, sfh, out)

    rec = {
        "target_id": tid, "z": float(z), "status": "ok", "wave": WAVE_OBS,
        "flux": flux, "unc": unc, "mask": mask, "line_pix": line_pix, "tight_pix": tight,
        "model": sp, "theta": best.x, "theta_dict": td, "labels": model.theta_labels(),
        "ndim": len(best.x), "stats": stats, "stats_tight": stats_tight,
        "line_ratios": lr, "lnp": lnp_split(model, best.x, obs, sps_use),
        "optim": info, "sfh": sfh, "lines": lines,
        "nlines": 0 if cont_only else len(model.emline_info),
        "nebemlineinspec": bool(np.any(model.params.get("nebemlineinspec"))),
        "lsf": "median desi_resolution",
        "nebular": "none (continuum only)" if cont_only else "cue_stellar_nebular",
        "cont_only": bool(cont_only),
        "fit_mask_npix": int(fit_mask.sum()),
        # unc above is POST-floor, so chi2 in this record is already the floored one.
        "error_floor": float(error_floor or 0.0),
        "optimizer": str(method),
        "zcontinuous": int(zcontinuous),
        # Record the VALUES, not just "frozen". build_continuum_model freezes at
        # DEFAULT_SET_VALS (sigma_reg 0.17, sigma_dyn 0.005); --freeze-hypers overrides
        # to FROZEN_HYPERS (1.5, 0.1). Those differ by 8.8x and 20x and neither has a
        # recorded source, so which one produced a given fit has to be on the record.
        "hypers": "frozen" if (frozen or cont_only) else "free",
        "hyper_values": {k: float(np.atleast_1d(model.params[k])[0])
                         for k in ("sigma_reg", "tau_eq", "tau_in", "sigma_dyn", "tau_dyn")
                         if k in model.params},
        "fixed": dict(fixed or {}),
        "sfh_prior": f"tophat+/-{FLAT_SFH_RANGE}" if flat_sfh else "gp_stochastic",
        "oii_fix": None if cont_only else bool(
            np.min(np.abs(np.asarray(cue_sps.emline_wavelengths, float)
                          - OII_FIX[1][1])) < 1e-3),
        "eline_waves": None if cont_only else
            np.asarray(cue_sps.emline_wavelengths, float).copy(),
    }
    with open(out / f"{tid}.pkl", "wb") as f:
        pickle.dump(rec, f)
    return rec


def _worker(tid, n_seeds, maxfev, outdir, seeds, frozen, fixed, warm_from, fix_oii=False,
            wave_file=None, flat_sfh=False, cont_only=False, error_floor=0.0,
            method="Powell", zcontinuous=1):
    S = get_sps(fix_oii=fix_oii, wave_file=wave_file, zcontinuous=zcontinuous)
    return fit_one(tid, S["sps"], S["cue"], S["lines"], S["line_waves"],
                   n_seeds, maxfev, Path(outdir), seeds=seeds, frozen=frozen, fixed=fixed,
                   warm_from=warm_from, flat_sfh=flat_sfh, cont_only=cont_only,
                   error_floor=error_floor, method=method, zcontinuous=zcontinuous)


def main(argv=None):
    p = argparse.ArgumentParser(description="MAP fits (Cue, free PSD) for emission-line OOD outliers.")
    p.add_argument("-s", "--sample", type=str,
                   default=str(PATHS["RESULTS"] / "emline_outlier_sample20.npz"))
    p.add_argument("-o", "--outdir", type=str, default=str(PATHS["RESULTS"] / "emline_map_fits"))
    p.add_argument("-n", "--n-seeds", type=int, default=6)
    p.add_argument("-m", "--maxfev", type=int, default=120_000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--fix-oii", action="store_true",
                   help="correct the [O II] 3726/3729 wavelengths in cue_emlines_info.dat")
    p.add_argument("--wave-file", default=None,
                   help="replace the whole Cue wavelength array from this .dat "
                        "(e.g. src/hubersed/data/cue_emlines_info_corrected.dat); "
                        "pair with --fix eline_delta_zred=0")
    p.add_argument("-w", "--workers", type=int, default=1)
    p.add_argument("--freeze-hypers", action="store_true",
                   help="fix the 5 PSD hyperparameters (Run A)")
    p.add_argument("--fix", action="append", default=[], metavar="NAME=VALUE",
                   help="fix a parameter at ONE value for the whole sample, "
                        "e.g. --fix logzsol=-2.5 (Run B)")
    p.add_argument("--fix-from-sample", default="", metavar="NAME[,NAME...]",
                   help="fix parameters at the PER-GALAXY value held in the sample npz "
                        "column of the same name, matched by TARGETID. Use to pin "
                        "logzsol to what alf measured for each object and see what chi2 "
                        "and the SFH do with metallicity out of the degeneracy.")
    p.add_argument("--flat-sfh-prior", action="store_true",
                   help=f"replace the GP stochastic prior on logsfr_ratios with "
                        f"TopHat(+/-{FLAT_SFH_RANGE} dex) and use SpecModel instead of "
                        f"HyperSpecModel. Removes the young-bin clamp (see "
                        f"knowledge/stochastic_prior_young_bin_clamp.md) but also removes "
                        f"bin-count insensitivity -- do not compare across nbins.")
    p.add_argument("--warm-from", default=None, metavar="DIR",
                   help="seed start 0 from the MAP in DIR/<tid>.pkl instead of the prior init")
    p.add_argument("--zcontinuous", type=int, default=1, choices=[1, 2],
                   help="1 = linear interpolation in log Z (kink at every MIST node). "
                        "2 = convolve with a closed-box MDF, smooth in logzsol, but a "
                        "MODEL change: logzsol becomes an MDF scale parameter with a "
                        "built-in metallicity spread, NOT comparable to zcontinuous=1 "
                        "or to alf [Z/H]. See build_sps.")
    p.add_argument("--optimizer", default="Powell", choices=["Powell", "Nelder-Mead"],
                   help="scipy minimize method. Powell (default) does 1-D Brent line "
                        "searches, which stall on the kinks ztinterp puts at every "
                        "MIST node. Nelder-Mead does no line search.")
    p.add_argument("--error-floor", type=float, default=0.0, metavar="FRAC",
                   help="fractional error floor added IN QUADRATURE: "
                        "sigma_eff^2 = sigma^2 + (FRAC*flux)^2. 0.037 is the value that "
                        "takes 42580 to chi2_red = 1 and matches alf's fitted jitter of "
                        "1.39. A pure multiplicative rescale would change nothing; this "
                        "reweights high-S/N pixels against low-S/N ones.")
    p.add_argument("--continuum-only", action="store_true",
                   help="fit build_continuum_model (15 free: logzsol, dust2, logmass, "
                        "9x logsfr_ratios, dust_ratio, dust_index, sigma_smooth) on "
                        "line-masked pixels, with plain FSPS and no Cue nebular. The 5 "
                        "PSD hyperparameters are frozen -- with them free the MAP "
                        "objective is unbounded (see fit_one's docstring).")
    args = p.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.sample)
    tids = d["target_ids"].astype(np.int64)
    if args.limit:
        tids = tids[: args.limit]

    # Per-galaxy logzsol seed if the sample file carries one, else the old -1.0. The
    # jitter is 2% of the prior range (+/-0.06 dex), so a seed 1.3 dex from the solution
    # -- which -1.0 is for the continuum outliers -- starts every restart in the same
    # wrong basin. Backward compatible: files without the column behave as before.
    _z0 = d["logzsol"] if "logzsol" in d.files else np.full(len(d["target_ids"]), -1.0)
    seeds = {int(t): {"logmass": float(m), "eline_sigma": float(s), "logzsol": float(zz)}
             for t, m, s, zz in zip(d["target_ids"], d["logmstar"],
                                    d["narrow_sigma"], _z0)}

    fixed = dict(kv.split("=") for kv in args.fix)
    fixed = {k: float(v) for k, v in fixed.items()}

    # Per-galaxy fixed values from a column of the sample npz. --fix is one value for the
    # whole sample; this is for "fix logzsol at the value ANOTHER code measured for THIS
    # galaxy". Matched by TARGETID, never by row position.
    persist = {}
    for k in (s.strip() for s in args.fix_from_sample.split(",") if s.strip()):
        if k not in d.files:
            raise SystemExit(f"--fix-from-sample {k}: no column '{k}' in {args.sample}. "
                             f"has: {sorted(d.files)}")
        persist[k] = {int(t): float(v) for t, v in zip(d["target_ids"], d[k])}
        print(f"fixing {k} per galaxy from {args.sample}", flush=True)

    def fixed_for(tid):
        f = dict(fixed)
        for k, m in persist.items():
            if tid not in m or not np.isfinite(m[tid]):
                raise SystemExit(f"--fix-from-sample {k}: no finite value for {tid}")
            f[k] = m[tid]
        return f

    if args.freeze_hypers or fixed or persist:
        print(f"frozen hypers: {args.freeze_hypers}   fixed: {fixed}"
              f"   per-galaxy: {sorted(persist)}", flush=True)

    todo = [int(t) for t in tids
            if not (args.skip_existing and (out / f"{t}.pkl").exists())]
    print(f"{len(todo)}/{len(tids)} to fit, {args.workers} worker(s)", flush=True)

    if args.workers > 1:
        results = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, t, args.n_seeds, args.maxfev, str(out),
                              seeds.get(t), args.freeze_hypers, fixed_for(t),
                              args.warm_from, args.fix_oii, args.wave_file,
                              flat_sfh=args.flat_sfh_prior,
                              cont_only=args.continuum_only,
                              error_floor=args.error_floor,
                              method=args.optimizer,
                              zcontinuous=args.zcontinuous): t for t in todo}
            for n, fu in enumerate(as_completed(futs), 1):
                t = futs[fu]
                try:
                    results[t] = fu.result()
                except Exception as e:
                    results[t] = {"target_id": t, "status": f"error:{type(e).__name__}"}
                print(f"[{n}/{len(todo)}] {t} done: {results[t].get('status')}", flush=True)
        recs = [results[t] for t in todo if t in results]
    else:
        S = get_sps(fix_oii=args.fix_oii, wave_file=args.wave_file,
                    zcontinuous=args.zcontinuous)
        recs = []
        for i, tid in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {tid}", flush=True)
            try:
                recs.append(fit_one(tid, S["sps"], S["cue"], S["lines"], S["line_waves"],
                                    args.n_seeds, args.maxfev, out, seeds=seeds.get(tid),
                                    frozen=args.freeze_hypers, fixed=fixed_for(tid),
                                    warm_from=args.warm_from, flat_sfh=args.flat_sfh_prior,
                                    cont_only=args.continuum_only,
                                    error_floor=args.error_floor,
                                    method=args.optimizer,
                                    zcontinuous=args.zcontinuous))
            except Exception as e:
                print(f"    FAILED {type(e).__name__}: {e}", flush=True)
                recs.append({"target_id": int(tid), "status": f"error:{type(e).__name__}"})

    summary = []
    def par(rec, name, default=np.nan):
        """A parameter's value whether it was free, --fix'd, or frozen in the template.

        Order: theta_dict (free) -> rec['fixed'] (--fix / --fix-from-sample) ->
        rec['hyper_values'] (frozen PSD hypers) -> default.
        """
        td = rec.get("theta_dict") or {}
        if name in td:
            return float(np.atleast_1d(td[name])[0])
        for k in ("fixed", "hyper_values"):
            m = rec.get(k) or {}
            if name in m:
                return float(m[name])
        return float(default)

    lines = next((r["lines"] for r in recs if r.get("status") == "ok"), AIR_LINES)
    for rec in recs:
        tid = rec["target_id"]
        if rec["status"] != "ok":
            summary.append({"target_id": int(tid), "status": rec["status"]})
            continue
        v = np.array([rec["line_ratios"][k] for k in lines])
        b = np.array([rec["line_ratios"][k] for k in BALMER])
        summary.append({
            "target_id": int(tid), "z": rec["z"], "status": "ok",
            "chi2_red": rec["stats"]["chi2_red"],
            "line_per_pix": rec["stats_tight"]["chi2_per_pix_line"],
            "cont_per_pix": rec["stats_tight"]["chi2_per_pix_cont"],
            "rms_all": float(np.sqrt(np.nanmean((v - 1) ** 2))),
            "rms_balmer": float(np.sqrt(np.nanmean((b - 1) ** 2))),
            # Every parameter goes through par(): theta_dict holds only what THIS fit was
            # free to move, and which parameters those are changes with --continuum-only,
            # --freeze-hypers and --fix-from-sample. An unguarded lookup KeyErrors AFTER
            # all the per-galaxy pkls are written, which has now happened twice
            # (eline_sigma, then logzsol). Do not add a bare theta_dict[...] here.
            "eline_sigma": par(rec, "eline_sigma"),
            "logzsol": par(rec, "logzsol"),
            "logmass": par(rec, "logmass"),
            "sigma_reg": par(rec, "sigma_reg", FROZEN_HYPERS["sigma_reg"]),
            "cont_only": bool(rec.get("cont_only", False)),
            "cmf_dev": float(np.max(np.abs(rec["sfh"]["cmf"] - rec["sfh"]["cmf_flat_null"]))),
            "n_converged": rec["optim"]["n_converged"],
            "gap_best_second": rec["optim"]["gap_best_second"],
            "seconds": rec["optim"].get("seconds", np.nan),
        })

    with open(out / "summary.pkl", "wb") as f:
        pickle.dump(summary, f)

    okr = [s for s in summary if s["status"] == "ok"]
    print(f"\n{'TARGETID':>19}{'z':>8}{'chi2':>8}{'line':>8}{'cont':>7}{'rmsAll':>8}"
          f"{'rmsBal':>8}{'e_sig':>7}{'logZ':>7}{'cmfdev':>8}{'conv':>6}")
    for s in okr:
        print(f"{s['target_id']:>19d}{s['z']:>8.4f}{s['chi2_red']:>8.3f}{s['line_per_pix']:>8.2f}"
              f"{s['cont_per_pix']:>7.3f}{s['rms_all']:>8.4f}{s['rms_balmer']:>8.4f}"
              f"{s['eline_sigma']:>7.1f}{s['logzsol']:>7.2f}{s['cmf_dev']:>8.3f}"
              f"{s['n_converged']:>6d}")
    print(f"\n{len(okr)}/{len(tids)} ok -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
