import numpy as np
import warnings
import emcee
from scipy.optimize import minimize
from prospect.fitting import lnprobfn
from prospect.models.transforms import logsfr_ratios_to_sfrs

from fit_config import build_continuum_model, build_full_model

from hubersed.conversion import flambda_to_maggies, ivar_flambda_to_ivar_maggies

import parameter_file as P


def run_optimizer(neg_lnp, theta_init, n_seeds=5, jitter=0.05, 
                  maxfev=30_000, ftol=1e-6):
    """Multi-start Powell optimizer."""
    results = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        theta_start = theta_init + rng.normal(0, jitter, size=theta_init.shape)
        if not np.isfinite(-neg_lnp(theta_start)):
            theta_start = theta_init
        res = minimize(neg_lnp, theta_start, method="Powell",
                      options={"maxiter": maxfev//10, 
                               "maxfev": maxfev, "ftol": ftol})
        if np.isfinite(res.fun) and res.fun < 1e10:
            results.append(res)
    return min(results, key=lambda r: r.fun) if results else None

def run_emcee(lnp_fn, theta_map, ndim, model=None, nwalkers=64, 
              nburn=300, nprod=1000):
    """Run emcee seeded from MAP, with prior-spread initialization."""
    p0 = np.empty((nwalkers, ndim))
    
    for i in range(nwalkers):
        if model is not None:
            # Draw from prior, then pull toward MAP
            # 75% of walkers start near MAP, 25% start from prior to encourage exploration
            if i < 3 * nwalkers // 4:
                p0[i] = theta_map + 1e-4 * np.random.randn(ndim)
            else:
                p0[i] = model.prior_transform(np.random.uniform(size=ndim))
        else:
            p0[i] = theta_map + 1e-4 * np.random.randn(ndim)
        
        # Ensure valid starting position
        attempts = 0
        while not np.isfinite(lnp_fn(p0[i])) and attempts < 50:
            if model is not None:
                p0[i] = model.prior_transform(np.random.uniform(size=ndim))
            else:
                p0[i] = theta_map + 1e-4 * np.random.randn(ndim)
            attempts += 1
        if attempts == 50:
            p0[i] = theta_map

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp_fn)
    p0, _, _ = sampler.run_mcmc(p0, nburn, progress=False,
                                 skip_initial_state_check=True)
    sampler.reset()
    sampler.run_mcmc(p0, nprod, progress=False,
                     skip_initial_state_check=True)
    return sampler

def extract_chain(sampler, nburn=100, thin=5):
    """Extract flat chain with autocorr-based thinning if possible."""
    try:
        tau    = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        thin   = int(0.5 * np.min(tau))
    except Exception:
        burnin, thin = nburn, thin
    flat_samples  = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    flat_logprobs = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
    return flat_samples, flat_logprobs

def compute_sfh(flat_samples, model, n_thin=10):
    """Compute SFH percentiles from posterior samples."""
    agebins      = model.params["agebins"]
    logmass_idx  = model.theta_index["logmass"]
    logratio_idx = model.theta_index["logsfr_ratios"]

    all_sfrs = []
    for theta in flat_samples[::n_thin]:
        sfrs = logsfr_ratios_to_sfrs(
            theta[logmass_idx][0], theta[logratio_idx], agebins
        )
        all_sfrs.append(sfrs)
    all_sfrs = np.array(all_sfrs)
    return {
        "agebins": agebins,
        "sfr_16":  np.percentile(all_sfrs, 16, axis=0),
        "sfr_50":  np.percentile(all_sfrs, 50, axis=0),
        "sfr_84":  np.percentile(all_sfrs, 84, axis=0),
    }

def fit_galaxy(outlier_idx,
               run_continuum=True, run_full=True,
               cont_nseeds=3, cont_maxfev=30_000,
               full_nseeds=3, full_maxfev=30_000,
               full_nburn=500, full_nprod=3000):
    """
    Full pipeline for one galaxy.
    Returns dict with all results.
    """
    # ── Load data ────────────────────────────────────────────────────────────
    spec, unc, redshift, _, gal_id = P.get_outlier_info(outlier_idx, streaming=False)
    wave_A = P.WAVE_OBS

    spec_maggies  = flambda_to_maggies(wave_A, spec)
    ivar_maggies  = ivar_flambda_to_ivar_maggies(wave_A, unc)
    sigma_maggies = 1 / np.sqrt(np.where(ivar_maggies > 0, ivar_maggies, np.inf))

    sps = P.build_sps()
    fsps_waves = sps.ssp.emline_wavelengths
    fsps_optical = fsps_waves[(fsps_waves > 3600) & (fsps_waves < 9824)]

    mask      = (sigma_maggies > 0) & np.isfinite(sigma_maggies)
    mask_em   = P.mask_spectral_lines(wave_A, mask, redshift, halfwidth_kms=1500.0, line_waves=fsps_optical)

    print(f"Fitting galaxy {gal_id} (outlier index {outlier_idx}) at z={redshift:.3f}")
    obs = P.build_obs(spec=spec_maggies, unc=sigma_maggies, mask=mask_em)

    results = {
        "outlier_idx": outlier_idx,
        "id":          gal_id,
        "redshift":    redshift,
    }

    # ── Continuum fit ────────────────────────────────────────────────────────
    if run_continuum:
        print(f"Running continuum fit for galaxy {gal_id}...")
        model, template = build_continuum_model(redshift)
        theta_init = model.theta.copy()

        def neg_lnp_cont(theta):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    lp = lnprobfn(theta, model=model, obs=obs,
                                  sps=sps, nested=False)
                    return -lp if np.isfinite(lp) else 1e18
                except Exception:
                    return 1e18

        # Optimizer
        print("Running optimizer for continuum fit...")
        best_res = run_optimizer(neg_lnp_cont, theta_init,
                                 n_seeds=cont_nseeds, maxfev=cont_maxfev)
        if best_res is None:
            results["continuum_status"] = "optimizer_failed"
            return results

        theta_map_cont = best_res.x

        results.update({
            "continuum_status":    "success",
            "theta_map_cont":      theta_map_cont,
        })

    # ── Full nebular fit ─────────────────────────────────────────────────────
    if run_full and results.get("continuum_status") == "success":
        print(f"Running full fit for galaxy {gal_id}...")
        obs_full   = P.build_obs(spec=spec_maggies, unc=sigma_maggies, mask=mask)
        full_model, full_template = build_full_model(
            template, theta_map_cont, model, redshift
        )
        theta_init_full = full_model.theta.copy()

        def neg_lnp_full(theta):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    lp = lnprobfn(theta, model=full_model, obs=obs_full,
                                  sps=sps, nested=False)
                    return -lp if np.isfinite(lp) else 1e18
                except Exception:
                    return 1e18
        print("Running optimizer for full fit...")
        best_res_full = run_optimizer(neg_lnp_full, theta_init_full,
                                      n_seeds=full_nseeds, 
                                      maxfev=full_maxfev, jitter=0.01)
        if best_res_full is None:
            results["full_status"] = "optimizer_failed"
            return results

        theta_map_full = best_res_full.x

        def lnp_full(theta):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    lp = lnprobfn(theta, model=full_model, obs=obs_full,
                                  sps=sps, nested=False)
                    return lp if np.isfinite(lp) else -np.inf
                except Exception:
                    return -np.inf

        print("Running MCMC for full fit...")
        sampler_full = run_emcee(lnp_full, theta_map_full,
                                 ndim=len(theta_map_full),
                                 nburn=full_nburn, nprod=full_nprod, model=full_model)

        print("Extracting chain for full fit...")
        flat_samples_full, flat_lp_full = extract_chain(sampler_full)
        theta_best_full = flat_samples_full[np.argmax(flat_lp_full)]

        spec_full, _, _ = full_model.predict(theta_best_full, 
                                              obs=obs_full, sps=sps)
        resid_full = (obs_full['spectrum'][obs_full['mask']] \
                      - spec_full[obs_full['mask']]) \
                     / obs_full['unc'][obs_full['mask']]
        chi2_full  = float(np.nansum(resid_full**2))
        ndof_full  = int(obs_full['mask'].sum()) - len(theta_best_full)
        print(f"Full fit chi2_red = {chi2_full / ndof_full:.3f}")

        params_full = {}
        for name, idx in full_model.theta_index.items():
            vals = flat_samples_full[:, idx]
            if vals.ndim == 1:
                q16, q50, q84 = np.percentile(vals, [16, 50, 84])
                params_full[name] = {"q16": q16, "q50": q50, "q84": q84}
            else:
                params_full[name] = {
                    "q16": np.percentile(vals, 16, axis=0),
                    "q50": np.percentile(vals, 50, axis=0),
                    "q84": np.percentile(vals, 84, axis=0),
                }

        # ISM residuals at key absorption lines
        ism_lines = {
            'NaD': 5893.0, 'MgI': 5183.6,
            'CaK': 3933.7, 'CaH': 3968.5,
        }
        ism_residuals = {}
        for line_name, wave_rest in ism_lines.items():
            wave_obs = wave_rest * (1 + redshift)
            region = (wave_A >= wave_obs - 15) & (wave_A <= wave_obs + 15) \
                     & obs_full['mask']
            if region.sum() >= 3:
                ism_residuals[line_name] = float(np.median(
                    (spec_maggies[region] - spec_full[region]) / spec_full[region]
                ))

        results.update({
            "full_status":       "success",
            "theta_map_full":    theta_map_full,
            "theta_best_full":   theta_best_full,
            "flat_samples_full": flat_samples_full,
            "flat_lp_full":      flat_lp_full,
            "params_full":       params_full,
            "chi2_full":         chi2_full,
            "chi2_red_full":     chi2_full / ndof_full,
            "ndof_full":         ndof_full,
            "spec_full":         spec_full,
            "ism_residuals":     ism_residuals,
            "acceptance_full":   float(np.mean(sampler_full.acceptance_fraction)),
        })

    return results