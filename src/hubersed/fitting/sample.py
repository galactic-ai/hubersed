"""emcee posterior sampling, seeded from a MAP solution.

Was bin/prospector/fit_single.py.

DROPPED in the move (recover from 77b3e31:bin/prospector/fit_single.py):
  - fit_galaxy()    -- called P.get_outlier_info(outlier_idx): POSITIONAL, the
    2026-06-10 bug, and get_outlier_info no longer exists. Its only caller was
    run_fits.py (dead SLURM arm). Use chi2.tids_to_indices + chi2.load_by_index.
  - run_optimizer() -- duplicated chi2._map_optimize. Two optimizers with different
    defaults is what produced the "free-SFH ceiling" retraction (unequal
    NSEEDS/MAXFEV between arms).
  - compute_sfh()   -- zero callers in bin/, src/, tmp/, nb/.

⚠️ UNVERIFIED: this code's acceptance fraction collapsed to 0.00-0.01 in earlier
runs. The log blames DE moves + over-dispersed init. Two better candidates, neither
tested: (1) emcee calls lnprobfn(nested=False) -> prospect's
MultiVariateNormal.__call__ has no override, so it hits Prior.__call__ =
scipy.stats.norm.logpdf(x, scale=Sigma) -> 9x9 NaN matrix (priors.py:269).
WorkingMVN in tmp/alpha_tilt_map.py is the workaround and is NOT wired in here.
(2) skip_initial_state_check was suppressing emcee's own degenerate-walker guard.
Check sampler.acceptance_fraction before trusting any chain from this.
"""

import emcee
import numpy as np


def run_emcee(
    lnp_fn,
    theta_map,
    ndim,
    model=None,
    nwalkers=64,
    nburn=300,
    nprod=1000,
    *,
    rng,
    frac_near_map=0.75,
    max_attempts=50,
    skip_state_check=False,
    progress=True,
):
    """Run emcee seeded from MAP, with prior-spread initialization.

    Parameters
    ----------
    lnp_fn : callable
        ln-posterior.
    theta_map : np.ndarray
        MAP solution to seed walkers around.
    ndim : int
        Number of free parameters.
    model : prospect model, optional
        If given, `frac_near_map` of walkers start near theta_map and the rest are
        drawn from the prior via model.prior_transform, to encourage exploration.
    rng : np.random.Generator
        Required, keyword-only. The caller owns it and must record the seed -- the
        chain is not reproducible otherwise (SPEC 7). Was np.random.* (global,
        unseeded) before the 2026-07-17 move.
    skip_state_check : bool
        Passed to emcee as skip_initial_state_check. Default False. This was
        hard-coded True; emcee's check catches linearly-dependent / degenerate
        walker configurations, which is a prime suspect for the historical
        acceptance collapse. Only set True if you know why the check fires.

    Returns
    -------
    emcee.EnsembleSampler
        Check `.acceptance_fraction` before using the chain.
    """
    p0 = np.empty((nwalkers, ndim))
    n_near = int(frac_near_map * nwalkers)

    for i in range(nwalkers):
        if model is not None and i >= n_near:
            p0[i] = model.prior_transform(rng.uniform(size=ndim))
        else:
            p0[i] = theta_map + 1e-4 * rng.standard_normal(ndim)

        # ensure a valid starting position
        attempts = 0
        while not np.isfinite(lnp_fn(p0[i])) and attempts < max_attempts:
            if model is not None:
                p0[i] = model.prior_transform(rng.uniform(size=ndim))
            else:
                p0[i] = theta_map + 1e-4 * rng.standard_normal(ndim)
            attempts += 1
        if attempts == max_attempts:
            # every draw invalid -> fall back to the MAP. If this fires for many
            # walkers the ensemble is degenerate, and emcee's state check (see
            # skip_state_check) is what will tell you.
            p0[i] = theta_map

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp_fn)
    p0, _, _ = sampler.run_mcmc(
        p0, nburn, progress=progress, skip_initial_state_check=skip_state_check
    )
    sampler.reset()
    sampler.run_mcmc(
        p0, nprod, progress=progress, skip_initial_state_check=skip_state_check
    )
    return sampler


def extract_chain(sampler, nburn=0, thin=5):
    """Extract flat chain with autocorr-based thinning if possible.

    `nburn` defaults to 0: run_emcee already burns in and calls sampler.reset(), so
    the chain handed here has no burn-in left to discard. The old default of 100
    silently threw away another 100 steps on top of run_emcee's 300.
    """
    try:
        tau = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
    except Exception:
        burnin, thin = nburn, thin
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    flat_logprobs = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
    return flat_samples, flat_logprobs
