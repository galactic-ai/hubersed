"""Draw the stochastic-SFH prior sample that make_model_seds turns into mock spectra.

The npz is written by ``scripts/get_stochastic_priors.py``.
"""

import numpy as np

from hubersed.mocks.distributions import (
    sample_log_uniform,
    sample_truncated_normal,
    sample_uniform,
)
from hubersed.sps.utils import universe_age_gyr


def draw_priors(n, seed, cue):
    """Draw every prior parameter with one seeded generator.

    Parameters
    ----------
    n : int
        Number of samples.
    seed : int
        Seed of the numpy generator. The same seed gives the same draws.
    cue : bool
        Draw the gas metallicity over the Cue range and add the Cue (Li+24) nebular
        parameters. Otherwise use the FSPS nebular grid range.

    Returns
    -------
    dict
        Parameter arrays of length n, plus ``_seed``, ``_sample_size`` and ``_cue``.
    """
    rng = np.random.default_rng(seed)

    # from Wan+24 Stochastic prior model

    # redshift 0.01 to 0.6 uniform
    redshifts = sample_uniform(0.01, 0.6, size=n, rng=rng)

    # stellar mass 7 to 12 uniform
    stellar_masses = sample_uniform(7, 12, size=n, rng=rng)

    # stellar metallicity -2.5 to 0.5 uniform
    stellar_metallicities = sample_uniform(-2.5, 0.5, size=n, rng=rng)

    # sigma_reg log uniform 0.1 to 5
    sigma_regs = sample_log_uniform(0.1, 5, size=n, rng=rng)

    # tau_eq uniform 0.01 to t_H
    t_h = universe_age_gyr(redshifts)
    tau_eqs = sample_uniform(0.01, t_h, size=n, rng=rng)
    tau_ins = sample_uniform(0.01, t_h, size=n, rng=rng)

    # sigma_dyn log uniform 0.001 to 0.5
    sigma_dyns = sample_log_uniform(0.001, 0.5, size=n, rng=rng)

    # tau_dyn truncated normal min 0.005 max 0.2 mu 0.01 sigma 0.02
    tau_dyns = sample_truncated_normal(0.01, 0.02, 0.005, 0.2, size=n, rng=rng)

    # dust_index uniform -2.5 to 0.4, the same range as the fit prior (sps/config.py)
    ns = sample_uniform(-2.5, 0.4, size=n, rng=rng)

    # tau_dust,2 truncated normal min 0.0 max 4 mu 0.3 sigma 1.0
    tau_dust_2s = sample_truncated_normal(0.3, 1.0, 0.0, 4.0, size=n, rng=rng)

    # tau_dust,1 truncated normal min 0.0 max 2 mu 1.0 sigma 0.3 (actually dust_ratio)
    tau_dust_1s = sample_truncated_normal(1.0, 0.3, 0.0, 2.0, size=n, rng=rng)

    # U_min truncated normal min 0.1 max 15 mu 2.0 sigma 1.0
    u_mins = sample_truncated_normal(2.0, 1.0, 0.1, 15.0, size=n, rng=rng)

    # gamma_e log uniform 1e-4 to 0.1
    gamma_es = sample_log_uniform(1e-4, 0.1, size=n, rng=rng)

    # q_pah uniform 0.5 to 7.0
    q_pahs = sample_uniform(0.5, 7.0, size=n, rng=rng)

    # sigma_gas uniform 10 to 250
    sigma_gass = sample_uniform(10, 250, size=n, rng=rng)

    # gas phase metallicity. Cue allows -2.2 to 0.5. The FSPS nebular grid covers -1.3 to
    # 0.3 and FSPS clamps outside it (add_nebular.f90:27-29).
    gas_lo, gas_hi = (-2.2, 0.5) if cue else (-1.3, 0.3)
    gas_metallicities = sample_uniform(gas_lo, gas_hi, size=n, rng=rng)

    # gas ionization parameter -4 to -1
    gas_ionization_parameters = sample_uniform(-4.0, -1.0, size=n, rng=rng)

    # top hat min 10 max 400 (not used in Wan+24 but included for completeness)
    sigma_smooths = sample_uniform(10, 400, size=n, rng=rng)

    # save to npz (separate filename for the Cue sample)
    arrays = dict(
        redshifts=redshifts,
        stellar_masses=stellar_masses,
        stellar_metallicities=stellar_metallicities,
        sigma_regs=sigma_regs,
        tau_eqs=tau_eqs,
        tau_ins=tau_ins,
        sigma_dyns=sigma_dyns,
        tau_dyns=tau_dyns,
        ns=ns,
        tau_dust_2s=tau_dust_2s,
        tau_dust_1s=tau_dust_1s,
        u_mins=u_mins,
        gamma_es=gamma_es,
        q_pahs=q_pahs,
        sigma_gass=sigma_gass,
        gas_metallicities=gas_metallicities,
        gas_ionization_parameters=gas_ionization_parameters,
        sigma_smooths=sigma_smooths,
    )

    # --- Cue (Li+24) free nebular params (arXiv:2405.04598 Table 1) ---
    # gas_logno/gas_logco are LOG10 of (N/O)/(N-O)_sun ; grid is linear [0.1, 5.4] -> log [-1, log10(5.4)]
    if cue:
        arrays["gas_lognHs"] = sample_uniform(1.0, 4.0, size=n, rng=rng)  # log nH [cm^-3]
        arrays["gas_lognos"] = sample_uniform(-1.0, np.log10(5.4), size=n, rng=rng)  # log [N/O]
        arrays["gas_logcos"] = sample_uniform(-1.0, np.log10(5.4), size=n, rng=rng)  # log [C/O]

    arrays["_seed"] = seed
    arrays["_sample_size"] = n
    arrays["_cue"] = cue

    return arrays
