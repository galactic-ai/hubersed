"""Soft tie of Cue's C/O to O/H through the Nicholls et al. (2017) relation.

Optical lines alone barely constrain C/O, and in fits it rails at the top of Cue's grid. Carbon
and oxygen come from stars of different masses, so C/O follows O/H with scatter (Berg et al.
2019). The tie sets ``gas_logco`` from ``gas_logz`` with Nicholls et al. (2017) Eq. 3,

    log(C/O) = log10(10^-0.8 + 10^(log(O/H) + 2.72)),

plus a free offset ``gas_dlogco`` with a Normal(0, 0.17 dex) prior, the C/O dispersion of the
Berg et al. (2019) 40-galaxy sample. Cue's [O/H] and [C/O] are relative to its solar values
log(O/H) = -3.07 and log(C/O) = -0.37 (Li et al. 2024), which are totals before dust depletion,
like the stellar abundances Eq. 3 was fitted to. The result is clipped to Cue's C/O grid.
Sources and checks are in knowledge/notes_2026-10-10_er37084_reading.md, Step 1 (UberSED notes).
The same function was first used in experiments/2026-09-26_nautilus_c3k_broadline/map_co_oh.py.
"""

import numpy as np
from prospect.models import priors

CUE_LOGOH_SUN = -3.07
CUE_LOGCO_SUN = -0.37
CUE_LOGCO_RANGE = (-1.0, np.log10(5.4))
NICHOLLS_CO_A, NICHOLLS_CO_B = -0.8, 2.72
CO_SCATTER_DEX = 0.17


def co_from_oh(gas_logz=0.0, gas_dlogco=0.0, **extras):
    """Return Cue's ``gas_logco`` from ``gas_logz`` with the Nicholls et al. (2017) relation.

    Parameters
    ----------
    gas_logz : float or np.ndarray
        Cue's log (O/H) relative to its solar value.
    gas_dlogco : float or np.ndarray
        Offset in dex added to the relation.
    **extras
        Other model parameters, ignored (prospect passes them to ``depends_on`` functions).

    Returns
    -------
    np.ndarray
        log (C/O) relative to Cue's solar value, clipped to Cue's grid.
    """
    log_oh = CUE_LOGOH_SUN + np.atleast_1d(gas_logz)
    log_co = np.log10(10**NICHOLLS_CO_A + 10 ** (log_oh + NICHOLLS_CO_B))
    return np.clip(log_co - CUE_LOGCO_SUN + np.atleast_1d(gas_dlogco), *CUE_LOGCO_RANGE)


def add_co_tie(params, scatter=CO_SCATTER_DEX):
    """Tie ``gas_logco`` to ``gas_logz`` with a free offset, in place.

    Parameters
    ----------
    params : dict
        Prospect model parameter specification with ``gas_logz`` and ``gas_logco``.
    scatter : float
        Width in dex of the Normal prior on the offset ``gas_dlogco``.

    Returns
    -------
    dict
        ``params``, with ``gas_logco`` fixed through ``co_from_oh`` and ``gas_dlogco`` free.
    """
    params["gas_dlogco"] = dict(
        N=1, isfree=True, init=0.0, units="dex", prior=priors.Normal(mean=0.0, sigma=scatter)
    )
    params["gas_logco"] = dict(params["gas_logco"], isfree=False, depends_on=co_from_oh)
    params["gas_logco"].pop("prior", None)
    return params
