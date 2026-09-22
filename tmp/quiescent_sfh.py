"""
Tabulated declining ("quiescent") SFH constructor for the cont-only OOD work.

Purpose
-------
The stochastic SFH prior (Wan+24) has mean vector mu_{A/B} = 0 (flat-baseline
SFR(t)); see adjust_stochastic_params in prospect (mean = np.zeros). That baseline
expectation is star-forming (log sSFR100 ~ -10), so the massive-quiescent cont-only
outliers (DN4000~2, log sSFR100 <~ -11.4) are under-sampled by the mock prior.

This module builds a *tabulated* declining SFH (exp-decline-since-formation) in the
SAME 10-bin agebins used everywhere else, and returns it as a 9-vector of
logsfr_ratios[j] = log10(SFR_j / SFR_{j+1}), j=0 = youngest bin (prospect convention).

Two uses:
  (step 2) FIX logsfr_ratios to this vector in the MAP fit (forward-model reach test
           decoupled from the flat-baseline prior), or seed/center a prior on it.
  (step 3) use it (or its mean over a population) as the non-zero mu_{A/B} for a
           quiescent component of a MIXTURE prior in mock generation.

Epistemic status: SFH-space construction + diagnostics are exact (numpy). Whether the
resulting *spectrum* matches the data (DN4000, breaks) requires the FSPS/Cue forward
model -- that is the step-2 fit, not done here.
"""

import numpy as np

from hubersed.prospector.utils import make_stochastic_agebins, universe_age_gyr


def _masses_from_logsfr_ratios(logmass, logsfr_ratios, agebins):
    """Replica of prospect.models.transforms.logsfr_ratios_to_masses
    (j=0 = most recent bin). Returns per-bin stellar mass formed."""
    nbins = agebins.shape[0]
    sr = 10 ** np.clip(logsfr_ratios, -10, 10)
    dt = 10 ** agebins[:, 1] - 10 ** agebins[:, 0]
    coeffs = np.array(
        [
            (1.0 / np.prod(sr[:i])) * (np.prod(dt[1 : i + 1]) / np.prod(dt[:i]))
            for i in range(nbins)
        ]
    )
    return (10**logmass) / coeffs.sum() * coeffs


def declining_logsfr_ratios(z, tau_gyr, tform_frac=0.95):
    """Tabulated exp-declining-since-formation SFH -> logsfr_ratios (9-vector).

    SFR(t_lb) ∝ exp(-(t_form - t_lb)/tau) for lookback t_lb < t_form, else ~0,
    with t_form = tform_frac * t_univ(z). Larger tau = less quenched.
    """
    ab = make_stochastic_agebins(z)  # log10(yr), shape (10,2)
    t_univ = universe_age_gyr(z)
    mid_lb = 0.5 * (10 ** ab[:, 0] + 10 ** ab[:, 1]) / 1e9  # Gyr lookback
    age_since_form = tform_frac * t_univ - mid_lb  # Gyr since formation
    sfr = np.where(age_since_form > 0, np.exp(-age_since_form / tau_gyr), 0.0)
    sfr = np.clip(sfr, 1e-12, None)
    return np.log10(sfr[:-1] / sfr[1:])  # j = 0..8


def sfh_diagnostics(z, logsfr_ratios, logmass=10.6):
    """Return (mass-weighted lookback age [Gyr], sSFR100 [/yr], mass fractions young->old)."""
    ab = make_stochastic_agebins(z)
    dt = 10 ** ab[:, 1] - 10 ** ab[:, 0]
    m = _masses_from_logsfr_ratios(logmass, logsfr_ratios, ab)
    sfr = m / dt
    lb = 10**ab / 1e9
    mid = 0.5 * (lb[:, 0] + lb[:, 1])
    mwa = np.sum(m * mid) / np.sum(m)
    w100 = np.clip(np.minimum(lb[:, 1], 0.1) - lb[:, 0], 0, None) / (
        lb[:, 1] - lb[:, 0]
    )
    ssfr100 = np.sum(sfr * w100 * dt) / 1e8 / 10**logmass
    return mwa, ssfr100, m / np.sum(m)


if __name__ == "__main__":
    for z, tau in [(0.029, 1.0), (0.029, 2.0), (0.029, 3.0)]:
        r = declining_logsfr_ratios(z, tau)
        mwa, ssfr, frac = sfh_diagnostics(z, r)
        print(f"z={z} tau={tau}: log sSFR100={np.log10(ssfr):.2f}  mwa={mwa:.2f} Gyr")
