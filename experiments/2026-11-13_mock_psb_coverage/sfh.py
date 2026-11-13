"""Method 1: how many v2 mocks have a post-starburst star formation history?

QUESTION: What fraction of the 500,000 v2 mocks (2026-11-09_mocks_v2_outliers) has an SFH that
counts as post-starburst by published criteria, overall and at z 0.45-0.60 where the gallery105
PSBs sit?
HYPOTHESIS: none registered.
INPUTS: mocks_v2_500000.h5 (priors/theta, priors/redshifts, failed), path as the first argument.
METHOD: each mock's SFR per bin from prospect's logsfr_ratios_to_sfrs (logmass, the 9 logsfr_ratios,
hubersed make_stochastic_agebins(z)), piecewise constant in lookback time. A mock is PSB-like when all
three hold:
1. quenched: SFR averaged over the last 100 Myr / M_formed < 0.2 / t_H(z) (Pacifici et al. 2016,
   arXiv:1609.03572, l.229; 100 Myr as Carnall et al. 2019, arXiv:1903.11082, l.372-377, and Wild
   et al. 2020 l.295; the 10 Myr window of Pacifici l.184 is also reported);
2. burst-like: mean SFR within +-50 Myr of the SFR peak in the last 2 Gyr above the mean SFR 200-300 Myr
   before the peak (Wild et al. 2020, arXiv:2001.09154, l.358, l.214);
3. more than 40% of the mass formed in the last 1.5 Gyr (Wild et al. 2020 l.362).
Assumptions: sSFR uses mass formed (logmass), not living mass, so it is low by the mass-loss
fraction; the peak is located on the bin grid (factor ~2.3 wide bins at z~0.5); Wild's criteria come
from z~1 galaxies.
SEED: none.
COMMAND: ``python sfh.py <mocks_v2_500000.h5> <out dir>`` (stampede3, run.slurm).
RESULT: pending.
FIGURES: none (sfh_psb.txt, sfh_psb.npz).
"""

import sys
from pathlib import Path

import h5py
import numpy as np
from prospect.models.transforms import logsfr_ratios_to_sfrs

from hubersed.sps.utils import make_stochastic_agebins, universe_age_gyr

SSFR_CUT = 0.2  # times 1 / t_H
WINDOWS = (0.1, 0.01)  # Gyr, main and Pacifici
BURST_HALF, PRE = 0.05, (0.2, 0.3)  # Gyr, Wild l.358
PEAK_MAX, F15_MIN, F15_T = 2.0, 0.4, 1.5  # Gyr, fraction, Gyr


def mean_sfr(edges, sfr, t0, t1):
    """Return the time-weighted mean SFR between lookback times t0 and t1 (Gyr)."""
    lo, hi = np.maximum(edges[:, 0], t0), np.minimum(edges[:, 1], t1)
    w = np.clip(hi - lo, 0, None)
    return float(np.sum(w * sfr) / (t1 - t0)) if t1 > t0 else np.nan


def classify(z, logmass, ratios):
    """Return the criteria values of one mock."""
    agebins = make_stochastic_agebins(z)
    sfr = logsfr_ratios_to_sfrs(logmass=logmass, logsfr_ratios=ratios, agebins=agebins)  # Msun/yr
    edges = 10**agebins / 1e9  # Gyr
    m_formed = 10**logmass
    t_h = universe_age_gyr(z)
    ssfr = {w: mean_sfr(edges, sfr, 0.0, w) / m_formed * 1e9 for w in WINDOWS}  # per Gyr
    recent = edges[:, 0] < PEAK_MAX
    k = np.flatnonzero(recent)[np.argmax(sfr[recent])]
    t_peak = float(np.mean(edges[k]))
    burst = mean_sfr(edges, sfr, max(t_peak - BURST_HALF, 0.0), t_peak + BURST_HALF)
    pre = mean_sfr(edges, sfr, t_peak + PRE[0], t_peak + PRE[1])
    dt = (edges[:, 1] - edges[:, 0]) * 1e9
    mass = sfr * dt
    lo, hi = edges[:, 0], np.minimum(edges[:, 1], F15_T)
    f15 = float(np.sum(sfr * np.clip(hi - lo, 0, None) * 1e9) / mass.sum())
    return ssfr[0.1] * t_h, ssfr[0.01] * t_h, burst / pre if pre > 0 else np.inf, f15, t_peak


def main(path, out):
    """Classify every mock and write the table."""
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as h:
        labels = [s.decode() if isinstance(s, bytes) else str(s) for s in h.attrs["theta_labels"]]
        theta = h["priors/theta"][:]
        z = h["priors/redshifts"][:]
        failed = h["failed"][:]
    im = labels.index("logmass")
    ir = [labels.index(f"logsfr_ratios_{j}") for j in range(1, 10)]
    rows = np.array([classify(z[i], theta[i, im], theta[i, ir]) for i in range(z.size)])
    nssfr100, nssfr10, ratio, f15, t_peak = rows.T
    quench100, quench10 = nssfr100 < SSFR_CUT, nssfr10 < SSFR_CUT
    burst, young = ratio > 1, f15 > F15_MIN
    good = ~failed
    np.savez(out / "sfh_psb.npz", z=z, nssfr100=nssfr100, nssfr10=nssfr10, ratio=ratio, f15=f15,
             t_peak=t_peak, failed=failed)  # fmt: skip
    zsel = (z >= 0.45) & (z <= 0.60)
    lines = [f"mocks {good.sum()} (failed {failed.sum()})"]
    for name, q in (("100 Myr", quench100), ("10 Myr", quench10)):
        psb = q & burst & young & good
        lines.append(
            f"{name}: quenched {np.mean(q[good]):.4f}, burst-like {np.mean(burst[good]):.4f}, "
            f"f1.5>0.4 {np.mean(young[good]):.4f}; all three {psb.sum()} = {np.mean(psb[good]):.5f}; "
            f"at z 0.45-0.60: {np.sum(psb & zsel)} of {np.sum(good & zsel)} = "
            f"{np.sum(psb & zsel) / np.sum(good & zsel):.5f}"
        )
    text = "\n".join(lines)
    (out / "sfh_psb.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
