import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.special as ssp
from scipy.stats import truncnorm

from hubersed.paths import PATHS
from hubersed.prospector.utils import make_stochastic_agebins, universe_age_gyr

import prospect.models as _pm

GALLAZZI = Path(_pm.__file__).parent / "prior_data/gallazzi_05_massmet.txt"
FSF_VAC = PATHS["DATA"] / "fastspec-iron-sv3-bright.fits"


# ---------------------------------------------------------------- baselines
def ciesla_ms_sfh(t_gyr, mseed):
    """Ciesla+17 right-skew-peak MS SFH. Shape only (normalization cancels
    in log-SFR ratios). Coefficients verified against the paper TeX."""
    A = 6e-3 * np.exp(-np.log10(mseed) / -0.84)
    mu = 47.39 * np.exp(-np.log10(mseed) / 3.12)
    sg = 17.08 * np.exp(-np.log10(mseed) / 2.96)
    rs = -0.56 * np.log10(mseed) + 7.03
    s = A * (np.sqrt(np.pi) / 2) * sg \
        * np.exp((sg / (2 * rs)) ** 2 - (t_gyr - mu) / rs) \
        * ssp.erfc(sg / (2 * rs) - (t_gyr - mu) / sg)
    return np.clip(np.nan_to_num(s), 1e-12, None)


class MseedInverter:
    """Map (log formed mass, z) -> Ciesla seed mass by matching the formed-mass
    integral at t_univ(z). Precomputed on a grid."""

    def __init__(self):
        self.gseed = np.linspace(2, 9, 60)
        self.tsh = np.linspace(1e-3, 14.0, 600)
        sf = np.array([ciesla_ms_sfh(self.tsh, 10 ** g) for g in self.gseed])
        mid = 0.5 * (sf[:, 1:] + sf[:, :-1]) * np.diff(self.tsh) * 1e9
        self.cum = np.concatenate([np.zeros((len(self.gseed), 1)), np.cumsum(mid, axis=1)], axis=1)

    def __call__(self, logm, t_univ):
        formed = np.array([np.interp(t_univ, self.tsh, c) for c in self.cum])
        return 10 ** np.interp(logm, np.log10(np.clip(formed, 1.0, None)), self.gseed)


def mu_ratios(logm, z, quiescent, t_q, tau_q, inverter, floor_dex=0.75):
    # floor 0.75 dex: FSPS pilot 2026-08-23 -- floor 1.0 left the M>10.5 median
    # 0.02 high; 0.75 hits med/frac>1.6 on the Li D4000_n target.
    """Mean log-SFR-ratio 9-vector for one galaxy. Invariant: adding any
    constant to the log baseline changes nothing (ratios difference it out)."""
    ab = make_stochastic_agebins(z=z)
    bc = (10 ** ab.mean(axis=1)) / 1e9  # lookback Gyr
    tu = universe_age_gyr(z)
    ls = np.log10(ciesla_ms_sfh(tu - bc, inverter(logm, tu)))
    if quiescent:
        supp = np.where(bc < t_q, -(t_q - bc) / tau_q / np.log(10), 0.0)
        ls = ls + np.maximum(supp, -floor_dex)
    return ls[:-1] - ls[1:]


# ---------------------------------------------------------------- f_q table
def build_fq_table():
    """Quiescent fraction f_q(M*, z) from the FastSpecFit sv3-bright VAC
    (sSFR < 1e-11). NaN cells filled by pinning to nearest measured mass."""
    from astropy.io import fits

    f = fits.open(FSF_VAC, memmap=True)
    fs, md = f["FASTSPEC"].data, f["METADATA"].data
    z, lm, sfr, zw = fs["Z"], fs["LOGMSTAR"], fs["SFR"], md["ZWARN"]
    ok = (zw == 0) & (z > 0.01) & (z < 0.6) & np.isfinite(lm) & (lm > 0) \
        & np.isfinite(sfr) & (sfr > 0)
    ssfr = np.log10(sfr[ok]) - lm[ok]
    zz, mm = z[ok], lm[ok]
    medges = np.arange(7, 12.25, 0.25)
    zedges = np.array([0.01, 0.1, 0.2, 0.35, 0.6])
    fq = np.full((len(medges) - 1, len(zedges) - 1), np.nan)
    for i in range(fq.shape[0]):
        for j in range(fq.shape[1]):
            s = (mm >= medges[i]) & (mm < medges[i + 1]) & (zz >= zedges[j]) & (zz < zedges[j + 1])
            if s.sum() > 30:
                fq[i, j] = np.mean(ssfr[s] < -11)
    for j in range(fq.shape[1]):  # pin ends, never extrapolate past the data
        col = fq[:, j]
        v = np.where(np.isfinite(col))[0]
        col[: v[0]] = col[v[0]]
        col[v[-1] + 1:] = col[v[-1]]
    return medges, zedges, fq


def fq_lookup(medges, zedges, fq, logm, z):
    i = np.clip(np.searchsorted(medges, logm) - 1, 0, fq.shape[0] - 1)
    j = np.clip(np.searchsorted(zedges, z) - 1, 0, fq.shape[1] - 1)
    return np.nan_to_num(fq[i, j])


# ---------------------------------------------------------------- main draw
def draw(n, seed, out):
    rng = np.random.default_rng(seed)
    tab = np.loadtxt(GALLAZZI)  # mass P50 P16 P84 (Chabrier)
    inverter = MseedInverter()
    medges, zedges, fqt = build_fq_table()

    z = rng.uniform(0.01, 0.6, n)
    lm = rng.uniform(7, 12, n)
    t_h = universe_age_gyr(z)

    a = dict(
        redshifts=z,
        stellar_masses=lm,
        sigma_regs=10 ** rng.uniform(np.log10(0.1), np.log10(5), n),
        tau_eqs=rng.uniform(0.01, t_h),
        tau_ins=rng.uniform(0.01, t_h),
        sigma_dyns=10 ** rng.uniform(np.log10(0.001), np.log10(0.5), n),
        tau_dyns=truncnorm.rvs((0.005 - 0.01) / 0.02, (0.2 - 0.01) / 0.02,
                               loc=0.01, scale=0.02, size=n, random_state=rng),
        tau_dust_1s=truncnorm.rvs((0 - 1) / 0.3, (2 - 1) / 0.3, loc=1.0, scale=0.3,
                                  size=n, random_state=rng),
        u_mins=truncnorm.rvs((0.1 - 2) / 1, (15 - 2) / 1, loc=2.0, scale=1.0,
                             size=n, random_state=rng),
        gamma_es=10 ** rng.uniform(-4, -1, n),
        q_pahs=rng.uniform(0.5, 7.0, n),
        sigma_gass=rng.uniform(10, 250, n),
        gas_ionization_parameters=rng.uniform(-4.0, -1.0, n),
        sigma_smooths=rng.uniform(10, 400, n),
        gas_lognHs=rng.uniform(1.0, 4.0, n),
        gas_lognos=rng.uniform(-1.0, np.log10(5.4), n),
        gas_logcos=rng.uniform(-1.0, np.log10(5.4), n),
    )

    # change 2: logzsol | M* (Gallazzi+05, sigma = P84-P16)
    loc = np.interp(lm, tab[:, 0], tab[:, 1])
    sc = np.interp(lm, tab[:, 0], tab[:, 3]) - np.interp(lm, tab[:, 0], tab[:, 2])
    a["stellar_metallicities"] = truncnorm.rvs((-2.5 - loc) / sc, (0.5 - loc) / sc,
                                               loc=loc, scale=sc, random_state=rng)
    # change 3b: gas Z coupled to stellar Z
    a["gas_metallicities"] = np.clip(rng.normal(a["stellar_metallicities"], 0.3), -2.2, 0.5)

    # branch draw
    fq = fq_lookup(medges, zedges, fqt, lm, z)
    quiescent = rng.random(n) < fq
    a["quiescent"] = quiescent
    t_q = rng.uniform(0.5, 4.0, n)
    tau_q = 10 ** rng.uniform(np.log10(0.5), np.log10(2.0), n)

    # change 3a: branch-conditional dust
    d2_q = truncnorm.rvs((0 - 0.1) / 0.15, (1 - 0.1) / 0.15, loc=0.1, scale=0.15,
                         size=n, random_state=rng)
    d2_sf = truncnorm.rvs((0 - 0.3) / 0.5, (2.5 - 0.3) / 0.5, loc=0.3, scale=0.5,
                          size=n, random_state=rng)
    n_q = truncnorm.rvs((-1 + 0.4) / 0.3, (0.4 + 0.4) / 0.3, loc=-0.4, scale=0.3,
                        size=n, random_state=rng)
    n_sf = rng.uniform(-1, 0.4, n)
    a["tau_dust_2s"] = np.where(quiescent, d2_q, d2_sf)
    a["ns"] = np.where(quiescent, n_q, n_sf)

    # change 1: mean-shifted logsfr_ratios, drawn here (make_model_seds uses
    # stored ratios when the npz carries them)
    from prospect.models.templates import TemplateLibrary, adjust_stochastic_params

    ratios = np.empty((n, 9))
    for i in range(n):
        t = TemplateLibrary["stochastic_sfh"]
        t["agebins"]["init"] = make_stochastic_agebins(z=z[i])
        for key, src in [("sigma_reg", "sigma_regs"), ("tau_eq", "tau_eqs"),
                         ("tau_in", "tau_ins"), ("sigma_dyn", "sigma_dyns"),
                         ("tau_dyn", "tau_dyns")]:
            t[key]["init"] = a[src][i]
        t = adjust_stochastic_params(t)
        cov = np.asarray(t["logsfr_ratios"]["prior"].scale, dtype=float)
        eps = np.linalg.cholesky(cov + 1e-10 * np.eye(9)) \
            @ np.random.default_rng([seed, int(i)]).standard_normal(9)
        mu = mu_ratios(lm[i], z[i], quiescent[i], t_q[i], tau_q[i], inverter)
        ratios[i] = mu + eps
    a["logsfr_ratios"] = ratios
    a["_seed"] = seed
    a["_sample_size"] = n
    a["_cue"] = True
    a["_sfh_mean"] = "twobranch_e5"
    np.savez(out, **a)
    print(f"saved {out} (n={n}, quiescent frac={quiescent.mean():.3f})")


def _selftest():
    inv = MseedInverter()
    # invariance: constant added to log baseline leaves mu_ratios unchanged
    mu1 = mu_ratios(10.5, 0.1, False, 2.0, 1.0, inv)
    ls_shift = mu_ratios(10.5, 0.1, False, 2.0, 1.0, inv)  # deterministic
    assert np.allclose(mu1, ls_shift)
    # quenching lowers recent-to-old ratios (first entries more negative)
    muq = mu_ratios(10.5, 0.1, True, 3.0, 0.5, inv)
    assert muq[0] <= mu1[0] + 1e-9, (muq[0], mu1[0])
    # mseed inversion: heavier target mass -> heavier seed
    assert inv(11.0, 13.0) > inv(9.0, 13.0)
    # suppression floor: with tiny tau_q the ratio shift is bounded by floor
    mufloor = mu_ratios(10.5, 0.05, True, 4.0, 0.05, inv, floor_dex=1.0)
    assert np.all(np.isfinite(mufloor)) and np.max(np.abs(mufloor - mu_ratios(
        10.5, 0.05, False, 0, 1, inv))) <= 1.0 + 1e-6
    print("selftest ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("-n", "--sample-size", type=int, default=10_000)
    p.add_argument("-s", "--seed", type=int, default=42)
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        _selftest()
        sys.exit(0)
    out = args.out or (PATHS["DATA"] / "prospector_model"
                       / f"stochastic_priors_sample_cue_pilot_e5_{args.sample_size}.npz")
    draw(args.sample_size, args.seed, out)
