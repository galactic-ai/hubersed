"""
Plot best-fit SFHs from the SFH-resolution run in SFR/M_tot vs COSMIC TIME style
(+ cumulative mass), matching the engineered-break figure.

Rebuilds each binning's model (no fitting/FSPS) to recover theta indexing if the
pkl lacks it; if the pkl already has logsfr_ratios/agebins (new runs), uses those.

Run from hubersed root:
    python tmp/plot_sfh_resolution_42580.py
"""

import sys, pickle
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

sys.path.insert(0, "bin/prospector")
from hubersed.fitting import config as FC
from prospect.models.sedmodel import HyperSpecModel
from prospect.models.templates import adjust_stochastic_params
from prospect.models.transforms import logsfr_ratios_to_masses

d = pickle.load(open("results/sfh_resolution_parallel_42580.pkl", "rb"))
z = d["z"]
best = d["best"]
t_obs = cosmo.age(z).to_value(u.Gyr)
split = t_obs - 11.0
FREE = [
    "logsfr_ratios",
    "logzsol",
    "dust2",
    "dust_ratio",
    "dust_index",
    "logmass",
    "sigma_smooth",
    "gas_logz",
    "gas_logu",
    "gas_lognH",
    "gas_logno",
    "gas_logco",
    "eline_sigma",
]
HYP = {
    "sigma_reg": 1.5,
    "sigma_dyn": 0.1,
    "tau_eq": 1.0,
    "tau_in": 1.0,
    "tau_dyn": 0.025,
}


def agebins(n_sub):
    from hubersed.prospector.utils import make_stochastic_agebins

    std = make_stochastic_agebins(z)
    if n_sub <= 1:
        return std
    edges = np.unique(np.round(10 ** std.ravel() / 1e9, 6))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > split:
            out.extend(np.linspace(lo, hi, n_sub + 1)[:-1])
        else:
            out.append(lo)
    out.append(edges[-1])
    return np.log10(np.array([[out[i], out[i + 1]] for i in range(len(out) - 1)]) * 1e9)


def build(ab):
    FC.make_stochastic_agebins = lambda zz, ab=ab: ab
    cmodel, ct = FC.build_continuum_model(z)
    _, ft = FC.build_full_cue_model(ct, cmodel.theta.copy(), cmodel, z)
    for k in list(ft.keys()):
        if isinstance(ft[k], dict) and "isfree" in ft[k]:
            ft[k]["isfree"] = k in FREE
    for k, vv in HYP.items():
        ft[k]["isfree"] = False
        ft[k]["init"] = vv
    ft = adjust_stochastic_params(ft)
    return HyperSpecModel(ft)


def get_sfh(ns):
    r = best[ns]
    if "logsfr_ratios" in r and "agebins" in r:  # new-format pkl
        lr = np.asarray(r["logsfr_ratios"], float)
        lm = float(r["logmass"])
        ab = np.asarray(r["agebins"], float)
    else:  # old pkl: rebuild to recover indexing
        ab = agebins(ns)
        fm = build(ab)
        th = np.asarray(r["theta"], float)
        lr = th[fm.theta_index["logsfr_ratios"]]
        lm = float(np.atleast_1d(th[fm.theta_index["logmass"]])[0])
    mass = logsfr_ratios_to_masses(logmass=lm, logsfr_ratios=lr, agebins=ab)
    dt = 10 ** ab[:, 1] - 10 ** ab[:, 0]  # yr
    ssfr_gyr = (mass / dt) / 10**lm * 1e9  # SFR/M_tot [Gyr^-1]
    lb = 10**ab / 1e9  # lookback Gyr
    return ab, mass, ssfr_gyr, lb, lm


cols = {1: "C0", 3: "C2", 5: "C1", 8: "C3"}
fig, (a1, a2) = plt.subplots(
    2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)
for ns in sorted(best):
    ab, mass, ssfr, lb, lm = get_sfh(ns)
    c = cols[ns]
    nb = best[ns]["nb"]
    for i in range(ab.shape[0]):  # SFR/Mtot as step segments vs cosmic time
        tlo, thi = t_obs - lb[i, 1], t_obs - lb[i, 0]
        a1.plot(
            [tlo, thi],
            [ssfr[i], ssfr[i]],
            color=c,
            lw=1.8,
            label=f"{nb} bins (χ²r={best[ns]['chi2_red']:.2f})" if i == 0 else None,
        )
    o = np.argsort(-lb[:, 0])  # oldest->youngest (increasing cosmic time)
    ct = t_obs - lb[o, 0]
    cum = 100 * np.cumsum(mass[o]) / mass.sum()
    ct = np.concatenate(
        [[t_obs - lb[o[0], 1]], ct]
    )  # anchor at old edge of oldest bin, 0%
    cum = np.concatenate([[0.0], cum])
    a2.plot(ct, cum, color=c, lw=1.5, marker="o", ms=3)
a1.set_ylabel("SFR / M$_{tot}$  [Gyr$^{-1}$]")
a1.legend(fontsize=8)
a1.set_ylim(bottom=0)
a2.set_ylabel("Cumulative mass\nformed [%]")
a2.set_xlabel("Cosmic time [Gyr]")
a2.set_ylim(0, 105)
a2.set_xlim(0, t_obs)
a2.grid(alpha=0.3)
# fig.suptitle(f"42580 (z={z:.3f}) — best-fit SFHs vs old-bin resolution")
fig.tight_layout()
fig.savefig("tmp/sfh_resolution_bestfit_sfhs_42580.png", dpi=140, bbox_inches="tight")
print("saved tmp/sfh_resolution_bestfit_sfhs_42580.png")
