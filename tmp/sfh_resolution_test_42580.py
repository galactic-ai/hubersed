"""
Does finer OLD-age SFH resolution (more bins for cosmic time < 11 Gyr) help the
6500-7200 A continuum bump in the continuum outlier 42580?

Method: keep the YOUNG bins identical, SUBDIVIDE the old bins (lookback > t_obs-11
Gyr) by a factor, free SFH (moderate fixed stochastic hypers, no variance-collapse),
tied Cue, dust free. Compare standard vs finer binnings.

Honest read: more bins lowers chi2 trivially (overfitting). So compare the 6500-7200
region vs a CONTROL region (4500-5500), and the dof-adjusted chi2_red. Resolution
"helps the bump" only if 6500-7200 improves SELECTIVELY.

Run from hubersed root:
    python tmp/sfh_resolution_test_42580.py
"""

import sys, pickle, warnings
import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

sys.path.insert(0, "bin/prospector")
from hubersed.prospector import parameter_file as P
from hubersed.fitting import config as FC
from hubersed.fitting import chi2 as MC
from prospect.models.templates import adjust_stochastic_params
from prospect.fitting import lnprobfn

PKL, KEY = "results/mapfit_cont_line_examples.pkl", "continuum-only"
NSEEDS, MAXFEV = 5, 40000
HYP = {
    "sigma_reg": 1.5,
    "sigma_dyn": 0.1,
    "tau_eq": 1.0,
    "tau_in": 1.0,
    "tau_dyn": 0.025,
}

d = pickle.load(open(PKL, "rb"))
wave = np.asarray(d["wave"], float)
r = d["results"][KEY]
z = r["z"]
flux = np.asarray(r["flux"], float)
unc = np.asarray(r["unc"], float)
mask = np.asarray(r["mask"], bool)
print(f"TID {r['id']}  z={z:.4f}  standard-bin MAP chi2_red was {r['chi2_red']:.2f}")
t_obs = cosmo.age(z).to_value(u.Gyr)
split = max(t_obs - 11.0, 0.5)  # lookback beyond which = "cosmic time < 11 Gyr"
print(f"t_obs={t_obs:.2f} Gyr -> subdividing old bins at lookback > {split:.2f} Gyr")


def std_agebins():
    from hubersed.prospector.utils import make_stochastic_agebins

    return make_stochastic_agebins(z)


def finer_agebins(n_sub):
    edges = np.unique(np.round(10 ** std_agebins().ravel() / 1e9, 6))  # Gyr edges
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > split:  # subdivide any bin OVERLAPPING cosmic time < 11 (bins 8 and 9)
            out.extend(np.linspace(lo, hi, n_sub + 1)[:-1])
        else:
            out.append(lo)
    out.append(edges[-1])
    ab = np.array([[out[i], out[i + 1]] for i in range(len(out) - 1)])
    return np.log10(ab * 1e9)


MC._fsps()
obs = P.build_obs(spec=flux, unc=unc, mask=mask, resolution=MC._lsf_sigma_kms())
sps = MC._cue()
sps.ssp.params["tpagb_norm_type"] = 2
sps.ssp.params["add_agb_dust_model"] = True
wr = wave / (1 + z)
m = obs[0].mask


def chi2win(sp, lo, hi):
    s = m & (wr >= lo) & (wr < hi)
    return (
        float(np.nansum(((obs[0].flux[s] - sp[s]) / obs[0].uncertainty[s]) ** 2))
        / s.sum()
    )


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


def fit_with(agebins, label):
    nb = agebins.shape[0]
    FC.make_stochastic_agebins = lambda zz, ab=agebins: (
        ab
    )  # patch so build_* uses these bins
    cmodel, ctemplate = FC.build_continuum_model(z)
    fmodel, ft = FC.build_full_cue_model(ctemplate, cmodel.theta.copy(), cmodel, z)
    for k in list(ft.keys()):
        if isinstance(ft[k], dict) and "isfree" in ft[k]:
            ft[k]["isfree"] = k in FREE
    for k, vv in HYP.items():
        ft[k]["isfree"] = False
        ft[k]["init"] = vv
    ft = adjust_stochastic_params(ft)
    from prospect.models.sedmodel import HyperSpecModel

    fm = HyperSpecModel(ft)

    def neg(th):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lp = lnprobfn(th, model=fm, observations=obs, sps=sps, nested=False)
                return -lp if np.isfinite(lp) else 1e18
            except Exception:
                return 1e18

    # SEED an OLD declining SFH, equivalent old-bias across binnings:
    # total decline = Σ(logsfr_ratios) = -10.8 (the engineered Σ=-10.8), spread over nb-1 ratios
    th0 = fm.theta.copy()
    th0[fm.theta_index["logsfr_ratios"]] = -10.8 / (nb - 1)
    bf = MC._map_optimize(neg, th0, n_seeds=NSEEDS, maxfev=MAXFEV)
    preds, _ = fm.predict(bf.x, observations=obs, sps=sps)
    sp = np.asarray(preds[0], float)
    chi2 = float(np.nansum(((obs[0].flux[m] - sp[m]) / obs[0].uncertainty[m]) ** 2))
    ndof = int(m.sum()) - len(bf.x)
    print(
        f"{label:16s} Nbins={nb:2d}  chi2_red={chi2 / ndof:.3f}  "
        f"6500-7200/pix={chi2win(sp, 6500, 7200):.2f}  CONTROL 4500-5500/pix={chi2win(sp, 4500, 5500):.2f}"
    )
    return dict(
        label=label,
        nbins=nb,
        agebins=agebins,
        model=sp,
        theta=np.asarray(bf.x, float),
        theta_labels=list(fm.free_params),
        chi2=chi2,
        ndof=ndof,
        chi2_red=chi2 / ndof,
        bump_6500_7200=chi2win(sp, 6500, 7200),
        control_4500_5500=chi2win(sp, 4500, 5500),
    )


print(
    f"\n{'binning':16s} {'':6s} {'chi2_red':>9s}  {'bump 6500-7200':>14s}  {'control 4500-5500':>17s}"
)
runs = {}
runs["standard10"] = fit_with(std_agebins(), "standard(10)")
runs["old_x3"] = fit_with(finer_agebins(3), "old x3")
runs["old_x5"] = fit_with(finer_agebins(5), "old x5")
import pickle

pickle.dump(
    dict(wave=wave, z=z, flux=flux, unc=unc, mask=mask, runs=runs),
    open("results/sfh_resolution_test_42580.pkl", "wb"),
)
print(
    "\nsaved results/sfh_resolution_test_42580.pkl (models + agebins + theta for all 3)"
)
print(
    "\nRead: bump/pix should drop MORE than control/pix if old-SFH resolution targets the 6500-7200 bump."
)
print(
    "If both drop together (or barely move), it's overfitting / not an SFH-resolution effect."
)
