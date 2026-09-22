"""
MAP fit at the engineered OLD SFH (fixed logsfr_ratios), with the AGB params
FIXED at their profile-best values, and the cheap params (dust/Z/gas/eline) free.

Because agb/pagb/agb_dust are held fixed, the SSPs are built once -> the inner
MAP is fast (no per-step recompute). Saves the fit to a pkl.

Run from hubersed root:
    python tmp/mapfit_bestAGB_42580.py
"""

import sys, pickle, warnings
import numpy as np

sys.path.insert(0, "bin/prospector")
from hubersed.prospector import parameter_file as P
from hubersed.fitting import config as FC
from hubersed.fitting import chi2 as MC
from prospect.models.sedmodel import SpecModel
from prospect.fitting import lnprobfn

PKL, VARIANT = (
    "results/engineered_break_3sfh_39633140817331167.pkl",
    "Sigma=-10.8 (fixed)",
)
PROF = "results/agb_pagb_profilemap_42580.pkl"
OUT = "results/mapfit_bestAGB_42580.pkl"
NSEEDS, MAXFEV = 3, 10000

# best AGB values from the profile (fallback to these if the profile pkl is absent)
best = {"agb": 0.6, "pagb": 1.0, "agb_dust": 2.0}
try:
    pf = pickle.load(open(PROF, "rb"))
    best = {k: pf[k]["best_val"] for k in ("agb", "pagb", "agb_dust")}
except Exception as e:
    print("(profile pkl not found, using defaults)", e)
print(f"AGB params fixed at: {best}")

d = pickle.load(open(PKL, "rb"))
wave = np.asarray(d["wave"], float)
z = d["z"]
flux = np.asarray(d["flux"], float)
unc = np.asarray(d["unc"], float)
mask = np.asarray(d["mask"], bool)
v = d["fits"][VARIANT]
labels = list(v["labels"])
old_lr = np.asarray(v["old_lr"], float)
td = v["theta_dict"]
print(f"TID {d['targetid']}  z={z:.4f}  engineered(ref) chi2_red={v['chi2_red']:.3f}")

MC._fsps()
obs = P.build_obs(spec=flux, unc=unc, mask=mask, resolution=MC._lsf_sigma_kms())
cmodel, ctemplate = FC.build_continuum_model(z)
_, ft = FC.build_full_cue_model(ctemplate, cmodel.theta.copy(), cmodel, z)
ft["logsfr_ratios"]["isfree"] = False
ft["logsfr_ratios"]["init"] = old_lr
for k in list(ft.keys()):
    if isinstance(ft[k], dict) and "isfree" in ft[k]:
        ft[k]["isfree"] = k in labels
        if k in labels:
            ft[k]["init"] = float(np.atleast_1d(td[k])[0])
fmodel = SpecModel(ft)
assert list(fmodel.free_params) == labels, (list(fmodel.free_params), labels)

sps = MC._cue()
sps.ssp.params["tpagb_norm_type"] = 2
sps.ssp.params["add_agb_dust_model"] = True
sps.ssp.params["agb"] = float(best["agb"])
sps.ssp.params["pagb"] = float(best["pagb"])
sps.ssp.params["agb_dust"] = float(best["agb_dust"])


def neg(th):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lp = lnprobfn(th, model=fmodel, observations=obs, sps=sps, nested=False)
            return -lp if np.isfinite(lp) else 1e18
        except Exception:
            return 1e18


bf = MC._map_optimize(neg, fmodel.theta.copy(), n_seeds=NSEEDS, maxfev=MAXFEV)
th = bf.x
preds, _ = fmodel.predict(th, observations=obs, sps=sps)
sp = np.asarray(preds[0], float)
m = obs[0].mask
chi2 = float(np.nansum(((obs[0].flux[m] - sp[m]) / obs[0].uncertainty[m]) ** 2))
ndof = int(m.sum()) - len(th)
theta_dict = {k: np.asarray(th[i], float) for k, i in fmodel.theta_index.items()}
print(
    f"\nbest-AGB MAP chi2_red = {chi2 / ndof:.4f}   (engineered ref {v['chi2_red']:.3f})"
)
print(
    f"  dust_index={float(theta_dict['dust_index']):.3f}  dust2={float(theta_dict['dust2']):.3f}  logzsol={float(theta_dict['logzsol']):.3f}"
)

pickle.dump(
    dict(
        wave=wave,
        z=z,
        targetid=d["targetid"],
        flux=flux,
        unc=unc,
        mask=mask,
        model=sp,
        theta=th,
        free_params=labels,
        theta_dict=theta_dict,
        agb_params=best,
        old_lr=old_lr,
        chi2=chi2,
        ndof=ndof,
        chi2_red=chi2 / ndof,
        ref_chi2_red=v["chi2_red"],
    ),
    open(OUT, "wb"),
)
print(f"saved {OUT}")
