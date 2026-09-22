"""
Profile-MAP over the TP-AGB ssp_params at the ENGINEERED OLD-SFH theta of 42580.

Why a profile and not a plain free-parameter MAP:
  agb/pagb/agb_dust are ssp_params -> setting them rebuilds the SSPs. Freeing them
  in a normal optimizer = SSP recompute every step (hours). Instead we GRID each
  one and, at each grid value, run a fast MAP over the CHEAP params (the saved
  free list: logzsol, dust2, logmass, dust_ratio, dust_index, sigma_smooth,
  gas_*, eline_sigma). SSP rebuilds only once per grid value.

logsfr_ratios stays FIXED at the engineered old SFH (old_lr). tpagb_norm_type=2.

Grids: agb 0-2/0.2, pagb 0-1/0.2, agb_dust 0-2/0.2 (one varied at a time, others=1).

Run from hubersed root in the venv:
    python tmp/agb_pagb_profilemap_engineered.py
"""

import sys, pickle, warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "bin/prospector")
from hubersed.prospector import parameter_file as P
from hubersed.fitting import config as FC
from hubersed.fitting import chi2 as MC
from prospect.models.sedmodel import SpecModel
from prospect.fitting import lnprobfn

PKL = "results/engineered_break_3sfh_39633140817331167.pkl"
VARIANT = "Sigma=-10.8 (fixed)"
NSEEDS, MAXFEV = 1, 2500  # inner-MAP budget (keep modest; 28 inner fits total)

d = pickle.load(open(PKL, "rb"))
wave = np.asarray(d["wave"], float)
z = d["z"]
flux = np.asarray(d["flux"], float)
unc = np.asarray(d["unc"], float)
mask = np.asarray(d["mask"], bool)
v = d["fits"][VARIANT]
labels = list(v["labels"])
old_lr = np.asarray(v["old_lr"], float)
print(
    f"TID {d['targetid']}  z={z:.4f}  variant '{VARIANT}'  saved chi2_red={v['chi2_red']:.3f}"
)

MC._fsps()
obs = P.build_obs(spec=flux, unc=unc, mask=mask, resolution=MC._lsf_sigma_kms())
cmodel, ctemplate = FC.build_continuum_model(z)
_, ft = FC.build_full_cue_model(ctemplate, cmodel.theta.copy(), cmodel, z)
ft["logsfr_ratios"]["isfree"] = False
ft["logsfr_ratios"]["init"] = old_lr
for k in list(ft.keys()):
    if isinstance(ft[k], dict) and "isfree" in ft[k]:
        ft[k]["isfree"] = k in labels
fmodel = SpecModel(ft)
assert list(fmodel.free_params) == labels, (list(fmodel.free_params), labels)
sps = MC._cue()


def ndof():
    m = obs[0].mask
    return int(m.sum()) - len(fmodel.free_params)


def map_at(agb=1.0, agb_dust=1.0, pagb=1.0):
    sps.ssp.params["tpagb_norm_type"] = 2
    sps.ssp.params["add_agb_dust_model"] = True
    sps.ssp.params["agb"] = float(agb)
    sps.ssp.params["agb_dust"] = float(agb_dust)
    sps.ssp.params["pagb"] = float(pagb)

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
    return chi2 / ndof(), th


base_chi2red, _ = map_at()
print(f"\nbaseline (all=1.0) re-fit chi2_red = {base_chi2red:.4f}")

grids = {
    "agb": np.round(np.arange(0, 2.0001, 0.2), 2),
    "pagb": np.round(np.arange(0, 1.0001, 0.2), 2),
    "agb_dust": np.round(np.arange(0, 2.0001, 0.2), 2),
}
res = {}
for name, grid in grids.items():
    print(f"\n--- profile {name} ---\n{name:>9s} {'chi2_red':>10s}")
    cr = []
    for val in grid:
        c, _ = map_at(**{name: val})
        cr.append(c)
        print(f"{val:9.1f} {c:10.4f}")
    res[name] = (grid, np.array(cr))

# save numerical results
save = {
    "TID": d["targetid"],
    "z": z,
    "variant": VARIANT,
    "baseline_refit_chi2_red": base_chi2red,
    "saved_engineered_chi2_red": v["chi2_red"],
}
for name, (g, cr) in res.items():
    i = int(np.argmin(cr))
    save[name] = {
        "grid": g,
        "chi2_red": cr,
        "best_val": float(g[i]),
        "best_chi2_red": float(cr[i]),
    }
pickle.dump(save, open("results/agb_pagb_profilemap_42580.pkl", "wb"))
print("saved results/agb_pagb_profilemap_42580.pkl")

fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for j, name in enumerate(grids):
    g, cr = res[name]
    ax[j].plot(g, cr, "o-", color="k")
    ax[j].axhline(base_chi2red, ls=":", color="0.6")
    ax[j].set_xlabel(name)
    ax[j].set_title(f"best chi2_red={cr.min():.3f} @ {name}={g[np.argmin(cr)]:.1f}")
ax[0].set_ylabel("chi2_red (cheap params re-fit)")
fig.suptitle(
    f"Profile-MAP over TP-AGB params @ engineered '{VARIANT}'  TID {d['targetid']}"
)
fig.tight_layout()
fig.savefig("tmp/agb_pagb_profilemap_engineered.png", dpi=140, bbox_inches="tight")
print(
    f"\nbaseline chi2_red={base_chi2red:.4f}; saved tmp/agb_pagb_profilemap_engineered.png"
)
