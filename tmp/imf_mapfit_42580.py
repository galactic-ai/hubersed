"""
IMF sensitivity via MAP fits at the engineered OLD SFH (fixed logsfr_ratios).
For each imf_type, re-fit the cheap params (dust/Z/gas/eline) and record chi2_red
plus the Na D / Mg b absorption residuals (the IMF/gravity-sensitive lines).

imf_type is discrete -> set on the SSP (one recompute each), inner MAP is fast.
tpagb_norm_type=2.  AGB params left at default 1.0.

Run from hubersed root:
    python tmp/imf_mapfit_42580.py
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
OUT = "results/imf_mapfit_42580.pkl"
IMF_TYPES = {0: "Salpeter", 1: "Chabrier", 2: "Kroupa", 3: "vanDokkum"}
NSEEDS, MAXFEV = 3, 8000

d = pickle.load(open(PKL, "rb"))
wave = np.asarray(d["wave"], float)
z = d["z"]
wr = wave / (1 + z)
flux = np.asarray(d["flux"], float)
unc = np.asarray(d["unc"], float)
mask = np.asarray(d["mask"], bool)
v = d["fits"][VARIANT]
labels = list(v["labels"])
old_lr = np.asarray(v["old_lr"], float)
td = v["theta_dict"]
print(
    f"TID {d['targetid']}  z={z:.4f}  engineered(Kroupa) chi2_red={v['chi2_red']:.3f}"
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
        if k in labels:
            ft[k]["init"] = float(np.atleast_1d(td[k])[0])
fmodel = SpecModel(ft)
assert list(fmodel.free_params) == labels
sps = MC._cue()
sps.ssp.params["tpagb_norm_type"] = 2
sps.ssp.params["add_agb_dust_model"] = True

m = obs[0].mask


def feat_resid(sp, L):
    s = mask & (unc > 0) & (np.abs(wr - L) < 8)
    return float(np.nanmean(((flux - sp) / unc)[s]))


def redchi(sp, lo, hi):
    s = m & (wr >= lo) & (wr < hi)
    return (
        float(np.nansum(((obs[0].flux[s] - sp[s]) / obs[0].uncertainty[s]) ** 2))
        / s.sum()
    )


def map_at_imf(it):
    sps.ssp.params["imf_type"] = it

    def neg(th):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lp = lnprobfn(th, model=fmodel, observations=obs, sps=sps, nested=False)
                return -lp if np.isfinite(lp) else 1e18
            except Exception:
                return 1e18

    bf = MC._map_optimize(neg, fmodel.theta.copy(), n_seeds=NSEEDS, maxfev=MAXFEV)
    preds, _ = fmodel.predict(bf.x, observations=obs, sps=sps)
    sp = np.asarray(preds[0], float)
    chi2 = float(np.nansum(((obs[0].flux[m] - sp[m]) / obs[0].uncertainty[m]) ** 2))
    cr = chi2 / (int(m.sum()) - len(labels))
    return cr, sp, {k: float(bf.x[i]) for k, i in fmodel.theta_index.items()}


res = {}
print(
    f"\n{'imf':10s} {'chi2_red':>9s} {'NaD σ':>7s} {'Mgb σ':>7s} {'redχ²/pix':>9s} {'dust_index':>10s}"
)
for it, nm in IMF_TYPES.items():
    cr, sp, thd = map_at_imf(it)
    nad, mgb, red = feat_resid(sp, 5890), feat_resid(sp, 5175), redchi(sp, 6500, 7200)
    res[it] = dict(
        name=nm,
        chi2_red=cr,
        NaD_sig=nad,
        Mgb_sig=mgb,
        red_chi2pix=red,
        model=sp,
        theta_dict=thd,
    )
    print(
        f"{nm:10s} {cr:9.3f} {nad:7.1f} {mgb:7.1f} {red:9.2f} {thd['dust_index']:10.3f}"
    )
pickle.dump(
    dict(
        wave=wave,
        z=z,
        flux=flux,
        unc=unc,
        mask=mask,
        results=res,
        ref_chi2_red=v["chi2_red"],
    ),
    open(OUT, "wb"),
)
print(f"\nsaved {OUT}")
print(
    "(Kroupa is the baseline; Na D residual ~ -5σ at Kroupa. Watch if any imf_type pulls it toward 0.)"
)
