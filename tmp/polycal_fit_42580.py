"""
Spectro-photometric CALIBRATION-POLYNOMIAL fit at the engineered OLD SFH (42580).

Wraps the DESI Spectrum in prospect's PolyOptCal mixin: at each likelihood call a
Chebyshev polynomial (order N) is fit by least-squares to (data/model) and applied
as the response, absorbing the smooth continuum shape. Then the FEATURES drive the
fit instead of the continuum tilt.

Question: does this (a) drop chi2_red and (b) let dust_index come OFF the -2.5 rail
(into the physical range)?  We compare polynomial orders.

logsfr_ratios FIXED (old SFH), AGB default, tpagb_norm_type=2.
Run from hubersed root:
    python tmp/polycal_fit_42580.py
"""

import sys, pickle, warnings
import numpy as np

sys.path.insert(0, "bin/prospector")
from hubersed.prospector import parameter_file as P
from hubersed.fitting import config as FC
from hubersed.fitting import chi2 as MC
from prospect.models.sedmodel import SpecModel
from prospect.fitting import lnprobfn
from prospect.observation import Spectrum
from prospect.observation.observation import PolyOptCal


class PolyCalSpectrum(PolyOptCal, Spectrum):
    pass


PKL, VARIANT = (
    "results/engineered_break_3sfh_39633140817331167.pkl",
    "Sigma=-10.8 (fixed)",
)
OUT = "results/polycal_fit_42580.pkl"
ORDERS = [4, 8, 12]  # order 0 = "no polynomial" (constant) = the baseline; skip it
NSEEDS, MAXFEV = 3, 8000

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
print(
    f"TID {d['targetid']}  z={z:.4f}  engineered(no cal) chi2_red={v['chi2_red']:.3f}, dust_index={float(np.atleast_1d(td['dust_index'])[0]):.3f}"
)

MC._fsps()
res_lsf = MC._lsf_sigma_kms()
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


def make_obs(order):
    s = PolyCalSpectrum(
        wavelength=P.WAVE_OBS,
        flux=flux,
        uncertainty=unc,
        mask=mask,
        resolution=res_lsf,
        polynomial_order=order,
    )
    s.rectify()
    return [s]


def fit_order(order):
    obs = make_obs(order)

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
    sp = np.asarray(preds[0], float)  # already includes the response/calibration
    m = obs[0].mask
    chi2 = float(np.nansum(((obs[0].flux[m] - sp[m]) / obs[0].uncertainty[m]) ** 2))
    ndof = (
        int(m.sum()) - len(labels) - (order + 1)
    )  # account for the marginalized poly terms
    thd = {k: float(bf.x[i]) for k, i in fmodel.theta_index.items()}
    resp = np.asarray(getattr(obs[0], "response", np.ones_like(flux)), float)
    return dict(
        order=order,
        chi2=chi2,
        chi2_red=chi2 / ndof,
        model=sp,
        response=resp,
        dust_index=thd["dust_index"],
        dust2=thd["dust2"],
        logzsol=thd["logzsol"],
        theta_dict=thd,
    )


res = {}
print(
    f"\n{'order':>6s} {'chi2_red':>9s} {'dust_index':>11s} {'dust2':>7s} {'logzsol':>8s} {'resp range':>16s}"
)
for o in ORDERS:
    r = fit_order(o)
    res[o] = r
    rr = r["response"][mask]
    print(
        f"{o:6d} {r['chi2_red']:9.3f} {r['dust_index']:11.3f} {r['dust2']:7.3f} {r['logzsol']:8.3f}  [{rr.min():.3f},{rr.max():.3f}]"
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
        ref_dust_index=float(np.atleast_1d(td["dust_index"])[0]),
    ),
    open(OUT, "wb"),
)
print(f"\nsaved {OUT}")
print(
    "watch: does chi2_red drop AND does dust_index leave the -2.5 floor toward physical (>-1)?"
)
