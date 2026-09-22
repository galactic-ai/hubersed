"""
Bursty-SFH MAP fit of the emission-line outlier 94183 (z=0.0395), TIED ionization.

Goal: can the model reach the (2-3x under-predicted) emission lines by letting the
SFH burst, WITHOUT freeing gas_logqion (ionization stays tied to the stars)?

Design:
  - logsfr_ratios FREE (can burst), with the stochastic MVN prior built from
    MODERATE, FIXED hyperparameters (sigma_reg=1.5, sigma_dyn=0.1, tau_eq=tau_in=1,
    tau_dyn=0.025) -> wide enough for a recent burst, and can't variance-collapse
    (the failure mode of the original fit, where sigma_reg/sigma_dyn railed to floor).
  - eline_sigma FREE, seeded low (was stuck at 100), multi-start.
  - cheap params free: logzsol, dust2/ratio/index, logmass, sigma_smooth, gas_*.
  - Cue tied: cue_stellar_nebular, use_stellar_ionizing=True, NO gas_logqion.
  - tpagb_norm_type=2, IMF/AGB default. (all csp moves -> no SSP recompute -> fast)

Run from hubersed root:
    python tmp/burstysfh_fit_94183.py
"""

import sys, pickle, warnings
import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

sys.path.insert(0, "bin/prospector")
from hubersed.prospector import parameter_file as P
from hubersed.fitting import config as FC
from hubersed.fitting import chi2 as MC
from prospect.models.sedmodel import HyperSpecModel
from prospect.models.templates import adjust_stochastic_params
from prospect.fitting import lnprobfn

PKL, KEY = "results/mapfit_cont_line_examples.pkl", "emission-line-only"
OUT = "results/burstysfh_fit_94183.pkl"
NSEEDS, MAXFEV = 5, 15000
HYP = {
    "sigma_reg": 1.5,
    "sigma_dyn": 0.1,
    "tau_eq": 1.0,
    "tau_in": 1.0,
    "tau_dyn": 0.025,
}
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

d = pickle.load(open(PKL, "rb"))
wave = np.asarray(d["wave"], float)
r = d["results"][KEY]
z = r["z"]
flux = np.asarray(r["flux"], float)
unc = np.asarray(r["unc"], float)
mask = np.asarray(r["mask"], bool)
ref_cr = r["chi2_red"]
ref_model = np.asarray(r["model"], float)
print(
    f"TID {r['id']}  z={z:.4f}  ORIGINAL fit chi2_red={ref_cr:.2f} (lines 2-3x under)"
)

MC._fsps()
obs = P.build_obs(spec=flux, unc=unc, mask=mask, resolution=MC._lsf_sigma_kms())
cmodel, ctemplate = FC.build_continuum_model(z)
_, ft = FC.build_full_cue_model(ctemplate, cmodel.theta.copy(), cmodel, z)
for k in list(ft.keys()):
    if isinstance(ft[k], dict) and "isfree" in ft[k]:
        ft[k]["isfree"] = k in FREE
for k, vv in HYP.items():
    ft[k]["isfree"] = False
    ft[k]["init"] = vv
ft["eline_sigma"]["isfree"] = True
ft["eline_sigma"]["init"] = 50.0
ft = adjust_stochastic_params(
    ft
)  # rebuild MVN logsfr_ratios prior from MODERATE hypers
fmodel = HyperSpecModel(ft)
print("free params:", list(fmodel.free_params))
sps = MC._cue()
sps.ssp.params["tpagb_norm_type"] = 2
sps.ssp.params["add_agb_dust_model"] = True

# seed a recent burst + low eline_sigma
th0 = fmodel.theta.copy()
li = fmodel.theta_index["logsfr_ratios"]
th0[li] = np.array([0.7, 0.6, 0.4, 0.2, 0, 0, 0, 0, 0])
th0[fmodel.theta_index["eline_sigma"]] = 50.0


def neg(th):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lp = lnprobfn(th, model=fmodel, observations=obs, sps=sps, nested=False)
            return -lp if np.isfinite(lp) else 1e18
        except Exception:
            return 1e18


print(f"\noptimizing (NSEEDS={NSEEDS}, MAXFEV={MAXFEV}) ...")
bf = MC._map_optimize(neg, th0, n_seeds=NSEEDS, maxfev=MAXFEV)
th = bf.x
preds, _ = fmodel.predict(th, observations=obs, sps=sps)
sp = np.asarray(preds[0], float)
m = obs[0].mask
chi2 = float(np.nansum(((obs[0].flux[m] - sp[m]) / obs[0].uncertainty[m]) ** 2))
ndof = int(m.sum()) - len(th)
cr = chi2 / ndof
thd = {k: np.asarray(th[i], float) for k, i in fmodel.theta_index.items()}

# line-window chi2 fraction + data/model line flux
wr = wave / (1 + z)


def linefrac():
    neb = [3727, 4861, 4959, 5007, 6300, 6548, 6563, 6584, 6716, 6731]
    lp = np.zeros_like(mask)
    for L in neb:
        lp |= np.abs(wr - L) < L * 1500 / 3e5
    g = m & (unc > 0)
    chi = ((flux - sp) / unc) ** 2
    return 100 * np.nansum(chi[g & lp]) / np.nansum(chi[g])


c_AA = 2.998e18


def ratio(L0):
    core = (wr >= L0 - 9) & (wr <= L0 + 9) & mask
    lf = (wr >= L0 - 30) & (wr <= L0 - 15) & mask
    rt = (wr >= L0 + 15) & (wr <= L0 + 30) & mask
    cl = np.median(flux[lf | rt]) if (lf | rt).sum() else 0
    fd = np.nansum((flux - cl)[core])
    fm = np.nansum((sp - cl)[core])
    return fd / fm if fm > 0 else np.nan


# recent vs old SFR
def sfr_split():
    t = cosmo.age(z).to_value(u.Gyr)
    ab = np.zeros((10, 2))
    ab[0] = [0.001, 0.005]
    ab[1] = [0.005, 0.01]
    e = np.geomspace(0.01, 0.95 * t, 9)
    for i in range(2, 10):
        ab[i] = [e[i - 2], e[i - 1]]
    ab = np.log10(ab * 1e9)
    lr = np.atleast_1d(thd["logsfr_ratios"])
    lm = float(np.atleast_1d(thd["logmass"])[0])
    sr = 10 ** np.clip(lr, -10, 10)
    dt = 10 ** ab[:, 1] - 10 ** ab[:, 0]
    co = np.array(
        [
            (1 / np.prod(sr[:i])) * (np.prod(dt[1 : i + 1]) / np.prod(dt[:i]))
            for i in range(10)
        ]
    )
    mm = 10**lm / co.sum() * co
    sfr = mm / dt
    rec = mm[(10 ** ab[:, 1] <= 1.1e7)].sum() / 1e7
    avg = mm.sum() / dt.sum()
    return rec, avg, rec / avg


print(f"\n=== RESULT ===")
print(f"chi2_red  {cr:.2f}   (was {ref_cr:.2f})   line-window χ²: {linefrac():.0f}%")
print(
    f"data/model line flux:  Hβ {ratio(4861):.2f}   Hα {ratio(6563):.2f}   [OIII]5007 {ratio(5007):.2f}   (was ~2.3/2.6/3.2)"
)
rec, avg, b = sfr_split()
print(
    f"recent(0-10Myr) SFR {rec:.3f}  /  avg {avg:.3f}  =  {b:.1f}x  (lines wanted ~2.6x)"
)
print(
    f"eline_sigma {float(thd['eline_sigma']):.1f} km/s (was stuck 100)   logzsol {float(thd['logzsol']):.2f}   dust2 {float(thd['dust2']):.3f}"
)
pickle.dump(
    dict(
        wave=wave,
        z=z,
        targetid=r["id"],
        flux=flux,
        unc=unc,
        mask=mask,
        model=sp,
        theta=th,
        free_params=list(fmodel.free_params),
        theta_dict=thd,
        chi2=chi2,
        ndof=ndof,
        chi2_red=cr,
        ref_chi2_red=ref_cr,
        ref_model=ref_model,
        hypers_fixed=HYP,
    ),
    open(OUT, "wb"),
)
print(f"saved {OUT}")
