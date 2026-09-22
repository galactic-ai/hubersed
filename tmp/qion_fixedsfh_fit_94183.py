"""
Isolation test: decoupled ionization with SFH FIXED (no burst) for 94183.

Same as qion_freesfh but logsfr_ratios is held at the original fit's values and the
SFH is NOT free -> isolates whether gas_logqion ALONE reaches the lines, or whether
the 20.9x burst in the free-SFH run was also contributing.

cue_nebular (use_stellar_ionizing=False); free: gas_logqion + ionspec shape +
gas abundances + dust + logzsol + logmass + sigma_smooth + eline_sigma.
SpecModel (fixed logsfr_ratios -> no MVN prior issue).

Run from hubersed root:
    python tmp/qion_fixedsfh_fit_94183.py
"""

import sys, pickle, warnings, copy
import numpy as np

sys.path.insert(0, "bin/prospector")
from hubersed.prospector import parameter_file as P
from hubersed.fitting import config as FC
from hubersed.fitting import chi2 as MC
from prospect.models.sedmodel import SpecModel
from prospect.models.templates import TemplateLibrary
from prospect.models.priors import TopHat
from prospect.fitting import lnprobfn

PKL, KEY = "results/mapfit_cont_line_examples.pkl", "emission-line-only"
OUT = "results/qion_fixedsfh_fit_94183.pkl"
NSEEDS, MAXFEV = 3, 25000

d = pickle.load(open(PKL, "rb"))
wave = np.asarray(d["wave"], float)
r = d["results"][KEY]
z = r["z"]
td0 = r["theta_dict"]
flux = np.asarray(r["flux"], float)
unc = np.asarray(r["unc"], float)
mask = np.asarray(r["mask"], bool)
ref_cr = r["chi2_red"]
ref_model = np.asarray(r["model"], float)
orig_lr = np.atleast_1d(np.asarray(td0["logsfr_ratios"], float))
print(f"TID {r['id']}  z={z:.4f}  tied chi2_red={ref_cr:.2f};  free-SFH+qion gave 2.91")

MC._fsps()
obs = P.build_obs(spec=flux, unc=unc, mask=mask, resolution=MC._lsf_sigma_kms())
cmodel, ctemplate = FC.build_continuum_model(z)
ft = copy.deepcopy(ctemplate)
ft.update(copy.deepcopy(TemplateLibrary["cue_nebular"]))  # use_stellar_ionizing=False
ft["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}
ft["logsfr_ratios"]["isfree"] = False
ft["logsfr_ratios"]["init"] = orig_lr  # SFH FIXED (no burst)

FREE = [
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
    "gas_logqion",
    "ionspec_index1",
    "ionspec_index2",
    "ionspec_index3",
    "ionspec_index4",
    "ionspec_logLratio1",
    "ionspec_logLratio2",
    "ionspec_logLratio3",
    "eline_sigma",
]
for k in list(ft.keys()):
    if isinstance(ft[k], dict) and "isfree" in ft[k]:
        ft[k]["isfree"] = k in FREE
for k in [
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
]:
    if k in td0:
        ft[k]["init"] = float(np.atleast_1d(td0[k])[0])
ft["eline_sigma"] = {
    "N": 1,
    "isfree": True,
    "init": 50.0,
    "units": "km/s",
    "prior": TopHat(mini=20.0, maxi=250.0),
}
ft["gas_logqion"]["isfree"] = True
ft["gas_logqion"]["init"] = 52.0
ft["use_stellar_ionizing"]["init"] = False

fmodel = SpecModel(ft)
print("free params:", list(fmodel.free_params))
assert "gas_logqion" in fmodel.free_params and "logsfr_ratios" not in fmodel.free_params
sps = MC._cue()
sps.ssp.params["tpagb_norm_type"] = 2
sps.ssp.params["add_agb_dust_model"] = True


def neg(th):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lp = lnprobfn(th, model=fmodel, observations=obs, sps=sps, nested=False)
            return -lp if np.isfinite(lp) else 1e18
        except Exception:
            return 1e18


print(f"\noptimizing (NSEEDS={NSEEDS}, MAXFEV={MAXFEV}) ...")
bf = MC._map_optimize(neg, fmodel.theta.copy(), n_seeds=NSEEDS, maxfev=MAXFEV)
th = bf.x
preds, _ = fmodel.predict(th, observations=obs, sps=sps)
sp = np.asarray(preds[0], float)
m = obs[0].mask
chi2 = float(np.nansum(((obs[0].flux[m] - sp[m]) / obs[0].uncertainty[m]) ** 2))
cr = chi2 / (int(m.sum()) - len(th))
thd = {k: np.asarray(th[i], float) for k, i in fmodel.theta_index.items()}

wr = wave / (1 + z)


def linefrac():
    neb = [3727, 4861, 4959, 5007, 6300, 6548, 6563, 6584, 6716, 6731]
    lp = np.zeros_like(mask)
    for L in neb:
        lp |= np.abs(wr - L) < L * 1500 / 3e5
    g = m & (unc > 0)
    chi = ((flux - sp) / unc) ** 2
    return 100 * np.nansum(chi[g & lp]) / np.nansum(chi[g])


def ratio(L0):
    core = (wr >= L0 - 9) & (wr <= L0 + 9) & mask
    lf = (wr >= L0 - 30) & (wr <= L0 - 15) & mask
    rt = (wr >= L0 + 15) & (wr <= L0 + 30) & mask
    cl = np.median(flux[lf | rt]) if (lf | rt).sum() else 0
    fd = np.nansum((flux - cl)[core])
    fm = np.nansum((sp - cl)[core])
    return fd / fm if fm > 0 else np.nan


print(f"\n=== RESULT (qion free, SFH FIXED) ===")
print(
    f"chi2_red {cr:.2f}  (tied {ref_cr:.2f}; free-SFH+qion 2.91)   line-window χ²: {linefrac():.0f}%"
)
print(
    f"data/model line flux:  Hβ {ratio(4861):.2f}  Hα {ratio(6563):.2f}  [OIII]5007 {ratio(5007):.2f}  (target ~1.0)"
)
print(
    f"gas_logqion {float(thd['gas_logqion']):.2f}   eline_sigma {float(thd['eline_sigma']):.1f}   dust2 {float(thd['dust2']):.3f}   logzsol {float(thd['logzsol']):.2f}"
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
        chi2_red=cr,
        ref_chi2_red=ref_cr,
        ref_model=ref_model,
    ),
    open(OUT, "wb"),
)
print(f"saved {OUT}")
