"""
Uniform-in-sSFR (sSFR-support-bounded) + fixed-qion fit of 42580 / 94183 / 702814.

Same as freesfh_fixedqion_fit.py (free TopHat SFH, cue_stellar_nebular tied ionization,
KC13 dust + order-4 calibration polynomial, DESI LSF) BUT the log sSFR_100 is bounded to
[SSFR_LO, CEIL] = [-13, -7.78] in the objective. For a MAP this is the operative difference
vs the free-SFH run: it stops the quiescent SFH from railing to log sSFR ~ -25 (the flat-ratio
prior edge) while leaving the EELGs (sSFR ~ -10, well inside the bound) essentially unchanged.
NB this bounds the sSFR *support* (approx uniform-in-sSFR); a strict uniform-in-sSFR prior
would also flatten the within-bound density (Jacobian reweight) -- irrelevant for the MAP.

Question: does bounding sSFR give a more sensible mass (esp. the quiescent) while still
closing the EELG lines?  Run from hubersed root (venv):  python tmp/uniform_ssfr_fixedqion_fit.py
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
from prospect.observation import Spectrum
from prospect.observation.observation import PolyOptCal
from hubersed.conversion import flambda_to_maggies, ivar_flambda_to_ivar_maggies
from hubersed.prospector.derived_quantities import compute_logssfr


class PolyCalSpectrum(PolyOptCal, Spectrum):
    pass


POLY_ORDER = 4
NSEEDS, MAXFEV = 8, 30000
SFH_RANGE = 5.0
TO = 1e8
R = 0.4
SSFR_LO, SSFR_HI = -13.0, float(np.log10(1.0 / (TO * (1 - R))))  # ~[-13, -7.78]
EXTREME_EELG_TID = 39633149675702814
OUT = "results/uniform_ssfr_fixedqion_fit.pkl"

MC._fsps()
d = pickle.load(open("results/mapfit_cont_line_examples.pkl", "rb"))
gals = []
for key, label in [
    ("continuum-only", "42580 quiescent (control)"),
    ("emission-line-only", "94183 EELG (moderate, EW~342)"),
]:
    r = d["results"][key]
    gals.append(
        dict(
            label=label,
            tid=int(r["id"]),
            z=float(r["z"]),
            flux=np.asarray(r["flux"], float),
            unc=np.asarray(r["unc"], float),
            mask=np.asarray(r["mask"], bool),
            ref=float(r["chi2_red"]),
        )
    )
gidx = int(MC.tids_to_indices(np.array([EXTREME_EELG_TID], dtype=np.int64))[0])
spec, ivar, zt, tid2 = MC.load_by_index(gidx)
sm = flambda_to_maggies(P.WAVE_OBS, spec)
iv = ivar_flambda_to_ivar_maggies(P.WAVE_OBS, ivar)
sig = 1.0 / np.sqrt(np.where(iv > 0, iv, np.inf))
mk = (sig > 0) & np.isfinite(sig) & np.isfinite(sm)
gals.append(
    dict(
        label="702814 EELG (extreme, EW~640)",
        tid=int(tid2),
        z=float(zt),
        flux=sm,
        unc=sig,
        mask=mk,
        ref=None,
    )
)

res_lsf = MC._lsf_sigma_kms()
sps = MC._cue()
sps.ssp.params["tpagb_norm_type"] = 2
sps.ssp.params["add_agb_dust_model"] = True


def build_model(z):
    cmodel, ctemplate = FC.build_continuum_model(z)
    ft = copy.deepcopy(ctemplate)
    ft.update(
        copy.deepcopy(TemplateLibrary["cue_stellar_nebular"])
    )  # tied qion (FIXED)
    ft["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}
    ft["use_stellar_ionizing"]["init"] = True
    nrat = len(ft["agebins"]["init"]) - 1
    ft["logsfr_ratios"]["isfree"] = True
    ft["logsfr_ratios"]["init"] = np.zeros(nrat)
    ft["logsfr_ratios"]["prior"] = TopHat(
        mini=np.full(nrat, -SFH_RANGE), maxi=np.full(nrat, SFH_RANGE)
    )
    FREE = [
        "logsfr_ratios",
        "logmass",
        "logzsol",
        "dust2",
        "dust_ratio",
        "dust_index",
        "sigma_smooth",
        "gas_logz",
        "gas_logu",
        "gas_lognH",
        "gas_logno",
        "gas_logco",
        "eline_sigma",
    ]
    for k in list(ft.keys()):
        if isinstance(ft[k], dict) and "isfree" in ft[k]:
            ft[k]["isfree"] = k in FREE
    ft["eline_sigma"] = {
        "N": 1,
        "isfree": True,
        "init": 80.0,
        "units": "km/s",
        "prior": TopHat(mini=20.0, maxi=250.0),
    }
    return SpecModel(ft)


def make_obs(g):
    s = PolyCalSpectrum(
        wavelength=P.WAVE_OBS,
        flux=g["flux"],
        uncertainty=g["unc"],
        mask=g["mask"],
        resolution=res_lsf,
        polynomial_order=POLY_ORDER,
    )
    s.rectify()
    return [s]


def line_ratio(flux, sp, wr, mask, L0):
    core = (wr >= L0 - 9) & (wr <= L0 + 9) & mask
    lf = (wr >= L0 - 30) & (wr <= L0 - 15) & mask
    rt = (wr >= L0 + 15) & (wr <= L0 + 30) & mask
    cl = np.median(flux[lf | rt]) if (lf | rt).sum() else 0.0
    fd = np.nansum((flux - cl)[core])
    fm = np.nansum((sp - cl)[core])
    return fd / fm if fm > 0 else np.nan


def line_window_frac(flux, sp, unc, wr, mask):
    neb = [3727, 4861, 4959, 5007, 6300, 6548, 6563, 6584, 6716, 6731]
    lp = np.zeros_like(mask)
    for L in neb:
        lp |= np.abs(wr - L) < L * 1500 / 3e5
    g = mask & (unc > 0)
    chi = ((flux - sp) / unc) ** 2
    return 100 * np.nansum(chi[g & lp]) / np.nansum(chi[g])


results = {}
for g in gals:
    print(
        f"\n{'=' * 70}\n{g['label']}   TID {g['tid']}  z={g['z']:.4f}  (tied-ref {g['ref']})"
    )
    fmodel = build_model(g["z"])
    obs = make_obs(g)
    nfree = len(fmodel.free_params)
    assert (
        "logsfr_ratios" in fmodel.free_params
        and "gas_logqion" not in fmodel.free_params
    )

    def neg(th):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ss = float(compute_logssfr(fmodel, th))  # sSFR-support bound
                if not (SSFR_LO <= ss <= SSFR_HI):
                    return 1e18
                lp = lnprobfn(th, model=fmodel, observations=obs, sps=sps, nested=False)
                return -lp if np.isfinite(lp) else 1e18
            except Exception:
                return 1e18

    print(
        f"optimizing (NSEEDS={NSEEDS}, MAXFEV={MAXFEV}; sSFR bound [{SSFR_LO},{SSFR_HI:.2f}]) -- slow ..."
    )
    bf = MC._map_optimize(neg, fmodel.theta.copy(), n_seeds=NSEEDS, maxfev=MAXFEV)
    th = bf.x
    preds, _ = fmodel.predict(th, observations=obs, sps=sps)
    sp = np.asarray(preds[0], float)
    m = obs[0].mask
    chi2 = float(np.nansum(((obs[0].flux[m] - sp[m]) / obs[0].uncertainty[m]) ** 2))
    cr = chi2 / (int(m.sum()) - nfree - (POLY_ORDER + 1))
    thd = {k: np.asarray(th[i], float) for k, i in fmodel.theta_index.items()}
    wr = P.WAVE_OBS / (1 + g["z"])
    rHb, rHa, rO3 = (
        line_ratio(g["flux"], sp, wr, g["mask"], L) for L in (4861, 6563, 5007)
    )
    lwf = line_window_frac(g["flux"], sp, g["unc"], wr, g["mask"])
    try:
        logssfr = float(compute_logssfr(fmodel, th))
    except Exception:
        logssfr = np.nan
    di = float(np.atleast_1d(thd["dust_index"])[0])
    lz = float(np.atleast_1d(thd["logzsol"])[0])
    lmv = float(np.atleast_1d(thd["logmass"])[0])
    print(f"\n  chi2_red = {cr:.2f}   line-window chi2 = {lwf:.0f}%")
    print(
        f"  data/model:  Hb {rHb:.2f}  Ha {rHa:.2f}  [OIII] {rO3:.2f}   log sSFR100 = {logssfr:.2f}   logmass = {lmv:.2f}"
    )
    print(
        f"  logzsol {lz:+.2f}   dust_index {di:+.2f}   eline_sigma {float(np.atleast_1d(thd['eline_sigma'])[0]):.0f}"
    )
    results[g["label"]] = dict(
        tid=g["tid"],
        z=g["z"],
        chi2_red=cr,
        line_window_frac=lwf,
        dHb=rHb,
        dHa=rHa,
        dO3=rO3,
        logssfr=logssfr,
        theta_dict=thd,
        model=sp,
        flux=g["flux"],
        unc=g["unc"],
        mask=g["mask"],
        ref=g["ref"],
    )
pickle.dump(
    dict(
        results=results,
        poly_order=POLY_ORDER,
        sfh_range=SFH_RANGE,
        ssfr_bound=[SSFR_LO, SSFR_HI],
    ),
    open(OUT, "wb"),
)
print(f"\nsaved {OUT}")
