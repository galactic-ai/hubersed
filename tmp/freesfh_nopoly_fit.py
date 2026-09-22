"""
FREE SFH, NO PolyOptCal -- MAP re-fit of 42580 / 94183 / 702814.

Identical to freesfh_fixedqion_fit.py (truly free SFH: flat TopHat(+/-5) per logsfr_ratio,
ExReg ACF off, qion FIXED = cue_stellar_nebular, KC13 dust, DESI LSF) EXCEPT the order-4
PolyOptCal is REMOVED -- plain Spectrum via P.build_obs, so the absolute continuum
normalization is fit by the physical model (anchors mass) instead of being absorbed by a
multiplicative polynomial. No sSFR bound (truly free -> 42580 may rail old).

Purpose: does dropping the polynomial give sensible masses (as it did for the 5 non-outliers)
while still closing the EELG lines? MAP is a quick point-estimate on a degenerate ridge --
trust coarse shape + mass, not individual bins; dynesty is the trustworthy follow-up.

Run from hubersed root (venv):  python tmp/freesfh_nopoly_fit.py
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
from hubersed.conversion import flambda_to_maggies, ivar_flambda_to_ivar_maggies
from hubersed.prospector.derived_quantities import compute_logssfr

NSEEDS, MAXFEV = 8, 30000  # free SFH is degenerate -> many seeds
SFH_RANGE = 5.0
EXTREME_EELG_TID = 39633149675702814
OUT = "results/freesfh_nopoly_fit.pkl"

MC._fsps()

# ---- gather galaxies: two from the examples pkl + the extreme EELG by TARGETID ----
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
    """free-SFH (ACF off) + FIXED qion (cue_stellar_nebular) + KC13 dust."""
    cmodel, ctemplate = FC.build_continuum_model(z)
    ft = copy.deepcopy(ctemplate)
    ft.update(copy.deepcopy(TemplateLibrary["cue_stellar_nebular"]))
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
    return P.build_obs(
        spec=g["flux"], unc=g["unc"], mask=g["mask"], resolution=res_lsf
    )  # plain Spectrum, NO PolyOptCal


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
        f"\n{'=' * 70}\n{g['label']}   TID {g['tid']}  z={g['z']:.4f}  (tied-ref chi2_red={g['ref']})"
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
                lp = lnprobfn(th, model=fmodel, observations=obs, sps=sps, nested=False)
                return -lp if np.isfinite(lp) else 1e18
            except Exception:
                return 1e18

    print(f"optimizing (NSEEDS={NSEEDS}, MAXFEV={MAXFEV}; NO PolyOptCal) -- slow ...")
    bf = MC._map_optimize(neg, fmodel.theta.copy(), n_seeds=NSEEDS, maxfev=MAXFEV)
    th = bf.x
    preds, _ = fmodel.predict(th, observations=obs, sps=sps)
    sp = np.asarray(preds[0], float)
    m = obs[0].mask
    chi2 = float(np.nansum(((obs[0].flux[m] - sp[m]) / obs[0].uncertainty[m]) ** 2))
    ndof = int(m.sum()) - nfree  # no calibration params
    cr = chi2 / ndof
    thd = {k: np.asarray(th[i], float) for k, i in fmodel.theta_index.items()}
    wr = P.WAVE_OBS / (1 + g["z"])
    rHb, rHa, rO3 = (
        line_ratio(g["flux"], sp, wr, g["mask"], L) for L in (4861, 6563, 5007)
    )
    lwf = line_window_frac(g["flux"], sp, g["unc"], wr, g["mask"])
    try:
        logssfr = float(compute_logssfr(fmodel, th))
    except Exception as e:
        logssfr = np.nan
        print("  (sSFR calc failed:", e, ")")
    di = float(np.atleast_1d(thd["dust_index"])[0])
    lz = float(np.atleast_1d(thd["logzsol"])[0])
    lmv = float(np.atleast_1d(thd["logmass"])[0])
    print(f"\n  chi2_red = {cr:.2f}   line-window chi2 = {lwf:.0f}%")
    print(
        f"  data/model line flux:  Hb {rHb:.2f}  Ha {rHa:.2f}  [OIII]5007 {rO3:.2f}   (target ~1.0)"
    )
    print(
        f"  logmass = {lmv:.2f}   log sSFR(100Myr) = {logssfr:.2f}   logzsol {lz:+.2f}   dust_index {di:+.2f}   eline_sigma {float(np.atleast_1d(thd['eline_sigma'])[0]):.0f}"
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
    dict(results=results, poly_order=None, sfh_range=SFH_RANGE), open(OUT, "wb")
)
print(f"\nsaved {OUT}  (NO PolyOptCal)")
