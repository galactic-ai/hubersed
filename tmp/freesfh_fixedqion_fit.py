"""
REACH TEST: does a TRULY FREE SFH (not the stochastic ExReg prior) reach the emission
lines with qion FIXED (tied to the stars)?

Motivation: the earlier `burstysfh_fit_94183` kept the stochastic prior's mean-zero ACF
(moderate fixed hyperparameters), which capped the burst at ~5.2x and kept a prior-imposed
OLD stellar mass -> dust re-attenuation trap -> lines stayed 2-3x under. That was NOT a free
SFH. Here we remove the ACF entirely (independent, wide Student-t on logsfr_ratios) so the
SFH can go young-DOMINATED, and we DO NOT free gas_logqion (cue_stellar_nebular = ionization
tied to the stars). Plus KC13 dust + an order-4 calibration polynomial (so the quiescent
continuum tilt doesn't rail the dust).

Question per galaxy:
  - continuum control 42580: should reach chi2~few with an OLD free SFH (SFH flexibility enough).
  - EELG 94183 (EW~342) / 702814 (EW~640, extreme): does a young-dominated free SFH reach the
    line fluxes (data/model -> 1) with FIXED qion? If yes -> EELG OOD is prior-coverage (the ACF
    was the blocker). If lines stay <1 while the SFH rails young -> genuine ionization
    incompleteness (needs free qion / BPASS), independent of the SFH prior.

Run from hubersed root (venv with FSPS + Cue + the prospect fork):
    python tmp/freesfh_fixedqion_fit.py
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
NSEEDS, MAXFEV = (
    8,
    30000,
)  # free SFH is degenerate -> many seeds; slow (Cue + poly refit)
SFH_RANGE = (
    5.0  # FLAT TopHat half-range on each logsfr_ratio: |log10(SFR_j/SFR_j+1)| < this.
)
# No mean-zero pull => truly free (continuity's Student-t centers on 0 = flat-baseline bias).
EXTREME_EELG_TID = 39633149675702814
OUT = "results/freesfh_fixedqion_fit.pkl"

MC._fsps()  # patch SSPBasis resolution before building anything

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
    cmodel, ctemplate = FC.build_continuum_model(
        z
    )  # KC13 dust (dust_type=4), stochastic agebins
    ft = copy.deepcopy(ctemplate)
    ft.update(
        copy.deepcopy(TemplateLibrary["cue_stellar_nebular"])
    )  # ionization TIED to stars (qion fixed)
    ft["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}
    ft["use_stellar_ionizing"]["init"] = True

    # TRULY FREE SFH: FLAT (TopHat) independent prior on each log-SFR ratio -> NO mean-zero /
    # flat-baseline pull (a Student-t centered at 0 would still bias toward flat, the very thing
    # we're removing) and NO ExReg ACF. Every rising/falling/bursty shape is a priori equal.
    nrat = len(ft["agebins"]["init"]) - 1
    ft["logsfr_ratios"]["isfree"] = True
    ft["logsfr_ratios"]["init"] = np.zeros(nrat)
    ft["logsfr_ratios"]["prior"] = TopHat(
        mini=np.full(nrat, -SFH_RANGE), maxi=np.full(nrat, SFH_RANGE)
    )  # length-9 flat, independent per bin-ratio

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
            ft[k]["isfree"] = (
                k in FREE
            )  # freezes ExReg hyperparameters (sigma_reg, tau_*) -> ACF unused
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
        f"\n{'=' * 70}\n{g['label']}   TID {g['tid']}  z={g['z']:.4f}  (tied-ref chi2_red={g['ref']})"
    )
    fmodel = build_model(g["z"])
    obs = make_obs(g)
    nfree = len(fmodel.free_params)
    print("free params:", list(fmodel.free_params))
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

    print(f"optimizing (NSEEDS={NSEEDS}, MAXFEV={MAXFEV}) -- slow ...")
    bf = MC._map_optimize(neg, fmodel.theta.copy(), n_seeds=NSEEDS, maxfev=MAXFEV)
    th = bf.x
    preds, _ = fmodel.predict(th, observations=obs, sps=sps)
    sp = np.asarray(preds[0], float)
    m = obs[0].mask
    chi2 = float(np.nansum(((obs[0].flux[m] - sp[m]) / obs[0].uncertainty[m]) ** 2))
    ndof = int(m.sum()) - nfree - (POLY_ORDER + 1)
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
    railed_di = "RAIL" if di <= -2.45 or di >= 0.35 else "ok"
    railed_lz = "RAIL" if lz <= -2.45 or lz >= 0.45 else "ok"
    print(f"\n  chi2_red = {cr:.2f}   line-window chi2 = {lwf:.0f}%")
    print(
        f"  data/model line flux:  Hb {rHb:.2f}  Ha {rHa:.2f}  [OIII]5007 {rO3:.2f}   (target ~1.0; <1 = model under-predicts)"
    )
    print(
        f"  recovered log sSFR(100Myr) = {logssfr:.2f}   (young-dominated burst if >> -9)"
    )
    print(
        f"  logzsol {lz:+.2f} [{railed_lz}]   dust_index {di:+.2f} [{railed_di}]   eline_sigma {float(np.atleast_1d(thd['eline_sigma'])[0]):.0f}"
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
    dict(results=results, poly_order=POLY_ORDER, sfh_range=SFH_RANGE), open(OUT, "wb")
)
print(f"\nsaved {OUT}")
print(
    "\nREAD: EELG lines -> data/model ~1.0 with a young-dominated SFH and FIXED qion => prior-coverage"
)
print(
    "      (the ACF was the blocker). Lines stay <1 while SFH rails young => ionization incompleteness."
)
