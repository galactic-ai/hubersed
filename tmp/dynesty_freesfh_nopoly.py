"""
dynesty posteriors: FREE SFH, NO PolyOptCal, for the three outliers 42580 / 94183 / 702814.

Same model as freesfh_nopoly_fit.py (flat TopHat(+/-5) logsfr_ratios, ACF off, qion FIXED =
cue_stellar_nebular, KC13 dust, DESI LSF, plain Spectrum = NO PolyOptCal, NO sSFR bound = truly
free). dynesty (rslice, nlive 250, dlogz 1.0) gives the full posterior so we can see how WIDE
the mass/sSFR really are given the +/-1.5 dex calibration sensitivity the MAP exposed.

Saves per galaxy: eq_samples (full posterior), the max-loglike model spectrum + data for plots,
derived logmass/sSFR posteriors, theta_index. Incremental save + resume.  Slow (~1-3 h each).

Run from hubersed root (venv):  python tmp/dynesty_freesfh_nopoly.py [smoke]
"""

import sys, os, pickle, warnings, copy, time
import numpy as np, dynesty
from dynesty.utils import resample_equal

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

SFH_RANGE = 5.0
NLIVE, DLOGZ, MAXCALL = 250, 1.0, None
SMOKE = "smoke" in sys.argv
if SMOKE:
    NLIVE, MAXCALL = 100, 20000
    print("*** SMOKE: capped, 1 galaxy, plumbing only ***")
OUT = (
    "results/dynesty_freesfh_nopoly_smoke.pkl"
    if SMOKE
    else "results/dynesty_freesfh_nopoly.pkl"
)
EXTREME_EELG_TID = 39633149675702814

MC._fsps()
res_lsf = MC._lsf_sigma_kms()
sps = MC._cue()
sps.ssp.params["tpagb_norm_type"] = 2
sps.ssp.params["add_agb_dust_model"] = True

# ---- gather the three galaxies (identical to freesfh_nopoly_fit.py) ----
d = pickle.load(open("results/mapfit_cont_line_examples.pkl", "rb"))
gals = []
for k, label in [
    ("continuum-only", "42580 quiescent (control)"),
    ("emission-line-only", "94183 EELG (moderate, EW~342)"),
]:
    r = d["results"][k]
    gals.append(
        dict(
            label=label,
            tid=int(r["id"]),
            z=float(r["z"]),
            flux=np.asarray(r["flux"], float),
            unc=np.asarray(r["unc"], float),
            mask=np.asarray(r["mask"], bool),
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
    )
)
if SMOKE:
    gals = gals[1:2]  # smoke = 94183 (has lines, moderate)


def build_model(z):
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


results = {}
if os.path.exists(OUT) and not SMOKE:
    try:
        results = pickle.load(open(OUT, "rb")).get("results", {})
    except Exception:
        results = {}

for g in gals:
    tag = str(g["tid"])
    if tag in results:
        print(f"[skip {tag}]")
        continue
    t0 = time.time()
    z = g["z"]
    fmodel = build_model(z)
    ndim = len(fmodel.theta)
    obs = P.build_obs(spec=g["flux"], unc=g["unc"], mask=g["mask"], resolution=res_lsf)
    print(f"\n=== {g['label']}  TID {g['tid']}  z={z:.4f}  ndim={ndim} ===")

    def loglike(x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ll = lnprobfn(
                    x, model=fmodel, observations=obs, sps=sps, nested=True
                )  # NO sSFR bound (truly free)
                return float(ll) if np.isfinite(ll) else -np.inf
            except Exception:
                return -np.inf

    ds = dynesty.NestedSampler(
        loglike,
        lambda u: fmodel.prior_transform(u),
        ndim,
        nlive=NLIVE,
        bound="multi",
        sample="rslice",
    )
    ds.run_nested(dlogz=DLOGZ, maxcall=MAXCALL, print_progress=True)
    res = ds.results
    wv = np.exp(res.logwt - res.logz[-1])
    eq = resample_equal(res.samples, wv / wv.sum())
    ti = {
        k: (
            int(np.atleast_1d(np.arange(ndim)[v])[0]),
            int(np.atleast_1d(np.arange(ndim)[v])[-1]) + 1,
        )
        for k, v in fmodel.theta_index.items()
    }
    lm_post = eq[:, ti["logmass"][0]]
    ss_post = np.array([float(compute_logssfr(fmodel, x)) for x in eq[::5]])
    best = res.samples[np.argmax(res.logl)]  # max-loglike sample for the model plot
    mpred, _ = fmodel.predict(best, observations=obs, sps=sps)
    model = np.asarray(mpred[0], float)
    m = obs[0].mask
    chi2 = float(
        np.nansum(
            (
                (np.asarray(obs[0].flux)[m] - model[m])
                / np.asarray(obs[0].uncertainty)[m]
            )
            ** 2
        )
    )

    def q(p):
        p = np.asarray(p, float)
        p = p[np.isfinite(p)]
        return (
            float(np.percentile(p, 16)),
            float(np.median(p)),
            float(np.percentile(p, 84)),
        )

    results[tag] = dict(
        label=g["label"],
        tid=g["tid"],
        z=z,
        eq_samples=np.asarray(eq, np.float32),
        theta_index=ti,
        free_params=list(fmodel.free_params),
        best_theta=np.asarray(best, float),
        model=np.asarray(model, np.float32),
        flux=np.asarray(g["flux"], np.float32),
        unc=np.asarray(g["unc"], np.float32),
        mask=np.asarray(g["mask"], bool),
        chi2_red=chi2 / (int(m.sum()) - ndim),
        logmass_q=q(lm_post),
        logssfr_q=q(ss_post),
        flat_logmass=lm_post,
        flat_logssfr=ss_post,
        logz=float(res.logz[-1]),
        ncall=int(np.sum(res.ncall)),
        minutes=(time.time() - t0) / 60,
    )
    lm = results[tag]["logmass_q"]
    ss = results[tag]["logssfr_q"]
    print(
        f"  logmass  {lm[1]:.2f} [{lm[0]:.2f},{lm[2]:.2f}]   logsSFR {ss[1]:+.2f} [{ss[0]:+.2f},{ss[2]:+.2f}]   chi2_red {results[tag]['chi2_red']:.2f}   ({results[tag]['minutes']:.0f} min)"
    )
    pickle.dump(dict(results=results), open(OUT, "wb"))
    print(f"  [saved {tag} -> {OUT}]")

print(f"\nsaved {OUT}")
