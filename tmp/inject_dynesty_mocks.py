"""
Injection-recovery on the REAL mock catalog (not hand-rolled).

Pulls (truth, DESI-noised spectrum) straight from your existing mocks:
  - truth       : data/prospector_model/prospector_stochastic_model_seds_cue_500000.h5  priors/*
  - noised spec : data/prospector_model/DESIcueprospector1024_{gidx//1024}.pkl  row gidx%1024
                  (target_id in the pkl == global catalog index -> maps back to priors/)
Un-normalizes exactly like map_chi2.load_by_index (s*norm, ivar/norm^2), pads the 7780
instrument grid to the 7781 WAVE_OBS grid (last pixel masked), converts to maggies, and fits
with the UNIFORM-in-sSFR prior + dynesty (rslice, nlive 250, dlogz 1.0).

So: stochastic-prior TRUTH (realistic SFHs), YOUR DESI noise, fit with the wide prior ->
does the wide-prior posterior recover the truth of realistic galaxies?  These theta are
genuine prior draws, so coverage statements are meaningful (unlike the hand-picked ladder).

Run from hubersed root (venv):  python tmp/inject_dynesty_mocks.py [smoke]
"""

import sys, os, pickle, warnings, copy, time
import numpy as np
import h5py, dynesty
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
from hubersed.prospector.utils import make_stochastic_agebins

TO = 1e8
R = 0.4
SFH_RANGE = 5.0
SSFR_LO, SSFR_HI = -13.0, float(np.log10(1.0 / (TO * (1 - R))))
NLIVE, DLOGZ, MAXCALL = 250, 1.0, None
SMOKE = "smoke" in sys.argv
if SMOKE:
    NLIVE, MAXCALL = 100, 20000
    print("*** SMOKE: capped, 1 mock, plumbing only ***")
OUT = "results/inject_mocks_smoke.pkl" if SMOKE else "results/inject_mocks.pkl"
DATA = MC.DATA_PATH
H5 = DATA / "prospector_model" / "prospector_stochastic_model_seds_cue_500000.h5"
CHUNK = 1024
NPIX = len(P.WAVE_OBS)  # 7781
N_MOCKS = 5  # selected across the sSFR range
POOL = 20000  # search this many for the sSFR spread

MC._fsps()
res_lsf = MC._lsf_sigma_kms()
sps = MC._cue()
sps.ssp.params["tpagb_norm_type"] = 2
sps.ssp.params["add_agb_dust_model"] = True


# ---------- SFH truth helpers (bin0 youngest; matches compute_logssfr) ----------
def masses_from_lsr(z, logmass, lsr):
    ab = 10 ** make_stochastic_agebins(z)
    dt = ab[:, 1] - ab[:, 0]
    sr = 10 ** np.clip(lsr, -100, 100)
    c = np.ones(10)
    for i in range(10):
        num = np.prod(dt[1 : i + 1]) if i >= 1 else 1.0
        den = np.prod(dt[:i]) if i >= 1 else 1.0
        sd = np.prod(sr[:i]) if i >= 1 else 1.0
        c[i] = (1 / sd) * (num / den)
    return 10**logmass / c.sum() * c, ab


def true_ssfr(z, logmass, lsr):
    m, ab = masses_from_lsr(z, logmass, lsr)
    mid = ab.mean(1)
    M = m.sum()
    return float(np.log10((m[mid <= TO].sum() / TO) / (M * (1 - R))))


# ---------- model (uniform-in-sSFR), identical to inject_dynesty ----------
def build_uniform_model(z):
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


# ---------- load a noised mock by GLOBAL catalog index (mirror load_by_index) ----------
def load_mock(gidx):
    chunk, row = gidx // CHUNK, gidx % CHUNK
    with open(
        DATA / "prospector_model" / f"DESIcueprospector1024_{chunk}.pkl", "rb"
    ) as f:
        f_noisy, ivar, z, tid, norm, *_ = pickle.load(f)
    to = lambda x: x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    s = to(f_noisy)[row].astype(float) * float(
        to(norm)[row]
    )  # un-normalize -> DESI-unit flambda (7780)
    w = (
        to(ivar)[row].astype(float) / float(to(norm)[row]) ** 2
    )  # -> flambda ivar (7780)
    if s.shape[0] == NPIX - 1:  # pad the dropped last pixel (masked)
        s = np.append(s, 0.0)
        w = np.append(w, 0.0)
    return s, w, float(to(z)[row]), int(to(tid)[row])


# ---------- pick mocks spanning the true sSFR range ----------
h = h5py.File(H5, "r")
lm = h["priors/stellar_masses"][:POOL].astype(float)
zz = h["priors/redshifts"][:POOL].astype(float)
lsr_pool = h["priors/logsfr_ratios"][:POOL].astype(float)
ss_pool = np.array([true_ssfr(zz[i], lm[i], lsr_pool[i]) for i in range(POOL)])
pcts = [3, 27, 50, 73, 97][:N_MOCKS]
SEL = [int(np.argmin(np.abs(ss_pool - np.percentile(ss_pool, p)))) for p in pcts]
if SMOKE:
    SEL = SEL[:1]
print(
    f"pool sSFR range [{ss_pool.min():.2f}, {ss_pool.max():.2f}]  selected gidx {SEL} "
    f"sSFR {[round(float(ss_pool[i]), 2) for i in SEL]}"
)


def truth_of(gidx):
    p = h["priors"]
    return dict(
        logmass=float(p["stellar_masses"][gidx]),
        logzsol=float(p["stellar_metallicities"][gidx]),
        dust2=float(p["tau_dust_2s"][gidx]),
        z=float(p["redshifts"][gidx]),
        logssfr=true_ssfr(
            float(p["redshifts"][gidx]),
            float(p["stellar_masses"][gidx]),
            p["logsfr_ratios"][gidx].astype(float),
        ),
        logsfr_ratios=p["logsfr_ratios"][gidx].astype(float),
    )


META = dict(H5=str(H5), settings=dict(nlive=NLIVE, dlogz=DLOGZ, prior="uniform_pm5"))
results = {}
if os.path.exists(OUT) and not SMOKE:
    try:
        results = pickle.load(open(OUT, "rb")).get("results", {})
    except Exception:
        results = {}

for gidx in SEL:
    key = f"mock{gidx}"
    if key in results:
        print(f"[skip {key}]")
        continue
    t0 = time.time()
    spec, ivar, z, tid = load_mock(gidx)
    assert int(tid) == int(gidx), f"tid {tid} != gidx {gidx}"
    sm = flambda_to_maggies(P.WAVE_OBS, spec)
    iv = ivar_flambda_to_ivar_maggies(P.WAVE_OBS, ivar)
    sig = 1.0 / np.sqrt(np.where(iv > 0, iv, np.inf))
    mask = (sig > 0) & np.isfinite(sig) & np.isfinite(sm)
    tp = truth_of(gidx)
    # self-check: noised maggies should track the CLEAN catalog fluxes (same object)
    clean = h["fluxes"][gidx].astype(float)
    g = mask & np.isfinite(clean) & (clean > 0)
    ratio = float(np.nanmedian(sm[g]) / np.nanmedian(clean[g]))
    print(
        f"\n=== {key}  z={z:.4f}  npix={int(mask.sum())}  noised/clean median={ratio:.2f} (want ~1) ==="
    )
    print(
        f"  truth: logM {tp['logmass']:.2f}  sSFR {tp['logssfr']:+.2f}  logzsol {tp['logzsol']:+.2f}  dust2 {tp['dust2']:.2f}"
    )
    fmodel = build_uniform_model(z)
    ndim = len(fmodel.theta)
    obs = P.build_obs(spec=sm, unc=sig, mask=mask, resolution=res_lsf)

    def loglike(th):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ss = float(compute_logssfr(fmodel, th))
                if not (SSFR_LO <= ss <= SSFR_HI):
                    return -np.inf
                ll = lnprobfn(th, model=fmodel, observations=obs, sps=sps, nested=True)
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
    wts = np.exp(res.logwt - res.logz[-1])
    eq = resample_equal(res.samples, wts / wts.sum())
    ti = {
        k: (
            int(np.atleast_1d(np.arange(ndim)[v])[0]),
            int(np.atleast_1d(np.arange(ndim)[v])[-1]) + 1,
        )
        for k, v in fmodel.theta_index.items()
    }
    lm_post = eq[:, ti["logmass"][0]]
    ss_post = np.array([float(compute_logssfr(fmodel, th)) for th in eq[::5]])
    lz_post = eq[:, ti["logzsol"][0]]
    d2_post = eq[:, ti["dust2"][0]]

    def cover(true, post):
        post = np.asarray(post, float)
        post = post[np.isfinite(post)]
        if post.size < 10:
            return dict(q50=np.nan, q16=np.nan, q84=np.nan, in68=False, in95=False)
        lo, hi = np.percentile(post, [16, 84])
        lo2, hi2 = np.percentile(post, [2.5, 97.5])
        return dict(
            q50=float(np.median(post)),
            q16=float(lo),
            q84=float(hi),
            in68=bool(lo <= true <= hi),
            in95=bool(lo2 <= true <= hi2),
        )

    cov = {
        k: cover(tp[k], p)
        for k, p in [
            ("logmass", lm_post),
            ("logssfr", ss_post),
            ("logzsol", lz_post),
            ("dust2", d2_post),
        ]
    }
    results[key] = dict(
        gidx=gidx,
        z=z,
        truth=tp,
        cover=cov,
        theta_index=ti,
        free_params=list(fmodel.free_params),
        eq_samples=np.asarray(eq, np.float32),
        logz=float(res.logz[-1]),
        logzerr=float(res.logzerr[-1]),
        ncall=int(np.sum(res.ncall)),
        noised_clean_ratio=ratio,
        minutes=(time.time() - t0) / 60,
    )
    for k in ["logmass", "logssfr", "logzsol", "dust2"]:
        c = cov[k]
        print(
            f"  {k:9s} true {tp[k]:+.2f}  post {c['q50']:+.2f} [{c['q16']:+.2f},{c['q84']:+.2f}]  in68={c['in68']} in95={c['in95']}"
        )
    print(
        f"  logZ {results[key]['logz']:.0f}  ncall {results[key]['ncall']:,}  ({results[key]['minutes']:.0f} min)"
    )
    pickle.dump(dict(results=results, **META), open(OUT, "wb"))
    print(f"  [saved {key} -> {OUT}]")

print(f"\nsaved {OUT}")
