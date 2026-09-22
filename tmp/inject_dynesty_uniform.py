"""
SELF-CONSISTENT injection-recovery of the UNIFORM-in-sSFR prior.

Matched priors (the fix): draw 5 theta from the SAME bounded uniform prior the fit uses
(flat +/-5 logsfr_ratios + sSFR bound via rejection), generate the SED with FSPS+Cue,
paste YOUR DESI noise with the trained decoder (verbatim from make_prospector_noisy_sed),
then fit each noisy mock with the uniform prior + dynesty and check recovery of the KNOWN truth.

theta ~ uniform, data ~ p(.|theta), infer with uniform  ==> coverage is VALID (SBC condition).
Caveats: 5 = bias/coverage spot-check (not a statistical coverage statement -> that needs the
flow); self-consistent -> validates inference/sampler, NOT model realism.

Run from hubersed root (venv):  python tmp/inject_dynesty_uniform.py [smoke]
"""

import sys, os, pickle, warnings, copy, time
import numpy as np, torch, dynesty
from dynesty.utils import resample_equal

sys.path.insert(0, "bin/prospector")
from hubersed.prospector import parameter_file as P
from hubersed.fitting import config as FC
from hubersed.fitting import chi2 as MC
from prospect.models.sedmodel import SpecModel
from prospect.models.templates import TemplateLibrary
from prospect.models.priors import TopHat
from prospect.fitting import lnprobfn
from hubersed.conversion import (
    flambda_to_maggies,
    ivar_flambda_to_ivar_maggies,
    maggies_to_flambda,
)
from hubersed.prospector.derived_quantities import compute_logssfr

# --- your noise pipeline (same imports as make_prospector_noisy_sed) ---
from hubersed.spender.utils import load_models
from hubersed.spender.quantities import normalize_spectra
from hubersed.paths import PATHS
from spender.data import desi
from spender.instrument import get_skyline_mask

TO = 1e8
R = 0.4
SFH_RANGE = 5.0
SSFR_LO, SSFR_HI = -13.0, float(np.log10(1.0 / (TO * (1 - R))))
NLIVE, DLOGZ, MAXCALL = 250, 1.0, None
SMOKE = "smoke" in sys.argv
if SMOKE:
    NLIVE, MAXCALL = 100, 20000
    print("*** SMOKE: capped, 1 mock, plumbing only ***")
OUT = "results/inject_uniform_smoke.pkl" if SMOKE else "results/inject_uniform.pkl"
Z_LIST = [0.06, 0.13, 0.20, 0.29, 0.40][: 1 if SMOKE else 5]
DATA = PATHS["DATA"]
NPIX = len(P.WAVE_OBS)
rng = np.random.default_rng(20)
torch.manual_seed(42)
gen = torch.Generator()
gen.manual_seed(42)

MC._fsps()
res_lsf = MC._lsf_sigma_kms()
sps = MC._cue()
sps.ssp.params["tpagb_norm_type"] = 2
sps.ssp.params["add_agb_dust_model"] = True

# ---------- noise decoder (load once) ----------
instrument = desi.DESI()
wave_obs = instrument._wave_obs
sky_mask = get_skyline_mask(wave_obs)
NDE_theta, model_spender = load_models(
    flow_file=str(DATA / "desi_noise_spender_10latent_flow.pt"),
    spender_file=str(DATA / "desi_noise_spender_10latent.pt"),
    flow_latent=10,
    instrument=instrument,
    map_location=torch.device("cpu"),
    weights_only=False,
)
model_spender.eval()
instrument.eval()
WAVE_T = torch.tensor(np.asarray(P.WAVE_OBS, float))


def paste_desi_noise(clean_maggies, zval):
    """clean model SED (maggies, 7781) -> noised DESI-unit flambda (7780), ivar (7780), norm. Verbatim from make_prospector_noisy_sed."""
    flambda = maggies_to_flambda(WAVE_T, torch.tensor(clean_maggies[None, :])) / 1e-17
    fn, norms, good = normalize_spectra(
        flambda, torch.tensor([zval]), WAVE_T, inplace=False
    )
    fn = fn[:, :-1]  # drop last pixel -> 7780 instrument grid
    with torch.no_grad():
        samp = (
            NDE_theta.sample(1, context=norms.unsqueeze(1).float())
            .permute(1, 0, 2)
            .float()
        )
        snr = model_spender.decode(samp).squeeze(0)  # (1, 7780)
    snr[:, sky_mask[:-1]] = float("nan")
    snr[snr <= 0] = float("nan")
    sigma = (fn / snr).abs()
    sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = torch.minimum(sigma, torch.nanquantile(sigma, 0.995, dim=1, keepdim=True))
    ivar = torch.nan_to_num(1.0 / (sigma**2), nan=0.0, posinf=0.0, neginf=0.0)
    ivar[sigma == 0] = 0.0
    f_noisy = fn + torch.normal(mean=0.0, std=sigma, generator=gen)
    return f_noisy.numpy()[0], ivar.numpy()[0], float(norms[0])


# ---------- uniform model + truth helpers ----------
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


def true_props(fmodel, th):
    lm = float(np.atleast_1d(th[fmodel.theta_index["logmass"]])[0])
    return dict(
        logmass=lm,
        logssfr=float(compute_logssfr(fmodel, th)),
        logzsol=float(np.atleast_1d(th[fmodel.theta_index["logzsol"]])[0]),
        dust2=float(np.atleast_1d(th[fmodel.theta_index["dust2"]])[0]),
    )


META = dict(settings=dict(nlive=NLIVE, dlogz=DLOGZ, prior="uniform_pm5+ssfrbound"))
results = {}
if os.path.exists(OUT) and not SMOKE:
    try:
        results = pickle.load(open(OUT, "rb")).get("results", {})
    except Exception:
        results = {}

obs0 = P.build_obs(
    spec=np.ones(NPIX), unc=np.ones(NPIX), mask=np.ones(NPIX, bool), resolution=res_lsf
)  # for predict
for mi, z in enumerate(Z_LIST):
    key = f"unif{mi}_z{z:.2f}"
    if key in results:
        print(f"[skip {key}]")
        continue
    t0 = time.time()
    fmodel = build_uniform_model(z)
    ndim = len(fmodel.theta)
    # draw theta from the SAME bounded uniform prior (flat +/-5 ratios + sSFR bound)
    for _ in range(3000):
        th_true = fmodel.prior_transform(rng.uniform(size=ndim))
        try:
            ss = float(compute_logssfr(fmodel, th_true))
        except Exception:
            continue
        if SSFR_LO <= ss <= SSFR_HI:
            break
    tp = true_props(fmodel, th_true)
    _pr, _ = fmodel.predict(
        th_true, observations=obs0, sps=sps
    )  # returns (list, mfrac); unpack!
    clean = np.asarray(_pr[0], float).reshape(-1)  # clean maggies (7781,)
    f_noisy, ivar_f, norm = paste_desi_noise(clean, z)
    spec = f_noisy * norm  # un-normalize -> DESI flambda (7780)
    ivar = ivar_f / norm**2
    spec = np.append(spec, 0.0)
    ivar = np.append(ivar, 0.0)  # pad -> 7781 (last pixel masked)
    sm = flambda_to_maggies(P.WAVE_OBS, spec)
    iv = ivar_flambda_to_ivar_maggies(P.WAVE_OBS, ivar)
    sig = 1.0 / np.sqrt(np.where(iv > 0, iv, np.inf))
    mask = (sig > 0) & np.isfinite(sig) & np.isfinite(sm)
    g = mask & np.isfinite(clean) & (clean > 0)
    ratio = float(np.nanmedian(sm[g]) / np.nanmedian(clean[g]))
    print(
        f"\n=== {key}  npix={int(mask.sum())}  noised/clean median={ratio:.2f} (want ~1) ==="
    )
    print(
        f"  truth: logM {tp['logmass']:.2f}  sSFR {tp['logssfr']:+.2f}  logzsol {tp['logzsol']:+.2f}  dust2 {tp['dust2']:.2f}"
    )
    obs = P.build_obs(spec=sm, unc=sig, mask=mask, resolution=res_lsf)

    def loglike(x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ss = float(compute_logssfr(fmodel, x))
                if not (SSFR_LO <= ss <= SSFR_HI):
                    return -np.inf
                ll = lnprobfn(x, model=fmodel, observations=obs, sps=sps, nested=True)
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
    post = dict(
        logmass=eq[:, ti["logmass"][0]],
        logzsol=eq[:, ti["logzsol"][0]],
        dust2=eq[:, ti["dust2"][0]],
        logssfr=np.array([float(compute_logssfr(fmodel, x)) for x in eq[::5]]),
    )

    def cover(true, p):
        p = np.asarray(p, float)
        p = p[np.isfinite(p)]
        if p.size < 10:
            return dict(q50=np.nan, q16=np.nan, q84=np.nan, in68=False, in95=False)
        lo, hi = np.percentile(p, [16, 84])
        lo2, hi2 = np.percentile(p, [2.5, 97.5])
        return dict(
            q50=float(np.median(p)),
            q16=float(lo),
            q84=float(hi),
            in68=bool(lo <= true <= hi),
            in95=bool(lo2 <= true <= hi2),
        )

    cov = {k: cover(tp[k], post[k]) for k in ["logmass", "logssfr", "logzsol", "dust2"]}
    results[key] = dict(
        z=z,
        true_theta=np.asarray(th_true, float),
        truth=tp,
        cover=cov,
        theta_index=ti,
        free_params=list(fmodel.free_params),
        eq_samples=np.asarray(eq, np.float32),
        logz=float(res.logz[-1]),
        ncall=int(np.sum(res.ncall)),
        noised_clean_ratio=ratio,
        minutes=(time.time() - t0) / 60,
    )
    for k in ["logmass", "logssfr", "logzsol", "dust2"]:
        c = cov[k]
        print(
            f"  {k:9s} true {tp[k]:+.2f}  post {c['q50']:+.2f} [{c['q16']:+.2f},{c['q84']:+.2f}]  in68={c['in68']} in95={c['in95']}"
        )
    print(f"  ncall {results[key]['ncall']:,}  ({results[key]['minutes']:.0f} min)")
    pickle.dump(dict(results=results, **META), open(OUT, "wb"))
    print(f"  [saved {key} -> {OUT}]")

print(f"\nsaved {OUT}")
