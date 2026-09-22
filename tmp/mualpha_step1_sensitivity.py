"""
STEP 1 (sensitivity / identifiability): are mu_alpha (mean recent SFH slope) and sigma_reg
(PSD / burstiness amplitude) SEPARABLE in the DESI spectra-only features (Halpha EW, Dn4000)?

Burnham+2026 (arXiv:2601.20930) warns slope and burstiness "produce similar signatures" -- if they
are degenerate given our observables, the whole mu_alpha program is moot. This is that test.

Extends the existing mock machinery minimally (cf. bin/model_seds/make_cue_model_sed.py):
  logsfr_ratios = t['logsfr_ratios']['prior'].sample()   [mean-zero MVN(0, Sigma_ACF)]  +  mu_vec(alpha)
  mu_vec(alpha)_i = alpha * log10(mid_i / mid_{i+1})     [power-law slope in the ratio basis]
  alpha ~ N(mu_alpha, sigma_alpha);  sigma_reg set directly (free PSD amplitude)
Sign: alpha > 0 -> SFR rises with lookback -> DECLINING to present;  alpha < 0 -> rising (Burnham's sign).

NOISELESS, no selection -- this is the intrinsic-separability test only. Noise/selection/feature
matching come in Step 2 if this passes.

Run from hubersed root (venv):  python tmp/mualpha_step1_sensitivity.py
"""

import sys, copy, pickle, warnings, time
import numpy as np

sys.path.insert(0, "bin/prospector")
from hubersed.prospector import parameter_file as P
from hubersed.fitting import config as FC
from hubersed.fitting import chi2 as MC
from prospect.models.sedmodel import SpecModel
from prospect.models.templates import TemplateLibrary, adjust_stochastic_params
from prospect.models.priors import TopHat
from hubersed.prospector.utils import make_stochastic_agebins

# MODE: "sigreg" -> grid mu_alpha x sigma_reg (burstiness degeneracy; logzsol from prior)
#       "zmet"   -> grid mu_alpha x logzsol  (age-metallicity degeneracy; sigma_reg fixed)
MODE = (
    sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("sigreg", "zmet") else "sigreg"
)
Z_POP = 0.15  # population redshift (one bin for this test)
N_GAL = 300  # galaxies per population
SIGMA_A = 0.3  # per-galaxy scatter in alpha about mu_alpha
MU_GRID = [
    -0.5,
    0.0,
    +0.5,
    +1.0,
    +2.0,
]  # mean recent SFH slope (rising -> falling); +2 tests Dn4000 saturation
SIGREG_GRID = [0.3, 1.0, 3.0] if MODE == "sigreg" else [1.0]  # PSD amplitude
ZMET_GRID = (
    [None] if MODE == "sigreg" else [-1.0, -0.3, 0.0, +0.3]
)  # None => draw logzsol from prior
LOGMASS = 10.0
OUT = f"results/mualpha_step1_{MODE}.pkl"
print(f"MODE={MODE}  mu={MU_GRID}  sigreg={SIGREG_GRID}  logzsol={ZMET_GRID}")
rng = np.random.default_rng(3)

MC._fsps()
res_lsf = MC._lsf_sigma_kms()
sps = MC._cue()
sps.ssp.params["tpagb_norm_type"] = 2
sps.ssp.params["add_agb_dust_model"] = True

AB = 10 ** make_stochastic_agebins(Z_POP)
MID = AB.mean(1)  # yr
# mean-slope vector in the ratio basis: ratio_i = log10(SFR_i/SFR_{i+1}) = alpha*log10(mid_i/mid_{i+1})
DLOG = np.log10(MID[:-1] / MID[1:])  # (9,)


def mu_vec(alpha):
    return alpha * DLOG


_c, CTEMP = FC.build_continuum_model(Z_POP)  # stochastic template @ this z


def make_prior(sigma_reg):
    """ExReg stochastic prior at this PSD amplitude. Depends ONLY on sigma_reg -> build once, reuse."""
    t = copy.deepcopy(CTEMP)
    t["sigma_reg"]["init"] = sigma_reg  # free PSD amplitude
    t = adjust_stochastic_params(t)
    return t["logsfr_ratios"]["prior"]


def draw_theta(fmodel, alpha, prior, logzsol=None):
    """mean-zero stochastic draw + mu_vec(alpha). ALL nuisances (logzsol, dust, gas_*, eline_sigma...)
    drawn from their actual priors so they genuinely smear the features -- otherwise the test is
    optimistic. NO +/-5 clip: that is the FIT bound; clipping here would compress the sigma_reg axis.
    logzsol=None -> marginalized over its prior; a value -> FIXED (age-metallicity degeneracy test)."""
    ndim = len(fmodel.theta)
    th = np.asarray(
        fmodel.prior_transform(rng.uniform(size=ndim)), float
    )  # marginalize every nuisance
    ratios = np.asarray(prior.sample(), float).reshape(-1) + mu_vec(alpha)
    th[fmodel.theta_index["logsfr_ratios"]] = np.clip(
        ratios, -10, 10
    )  # +/-10 = numerical safety only
    th[fmodel.theta_index["logmass"]] = (
        LOGMASS  # irrelevant to EW/Dn4000 (ratios) but set for clarity
    )
    if logzsol is not None:
        th[fmodel.theta_index["logzsol"]] = logzsol  # fixed for the Z grid
    return th


def build_model(z):
    ft = copy.deepcopy(CTEMP)
    ft.update(copy.deepcopy(TemplateLibrary["cue_stellar_nebular"]))
    ft["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}
    ft["use_stellar_ionizing"]["init"] = True
    nrat = len(ft["agebins"]["init"]) - 1
    ft["logsfr_ratios"]["isfree"] = True
    ft["logsfr_ratios"]["init"] = np.zeros(nrat)
    ft["logsfr_ratios"]["prior"] = TopHat(
        mini=np.full(nrat, -5.0), maxi=np.full(nrat, 5.0)
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


fmodel = build_model(Z_POP)
obs0 = P.build_obs(
    spec=np.ones(len(P.WAVE_OBS)),
    unc=np.ones(len(P.WAVE_OBS)),
    mask=np.ones(len(P.WAVE_OBS), bool),
    resolution=res_lsf,
)
WREST = P.WAVE_OBS / (1 + Z_POP)


def features(spec):
    """Halpha EW (window method, ~matches FastSpecFit to ~5%) + Dn4000 (Balogh bands, Fnu)."""
    s = np.asarray(spec, float)
    s[s <= 0] = np.nan
    lm = np.abs(WREST - 6564.6) <= 12.0
    cm = ((WREST >= 6440) & (WREST <= 6500)) | ((WREST >= 6620) & (WREST <= 6680))
    cont = np.nanmedian(s[cm])
    ew = (
        np.nansum((s[lm] - cont) / cont) * (P.WAVE_OBS[1] - P.WAVE_OBS[0]) / (1 + Z_POP)
        if cont > 0
        else np.nan
    )
    b = np.nanmean(s[(WREST >= 3850) & (WREST <= 3950)])
    r = np.nanmean(s[(WREST >= 4000) & (WREST <= 4100)])
    return float(ew), float(r / b) if b > 0 else np.nan


PRIORS = {
    sr: make_prior(sr) for sr in SIGREG_GRID
}  # build once per sigma_reg, not per galaxy
res = {}
t0 = time.time()
for mu in MU_GRID:
    for sr in SIGREG_GRID:
        for zm in ZMET_GRID:
            EW, DN = [], []
            for n in range(N_GAL):
                a = rng.normal(mu, SIGMA_A)
                th = draw_theta(fmodel, a, PRIORS[sr], logzsol=zm)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        pr, _ = fmodel.predict(th, observations=obs0, sps=sps)
                        e, d = features(np.asarray(pr[0], float).reshape(-1))
                    except Exception:
                        e, d = np.nan, np.nan
                EW.append(e)
                DN.append(d)
            EW = np.array(EW)
            DN = np.array(DN)
            g = np.isfinite(EW) & np.isfinite(DN)
            key = f"{mu}_{sr}" if MODE == "sigreg" else f"{mu}_{zm}"
            res[key] = dict(ew=EW, dn=DN)
            tag = f"sigma_reg={sr:.1f}" if MODE == "sigreg" else f"logzsol={zm:+.2f}"
            print(
                f"mu_alpha={mu:+.1f} {tag}  N={int(g.sum())}  "
                f"med EW={np.nanmedian(EW[g]):7.2f}  med Dn4000={np.nanmedian(DN[g]):.3f}  "
                f"(EW 16-84: {np.nanpercentile(EW[g], 16):.2f}-{np.nanpercentile(EW[g], 84):.2f})"
            )
pickle.dump(
    dict(
        res=res,
        mu_grid=MU_GRID,
        sigreg_grid=SIGREG_GRID,
        zmet_grid=ZMET_GRID,
        mode=MODE,
        z=Z_POP,
        n_gal=N_GAL,
        sigma_alpha=SIGMA_A,
    ),
    open(OUT, "wb"),
)
print(f"\nsaved {OUT}   ({(time.time() - t0) / 60:.1f} min)")
print(
    "READ: if med EW / med Dn4000 shift strongly with mu_alpha at FIXED sigma_reg, and the"
)
print(
    "      sigma_reg direction moves them DIFFERENTLY (not just along the same track), the two"
)
print(
    "      are separable -> proceed. If the grid collapses onto one curve -> degenerate -> stop."
)
