"""Measure emcee runtime per galaxy for the uniform-in-sSFR full fit.
Times a single Cue likelihood eval + 30 emcee steps, extrapolates to 64x(500+3000).
Run from hubersed root (venv):  python tmp/time_emcee.py
"""

import sys, time, warnings, copy
import numpy as np, emcee

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

TO = 1e8
R = 0.4
SFH_RANGE = 5.0
SSFR_LO, SSFR_HI = -13.0, float(np.log10(1.0 / (TO * (1 - R))))
TID = 39627764281641737  # real BGS galaxy for obs (z, ivar, mask)
NW, NBURN, NPROD = 64, 500, 3000  # the fit_single defaults we'd use

MC._fsps()
res_lsf = MC._lsf_sigma_kms()
sps = MC._cue()
sps.ssp.params["tpagb_norm_type"] = 2
sps.ssp.params["add_agb_dust_model"] = True


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


gidx = int(MC.tids_to_indices(np.array([TID], dtype=np.int64))[0])
spec, ivar, z, tid = MC.load_by_index(gidx)
spec = np.asarray(spec, float).reshape(-1)
ivar = np.asarray(ivar, float).reshape(-1)
sm = flambda_to_maggies(P.WAVE_OBS, spec)
iv = ivar_flambda_to_ivar_maggies(P.WAVE_OBS, ivar)
sig = 1.0 / np.sqrt(np.where(iv > 0, iv, np.inf))
mask = (sig > 0) & np.isfinite(sig) & np.isfinite(sm)
fmodel = build_uniform_model(z)
obs = P.build_obs(spec=sm, unc=sig, mask=mask, resolution=res_lsf)
ndim = len(fmodel.theta)


def lnp(th):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ss = float(compute_logssfr(fmodel, th))
            if not (SSFR_LO <= ss <= SSFR_HI):
                return -np.inf
            lp = lnprobfn(th, model=fmodel, observations=obs, sps=sps, nested=False)
            return lp if np.isfinite(lp) else -np.inf
        except Exception:
            return -np.inf


# valid starting theta
th0 = fmodel.theta.copy()
if not np.isfinite(lnp(th0)):
    for _ in range(500):
        cand = fmodel.prior_transform(np.random.uniform(size=ndim))
        if np.isfinite(lnp(cand)):
            th0 = cand
            break

# single-eval cost (jitter each so FSPS can't cache an identical call)
t = time.time()
for _ in range(20):
    lnp(th0 + 1e-3 * np.random.randn(ndim))
single_ms = (time.time() - t) / 20 * 1000

p0 = np.array([th0 + 1e-4 * np.random.randn(ndim) for _ in range(NW)])
t0 = time.time()
s = emcee.EnsembleSampler(NW, ndim, lnp)
s.run_mcmc(p0, 30, progress=False)
dt = time.time() - t0
eps = 30 * NW / dt
tot = NW * (NBURN + NPROD)
print(f"\nndim={ndim}  single-eval {single_ms:.0f} ms  |  {eps:.0f} eval/s")
print(
    f"full {NW}x({NBURN}+{NPROD}) = {tot:,} evals  ->  {tot / eps / 60:.0f} min/gal serial,  ~{tot / eps / 60 / 8:.0f} min on 8 cores"
)
print(f"burn-in acceptance {np.mean(s.acceptance_fraction):.2f}")
