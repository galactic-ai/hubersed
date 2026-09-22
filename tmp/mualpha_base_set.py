"""
STEP 2a: generate the mu_alpha BASE SET (one per z-slice), measure features once.

Any target population (mu_alpha, sigma_alpha, mu_Z, sigma_Z) is then an IMPORTANCE REWEIGHTING
of this base -- no further FSPS calls. Key facts that shape this (all established/verified today):
  * Halpha EW and Dn4000 are RATIOS -> mass-independent -> base is per z-slice (7), NOT per (M,z)
    bin (37). Restricting to Pozzetti-complete bins removed the only other place M entered.
  * sigma_reg is GRIDDED (not per-draw) because adjust_stochastic_params depends only on it --
    building the prior per draw would dominate the runtime. The log-spaced grid == sampling its
    log-uniform prior, so sigma_reg is marginalized in the first pass.
  * logzsol drawn U(-1.5,+0.4): deliberately WIDER than BOTH the mock generator
    (get_stochastic_priors.py: U(-1.0,+0.19), a Wan+24 inheritance) and the observed need
    (Dn4000=2.04 requires +0.30). The base MUST cover or the reweighting ESS collapses.
  * alpha drawn U(-1,+2.5): brackets the observed (mu_alpha saturates above ~+1 -> lower limit there).

Run from hubersed root (venv):  python tmp/mualpha_base_set.py [quick]
"""

import sys, copy, time, warnings
import numpy as np

sys.path.insert(0, "bin/prospector")
from hubersed.prospector import parameter_file as P
from hubersed.fitting import config as FC
from hubersed.fitting import chi2 as MC
from prospect.models.sedmodel import SpecModel
from prospect.models.templates import TemplateLibrary, adjust_stochastic_params
from prospect.models.priors import TopHat
from hubersed.prospector.utils import make_stochastic_agebins

QUICK = "quick" in sys.argv
Z_SLICES = [
    0.075,
    0.125,
    0.175,
    0.225,
    0.275,
    0.325,
    0.375,
]  # centres of the 7 locked z bins
SIGREG_GRID = np.logspace(
    np.log10(0.1), np.log10(5.0), 15
)  # == log-uniform prior (marginalized)
N_PER_SR = 1000 if not QUICK else 40  # -> 15k per z-slice
ALPHA_LO, ALPHA_HI = -1.0, 2.5
LOGZ_LO, LOGZ_HI = -1.5, 0.4  # WIDER than mocks (-1.0,+0.19) and fits
LOGMASS = 10.0  # irrelevant: features are ratios
OUT = "results/mualpha_base_set.npz"
rng = np.random.default_rng(17)

MC._fsps()
res_lsf = MC._lsf_sigma_kms()
sps = MC._cue()
sps.ssp.params["tpagb_norm_type"] = 2
sps.ssp.params["add_agb_dust_model"] = True


def build_model(ctemp):
    ft = copy.deepcopy(ctemp)
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


REC = {
    k: []
    for k in [
        "z",
        "sigma_reg",
        "alpha",
        "logzsol",
        "dust2",
        "dust_index",
        "gas_logu",
        "ew",
        "dn4000",
    ]
}
t0 = time.time()
for z in Z_SLICES:
    _c, CTEMP = FC.build_continuum_model(z)
    fmodel = build_model(CTEMP)
    ndim = len(fmodel.theta)
    ti = fmodel.theta_index
    AB = 10 ** make_stochastic_agebins(z)
    MID = AB.mean(1)
    DLOG = np.log10(MID[:-1] / MID[1:])
    obs0 = P.build_obs(
        spec=np.ones(len(P.WAVE_OBS)),
        unc=np.ones(len(P.WAVE_OBS)),
        mask=np.ones(len(P.WAVE_OBS), bool),
        resolution=res_lsf,
    )
    WREST = P.WAVE_OBS / (1 + z)
    lmask = np.abs(WREST - 6564.6) <= 12.0
    cmask = ((WREST >= 6440) & (WREST <= 6500)) | ((WREST >= 6620) & (WREST <= 6680))
    bmask = (WREST >= 3850) & (WREST <= 3950)
    rmask = (WREST >= 4000) & (WREST <= 4100)
    dlam = P.WAVE_OBS[1] - P.WAVE_OBS[0]
    for sr in SIGREG_GRID:
        t = copy.deepcopy(CTEMP)
        t["sigma_reg"]["init"] = float(sr)
        prior = adjust_stochastic_params(t)["logsfr_ratios"][
            "prior"
        ]  # depends only on sigma_reg
        for _ in range(N_PER_SR):
            a = rng.uniform(ALPHA_LO, ALPHA_HI)
            lz = rng.uniform(LOGZ_LO, LOGZ_HI)
            th = np.asarray(
                fmodel.prior_transform(rng.uniform(size=ndim)), float
            )  # nuisances
            th[ti["logsfr_ratios"]] = np.clip(
                np.asarray(prior.sample(), float).reshape(-1) + a * DLOG, -10, 10
            )
            th[ti["logmass"]] = LOGMASS
            th[ti["logzsol"]] = lz
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    pr, _ = fmodel.predict(th, observations=obs0, sps=sps)
                    s = np.asarray(pr[0], float).reshape(-1)
                    s[s <= 0] = np.nan
                    cont = np.nanmedian(s[cmask])
                    ew = (
                        float(np.nansum((s[lmask] - cont) / cont) * dlam / (1 + z))
                        if cont > 0
                        else np.nan
                    )
                    b = np.nanmean(s[bmask])
                    rr = np.nanmean(s[rmask])
                    dn = float(rr / b) if b > 0 else np.nan
                except Exception:
                    ew = dn = np.nan
            REC["z"].append(z)
            REC["sigma_reg"].append(sr)
            REC["alpha"].append(a)
            REC["logzsol"].append(lz)
            REC["dust2"].append(float(np.atleast_1d(th[ti["dust2"]])[0]))
            REC["dust_index"].append(float(np.atleast_1d(th[ti["dust_index"]])[0]))
            REC["gas_logu"].append(float(np.atleast_1d(th[ti["gas_logu"]])[0]))
            REC["ew"].append(ew)
            REC["dn4000"].append(dn)
    n = len(REC["z"])
    ok = np.isfinite(REC["ew"][-N_PER_SR * len(SIGREG_GRID) :]).sum()
    print(
        f"z={z:.3f}  cumulative N={n:,}  finite(last slice)={ok}  elapsed={(time.time() - t0) / 60:.1f} min"
    )
np.savez_compressed(
    OUT,
    **{k: np.asarray(v, np.float32) for k, v in REC.items()},
    alpha_range=[ALPHA_LO, ALPHA_HI],
    logz_range=[LOGZ_LO, LOGZ_HI],
    sigreg_grid=SIGREG_GRID,
)
f = np.isfinite(REC["ew"]) & np.isfinite(REC["dn4000"])
print(
    f"\nsaved {OUT}   N={len(REC['z']):,}  finite={int(np.sum(f)):,}   ({(time.time() - t0) / 60:.1f} min)"
)
print(
    f"base coverage: EW {np.nanpercentile(REC['ew'], 1):.2f}..{np.nanpercentile(REC['ew'], 99):.1f}   "
    f"Dn4000 {np.nanpercentile(REC['dn4000'], 1):.3f}..{np.nanpercentile(REC['dn4000'], 99):.3f}"
)
print(
    "CHECK: base Dn4000 99th must EXCEED the observed max (1.958) or the massive bins are uncovered."
)
