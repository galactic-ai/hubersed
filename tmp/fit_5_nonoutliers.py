"""
Fit 5 CLEAN non-outlier galaxies (SF dwarf -> quiescent) with the STOCHASTIC prior and the
UNIFORM-in-sSFR prior, and compare recovered params to each other and to FastSpecFit (VAC).

Goal: does the new (uniform-in-sSFR) prior give sensible physical parameters for TYPICAL
galaxies (not just the OOD extremes), and does it agree with the stochastic fit + the VAC?

Run from hubersed root (venv):  python tmp/fit_5_nonoutliers.py    (slow: 10 MAP fits)
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
from astropy.io import fits


class PolyCalSpectrum(PolyOptCal, Spectrum):
    pass


GALS = {
    "SF dwarf": 39627782304566661,
    "main sequence": 39627764281641737,
    "massive SF": 39633145225543970,
    "green valley": 39627787719413869,
    "quiescent": 39633127663995476,
}
POLY_ORDER = 4
NSEEDS, MAXFEV = 4, 15000
SFH_RANGE = 5.0
TO = 1e8
R = 0.4
SSFR_LO, SSFR_HI = -13.0, float(np.log10(1.0 / (TO * (1 - R))))
OUT = "results/fit_5_nonoutliers.pkl"
VAC = (
    str(MC.DATA_PATH / "fastspec-iron-sv3-bright.fits")
    if hasattr(MC, "DATA_PATH")
    else "data/fastspec-iron-sv3-bright.fits"
)

MC._fsps()
res_lsf = MC._lsf_sigma_kms()
sps = MC._cue()
sps.ssp.params["tpagb_norm_type"] = 2
sps.ssp.params["add_agb_dust_model"] = True

# ---- VAC lookup for the 5 ----
h = fits.open("data/fastspec-iron-sv3-bright.fits", memmap=True)


def col(n):
    for hd in h:
        c = getattr(getattr(hd, "columns", None), "names", None)
        if c and n in c:
            return np.asarray(hd.data[n])


tid = col("TARGETID").astype(np.int64)
cols = {k: col(k) for k in ["LOGMSTAR", "SFR", "ZZSUN", "Z", "DN4000", "HALPHA_EW"]}
Vidx = {int(t): i for i, t in enumerate(tid)}


def vac(t):
    i = Vidx[int(t)]
    lm = float(cols["LOGMSTAR"][i])
    sfr = float(cols["SFR"][i])
    return dict(
        logmass=lm,
        logssfr=(np.log10(sfr) - lm) if sfr > 0 else np.nan,
        zzsun=float(cols["ZZSUN"][i]),
        dn4000=float(cols["DN4000"][i]),
        haew=float(cols["HALPHA_EW"][i]),
    )


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


def uniform_fit(flux, unc, mask, z):
    flux = np.asarray(flux, float).reshape(-1)
    unc = np.asarray(unc, float).reshape(-1)
    mask = np.asarray(mask, bool).reshape(-1)
    fmodel = build_uniform_model(z)
    nfree = len(fmodel.free_params)
    obs = P.build_obs(
        spec=flux, unc=unc, mask=mask, resolution=res_lsf
    )  # plain Spectrum, NO PolyOptCal (matches stochastic path)
    fl0 = np.asarray(obs[0].flux, float).reshape(-1)  # snapshot AFTER rectify,
    un0 = np.asarray(obs[0].uncertainty, float).reshape(
        -1
    )  # BEFORE predict can mutate obs
    mk0 = np.asarray(obs[0].mask, bool).reshape(-1)
    assert fl0.shape == un0.shape == mk0.shape, (
        f"obs shapes flux{fl0.shape} unc{un0.shape} mask{mk0.shape}"
    )

    def neg(th):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ss = float(compute_logssfr(fmodel, th))
                if not (SSFR_LO <= ss <= SSFR_HI):
                    return 1e18
                lp = lnprobfn(th, model=fmodel, observations=obs, sps=sps, nested=False)
                return -lp if np.isfinite(lp) else 1e18
            except Exception:
                return 1e18

    bf = MC._map_optimize(neg, fmodel.theta.copy(), n_seeds=NSEEDS, maxfev=MAXFEV)
    th = bf.x
    preds, _ = fmodel.predict(
        th, observations=obs, sps=sps
    )  # returns (list, mfrac); unpack!
    sp = np.asarray(preds[0], float).reshape(-1)
    chi2 = float(np.nansum(((fl0[mk0] - sp[mk0]) / un0[mk0]) ** 2))
    cr = chi2 / (int(mk0.sum()) - nfree)  # dof: no calibration params
    thd = {
        k: float(np.atleast_1d(th[i])[0])
        for k, i in fmodel.theta_index.items()
        if k != "logsfr_ratios"
    }
    thd["logsfr_ratios"] = np.asarray(
        th[fmodel.theta_index["logsfr_ratios"]], float
    )  # SAVE full SFH → multi-timescale sSFR
    thd["chi2_red"] = cr
    thd["logssfr"] = float(compute_logssfr(fmodel, th))
    thd["model"] = sp
    return thd


def stoch_ssfr(rs):
    z = rs["z"]
    tu = MC.universe_age_gyr(z) if hasattr(MC, "universe_age_gyr") else None
    from hubersed.prospector.utils import make_stochastic_agebins

    ab = 10 ** make_stochastic_agebins(z)
    dt = ab[:, 1] - ab[:, 0]
    mid = ab.mean(1)
    lsr = np.atleast_1d(rs["theta_dict"]["logsfr_ratios"])
    lmv = float(np.atleast_1d(rs["theta_dict"]["logmass"])[0])
    sr = 10 ** np.clip(lsr, -100, 100)
    c = np.ones(10)
    for i in range(10):
        num = np.prod(dt[1 : i + 1]) if i >= 1 else 1.0
        den = np.prod(dt[:i]) if i >= 1 else 1.0
        sd = np.prod(sr[:i]) if i >= 1 else 1.0
        c[i] = (1 / sd) * (num / den)
    mm = 10**lmv / c.sum() * c
    return float(np.log10((mm[mid <= TO].sum() / TO) / (mm.sum() * (1 - R))))


results = {}
for label, t in GALS.items():
    gidx = int(MC.tids_to_indices(np.array([t], dtype=np.int64))[0])
    print(f"\n{'=' * 66}\n{label}  TID {t}")
    spec, ivar, z, tid2 = MC.load_by_index(gidx)
    spec = np.asarray(spec, float).reshape(-1)
    ivar = np.asarray(ivar, float).reshape(-1)
    vac_z = float(cols["Z"][Vidx[t]])
    assert int(tid2) == int(t), f"TID MISMATCH: asked {t}, loaded {tid2}"
    assert abs(z - vac_z) < 0.002, f"z MISMATCH: loaded {z:.4f} vs VAC {vac_z:.4f}"
    assert spec.shape == P.WAVE_OBS.shape == ivar.shape, (
        f"SHAPE: spec{spec.shape} ivar{ivar.shape} wave{P.WAVE_OBS.shape}"
    )
    print(
        f"  [check] load_by_index TID {tid2}  z={z:.4f} == VAC z={vac_z:.4f}  shape={spec.shape}  OK"
    )
    rs = MC.map_chi2_one(gidx, use_cue=True, cont_nseeds=1, full_nseeds=2, maxfev=6000)
    sm = flambda_to_maggies(P.WAVE_OBS, spec)
    iv = ivar_flambda_to_ivar_maggies(P.WAVE_OBS, ivar)
    sig = 1.0 / np.sqrt(np.where(iv > 0, iv, np.inf))
    mk = (sig > 0) & np.isfinite(sig) & np.isfinite(sm)
    ru = uniform_fit(sm, sig, mk, z)
    vv = vac(t)

    def gz(d, k):
        return float(np.atleast_1d(d.get(k, np.nan))[0])

    ss_s = stoch_ssfr(rs) if rs.get("status") == "ok" else np.nan
    print(f"  {'':14}{'stochastic':>12}{'uniform':>12}{'VAC':>12}")
    print(
        f"  {'chi2_red':14}{rs.get('chi2_red', np.nan):12.2f}{ru['chi2_red']:12.2f}{'--':>12}"
    )
    print(
        f"  {'logmass':14}{gz(rs.get('theta_dict', {}), 'logmass'):12.2f}{ru['logmass']:12.2f}{vv['logmass']:12.2f}"
    )
    print(
        f"  {'logzsol':14}{gz(rs.get('theta_dict', {}), 'logzsol'):12.2f}{ru['logzsol']:12.2f}{vv['zzsun']:12.2f}"
    )
    print(f"  {'logsSFR100':14}{ss_s:12.2f}{ru['logssfr']:12.2f}{vv['logssfr']:12.2f}")
    results[label] = dict(tid=t, z=z, stoch=rs, uniform=ru, vac=vv, stoch_logssfr=ss_s)
pickle.dump(results, open(OUT, "wb"))
print(f"\nsaved {OUT}")
