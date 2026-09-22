# %% [markdown]
# # 94183 (EELG) — three emission-line fits
#
# Emission-line outlier TARGETID 39632991244258619 ("94183"), z=0.0395.
#
# | run | data | resolution handling | nebular model |
# |-----|------|---------------------|---------------|
# | 1 | native DESI, full range | OLD (LSF onto model, library zeroed) | FSPS/Cloudy (`build_full_model`) |
# | 2 | native DESI, full range | OLD (LSF onto model, library zeroed) | Cue (`build_full_cue_model`) |
# | 3 | degraded->MILES, MILES window | NEW (prep_spectrum, resolution=None) | Cue (`build_full_cue_model`) |
#
# Run 1 vs 2 = FSPS-nebular vs Cue (why Cue). Run 2 vs 3 = native vs MILES-degraded.
# Each nebular fit is seeded from a continuum MAP (lines masked). Nebular fits fit the lines
# (unmasked). FSPS nebular uses FastStepBasis; Cue uses NebStepBasis.
#
# ORDER: Run 3 (library intact) must complete before the native runs zero the library.

# %%
import warnings, pickle
import numpy as np
import matplotlib.pyplot as plt

from prospect.fitting import lnprobfn
from prospect.sources import SSPBasis

from hubersed.paths import PATHS
from hubersed.fitting.chi2 import load_by_index, tids_to_indices, _map_optimize, WAVE_OBS
from hubersed.fitting.config import build_continuum_model, build_full_model, build_full_cue_model
from hubersed.prospector.parameter_file import build_obs, build_sps, build_cue_sps, mask_spectral_lines
from hubersed.prospector.rebin import prep_spectrum, common_obs_edges, MILES_LAM_MIN, MILES_LAM_MAX
from hubersed.prospector.lsf import desi_resolution, C_KMS
from hubersed.conversion import flambda_to_maggies, ivar_flambda_to_ivar_maggies

TID   = 39632991244258619        # "94183"
EDGES = common_obs_edges()
LSF   = (C_KMS / (2.355 * desi_resolution(WAVE_OBS))).astype(np.float64)   # DESI LSF sigma [km/s]
OUT   = PATHS["RESULTS"]
REPORT = {}
LINES = {"[OII]": 3727.0, "[NeIII]": 3869.0, "Hb": 4861.0, "[OIII]": 5007.0,
         "Ha": 6563.0, "[NII]": 6584.0, "[SII]": 6716.0, "[SIII]": 9069.0}
CONT_KEYS = ("logzsol", "logmass", "sigma_smooth")
FSPS_KEYS = ("logzsol", "logmass", "sigma_smooth", "gas_logz", "gas_logu", "eline_sigma")
CUE_KEYS  = ("logzsol", "logmass", "sigma_smooth", "gas_logz", "gas_logu", "gas_logqion", "eline_sigma")

# %% [markdown]
# ## Load 94183 (native) + line inventory vs the MILES window

# %%
idx = int(tids_to_indices(np.array([TID], np.int64))[0])
spec, ivar, z, tid = load_by_index(idx)
assert int(tid) == TID
fm = flambda_to_maggies(WAVE_OBS, spec)
iv = ivar_flambda_to_ivar_maggies(WAVE_OBS, ivar)
ok = (iv > 0) & np.isfinite(fm)
iv = np.where(ok, iv, 0.0)
sig = 1.0 / np.sqrt(np.where(iv > 0, iv, np.inf))
print(f"TID {tid}  z={z:.4f}")
for name, lam in LINES.items():
    print(f"  {name:>7} rest {lam:6.0f}  obs {lam*(1+z):6.0f}  "
          f"{'IN MILES' if MILES_LAM_MIN <= lam <= MILES_LAM_MAX else 'DROPPED (Run3)'}")

# %% [markdown]
# ## Helpers

# %%
def map_fit(model, obs, sps, n_seeds=4, maxfev=20000, theta0=None):
    def neg(th):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lp = lnprobfn(th, model=model, observations=obs, sps=sps, nested=False)
                return -lp if np.isfinite(lp) else 1e18
            except Exception:
                return 1e18
    start = model.theta.copy() if theta0 is None else np.asarray(theta0, float)
    return _map_optimize(neg, start, n_seeds=n_seeds, maxfev=maxfev)

def build_cue_seeded(cont_tmpl, theta_seed, cont_model, z):
    """build_full_cue_model (tied cue_stellar_nebular, gas_logqion free per your config.py)
    + seed eline_sigma=50 (its config init 100 is a dead spot the simplex never leaves)
    and a burst SFH in the starting theta. No use_stellar_ionizing flip."""
    m, ft = build_full_cue_model(cont_tmpl, theta_seed, cont_model, z)
    th0 = m.theta.copy()
    th0[m.theta_index["eline_sigma"]] = 50.0
    li = m.theta_index["logsfr_ratios"]
    n = int(np.atleast_1d(th0[li]).size)
    th0[li] = np.array([0.7, 0.6, 0.4, 0.2, 0, 0, 0, 0, 0], float)[:n]
    return m, th0

def chi2_red(model, theta, obs, sps):
    preds, _ = model.predict(theta, observations=obs, sps=sps)   # (preds_list, extra)
    sp = np.asarray(preds[0], float)
    o = obs[0]
    flux = np.asarray(o.flux, float); unc = np.asarray(o.uncertainty, float)
    m = np.asarray(o.mask, bool) & np.isfinite(sp) & np.isfinite(unc) & (unc > 0)
    r = (flux[m] - sp[m]) / unc[m]
    ndof = max(int(m.sum()) - len(theta), 1)
    return float(np.sum(r**2) / ndof), sp

def grab(model, best, keys):
    if best is None:
        return {k: np.nan for k in keys}
    return {k: float(best.x[model.theta_index[k]][0]) for k in keys}

def emlines(sps):
    fw = sps.ssp.emline_wavelengths
    return fw[(fw > 3600) & (fw < 9824)]

_HAD_SR = "spectral_resolution" in SSPBasis.__dict__
def zero_library():
    SSPBasis.spectral_resolution = property(lambda self: np.zeros_like(self.ssp.wavelengths))
def restore_library():
    if "spectral_resolution" in SSPBasis.__dict__ and not _HAD_SR:
        delattr(SSPBasis, "spectral_resolution")

def plot_run(wave, flux, sigv, mask, model_spec, cont_spec, title, savepath):
    g = np.asarray(mask, bool)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(wave[g], flux[g], color="0.45", lw=0.7, label="data")
    if cont_spec is not None:
        ax.plot(wave[g], cont_spec[g], color="#2166ac", lw=0.9, label="continuum")
    ax.plot(wave[g], model_spec[g], color="#b2182b", lw=1.0, label="full model")
    ymax = np.nanpercentile(flux[g], 99.5) * 1.15
    ax.set_ylim(np.nanmin(flux[g]) * 1.1, ymax)
    for name, lam in LINES.items():
        lo = lam * (1 + z)
        if wave[g][0] < lo < wave[g][-1]:
            ax.axvline(lo, color="0.85", lw=0.8, zorder=0)
            ax.text(lo, ymax, name, rotation=90, va="top", ha="right", fontsize=7, color="0.55")
    ax.set_ylabel("flux [maggies]"); ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title(title)
    axr.axhline(0, color="0.6", lw=0.8)
    axr.plot(wave[g], ((flux - model_spec) / sigv)[g], color="#b2182b", lw=0.6)
    axr.set_ylim(-5, 5); axr.set_ylabel("resid/σ"); axr.set_xlabel("observed wavelength [Å]")
    fig.tight_layout(); fig.savefig(savepath, dpi=150, bbox_inches="tight"); plt.show()
    print("saved", savepath)

# %% [markdown]
# ## RUN 3 first (NEW: degrade->MILES, Cue) — library INTACT
# continuum MAP (masked) -> Cue MAP (lines fit)

# %%
restore_library()
sps = build_sps(); cue_sps = build_cue_sps()

wave_c, flux_c, ivar_c, good_c = prep_spectrum(WAVE_OBS, fm, iv, z, EDGES)
sig_c = 1.0 / np.sqrt(np.where(ivar_c > 0, ivar_c, np.inf))

m_cont = mask_spectral_lines(wave_c, good_c, z, halfwidth_kms=1500.0, line_waves=emlines(sps))
obs_c  = build_obs(spec=flux_c, unc=sig_c, mask=m_cont, resolution=None, wavelength=wave_c)
cont_model, cont_tmpl = build_continuum_model(z)
best_c = map_fit(cont_model, obs_c, sps)
_, cont_spec3 = chi2_red(cont_model, best_c.x, obs_c, sps)

obs_l = build_obs(spec=flux_c, unc=sig_c, mask=good_c, resolution=None, wavelength=wave_c)
cue_model, th0_3 = build_cue_seeded(cont_tmpl, best_c.x, cont_model, z)
lp0 = lnprobfn(th0_3, model=cue_model, observations=obs_l, sps=cue_sps, nested=False)
print(f"  Run3 cue lnprob(th0)={lp0:.1f}  finite={np.isfinite(lp0)}")
best_l = map_fit(cue_model, obs_l, cue_sps, theta0=th0_3)
assert best_l is not None, f"Cue MAP failed (None). lnprob(th0)={lp0}"
c2_3, pred3 = chi2_red(cue_model, best_l.x, obs_l, cue_sps)

REPORT["run3_MILES_cue"] = dict(chi2=c2_3, params=grab(cue_model, best_l, CUE_KEYS))
with open(OUT / "fit94183_run3_MILES_cue.pkl", "wb") as f:
    pickle.dump(dict(wave=wave_c, flux=flux_c, sig=sig_c, mask=good_c, model=pred3, cont=cont_spec3,
                     theta=best_l.x, labels=cue_model.theta_labels(), chi2=c2_3, z=float(z)), f)
print(f"[RUN3 MILES/Cue] chi2={c2_3:.2f}  {REPORT['run3_MILES_cue']['params']}")
plot_run(wave_c, flux_c, sig_c, good_c, pred3, cont_spec3,
         f"94183  z={z:.4f}  (Run 3: MILES + Cue)  chi2={c2_3:.2f}",
         OUT / "fit94183_run3_MILES_cue.png")

# %% [markdown]
# ## Native continuum seed (OLD: library zeroed) — used by Run 1 and Run 2

# %%
zero_library()
sps_old = build_sps(); cue_sps_old = build_cue_sps()

m_cont_n = mask_spectral_lines(WAVE_OBS, ok, z, halfwidth_kms=1500.0, line_waves=emlines(sps_old))
obs_cn   = build_obs(spec=fm, unc=sig, mask=m_cont_n, resolution=LSF, wavelength=WAVE_OBS)
cont_model_o, cont_tmpl_o = build_continuum_model(z)
best_cn = map_fit(cont_model_o, obs_cn, sps_old)
_, cont_spec_n = chi2_red(cont_model_o, best_cn.x, obs_cn, sps_old)
print(f"[native continuum seed] {grab(cont_model_o, best_cn, CONT_KEYS)}")

# %% [markdown]
# ## RUN 1 (native, FSPS nebular) — build_full_model + FastStepBasis

# %%
obs_1 = build_obs(spec=fm, unc=sig, mask=ok, resolution=LSF, wavelength=WAVE_OBS)
fsps_model, fsps_tmpl = build_full_model(cont_tmpl_o, best_cn.x, cont_model_o, z)
best_1 = map_fit(fsps_model, obs_1, sps_old)
assert best_1 is not None, "FSPS nebular MAP failed (None)."
c2_1, pred1 = chi2_red(fsps_model, best_1.x, obs_1, sps_old)
REPORT["run1_native_fsps"] = dict(chi2=c2_1, params=grab(fsps_model, best_1, FSPS_KEYS))
with open(OUT / "fit94183_run1_native_fsps.pkl", "wb") as f:
    pickle.dump(dict(wave=WAVE_OBS, flux=fm, sig=sig, mask=ok, model=pred1, cont=cont_spec_n,
                     theta=best_1.x, labels=fsps_model.theta_labels(), chi2=c2_1, z=float(z)), f)
print(f"[RUN1 native/FSPS] chi2={c2_1:.2f}  {REPORT['run1_native_fsps']['params']}")
plot_run(WAVE_OBS, fm, sig, ok, pred1, cont_spec_n,
         f"94183  z={z:.4f}  (Run 1: native + FSPS nebular)  chi2={c2_1:.2f}",
         OUT / "fit94183_run1_native_fsps.png")

# %% [markdown]
# ## RUN 2 (native, Cue nebular) — build_full_cue_model + NebStepBasis

# %%
obs_2 = build_obs(spec=fm, unc=sig, mask=ok, resolution=LSF, wavelength=WAVE_OBS)
cue_model_o, th0_2 = build_cue_seeded(cont_tmpl_o, best_cn.x, cont_model_o, z)
lp0 = lnprobfn(th0_2, model=cue_model_o, observations=obs_2, sps=cue_sps_old, nested=False)
print(f"  Run2 cue lnprob(th0)={lp0:.1f}  finite={np.isfinite(lp0)}")
best_2 = map_fit(cue_model_o, obs_2, cue_sps_old, theta0=th0_2)
assert best_2 is not None, f"Cue MAP failed (None). lnprob(th0)={lp0}"
c2_2, pred2 = chi2_red(cue_model_o, best_2.x, obs_2, cue_sps_old)
REPORT["run2_native_cue"] = dict(chi2=c2_2, params=grab(cue_model_o, best_2, CUE_KEYS))
with open(OUT / "fit94183_run2_native_cue.pkl", "wb") as f:
    pickle.dump(dict(wave=WAVE_OBS, flux=fm, sig=sig, mask=ok, model=pred2, cont=cont_spec_n,
                     theta=best_2.x, labels=cue_model_o.theta_labels(), chi2=c2_2, z=float(z)), f)
print(f"[RUN2 native/Cue] chi2={c2_2:.2f}  {REPORT['run2_native_cue']['params']}")
plot_run(WAVE_OBS, fm, sig, ok, pred2, cont_spec_n,
         f"94183  z={z:.4f}  (Run 2: native + Cue)  chi2={c2_2:.2f}",
         OUT / "fit94183_run2_native_cue.png")

# %% [markdown]
# ## Compare

# %%
def _p(d): return "  ".join(f"{k}={d['params'].get(k, np.nan):+.2f}" for k in
                            ("logzsol", "logmass", "sigma_smooth", "gas_logz", "gas_logu", "eline_sigma"))
print(f"{'run':>22} | {'chi2':>7} | params")
for r in ("run1_native_fsps", "run2_native_cue", "run3_MILES_cue"):
    print(f"{r:>22} | {REPORT[r]['chi2']:>7.2f} | {_p(REPORT[r])}")
print("\nRun1 vs Run2 = FSPS nebular vs Cue (native, emission convolved to DESI LSF).")
print("Run2 vs Run3 = native vs MILES-degraded (watch emission: MILES model emission is NOT")
print("re-broadened, so Run3 chi2 may blow up on the line cores).")
