"""One-galaxy diagnostic: find where the NEW path goes non-finite. Run: python tmp/debug_new.py"""
import numpy as np
from hubersed.paths import PATHS
from hubersed.fitting.config import build_continuum_model
from hubersed.fitting.chi2 import load_by_index, tids_to_indices, WAVE_OBS
from hubersed.prospector.parameter_file import build_obs, build_sps, mask_spectral_lines
from hubersed.prospector.rebin import prep_spectrum, common_obs_edges
from hubersed.prospector.lsf import desi_resolution, C_KMS
from hubersed.conversion import flambda_to_maggies, ivar_flambda_to_ivar_maggies
from prospect.fitting import lnprobfn

EDGES = common_obs_edges()
s = np.load(PATHS["RESULTS"] / "oldnew_sample.npz")
tid = int(s["sf_tids"][0])
idx = int(tids_to_indices(np.array([tid], np.int64))[0])
spec, ivar, z, tid = load_by_index(idx)
fm = flambda_to_maggies(WAVE_OBS, spec)
iv = ivar_flambda_to_ivar_maggies(WAVE_OBS, ivar)
ok = (iv > 0) & np.isfinite(fm)
print(f"[raw]  z={z:.4f}  flux finite={np.isfinite(fm).mean():.2f}  ivar>0={(iv>0).mean():.2f}")

wc, fc, ic, gc = prep_spectrum(WAVE_OBS, fm, np.where(ok, iv, 0.0), z, EDGES)
print(f"[rebin] npix={fc.size}  flux_c finite={np.isfinite(fc).mean():.2f}  good_c={int(gc.sum())}  "
      f"ivar_c>0={(ic>0).sum()}  flux_c range=[{np.nanmin(fc):.3g},{np.nanmax(fc):.3g}]  "
      f"ivar_c range=[{np.nanmin(ic):.3g},{np.nanmax(ic):.3g}]")

sps = build_sps()
model, _ = build_continuum_model(z)
fw = sps.ssp.emline_wavelengths; fopt = fw[(fw > 3600) & (fw < 9824)]
sig = 1.0 / np.sqrt(np.where(ic > 0, ic, np.inf))
m = mask_spectral_lines(wc, gc, z, halfwidth_kms=1500.0, line_waves=fopt)
print(f"[fit]  n_unmasked={int(m.sum())}  sig finite in mask={np.isfinite(sig[m]).all()}  sig>0 in mask={(sig[m]>0).all()}")

obs = build_obs(spec=fc, unc=sig, mask=m, resolution=None, wavelength=wc)
th = model.theta.copy()
try:
    pr = model.predict(th, observations=obs, sps=sps); sp = np.asarray(pr[0])
    print(f"[model] pred finite={np.isfinite(sp).mean():.2f}  range=[{np.nanmin(sp):.3g},{np.nanmax(sp):.3g}]")
except Exception as e:
    print("[model] PREDICT FAILED:", repr(e))
print("[prior] lnprior(init) =", float(model.prior_product(th)))
print("[NEW ] lnprob(init)  =", float(lnprobfn(th, model=model, observations=obs, sps=sps, nested=False)))

# --- OLD-style obs (native grid + LSF) for comparison ---
LSF = (C_KMS / (2.355 * desi_resolution(WAVE_OBS)))
mo = mask_spectral_lines(WAVE_OBS, ok, z, halfwidth_kms=1500.0, line_waves=fopt)
obso = build_obs(spec=fm, unc=1.0 / np.sqrt(np.where(iv > 0, iv, np.inf)), mask=mo,
                 resolution=LSF, wavelength=WAVE_OBS)
print("[OLD ] lnprob(init)  =", float(lnprobfn(th, model=model, observations=obso, sps=sps, nested=False)))
