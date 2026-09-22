# %% [markdown]
# # Is `gas_logqion` live under `cue_stellar_nebular`?
#
# Claim under test: freeing `gas_logqion` in `build_full_cue_model` is a no-op, because
# `prospect/sources/nebssp_basis.py:464` does
# `params.update(**fit_4loglinear_ionparam(wave, spec))`, and
# `cuejax/utils.py:318` returns `"gas_logqion": logQ`. Line 467 then uses the overwritten
# value. `cue_keys` (nebssp_basis.py:422-433) has 12 entries and does not include
# `gas_logqion`, so it never enters the emulator input either.
#
# No fitting here. Everything is `model.predict` at a fixed theta, so the whole notebook
# runs in seconds.
#
# | test | what it shows | expected if the claim is TRUE |
# |------|---------------|-------------------------------|
# | 0 | determinism control | identical theta -> diff exactly 0 |
# | A | perturb `gas_logqion` 3 dex, stock code | diff exactly 0 |
# | B | instrument `fit_4loglinear_ionparam` | returns `gas_logqion`; also exposes pre-clip `ionspec_*` |
# | C | amplitude-only patch, perturb again | diff LARGE (this is also the prototype fix) |
# | D | perturb `eline_sigma` and `sigma_smooth` | `eline_sigma` diff 0, `sigma_smooth` diff large |
#
# Test C is the control that makes Test A meaningful: it proves the comparison machinery
# can see a change when one exists.

# %%
import warnings
import numpy as np

import prospect.sources.nebssp_basis as nb
from prospect.sources import SSPBasis

from hubersed.fitting.chi2 import load_by_index, tids_to_indices, WAVE_OBS
from hubersed.fitting.config import build_continuum_model, build_full_cue_model
from hubersed.prospector.parameter_file import build_obs, build_cue_sps
from hubersed.prospector.lsf import desi_resolution, C_KMS
from hubersed.conversion import flambda_to_maggies, ivar_flambda_to_ivar_maggies

TID = 39632991244258619  # "94183", the EELG
LSF = (C_KMS / (2.355 * desi_resolution(WAVE_OBS))).astype(np.float64)
LINES = {"Hb": 4861.0, "[OIII]": 5007.0, "Ha": 6563.0}

# Same regime as fit_94183_three_runs.py runs 1 and 2: library resolution zeroed,
# DESI LSF supplied on the obs side.
_HAD_SR = "spectral_resolution" in SSPBasis.__dict__


def zero_library():
    SSPBasis.spectral_resolution = property(lambda self: np.zeros_like(self.ssp.wavelengths))


def restore_library():
    if "spectral_resolution" in SSPBasis.__dict__ and not _HAD_SR:
        delattr(SSPBasis, "spectral_resolution")


# %% [markdown]
# ## Load the galaxy and build the Cue model
#
# No continuum MAP is needed. `build_full_cue_model` only reads `logmass`, `logzsol` and
# `sigma_smooth` out of the continuum theta, so the continuum model's own default theta is
# a perfectly valid seed for a predict-only test.

# %%
idx = int(tids_to_indices(np.array([TID], np.int64))[0])
spec, ivar, z, tid = load_by_index(idx)
assert int(tid) == TID

fm = flambda_to_maggies(WAVE_OBS, spec)
iv = ivar_flambda_to_ivar_maggies(WAVE_OBS, ivar)
ok = (iv > 0) & np.isfinite(fm)
iv = np.where(ok, iv, 0.0)
sig = 1.0 / np.sqrt(np.where(iv > 0, iv, np.inf))

zero_library()
cue_sps = build_cue_sps()

obs = build_obs(spec=fm, unc=sig, mask=ok, resolution=LSF, wavelength=WAVE_OBS)
cont_model, cont_tmpl = build_continuum_model(z)
cue_model, cue_tmpl = build_full_cue_model(cont_tmpl, cont_model.theta, cont_model, z)

print(f"TID {tid}  z={z:.4f}  ndim={len(cue_model.theta)}")
print("free params:", cue_model.theta_labels())
print()
print("nebemlineinspec in template:", cue_tmpl["nebemlineinspec"]["init"])
print("use_stellar_ionizing       :", cue_tmpl["use_stellar_ionizing"]["init"])


# %% [markdown]
# ## Helper: predict at a theta, and compare two predictions

# %%
def predict_at(model, theta, sps=cue_sps, observations=obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds, _ = model.predict(np.asarray(theta, float), observations=observations, sps=sps)
    return np.asarray(preds[0], float)


def compare(a, b, label, mask=ok):
    d = np.abs(a - b)[mask]
    scale = np.maximum(np.abs(a)[mask], 1e-300)
    out = {
        "label": label,
        "max_abs": float(np.nanmax(d)),
        "max_frac": float(np.nanmax(d / scale)),
    }
    for name, lam in LINES.items():
        j = int(np.argmin(np.abs(WAVE_OBS - lam * (1 + z))))
        denom = a[j] if abs(a[j]) > 0 else np.nan
        out[name] = float(abs(a[j] - b[j]) / denom) if np.isfinite(denom) else np.nan
    print(
        f"{label:<38} max|d|={out['max_abs']:.6e}  max frac={out['max_frac']:.6e}  "
        + "  ".join(f"{k}={out[k]:.4e}" for k in LINES)
    )
    return out


RESULTS = {}
th0 = cue_model.theta.copy()
i_qion = cue_model.theta_index["gas_logqion"]
print("gas_logqion init:", float(np.atleast_1d(th0[i_qion])[0]))


# %% [markdown]
# ## Test 0 — determinism control
#
# Same theta twice. Anything but exactly 0 means the comparison itself is unreliable and
# every number below is meaningless.

# %%
p0a = predict_at(cue_model, th0)
p0b = predict_at(cue_model, th0)
RESULTS["0_determinism"] = compare(p0a, p0b, "TEST 0  same theta twice")
assert RESULTS["0_determinism"]["max_abs"] == 0.0, "predict is not deterministic; stop here"


# %% [markdown]
# ## Test A — perturb `gas_logqion` by 3 dex, stock code
#
# 3 dex is a factor of 1000 in ionizing photon rate. If the parameter were live the
# emission lines would change by that factor.

# %%
thA1, thA2 = th0.copy(), th0.copy()
thA1[i_qion] = 49.0
thA2[i_qion] = 52.0

pA1 = predict_at(cue_model, thA1)
pA2 = predict_at(cue_model, thA2)
RESULTS["A_stock"] = compare(pA1, pA2, "TEST A  gas_logqion 49 vs 52 (stock)")

if RESULTS["A_stock"]["max_abs"] == 0.0:
    print("\n  -> gas_logqion is DEAD under cue_stellar_nebular. Claim confirmed.")
else:
    print("\n  -> gas_logqion moved the spectrum. Claim REFUTED; report this back.")


# %% [markdown]
# ## Test B — instrument `fit_4loglinear_ionparam`
#
# Two things at once:
#
# 1. does the returned dict contain `gas_logqion` (the overwrite), and what `logQ` do the
#    stars actually give;
# 2. the `ionspec_*` values **before** the `np.clip` at `cuejax/utils.py:316-317`. Those
#    clips are silent, and a low-metallicity EELG is exactly the case that would hit them.
#    If a pre-clip value sits outside its bound, Cue is being evaluated at an ionizing
#    spectrum the stars do not have.

# %%
_ORIG_FIT = nb.fit_4loglinear_ionparam
CALLS = []

_BOUNDS = {  # cuejax/utils.py:316-317
    "ionspec_index1": (1.0, 42.0),
    "ionspec_index2": (-0.3, 30.0),
    "ionspec_index3": (-1.0, 14.0),
    "ionspec_index4": (-1.7, 8.0),
    "ionspec_logLratio1": (-1.0, 10.1),
    "ionspec_logLratio2": (-0.5, 1.9),
    "ionspec_logLratio3": (-0.4, 2.2),
}


def _fit_recording(wav, spec, **kw):
    d = _ORIG_FIT(wav, spec, **kw)
    coeff = np.asarray(d["powerlaw_params"], float)  # (4,2): [slope, norm] per segment
    CALLS.append({"returned": {k: v for k, v in d.items() if k != "powerlaw_params"},
                  "raw_slopes": coeff[:, 0].copy()})
    return d


nb.fit_4loglinear_ionparam = _fit_recording
CALLS.clear()
_ = predict_at(cue_model, thA1)
nb.fit_4loglinear_ionparam = _ORIG_FIT

print(f"fit_4loglinear_ionparam called {len(CALLS)} times per predict (young, old CSP)")
print("keys returned:", sorted(CALLS[0]["returned"].keys()))
print("'gas_logqion' in returned dict:", "gas_logqion" in CALLS[0]["returned"])
print()
for n, c in enumerate(CALLS):
    tag = ["young", "old"][n] if n < 2 else str(n)
    print(f"  [{tag}] gas_logqion set to {c['returned']['gas_logqion']:.3f} "
          f"(your theta was {float(np.atleast_1d(thA1[i_qion])[0]):.3f})")
    for k in ("ionspec_index1", "ionspec_index2", "ionspec_index3", "ionspec_index4"):
        lo, hi = _BOUNDS[k]
        post = float(np.atleast_1d(c["returned"][k])[0])
        raw = float(c["raw_slopes"][int(k[-1]) - 1])
        flag = "  <-- CLIPPED" if not (lo <= raw <= hi) else ""
        print(f"        {k:<20} raw={raw:+10.3f}  used={post:+10.3f}  bounds=[{lo}, {hi}]{flag}")
    for k in ("ionspec_logLratio1", "ionspec_logLratio2", "ionspec_logLratio3"):
        lo, hi = _BOUNDS[k]
        post = float(np.atleast_1d(c["returned"][k])[0])
        edge = "  <-- AT BOUND" if np.isclose(post, lo) or np.isclose(post, hi) else ""
        print(f"        {k:<20} used={post:+10.3f}  bounds=[{lo}, {hi}]{edge}")


# %% [markdown]
# ## Test C — amplitude-only patch: stars set the SHAPE, `gas_logqion` sets the NORM
#
# One-line change: drop `gas_logqion` from what `fit_4loglinear_ionparam` hands back, so
# `params.update` no longer clobbers the free value. The seven `ionspec_*` shape
# parameters still come from the stellar spectrum.
#
# This is legitimate inside Cue's own parameterization: `emulator.py:175` and
# `nebssp_basis.py:467` show Q_H enters purely multiplicatively (the emulator is trained
# at fixed log Q = 49.1), so it is a rescaling, not an extrapolation.
#
# Physically it is a free ionizing-photon-production-efficiency offset at fixed spectral
# shape: values above the stellar `logQ` mean the stars under-produce LyC (binaries,
# stripped stars), values below mean escape or covering-fraction losses.
#
# **Caveat:** `gas_logu` and `gas_logqion` both scale ionization amplitude and will be
# partially degenerate. If this run produces a ridge, fix `gas_logu` before drawing
# physical conclusions.

# %%
def _fit_keep_shape_only(wav, spec, **kw):
    d = _ORIG_FIT(wav, spec, **kw)
    d.pop("gas_logqion", None)  # keep the shape, let the free theta own the normalization
    return d


nb.fit_4loglinear_ionparam = _fit_keep_shape_only

pC1 = predict_at(cue_model, thA1)
pC2 = predict_at(cue_model, thA2)
RESULTS["C_patched"] = compare(pC1, pC2, "TEST C  gas_logqion 49 vs 52 (patched)")

nb.fit_4loglinear_ionparam = _ORIG_FIT  # restore before anything else runs

print()
if RESULTS["C_patched"]["max_abs"] > 0 and RESULTS["A_stock"]["max_abs"] == 0.0:
    print("  -> machinery CAN see a gas_logqion change. Test A's zero is a real no-op,")
    print("     not a broken comparison. The patch is the fix.")
elif RESULTS["C_patched"]["max_abs"] == 0.0:
    print("  -> patched run ALSO shows no change. Something else is blocking it;")
    print("     do not trust Test A either. Report back.")


# %% [markdown]
# ## Test D — is `eline_sigma` dead too, and what actually broadens the lines?
#
# `_cue_stellar_nebular_` sets `nebemlineinspec = True`, and `build_full_cue_model` never
# overrides it (`build_full_model`, the FSPS arm, does override it to `False` at
# `config.py:128-132`). With it True, `sedmodel.py:481` makes `_need_lines` False, so
# Prospector's analytic emission path is off and `eline_sigma` is unused. Lines are
# instead injected at `nebssp_basis.py:251-262` with a width set by **`sigma_smooth`**,
# the stellar velocity dispersion.
#
# If that holds, run 1 vs run 2 is currently a line-width comparison, not a nebular-model
# comparison.

# %%
print("model params  nebemlineinspec:", cue_model.params.get("nebemlineinspec"))
print("model         _need_lines    :", cue_model._need_lines)
print()

for name, lo, hi in [("eline_sigma", 30.0, 240.0), ("sigma_smooth", 30.0, 300.0)]:
    if name not in cue_model.theta_index:
        print(f"{name}: not a free parameter, skipped")
        continue
    j = cue_model.theta_index[name]
    t1, t2 = th0.copy(), th0.copy()
    t1[j], t2[j] = lo, hi
    RESULTS[f"D_{name}"] = compare(
        predict_at(cue_model, t1), predict_at(cue_model, t2),
        f"TEST D  {name} {lo:g} vs {hi:g}",
    )

print()
d_el = RESULTS.get("D_eline_sigma", {}).get("max_abs", np.nan)
d_ss = RESULTS.get("D_sigma_smooth", {}).get("max_abs", np.nan)
if d_el == 0.0 and d_ss > 0:
    print("  -> eline_sigma DEAD, sigma_smooth drives the line widths. Cue lines are being")
    print("     broadened to the STELLAR dispersion. Set nebemlineinspec=False in")
    print("     build_full_cue_model before comparing run 1 to run 2.")


# %% [markdown]
# ## Summary

# %%
print(f"{'test':<40} {'max|d|':>14} {'max frac':>14}")
for k, v in RESULTS.items():
    print(f"{v['label']:<40} {v['max_abs']:>14.6e} {v['max_frac']:>14.6e}")

restore_library()

verdict = (
    RESULTS["A_stock"]["max_abs"] == 0.0
    and RESULTS["C_patched"]["max_abs"] > 0.0
)
print()
print("gas_logqion is a NO-OP under cue_stellar_nebular:", verdict)
