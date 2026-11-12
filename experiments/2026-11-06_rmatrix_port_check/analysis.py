"""Does the prospector fork's resolution matrix path reproduce hubersed's ResolutionMatrixModel?

QUESTION: at one parameter point of 39627770174637084, do the fork's SpecModel through
prospect.observation.ResolutionMatrixSpectrum and the 10-13 hubersed ResolutionMatrixModel give the
same model spectrum and lnL on the same data, mask and R?
HYPOTHESIS: they agree to max |dmodel|/sigma < 0.1 and |dlnL| < 1; the old path smooths with
hubersed's smooth_log_uniform and the new one with sedpy's native Fourier path.
INPUTS: the 10-13 max-L point (_pred.npz `best`) with eline_fbroad = 0 and eline_sigma_forb =
eline_sigma, so the old two-component model has one narrow width; the data, uncertainty and
fitted pixels of that file; R from 10-13 model_setup.COADD (the recoadd). C3K_ER Cue, afe 0.
SEED: none.
COMMAND: from the repository root on the Mac,
``PYTHONPATH=<C3K_ER fsps site>:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-11-06_rmatrix_port_check/analysis.py``.
RESULT: 2026-11-06, check.txt. Pass: max |new - old|/sigma 0.001 (rms < 0.001) over 5530 pixels,
lnL -5169.31 old vs -5169.44 new (-0.13). The old path reproduces the saved max-L (sp_best exactly).
Assumptions: one parameter point; narrow lines only; Mac C3K_ER build; recoadd R.
FIGURES: results/2026-11-06_rmatrix_port_check/ check.txt and check.png.
"""

# %%
import matplotlib.pyplot as plt
import model_setup as ms
import numpy as np
from prospect.models.sedmodel import HyperSpecModel
from prospect.observation import ResolutionMatrixSpectrum

from hubersed.paths import PATHS
from hubersed.sps.resolution_matrix import load_resolution_npz

OUT = PATHS["RESULTS"] / "2026-11-06_rmatrix_port_check"
PRED = (
    PATHS["RESULTS"]
    / "2026-10-13_nautilus_resolution_matrix"
    / "39627770174637084_rmatrix_mask_er_broad_shared_split45_rest9000_seed0_pred.npz"
)
BROAD = (
    "eline_fbroad",
    "eline_sigma_broad",
    "eline_vbroad",
    "eline_sigma_forb",
    "eline_fbroad_forb",
)

# %% data, R and the parameter point
p = np.load(PRED, allow_pickle=True)
z, flux, unc, good = float(p["z"]), p["flux"], p["unc"], p["good"].astype(bool)
labels = [str(x) for x in p["labels"]]
best = dict(zip(labels, p["best"], strict=True))
best["eline_fbroad"] = 0.0
best["eline_sigma_forb"] = best["eline_sigma"]
r = load_resolution_npz(ms.COADD)
cue = ms.get_cue()

# %% old path: hubersed ResolutionMatrixModel on its own observation
old_model = ms.make_model(ms.make_params(z), "rmatrix")
old_obs = ms.make_obs(flux, unc, good, "rmatrix")
old_theta = np.array([best[k] for k in old_model.theta_labels()])
# prospect's lnL includes the Gaussian normalisation -0.5 sum ln(2 pi sigma^2)
norm = -0.5 * np.sum(np.log(2 * np.pi * unc[good] ** 2))
saved = old_model.predict(np.array(p["best"]), observations=old_obs, sps=cue)[0][0]
chi = -0.5 * np.sum(((saved - flux) / unc)[good] ** 2)
assert np.isclose(chi + norm, float(p["max_lnl"]), atol=0.01), "old path does not reproduce max-L"
old = old_model.predict(old_theta, observations=old_obs, sps=cue)[0][0]

# %% new path: fork SpecModel with the narrow lines only, on the fork's observation
params = ms.make_params(z)
for k in BROAD:
    params.pop(k, None)
new_model = HyperSpecModel(params)
new_obs = ResolutionMatrixSpectrum(
    wavelength=r["wave"], resolution_matrix=r["res"], flux=flux, uncertainty=unc, mask=good
)
new_obs.rectify()
new_theta = np.array([best[k] for k in new_model.theta_labels()])
new = new_model.predict(new_theta, observations=[new_obs], sps=cue)[0][0]

# %% compare
dz = (new - old)[good] / unc[good]
lnl_old = -0.5 * np.sum(((old - flux) / unc)[good] ** 2)
lnl_new = -0.5 * np.sum(((new - flux) / unc)[good] ** 2)
lines = [
    "39627770174637084, 10-13 max-L point with eline_fbroad = 0 and eline_sigma_forb = eline_sigma,",
    f"sigma_smooth {best['sigma_smooth']:.2f} km/s, eline_sigma {best['eline_sigma']:.2f} km/s, "
    f"{good.sum()} fitted pixels.",
    f"max |new - old| / sigma = {np.max(np.abs(dz)):.3f}, rms = {np.sqrt(np.mean(dz**2)):.3f}",
    f"lnL old {lnl_old:.2f}, new {lnl_new:.2f}, new - old {lnl_new - lnl_old:+.2f}",
    f"pass (max < 0.1 and |dlnL| < 1): {np.max(np.abs(dz)) < 0.1 and abs(lnl_new - lnl_old) < 1}",
]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "check.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# %% figure: both models and their difference in units of sigma
rest = r["wave"] / (1 + z)
fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
ax[0].plot(rest[good], flux[good], color="0.7", lw=0.5, label="data")
ax[0].plot(rest[good], old[good], lw=0.7, label="hubersed ResolutionMatrixModel")
ax[0].plot(
    rest[good], new[good], lw=0.7, ls="--", label="fork SpecModel + ResolutionMatrixSpectrum"
)
ax[1].plot(rest[good], dz, lw=0.6, color="k")
ax[1].axhline(0, color="0.5", lw=0.5)
ax[0].set_ylabel("flux [maggies]")
ax[1].set_ylabel("(new - old) / sigma")
ax[1].set_xlabel("rest wavelength [A]")
ax[0].legend(frameon=False)
fig.savefig(OUT / "check.png", dpi=110, bbox_inches="tight")
