"""Where do best-fit model spectra land in the cont10 latent space, next to their DESI spectrum or
with the v2 mocks?

QUESTION: DESI continua sit off the v2 mock bulk in cont10 space (61% below the mock p5 log p;
compare_flows.py). Can the model make DESI-like continua at all? If a galaxy's best-fit model lands
next to its own DESI spectrum, the offset comes from how the priors populate mock space; if it lands
away from DESI (toward the mocks), the model misses part of the continuum shape.
HYPOTHESIS: none stated before the run.
INPUTS: results/map_fits_outliers310_v2/<tid>.pkl (MAP fits of the 310: DESI flux, unc, best-fit
model, maggies on the DESI grid), data/checkpoints/spender_desi_cont_10latent_zmax.pt,
data/latents/spender_spec_cont10latent_snr3.h5 (stored DESI latents), the new cont10 flow
(flows/flow_nsf_cont10latent.pt) and the v2 mock latents.
METHOD: flux and model converted to f_lambda, normalized by the median over rest 5300-5850 A of good
pixels (unc > 0, outside the spender sky-line mask), as spender's prepare_spectra; encoded with
the cont10 encoder. Model spectra are encoded noiseless and with one noise draw from the galaxy's
unc. Check first: the re-encoded DESI flux must match the stored DESI latent.
SEED: 0 (noise draw).
COMMAND: ``uv run python experiments/2026-11-09_mocks_v2_outliers/model_spectra_test.py``.
RESULT: 310 fits; re-encoded DESI = stored latents (distance 0.000). Under the new cont10 flow the
best-fit models sit as far off the mock bulk as their DESI spectra: log p median -16.1 (noiseless)
vs DESI -15.6; 88% vs 81% below the mock p5 (-10.4). With one noise draw the models rise to -10.8
(55%). Model-DESI distance 1.38 (noisy 1.12) vs 0.69 to the nearest other galaxy. So the model
makes DESI-like, off-mock continua: the offset is mainly how the priors populate mock space.
Assumptions: the 310 are outliers, not typical DESI; MAP fits use the 10-2025 model, not the v2
mock setup; one noise draw per galaxy.
FIGURES: none (model_spectra_test.txt).
"""

# %%
import glob
import pickle

import astropy.units as u
import numpy as np
import torch
from spender import load_model
from spender.data import desi
from spender.instrument import get_skyline_mask

from hubersed.conversion import to_flambda
from hubersed.detect.flow_model import build_flow
from hubersed.io.latents import load_latents
from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-11-09_mocks_v2_outliers"
rng = np.random.default_rng(0)
lines = []

inst = desi.DESI()
sky = get_skyline_mask(inst._wave_obs).numpy()
enc = load_model(
    str(PATHS["DATA"] / "checkpoints" / "spender_desi_cont_10latent_zmax.pt"),
    inst,
    map_location="cpu",
    weights_only=False,
).float()
enc.eval()

# %% load the fits and build normalized spectra
spec_d, spec_m, spec_n, tids = [], [], [], []
for f in sorted(glob.glob(str(RES / "map_fits_outliers310_v2" / "*.pkl"))):
    d = pickle.load(open(f, "rb"))
    if not isinstance(d, dict) or d.get("status") != "ok":
        continue
    w = np.asarray(d["wave"], float)
    assert w.size == 7781 and np.allclose(w, inst._wave_obs.numpy()), f
    fl = to_flambda(w * u.AA, np.asarray(d["flux"], float) * u.mgy).value
    md = to_flambda(w * u.AA, np.asarray(d["model"], float) * u.mgy).value
    un = to_flambda(w * u.AA, np.asarray(d["unc"], float) * u.mgy).value
    good = np.isfinite(fl) & np.isfinite(un) & (un > 0) & ~sky
    rest = w / (1 + d["z"])
    sel = good & (rest > 5300) & (rest < 5850)
    if sel.sum() < 10:
        continue
    norm = np.median(fl[sel])
    noisy = md + rng.normal(0.0, np.where(good, un, 0.0))
    spec_d.append(np.nan_to_num(fl / norm))
    spec_m.append(np.nan_to_num(md / norm))
    spec_n.append(np.nan_to_num(noisy / norm))
    tids.append(int(d["target_id"]))
tids = np.array(tids)
lines.append(f"MAP fits used: {tids.size}")


def encode(x):
    """Return cont10 latents of normalized spectra."""
    with torch.no_grad():
        return enc.encode(torch.tensor(np.array(x), dtype=torch.float32)).numpy()


lat_d, lat_m, lat_n = encode(spec_d), encode(spec_m), encode(spec_n)

# %% check: re-encoded DESI vs stored DESI latents
xs, sid, _ = load_latents(PATHS["DATA"] / "latents" / "spender_spec_cont10latent_snr3.h5")
srow = {int(t): i for i, t in enumerate(sid)}
have = np.array([t in srow for t in tids])
stored = xs[[srow[t] for t in tids[have]]]
g = torch.load(OUT / "flows" / "flow_nsf_cont10latent.pt", weights_only=False, map_location="cpu")
sc = g["scaler_scale"]


def dist(a, b):
    """Return the distance between latents in units of the mock scaler."""
    return np.linalg.norm((a - b) / sc, axis=1)


dd = dist(lat_d[have], stored)
lines.append(
    f"check, re-encoded DESI vs stored ({have.sum()}): distance median {np.median(dd):.3f}, "
    f"p95 {np.percentile(dd, 95):.3f} (mock-scaler units)"
)

# %% positions: distances and log p under the new cont10 flow
nde = build_flow(g["method"], g["dim"], g["hidden"], g["num_transforms"], g["num_bins"])
nde.load_state_dict(g["state_dict"])
nde.eval()
xm, _, _ = load_latents(PATHS["ROOT"] / g["mock_file"])


def logp(x):
    """Return log p under the new cont10 flow."""
    z = ((x - g["scaler_mean"]) / sc).astype(np.float32)
    with torch.no_grad():
        return nde.log_prob(torch.from_numpy(z)).numpy()


lp_mock = logp(xm)
for name, lat in (("DESI", lat_d), ("model", lat_m), ("model+noise", lat_n)):
    lp = logp(lat)
    lines.append(
        f"{name:>12}: log p median {np.median(lp):7.1f} (p16 {np.percentile(lp, 16):7.1f}, "
        f"p84 {np.percentile(lp, 84):7.1f}); below the mock p5 {np.mean(lp < np.percentile(lp_mock, 5)):.2f}"
    )
d_dm, d_dn = dist(lat_d, lat_m), dist(lat_d, lat_n)
# typical spacing: distance from each DESI latent to its nearest other DESI latent in this set
dd_all = np.linalg.norm((lat_d[:, None, :] - lat_d[None, :, :]) / sc, axis=2)
np.fill_diagonal(dd_all, np.inf)
nn_desi = dd_all.min(axis=1)
lines.append(
    f"distance DESI-model median {np.median(d_dm):.2f}, DESI-(model+noise) {np.median(d_dn):.2f}; "
    f"nearest other DESI galaxy in this set {np.median(nn_desi):.2f}"
)
lines.append(f"mock log p median {np.median(lp_mock):.1f}, p5 {np.percentile(lp_mock, 5):.1f}")

text = "\n".join(lines)
print(text)
(OUT / "model_spectra_test.txt").write_text(text + "\n")
