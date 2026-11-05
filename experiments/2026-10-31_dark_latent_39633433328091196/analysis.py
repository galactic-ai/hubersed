"""Latent position and flow log p of the SV3 bright and SV3 dark spectra of 39633433328091196 (#17).

QUESTION: do the cont10 and cont15 latents and flow log p of the continuum outlier move when its SV3
dark coadd, which has no sky excess, is encoded instead of the SV3 bright coadd?
HYPOTHESIS: none registered.
INPUTS: bright spectrum from its DESIchunk1024 pkl (the spectra the DESI latents were made from).
Dark spectrum from data/sv3_dark_coadds/coadd-sv3-dark-10231.fits through spender
DESI.prepare_spectra, as 2026-10-30_sv3_bright_dark_39633433328091196. Encoders
data/checkpoints/spender_desi_cont_{10,15}latent_zmax.pt. Seed 0 flows
results/noised_cue_meanzero_wide_flow/flow_nsf_cont{10,15}latent.pt and their outlier files.
Stored latents data/latents/spender_spec_cont{10,15}latent_snr3.h5.
SEED: none.
COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-31_dark_latent_39633433328091196/analysis.py``.
RESULT: 2026-10-31, dark_latent.txt. The dark spectrum is not an outlier in either flow. log p goes
from -22.11 to -4.15 in cont10 (threshold -15.72, DESI percentile 0.8 to 55.7) and from -70.19 to
-4.87 in cont15 (threshold -25.24, percentile 0.04 to 6.8). Latent shift norm 3.0 (cont10) and 4.0
(cont15) mock scaler units. Median S/N is 5.06 bright and 5.64 dark. Assumptions: seed 0 flows only,
one dark exposure, Mac CPU encoding (cont15 log p tolerance 0.05).
FIGURES: results/2026-10-31_dark_latent_39633433328091196/dark_latent.png and dark_latent.txt.
"""

# %%
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from spender import load_model
from spender.data.desi import DESI

from hubersed.detect.flow_model import build_flow
from hubersed.io.desi import tids_to_indices
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39633433328091196
DATA = PATHS["DATA"]
FLOWS = PATHS["RESULTS"] / "noised_cue_meanzero_wide_flow"
OUT = PATHS["RESULTS"] / "2026-10-31_dark_latent_39633433328091196"
OUT.mkdir(parents=True, exist_ok=True)
TAGS = ("cont10", "cont15")
# The check stops the script when a re-encoded or rescored bright value differs by more than this.
# The cont15 latent file differs from a Mac CPU encode by 1e-4 to 6e-4 for every galaxy checked,
# probably GPU arithmetic, so its log p tolerance is 0.05 (Nikhil, 2026-10-31).
TOL_LATENT = 1e-3
TOL_LOGP = {"cont10": 1e-2, "cont15": 5e-2}


def median_snr(spec, w):
    """Return the median of spec * sqrt(w) over pixels with w > 0, as scripts/get_latent_space.py."""
    good = w > 0
    return float(np.median((spec * np.sqrt(w))[good]))


def load_flow(tag):
    """Return the seed 0 flow of a tag in eval mode, with its scaler mean and scale."""
    d = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu")
    nde = build_flow(d["method"], d["dim"], d["hidden"], d["num_transforms"], d["num_bins"])
    nde.load_state_dict(d["state_dict"])
    nde.eval()
    return nde, np.asarray(d["scaler_mean"]), np.asarray(d["scaler_scale"]), d["encoder"]


def log_p(nde, mean, scale, lat):
    """Return the flow log p of latents, scaled as in scripts/get_outliers_flow.py."""
    x = ((lat - mean) / scale).astype(np.float32)
    with torch.no_grad():
        return nde.log_prob(torch.from_numpy(np.atleast_2d(x))).numpy()


# %% the two spectra, both normalised by spender
gidx = int(tids_to_indices(np.array([TID], dtype=np.int64))[0])
with open(DATA / "desi_spectra" / f"DESIchunk1024_{gidx // 1024}.pkl", "rb") as f:
    s, w, z, tid, *_ = pickle.load(f)
row = gidx % 1024
assert int(tid[row]) == TID, "wrong chunk row"
bright = {"spec": np.asarray(s[row], np.float32), "w": np.asarray(w[row]), "z": float(z[row])}

spec, w, z, tid, _, _ = DESI.prepare_spectra(
    str(DATA / "sv3_dark_coadds" / "coadd-sv3-dark-10231.fits")
)
i = int(np.flatnonzero(tid.numpy() == TID)[0])
dark = {"spec": spec[i].numpy(), "w": w[i].numpy(), "z": float(z[i])}

# %% encode, score and check against the stored values
inst = DESI()
res = {}
for tag in TAGS:
    nde, mean, scale, enc = load_flow(tag)
    model = load_model(
        str(DATA / "checkpoints" / enc), inst, map_location="cpu", weights_only=False
    )
    model = model.float().eval()
    with torch.no_grad():
        lat = {k: model.encode(torch.from_numpy(d["spec"][None])).numpy()[0] for k, d in
               (("bright", bright), ("dark", dark))}  # fmt: skip
    lp = {k: float(log_p(nde, mean, scale, v)[0]) for k, v in lat.items()}

    o = torch.load(FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt", weights_only=False,
                   map_location="cpu")  # fmt: skip
    desi_lp = np.asarray(o["log_p_desi"], float)
    j = int(np.flatnonzero(np.asarray(o["desi_target_ids"]) == TID)[0])
    with h5py.File(DATA / "latents" / f"spender_spec_{tag}latent_snr3.h5", "r") as f:
        k = int(np.flatnonzero(f["target_ids"][:] == TID)[0])
        stored_lat = f["latents"][k]
    d_lat = float(np.abs(lat["bright"] - stored_lat).max())
    d_lp = abs(lp["bright"] - desi_lp[j])
    print(
        f"{tag}: bright latent max |re-encoded - stored| {d_lat:.2e}, |log p - stored| {d_lp:.2e}"
    )
    assert d_lat < TOL_LATENT and d_lp < TOL_LOGP[tag], (
        f"{tag}: bright does not reproduce the stored run"
    )

    shift = (lat["dark"] - lat["bright"]) / scale
    res[tag] = {"lp": lp, "stored_lp": desi_lp[j], "desi_lp": desi_lp, "thr": float(o["threshold"]),
                "shift": shift, "d_lat": d_lat, "d_lp": d_lp}  # fmt: skip

# %% table
lines = [
    f"TARGETID {TID}. Bright z {bright['z']:.5f}, median S/N {median_snr(**{k: bright[k] for k in ('spec', 'w')}):.2f}; "
    f"dark z {dark['z']:.5f}, median S/N {median_snr(dark['spec'], dark['w']):.2f} (latent files keep S/N > 3)",
    "Seed 0 flows. pct = percent of the 114693 DESI log p at or below the value.",
    "",
    f"{'flow':>6} {'threshold':>10} {'bright':>9} {'pct':>7} {'dark':>9} {'pct':>7} {'dark-bright':>11}"
    f" {'flagged b/d':>11}",
]
for tag, r in res.items():
    pct = {k: 100 * float(np.mean(r["desi_lp"] <= v)) for k, v in r["lp"].items()}
    flag = "/".join("yes" if r["lp"][k] <= r["thr"] else "no" for k in ("bright", "dark"))
    lines.append(
        f"{tag:>6} {r['thr']:10.3f} {r['lp']['bright']:9.3f} {pct['bright']:7.3f} "
        f"{r['lp']['dark']:9.3f} {pct['dark']:7.3f} {r['lp']['dark'] - r['lp']['bright']:+11.3f} {flag:>11}"
    )
lines.append("")
lines.append("Latent shift dark - bright in mock scaler units (per dimension, then the norm):")
for tag, r in res.items():
    dims = " ".join(f"{x:+.2f}" for x in r["shift"])
    lines.append(f"{tag:>6}: {dims}  | norm {np.linalg.norm(r['shift']):.2f}")
lines.append("")
lines.append("Check: bright re-encoded and rescored against the stored latents and log p:")
for tag, r in res.items():
    lines.append(f"{tag:>6}: max |d latent| {r['d_lat']:.2e}, |d log p| {r['d_lp']:.2e}")
text = "\n".join(lines)
(OUT / "dark_latent.txt").write_text(text + "\n")
print(text)

# %% figure: DESI log p histogram per flow, threshold, bright and dark
use_apj_style()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (tag, r) in zip(axes, res.items(), strict=True):
    ax.hist(r["desi_lp"], bins=200, color="0.7", label=r"DESI SV3 BGS, S/N $>$ 3")
    ax.axvline(r["thr"], color="k", ls="--", lw=1.2, label=r"threshold (0.1\% mock $\log p$)")
    ax.axvline(r["lp"]["bright"], color="#b2182b", lw=1.5, label="SV3 bright")
    ax.axvline(r["lp"]["dark"], color="#2166ac", lw=1.5, label="SV3 dark")
    ax.set_xlabel(rf"$\log p$, {tag}latent flow (seed 0)", fontsize=14)
    ax.set_ylabel("galaxies", fontsize=14)
axes[0].legend(frameon=False, fontsize=11, loc="upper left")
fig.savefig(OUT / "dark_latent.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'dark_latent.png'}")
