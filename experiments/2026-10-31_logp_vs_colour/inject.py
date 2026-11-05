"""Inject a blue or red power-law tilt into normal spectra and re-score them with both flows (#17).

Galaxies are unflagged by both seed 0 flows, in the VAC and with |dgr| < 0.05 (FIBERTOTFLUX), drawn
from 25 random chunk files (seed 0), at most 1000. Each normalised spender spectrum is multiplied
by (lambda / 5800 A)^alpha, re-normalised as spender does, encoded and scored. The injected dgr is
measured through the BASS g and r curves of 2026-10-30_photcolour_dark_39633433328091196/filters.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_logp_vs_colour/inject.py`` after analysis.py.
OUTPUT: results/2026-10-31_logp_vs_colour/inject.txt and inject.png.
RESULT: 2026-10-31. A tilt to dgr -0.48 flags 15% (cont10) and 0.4% (cont15) of 1000 normal
galaxies, against about 40% and 10-30% observed at that dgr. A smooth tilt explains part of the
cont10 trend and almost none of cont15.
"""

# %%
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table
from spender import load_model
from spender.data.desi import DESI

from hubersed.detect.flow_model import build_flow
from hubersed.io.desi import tids_to_indices
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

RES = PATHS["RESULTS"]
OUT = RES / "2026-10-31_logp_vs_colour"
FLOWS = RES / "noised_cue_meanzero_wide_flow"
FILT = PATHS["ROOT"] / "experiments" / "2026-10-30_photcolour_dark_39633433328091196" / "filters"
TAGS = ("cont10", "cont15")
ALPHAS = (-1.6, -1.2, -0.9, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6)
N_CHUNKS, N_MAX, PIVOT = 25, 1000, 5800.0
rng = np.random.default_rng(0)

# %% candidates: unflagged, in the VAC, |dgr| < 0.05
store = {}
for tag in TAGS:
    o = torch.load(
        RES / "wide_flow_corrected" / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt",
        weights_only=False,
        map_location="cpu",
    )
    store[tag] = dict(
        lp=dict(
            zip(
                np.asarray(o["desi_target_ids"]).tolist(),
                np.asarray(o["log_p_desi"]).tolist(),
                strict=True,
            )
        ),  # fmt: skip
        thr=float(o["threshold"]),
        flagged={int(t) for t in o["outlier_target_ids"]},
    )
h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)
M, S = h["METADATA"].data, h["FASTSPEC"].data
vrow = {int(t): i for i, t in enumerate(M["TARGETID"])}


def dgr_vac(t):
    """Return the VAC spectrum-minus-FIBERTOTFLUX g-r of one target, NaN without positive flux."""
    i = vrow[t]
    f = []
    for b in "GR":
        fs, ft = float(S[f"FLUX_SYNTH_{b}"][i]), float(M[f"FIBERTOTFLUX_{b}"][i])
        f.append(fs / ft if fs > 0 and ft > 0 else np.nan)
    return -2.5 * np.log10(f[0] / f[1])


flagged = store["cont10"]["flagged"] | store["cont15"]["flagged"]
tids = np.array(sorted(store["cont10"]["lp"]), np.int64)
gidx = tids_to_indices(tids)
chunks = rng.choice(np.unique(gidx // 1024), N_CHUNKS, replace=False)
spec, w, z, tsel = [], [], [], []
for c in sorted(chunks.tolist()):
    with open(PATHS["DATA"] / "desi_spectra" / f"DESIchunk1024_{c}.pkl", "rb") as f:
        s, ww, zz, tid, *_ = pickle.load(f)
    for r, t in enumerate(np.asarray(tid).tolist()):
        if t in store["cont10"]["lp"] and t not in flagged and t in vrow and abs(dgr_vac(t)) < 0.05:
            spec.append(np.asarray(s[r], np.float32))
            w.append(np.asarray(ww[r], np.float32))
            z.append(float(zz[r]))
            tsel.append(t)
keep = rng.permutation(len(tsel))[:N_MAX]
spec, w, z = np.array(spec)[keep], np.array(w)[keep], np.array(z)[keep]
tsel = np.array(tsel)[keep]
print(f"{len(tsel)} galaxies from {N_CHUNKS} chunks")

# %% tilt, re-normalise, measure the injected dgr
wave = DESI._wave_obs.numpy().astype(float)
band = {}
for b, name in (("g", "BASS-g"), ("r", "BASS-r")):
    tab = Table.read(FILT / f"{name}.ecsv")
    band[b] = np.interp(wave, np.asarray(tab["wavelength"], float), np.asarray(tab["response"]),
                        left=0, right=0) * wave  # fmt: skip


def tilted(alpha):
    """Return the tilted, re-normalised spectra and the injected dgr of each galaxy."""
    tilt = (wave / PIVOT) ** alpha
    out = spec * tilt
    for k in range(len(out)):
        rest = wave / (1 + z[k])
        sel = (w[k] > 0) & (rest > 5300) & (rest < 5850)
        out[k] /= np.median(out[k][sel])
    good = np.where(w > 0, spec, 0.0)
    ratio = {b: (good * tilt) @ band[b] / (good @ band[b]) for b in "gr"}
    return out.astype(np.float32), -2.5 * np.log10(ratio["g"] / ratio["r"])


# %% encode and score
inst = DESI()
res = {}
t0 = time.time()
for tag in TAGS:
    d = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu")
    nde = build_flow(d["method"], d["dim"], d["hidden"], d["num_transforms"], d["num_bins"])
    nde.load_state_dict(d["state_dict"])
    nde.eval()
    model = load_model(str(PATHS["DATA"] / "checkpoints" / d["encoder"]), inst, map_location="cpu",
                       weights_only=False).float().eval()  # fmt: skip
    for a in ALPHAS:
        x, inj = tilted(a)
        with torch.no_grad():
            lat = np.concatenate([model.encode(torch.from_numpy(x[i : i + 256])).numpy()
                                  for i in range(0, len(x), 256)])  # fmt: skip
            xs = ((lat - d["scaler_mean"]) / d["scaler_scale"]).astype(np.float32)
            lp = nde.log_prob(torch.from_numpy(xs)).numpy()
        res[(tag, a)] = (inj, lp)
        print(f"{tag} alpha {a:+.2f} done, {time.time() - t0:.0f} s", flush=True)
    stored = np.array([store[tag]["lp"][t] for t in tsel.tolist()])
    print(
        f"{tag} check: alpha 0 against stored log p, max |d| {np.abs(res[(tag, 0.0)][1] - stored).max():.3g}"
    )

# %% table and figure
lines = [f"{len(tsel)} galaxies, |dgr| < 0.05 and unflagged. Base flagged % is 0 by construction.",
         f"{'alpha':>6} {'inj dgr':>8}  " + "  ".join(f"{t}: flagged %, median d log p" for t in TAGS)]  # fmt: skip
for a in ALPHAS:
    inj = res[(TAGS[0], a)][0]
    cells = []
    for t in TAGS:
        lp = res[(t, a)][1]
        cells.append(
            f"{100 * np.mean(lp <= store[t]['thr']):6.2f} {np.median(lp - res[(t, 0.0)][1]):+8.2f}"
        )
    lines.append(f"{a:6.2f} {np.median(inj):+8.3f}  " + "  ".join(cells))
text = "\n".join(lines)
(OUT / "inject.txt").write_text(text + "\n")
print(text)

obs = np.load(OUT / "binned.npz")
mid = 0.5 * (np.clip(obs["edges"][:-1], -1, 0.5) + np.clip(obs["edges"][1:], -1, 0.5))
use_apj_style()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, t in zip(axes, TAGS, strict=True):
    ax.errorbar(mid, 100 * obs[f"{t}_flagged"], yerr=100 * obs[f"{t}_err"], fmt="o-", color="0.6",
                ms=3, label="observed, all DESI")  # fmt: skip
    inj = [np.median(res[(t, a)][0]) for a in ALPHAS]
    fl = [100 * np.mean(res[(t, a)][1] <= store[t]["thr"]) for a in ALPHAS]
    ax.plot(inj, fl, "s-", color="#b2182b", label="injected tilt, normal galaxies")
    ax.set_xlabel(r"$\Delta(g-r)$ [mag]", fontsize=13)
    ax.set_ylabel(f"flagged by {t}latent flow [\\%]", fontsize=13)
axes[0].legend(frameon=False, fontsize=11, loc="upper right")
fig.savefig(OUT / "inject.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'inject.png'}")
