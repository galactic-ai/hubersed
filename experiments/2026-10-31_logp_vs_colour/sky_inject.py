"""Add a sky-like blue excess to normal spectra and re-score them with both flows (#17).

The excess is the smoothed SV3 bright minus scaled SV3 dark spectrum of 39633433328091196 (dark
scaled to the bright median redward of 5500 A). Galaxies are those of inject.py. Each galaxy gets
the excess scaled to give a target dgr through the BASS g and r curves, then spender's
normalisation, encoding and scoring.
COMMAND: ``uv run --no-sync python -u experiments/2026-10-31_logp_vs_colour/sky_inject.py``.
OUTPUT: results/2026-10-31_logp_vs_colour/sky_inject.txt and sky_inject.png.
RESULT: 2026-10-31. At injected dgr -0.1 / -0.2 / -0.3, 8.8 / 42.9 / 62.7% (cont10) and
5.4 / 52.0 / 63.3% (cont15) of 1000 normal galaxies are flagged, above the observed rates.
"""

# %%
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table
from scipy.signal import medfilt
from spender import load_model
from spender.data.desi import DESI

from hubersed.detect.flow_model import build_flow
from hubersed.io.desi import load_by_index, tids_to_indices
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

RES = PATHS["RESULTS"]
OUT = RES / "2026-10-31_logp_vs_colour"
FLOWS = RES / "noised_cue_meanzero_wide_flow"
FILT = PATHS["ROOT"] / "experiments" / "2026-10-30_photcolour_dark_39633433328091196" / "filters"
TAGS = ("cont10", "cont15")
TARGETS = (0.0, -0.05, -0.1, -0.2, -0.3, -0.4, -0.5)  # injected dgr
N_CHUNKS, N_MAX, SMOOTH = 25, 1000, 201
TID = 39633433328091196
rng = np.random.default_rng(0)
wave = DESI._wave_obs.numpy().astype(float)

# %% the excess template, observed frame
fb, wb, *_ = load_by_index(int(tids_to_indices(np.array([TID]))[0]))
fb, wb = np.asarray(fb, float), np.asarray(wb, float)
spec, w, z, tid, norm, _ = DESI.prepare_spectra(
    str(PATHS["DATA"] / "sv3_dark_coadds" / "coadd-sv3-dark-10231.fits")
)
i = int(np.flatnonzero(tid.numpy() == TID)[0])
fd, wd = spec[i].numpy() * float(norm[i]), w[i].numpy() / float(norm[i]) ** 2
good = (wb > 0) & (wd > 0)
scale = np.median(fd[good & (wave > 5500)] / fb[good & (wave > 5500)])
diff = fb - fd / scale
excess = np.interp(wave, wave[good], medfilt(diff[good], SMOOTH))
print(f"dark/bright redward of 5500 A: {scale:.3f}")

# %% galaxies, as inject.py
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
print(f"{len(tsel)} galaxies from {N_CHUNKS} chunks", flush=True)

# %% scale the excess per galaxy to a target dgr, then re-normalise
band = {}
for b, name in (("g", "BASS-g"), ("r", "BASS-r")):
    tab = Table.read(FILT / f"{name}.ecsv")
    band[b] = np.interp(wave, np.asarray(tab["wavelength"], float), np.asarray(tab["response"]),
                        left=0, right=0) * wave  # fmt: skip
gal = np.where(w > 0, spec, 0.0)
a = (excess @ band["g"]) / (gal @ band["g"])
b = (excess @ band["r"]) / (gal @ band["r"])


def injected(target):
    """Return the spectra with the excess added for a target dgr, re-normalised as spender does."""
    r = 10 ** (-0.4 * target)
    k = (r - 1) / (a - r * b)  # (1 + k a) / (1 + k b) = r
    out = spec + k[:, None] * excess[None, :]
    for j in range(len(out)):
        rest = wave / (1 + z[j])
        sel = (w[j] > 0) & (rest > 5300) & (rest < 5850)
        out[j] /= np.median(out[j][sel])
    return out.astype(np.float32), k


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
    for tgt in TARGETS:
        x, k = injected(tgt)
        with torch.no_grad():
            lat = np.concatenate([model.encode(torch.from_numpy(x[i : i + 256])).numpy()
                                  for i in range(0, len(x), 256)])  # fmt: skip
            xs = ((lat - d["scaler_mean"]) / d["scaler_scale"]).astype(np.float32)
            res[(tag, tgt)] = nde.log_prob(torch.from_numpy(xs)).numpy()
        print(f"{tag} dgr {tgt:+.2f} done, {time.time() - t0:.0f} s", flush=True)
    stored = np.array([store[tag]["lp"][t] for t in tsel.tolist()])
    print(f"{tag} check: no excess against stored log p, max |d| "
          f"{np.abs(res[(tag, 0.0)] - stored).max():.3g}")  # fmt: skip

# %% table and figure
lines = [
    f"{len(tsel)} galaxies, |dgr| < 0.05 and unflagged; excess of {TID}, smoothed {SMOOTH} px.",
    f"{'inj dgr':>8}  " + "  ".join(f"{t}: flagged %, median d log p" for t in TAGS),
]
for tgt in TARGETS:
    cells = [f"{100 * np.mean(res[(t, tgt)] <= store[t]['thr']):6.2f} "
             f"{np.median(res[(t, tgt)] - res[(t, 0.0)]):+8.2f}" for t in TAGS]  # fmt: skip
    lines.append(f"{tgt:+8.2f}  " + "  ".join(cells))
text = "\n".join(lines)
(OUT / "sky_inject.txt").write_text(text + "\n")
print(text)

obs = np.load(OUT / "binned.npz")
mid = 0.5 * (np.clip(obs["edges"][:-1], -1, 0.5) + np.clip(obs["edges"][1:], -1, 0.5))
use_apj_style()
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].plot(wave, diff, color="0.8", lw=0.4, label="bright $-$ scaled dark")
axes[0].plot(wave, excess, color="#b2182b", lw=1.5, label=f"median filter {SMOOTH} px")
axes[0].set_xlabel(r"observed wavelength [\AA]", fontsize=13)
axes[0].set_ylabel(
    r"excess $f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]", fontsize=13
)
axes[0].legend(frameon=False, fontsize=11)
for ax, t in zip(axes[1:], TAGS, strict=True):
    ax.errorbar(mid, 100 * obs[f"{t}_flagged"], yerr=100 * obs[f"{t}_err"], fmt="o-", color="0.6",
                ms=3, label="observed, all DESI")  # fmt: skip
    fl = [100 * np.mean(res[(t, g)] <= store[t]["thr"]) for g in TARGETS]
    ax.plot(TARGETS, fl, "s-", color="#b2182b", label="injected sky-like excess")
    ax.set_xlabel(r"$\Delta(g-r)$ [mag]", fontsize=13)
    ax.set_ylabel(f"flagged by {t}latent flow [\\%]", fontsize=13)
axes[1].legend(frameon=False, fontsize=11, loc="upper right")
fig.savefig(OUT / "sky_inject.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'sky_inject.png'}")
