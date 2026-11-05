"""Flow log p and flux shape of 39627884985325077 (follow-up 27) on each fibre it was observed on (#17).

QUESTION: is follow-up 27 an outlier only in its SV3 bright spectra (fibres 2676 and 2666 of petal 5,
next to fibre 2675), and not in its main bright (2478), main dark (2995) and SV3 dark (2904) spectra?
Does the flux shape change with fibre?
HYPOTHESIS: none registered.
INPUTS: the five single-exposure tile coadds and redrocks in iron_urls.txt, in
data/iron_tile_spectra/; the SV3 bright healpix coadd row from its DESIchunk1024 pkl. Encoders and
seed 0 flows as 2026-10-31_dark_latent_39633433328091196.
SEED: none.
COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-11-01_fibre_neighbours_39627884985325077/analysis.py``.
RESULT: 2026-11-01, fibre_logp.txt. Only the fibre 2676 exposure (85093) is an outlier: cont10
-19.93, cont15 -35.22 (thresholds -15.72, -25.24). 2666, 2904, 2478 and 2995 give -5.0 to -5.4 and
+0.7 to +15.9. S/N does not explain it (2676 1.91, 2666 1.85). Against main dark, 2676 is +0.21 at
7550-8300 A and near 0 elsewhere red of 5800 A. 2666 has its z arm masked from 7621 A (BADAMPZ).
The stored SV3 bright coadd is -0.23 at 8300-9824 A where 2676 is -0.03, so the healpix coadd z arm
is not 2676 alone (cause not checked).
Assumptions: one galaxy, so fibre is confounded with exposure, sky and tile; encode() takes no
weights, so masked pixels (all of 2666's z arm) enter as zero flux; seed 0 flows; Mac CPU encoding.
FIGURES: results/2026-11-01_fibre_neighbours_39627884985325077/ fibre_logp.txt, fibre_diff.png.
"""

# %%
import csv
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from scipy.ndimage import median_filter
from spender import load_model
from spender.data.desi import DESI

from hubersed.detect.flow_model import build_flow
from hubersed.io.desi import tids_to_indices
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39627884985325077
DATA = PATHS["DATA"]
TILES = DATA / "iron_tile_spectra"
FLOWS = PATHS["RESULTS"] / "noised_cue_meanzero_wide_flow"
OUT = PATHS["RESULTS"] / "2026-11-01_fibre_neighbours_39627884985325077"
OUT.mkdir(parents=True, exist_ok=True)
TAGS = ("cont10", "cont15")
# Same tolerances as 2026-10-31_dark_latent_39633433328091196.
TOL_LATENT = 1e-3
TOL_LOGP = {"cont10": 1e-2, "cont15": 5e-2}
# label, tile, petal, last night
SPECTRA = (
    ("SV3 bright 2676", 285, 5, 20210416),
    ("SV3 bright 2666", 288, 5, 20210502),
    ("SV3 dark 2904", 271, 5, 20210406),
    ("main bright 2478", 20103, 4, 20210612),
    ("main dark 2995", 2694, 5, 20220330),
)
REF = "main dark 2995"
KERNEL = 101  # pixels of the median filter on the flux ratio


def prepare_one(coadd, redrock, tid):
    """Return spec, w, z of one target, with the steps of spender DESI.prepare_spectra.

    The keep cuts (fibre status, target bits, ZWARN) are not applied, so flagged spectra are kept.
    """
    h = fits.open(coadd)
    rr = fits.open(redrock)[1].data
    i = int(np.flatnonzero(h["FIBERMAP"].data["TARGETID"] == tid)[0])
    assert rr["TARGETID"][i] == tid, "redrock row order differs"
    z = torch.tensor(rr["Z"][i : i + 1].astype(np.float32))
    wv, fl, iv, mk = {}, {}, {}, {}
    for b in "brz":
        wv[b] = h[f"{b.upper()}_WAVELENGTH"].data
        fl[b] = h[f"{b.upper()}_FLUX"].data[i]
        iv[b] = h[f"{b.upper()}_IVAR"].data[i]
        mk[b] = h[f"{b.upper()}_MASK"].data[i]
    tol = 0.0001
    wave = wv["b"]
    for b in "brz":
        wave = np.append(wave, wv[b][wv[b] > wave[-1] + tol])
    assert np.abs(wave - DESI._wave_obs.numpy()).max() < tol, "wavelength grid differs"
    win, ncam = {}, np.zeros(wave.size)
    for b in "brz":
        imin = np.argmin(np.abs(wv[b][0] - wave))
        win[b] = np.arange(imin, imin + len(wv[b]))
        assert np.abs(wv[b] - wave[win[b]]).max() < tol
        ncam[win[b]] += 1
    flux = np.zeros(wave.size, fl["b"].dtype)
    ivar = np.zeros(wave.size, iv["b"].dtype)
    mask = np.zeros(wave.size, mk["b"].dtype)
    for b in "brz":
        j = win[b]
        good = mk[b] == 0
        ivar[j] += iv[b] * good
        flux[j] += iv[b] * good * fl[b]
        one = ncam[j] == 1
        mask[j[one]] = mk[b][one]
        many = ncam[j] > 1
        mask[j[many]] = mask[j[many]] & mk[b][many]
    ok = ivar > 0
    flux[ok] /= ivar[ok]
    bad = mask.astype(bool) | (ivar <= 1e-6)
    ivar[bad] = 0
    spec = torch.from_numpy(flux.astype(np.float32))
    w = torch.from_numpy(ivar.astype(np.float32))
    w[DESI._skyline_mask] = 0
    rest = DESI._wave_obs / (1 + z[0])
    sel = (w > 0) & (rest > 5300) & (rest < 5850)
    norm = torch.median(spec[sel]) if sel.count_nonzero() > 0 else torch.tensor(0.0)
    if not torch.isfinite(norm):
        norm = torch.tensor(0.0)
    else:
        spec /= norm
    w *= norm**2
    return spec.numpy(), w.numpy(), float(z[0])


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


# %% check prepare_one against spender on a healpix coadd it can read
hp = DATA / "sv3_dark_coadds" / "coadd-sv3-dark-10231.fits"
sp, wp, _, tp, _, _ = DESI.prepare_spectra(str(hp))
d_spec = d_w = 0.0
for k in range(len(tp)):
    s1, w1, _ = prepare_one(
        hp, DATA / "sv3_dark_redrocks" / "redrock-sv3-dark-10231.fits", int(tp[k])
    )
    d_spec = max(d_spec, float(np.abs(s1 - sp[k].numpy()).max()))
    d_w = max(d_w, float(np.abs(w1 - wp[k].numpy()).max()))
print(f"prepare_one vs prepare_spectra on {len(tp)} spectra: max |d spec| {d_spec}, |d w| {d_w}")
assert d_spec == 0 and d_w == 0, "prepare_one does not reproduce spender"

# %% the spectra
tiles = {int(r["TILEID"]): r for r in csv.DictReader(open(DATA / "iron_tiles" / "tiles-iron.csv"))}
gidx = int(tids_to_indices(np.array([TID], dtype=np.int64))[0])
with open(DATA / "desi_spectra" / f"DESIchunk1024_{gidx // 1024}.pkl", "rb") as f:
    s, w, z, tid, *_ = pickle.load(f)
row = gidx % 1024
assert int(tid[row]) == TID, "wrong chunk row"
spectra = {
    "SV3 bright coadd": {"spec": np.asarray(s[row], np.float32), "w": np.asarray(w[row]),
                         "z": float(z[row]), "info": "healpix 26068, 2 exposures, stored"},
}  # fmt: skip
for label, tile, petal, night in SPECTRA:
    coadd = TILES / f"coadd-{petal}-{tile}-thru{night}.fits"
    spec, wt, zz = prepare_one(coadd, TILES / f"redrock-{petal}-{tile}-thru{night}.fits", TID)
    h = fits.open(coadd)
    fm = h["FIBERMAP"].data
    i = int(np.flatnonzero(fm["TARGETID"] == TID)[0])
    ef = h["EXP_FIBERMAP"].data
    expid = ef["EXPID"][ef["TARGETID"] == TID]
    assert len(expid) == 1, "tile coadd has more than one exposure of the target"
    zwarn = fits.open(TILES / f"redrock-{petal}-{tile}-thru{night}.fits")[1].data["ZWARN"][i]
    info = (f"tile {tile} expid {expid[0]} night {night} petal {petal} fibre {fm['FIBER'][i]} "
            f"COADD_FIBERSTATUS {fm['COADD_FIBERSTATUS'][i]} ZWARN {zwarn} "
            f"tile EFFTIME_SPEC {float(tiles[tile]['EFFTIME_SPEC']):.0f} s")  # fmt: skip
    spectra[label] = {"spec": spec, "w": wt, "z": zz, "info": info}

# %% encode and score; check the stored SV3 bright coadd against the stored run
inst = DESI()
res = {}
for tag in TAGS:
    nde, mean, scale, enc = load_flow(tag)
    model = load_model(
        str(DATA / "checkpoints" / enc), inst, map_location="cpu", weights_only=False
    )
    model = model.float().eval()
    with torch.no_grad():
        lat = {k: model.encode(torch.from_numpy(d["spec"][None])).numpy()[0]
               for k, d in spectra.items()}  # fmt: skip
    lp = {k: float(log_p(nde, mean, scale, v)[0]) for k, v in lat.items()}
    o = torch.load(FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt", weights_only=False,
                   map_location="cpu")  # fmt: skip
    desi_lp = np.asarray(o["log_p_desi"], float)
    j = int(np.flatnonzero(np.asarray(o["desi_target_ids"]) == TID)[0])
    with h5py.File(DATA / "latents" / f"spender_spec_{tag}latent_snr3.h5", "r") as f:
        k = int(np.flatnonzero(f["target_ids"][:] == TID)[0])
        stored_lat = f["latents"][k]
    d_lat = float(np.abs(lat["SV3 bright coadd"] - stored_lat).max())
    d_lp = abs(lp["SV3 bright coadd"] - desi_lp[j])
    print(f"{tag}: stored coadd max |d latent| {d_lat:.2e}, |d log p| {d_lp:.2e}")
    assert d_lat < TOL_LATENT and d_lp < TOL_LOGP[tag], f"{tag}: coadd does not reproduce the run"
    res[tag] = {"lp": lp, "desi_lp": desi_lp, "thr": float(o["threshold"]), "d_lat": d_lat,
                "d_lp": d_lp}  # fmt: skip

# %% table
lines = [
    f"TARGETID {TID} (follow-up 27). Seed 0 flows. pct = percent of the "
    f"{len(res['cont10']['desi_lp'])} DESI log p at or below the value. good = fraction of "
    "pixels with w > 0 (skyline mask included).",
    "",
]
for k, d in spectra.items():
    lines.append(f"{k}: {d['info']}; z {d['z']:.5f}")
lines.append("")
head = f"{'spectrum':>18} {'S/N':>6} {'good':>5}"
for tag in TAGS:
    head += f" {tag + ' log p':>13} {'pct':>7} {'flag':>4}"
lines.append(head + f"   thresholds {res['cont10']['thr']:.3f} / {res['cont15']['thr']:.3f}")
for k, d in spectra.items():
    line = f"{k:>18} {median_snr(d['spec'], d['w']):6.2f} {np.mean(d['w'] > 0):5.3f}"
    for tag in TAGS:
        r = res[tag]
        v = r["lp"][k]
        line += f" {v:13.3f} {100 * np.mean(r['desi_lp'] <= v):7.3f} {'yes' if v <= r['thr'] else 'no':>4}"
    lines.append(line)
lines.append("")
lines.append(f"Check: prepare_one equals prepare_spectra on {len(tp)} spectra of {hp.name} "
             f"(max |d spec| {d_spec}, |d w| {d_w}).")  # fmt: skip
for tag, r in res.items():
    lines.append(
        f"Check: {tag} stored coadd max |d latent| {r['d_lat']:.2e}, |d log p| {r['d_lp']:.2e}"
    )
text = "\n".join(lines)
(OUT / "fibre_logp.txt").write_text(text + "\n")
print(text)

# %% median difference from the main dark spectrum in observed windows, and masked z pixels
WINDOWS = ((3600, 5800), (5800, 7550), (7550, 8300), (8300, 9824))
zarm = DESI._wave_obs.numpy() > 7550
lines = [f"Median of spectrum - {REF} (normalised flux) over pixels good in both:"]
for k, d in spectra.items():
    if k == REF:
        continue
    good = (d["w"] > 0) & (spectra[REF]["w"] > 0)
    parts = []
    for lo, hi in WINDOWS:
        m = good & (DESI._wave_obs.numpy() >= lo) & (DESI._wave_obs.numpy() < hi)
        parts.append(f"{lo}-{hi} {np.median(d['spec'][m] - spectra[REF]['spec'][m]):+.3f}")
    bad = zarm & (d["w"] == 0) & (spectra[REF]["w"] > 0)
    rng = (
        f"{DESI._wave_obs.numpy()[bad].min():.0f}-{DESI._wave_obs.numpy()[bad].max():.0f}"
        if bad.any()
        else "none"
    )
    lines.append(
        f"{k:>18}: " + ", ".join(parts) + f"; z pixels masked here only: {bad.sum()} ({rng} A)"
    )
text = "\n".join(lines)
with open(OUT / "fibre_logp.txt", "a") as f:
    f.write("\n" + text + "\n")
print(text)

# %% figure: smoothed spectra and their difference from the main dark spectrum
use_apj_style()
wave = DESI._wave_obs.numpy()
ref = spectra[REF]
colours = {"SV3 bright coadd": "k", "SV3 bright 2676": "#b2182b", "SV3 bright 2666": "#ef8a62",
           "SV3 dark 2904": "#2166ac", "main bright 2478": "#1b7837", REF: "0.5"}  # fmt: skip


def smooth(y, good, n):
    """Return the n px median of y over good pixels, NaN elsewhere so masked gaps stay empty."""
    out = np.full(y.shape, np.nan)
    out[good] = median_filter(y[good], n, mode="nearest")
    return out


fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for k, d in spectra.items():
    lab = f"{k} (cont10 {res['cont10']['lp'][k]:.1f}, cont15 {res['cont15']['lp'][k]:.1f})"
    axes[0].plot(wave, smooth(d["spec"], d["w"] > 0, KERNEL // 4), color=colours[k], lw=0.8,
                 label=lab)  # fmt: skip
    if k == REF:
        continue
    good = (d["w"] > 0) & (ref["w"] > 0)
    axes[1].plot(wave, smooth(d["spec"] - ref["spec"], good, KERNEL), color=colours[k], lw=1.2)
axes[1].axhline(0, color="0.5", ls="--", lw=1)
axes[0].set_ylabel(f"normalised flux, {KERNEL // 4} px median", fontsize=14)
axes[1].set_ylabel(f"minus {REF}, {KERNEL} px median", fontsize=14)
axes[1].set_xlabel(r"observed wavelength [\AA]", fontsize=14)
axes[0].legend(frameon=False, fontsize=10)
fig.savefig(OUT / "fibre_diff.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'fibre_diff.png'}")
