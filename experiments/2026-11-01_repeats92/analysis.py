"""Are the 92 canonical continuum outliers still outliers in their other DESI DR1 spectra (#17)?

QUESTION: for each of the 92 (cont of results/prescreened_outliers_postsimbad.npz), do its other DR1
coadds (main bright, main dark, SV3 dark, SV1) score as outliers in the seed 0 cont10 and cont15
flows, and how do they differ from its SV3 bright coadd?
HYPOTHESIS: none registered. An outlier set by one bad exposure, fibre or sky should not repeat;
one set by the galaxy (broad lines, real continuum) should.
INPUTS: results/2026-11-01_repeats92/sparcl_spectra.npz (fetch.py), the DESIchunk1024 pkl spectra
for the check, encoders and seed 0 flows as 2026-10-31_dark_latent_39633433328091196. Each SPARCL
spectrum goes through spender's steps after the arm merge (mask or ivar <= 1e-6 to zero weight,
skyline mask, median at rest 5300-5850 A) with its own redshift.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-01_repeats92/analysis.py``.
RESULT: pending.
FIGURES: results/2026-11-01_repeats92/ repeats.txt, check.txt and pairs/<NN>_<TARGETID>.png (flux, not
normalised) for the targets with another spectrum.
"""

# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import medfilt
from spender import load_model
from spender.data.desi import DESI

from hubersed.detect.flow_model import build_flow
from hubersed.io.desi import tids_to_indices
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

DATA = PATHS["DATA"]
RES = PATHS["RESULTS"]
OUT = RES / "2026-11-01_repeats92"
FLOWS = RES / "noised_cue_meanzero_wide_flow"
TAGS = ("cont10", "cont15")
N_MEDIAN = 9
# Same tolerances as 2026-10-31_dark_latent_39633433328091196 for the stored log p.
TOL_LOGP = {"cont10": 1e-2, "cont15": 5e-2}
COLOURS = {("main", "bright"): "#2166ac", ("main", "dark"): "#1b7837", ("sv3", "dark"): "#762a83",
           ("sv1", "bright"): "#e66101", ("sv1", "other"): "#e66101"}  # fmt: skip


def prepare(flux, ivar, mask, z):
    """Return spender's normalised spectrum and weights from one arm-merged DESI spectrum."""
    ivar = np.asarray(ivar, float).copy()
    bad = (np.asarray(mask) != 0) | (ivar <= 1e-6)
    ivar[bad] = 0
    spec = torch.from_numpy(np.asarray(flux, float).astype(np.float32))
    w = torch.from_numpy(ivar.astype(np.float32))
    w[DESI._skyline_mask] = 0
    rest = DESI._wave_obs / (1 + torch.tensor(np.float32(z)))
    sel = (w > 0) & (rest > 5300) & (rest < 5850)
    norm = torch.median(spec[sel]) if sel.count_nonzero() > 0 else torch.tensor(0.0)
    if not torch.isfinite(norm):
        norm = torch.tensor(0.0)
    else:
        spec /= norm
    w *= norm**2
    return spec.numpy(), w.numpy()


def load_flow(tag):
    """Return the seed 0 flow of a tag in eval mode, with its scaler mean and scale."""
    d = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu")
    nde = build_flow(d["method"], d["dim"], d["hidden"], d["num_transforms"], d["num_bins"])
    nde.load_state_dict(d["state_dict"])
    nde.eval()
    return nde, np.asarray(d["scaler_mean"]), np.asarray(d["scaler_scale"]), d["encoder"]


# %% spectra: SPARCL records through spender's steps, and the stored SV3 bright spectra
sp = np.load(OUT / "sparcl_spectra.npz", allow_pickle=True)
assert np.allclose(sp["wavelength"], DESI._wave_obs.numpy()), "SPARCL grid differs from spender's"
n = len(sp["targetid"])
key = [(str(s), str(p)) for s, p in zip(sp["survey"], sp["program"], strict=True)]
spec, w = np.zeros((n, len(sp["wavelength"])), np.float32), np.zeros_like(sp["flux"], np.float32)
for i in range(n):
    spec[i], w[i] = prepare(sp["flux"][i], sp["ivar"][i], sp["mask"][i], sp["redshift"][i])
tids = [int(t) for t in np.load(RES / "prescreened_outliers_postsimbad.npz")["cont"]]
stored = {}
for t in tids:
    g = int(tids_to_indices(np.array([t], dtype=np.int64))[0])
    with open(DATA / "desi_spectra" / f"DESIchunk1024_{g // 1024}.pkl", "rb") as f:
        s, ww, _, tt, *_ = pickle.load(f)
    assert int(tt[g % 1024]) == t, "wrong chunk row"
    stored[t] = (np.asarray(s[g % 1024], np.float32), np.asarray(ww[g % 1024], np.float32))

# %% encode and score every spectrum, and the stored SV3 bright spectra for the check
inst = DESI()
lp, thr, desi_lp, lp_stored, lp_run = {}, {}, {}, {}, {}
for tag in TAGS:
    nde, mean, scale, enc = load_flow(tag)
    model = load_model(
        str(DATA / "checkpoints" / enc), inst, map_location="cpu", weights_only=False
    )
    model = model.float().eval()

    def score(x, nde=nde, mean=mean, scale=scale, model=model):
        """Return the flow log p of normalised spectra ``x``."""
        with torch.no_grad():
            lat = model.encode(torch.from_numpy(np.atleast_2d(x))).numpy()
            z = torch.from_numpy(((lat - mean) / scale).astype(np.float32))
            return nde.log_prob(z).numpy()

    lp[tag] = score(spec)
    lp_stored[tag] = dict(zip(tids, score(np.array([stored[t][0] for t in tids])), strict=True))
    o = torch.load(FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt", weights_only=False,
                   map_location="cpu")  # fmt: skip
    thr[tag], desi_lp[tag] = float(o["threshold"]), np.asarray(o["log_p_desi"], float)
    lp_run[tag] = dict(zip(np.asarray(o["desi_target_ids"]).tolist(), desi_lp[tag], strict=True))

# %% check: the stored spectra reproduce the run, and SPARCL SV3 bright matches the stored spectra
lines = []
for tag in TAGS:
    d = [abs(lp_stored[tag][t] - lp_run[tag][t]) for t in tids]
    lines.append(f"{tag}: stored spectra re-encoded, max |log p - run| {max(d):.2e} over {len(d)}")
    assert max(d) < TOL_LOGP[tag], f"{tag}: stored spectra do not reproduce the run"
i_sv3 = {int(sp["targetid"][i]): i for i in range(n) if key[i] == ("sv3", "bright")}
assert len(i_sv3) == 92
d_spec = []
for t, i in i_sv3.items():
    s0, w0 = stored[t]
    good = (w0 > 0) & (w[i] > 0)
    d_spec.append((np.median(np.abs(spec[i][good] - s0[good])), np.mean((w0 > 0) != (w[i] > 0))))
d_spec = np.array(d_spec)
lines.append(f"SPARCL SV3 bright vs stored spender spectrum, over 92: median |d spec| on common "
             f"good pixels max {d_spec[:, 0].max():.2e}; fraction of pixels whose good/bad differs "
             f"max {d_spec[:, 1].max():.4f}")  # fmt: skip
for tag in TAGS:
    d = np.array([lp[tag][i] - lp_run[tag][t] for t, i in i_sv3.items()])
    lines.append(f"{tag}: SPARCL SV3 bright log p - run log p: median {np.median(d):+.3f}, "
                 f"max |.| {np.abs(d).max():.3f}")  # fmt: skip
text = "\n".join(lines)
(OUT / "check.txt").write_text(text + "\n")
print(text)

# %% table: every spectrum of each target, its log p, percentile and flag
lines = [
    f"Seed 0 flows, thresholds cont10 {thr['cont10']:.3f}, cont15 {thr['cont15']:.3f}. pct = percent "
    "of the 114693 DESI log p at or below. flag: yes when log p <= threshold. NE coadd_numexp.",
    "",
    f"{'#':>3} {'TARGETID':>17} {'spectrum':>11} {'z':>7} {'NE':>3} {'TSNR2_BGS':>9} "
    f"{'cont10':>8} {'pct':>6} {'flag':>4} {'cont15':>8} {'pct':>6} {'flag':>4}",
]
summary = {"repeat targets": 0, "repeat spectra": 0, "flag both": 0, "flag one": 0, "flag none": 0,
           "z differs": 0}  # fmt: skip
ZDIFF = 0.01  # |z_repeat - z_SV3| / (1 + z_SV3) above this: the repeat's score is not comparable
pairs, skipped = [], []
for k, t in enumerate(tids, 1):
    # other spectra spender would keep (ZWARN == 0, COADD_FIBERSTATUS == 0); the rest are listed
    others = [i for i in range(n) if int(sp["targetid"][i]) == t and i != i_sv3[t]]
    ok = [i for i in others
          if int(sp["redshift_warning"][i]) == 0 and int(sp["coadd_fiberstatus"][i]) == 0]  # fmt: skip
    skipped += [(t, "/".join(key[i]), int(sp["redshift_warning"][i]),
                 int(sp["coadd_fiberstatus"][i])) for i in others if i not in ok]  # fmt: skip
    rows = [i_sv3[t]] + ok
    if len(rows) > 1:
        summary["repeat targets"] += 1
        pairs.append((k, t, rows))
    for i in rows:
        cells = []
        for tag in TAGS:
            v = lp[tag][i]
            pct = 100 * np.mean(desi_lp[tag] <= v)
            cells.append(f"{v:8.2f} {pct:6.2f} {'yes' if v <= thr[tag] else 'no':>4}")
        lines.append(f"{k:3d} {t:17d} {'/'.join(key[i]):>11} {sp['redshift'][i]:7.4f} "
                     f"{int(sp['coadd_numexp'][i]):3d} {float(sp['tsnr2_bgs'][i]):9.1f} "
                     + " ".join(cells))  # fmt: skip
        if i != i_sv3[t]:
            summary["repeat spectra"] += 1
            z0 = float(sp["redshift"][i_sv3[t]])
            dz = abs(float(sp["redshift"][i]) - z0) / (1 + z0)
            if dz > ZDIFF or not all(np.isfinite(lp[tag][i]) for tag in TAGS):
                summary["z differs"] += 1
                continue
            nflag = sum(lp[tag][i] <= thr[tag] for tag in TAGS)
            summary[{2: "flag both", 1: "flag one", 0: "flag none"}[nflag]] += 1
lines += [
    "",
    "Other spectra (not SV3 bright): " + ", ".join(f"{k} {v}" for k, v in summary.items()),
]
lines += [f"Not scored, ZWARN or COADD_FIBERSTATUS set (target, spectrum, ZWARN, CFS): {s}"
          for s in skipped]  # fmt: skip
text = "\n".join(lines)
(OUT / "repeats.txt").write_text(text + "\n")
print(lines[-1])

# %% figures: SV3 bright against each other spectrum of the same target, in flux (not normalised)
use_apj_style()
wave = DESI._wave_obs.numpy()
(OUT / "pairs").mkdir(exist_ok=True)
flux = np.asarray(sp["flux"], float)
fgood = (np.asarray(sp["mask"]) == 0) & (np.asarray(sp["ivar"], float) > 0)
for k, t, rows in pairs:
    i0 = rows[0]
    z = float(sp["redshift"][i0])
    fig, ax = plt.subplots(
        2, 1, figsize=(16, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}
    )
    g0 = fgood[i0]
    ax[0].plot(wave[g0] / (1 + z), flux[i0][g0], color="0.6", lw=0.5)
    lab = f"SV3 bright (cont10 {lp['cont10'][i0]:.1f}, cont15 {lp['cont15'][i0]:.1f})"
    ax[0].plot(wave[g0] / (1 + z), medfilt(flux[i0][g0], N_MEDIAN), color="k", lw=0.6, label=lab)
    for i in rows[1:]:
        g = fgood[i]
        c = COLOURS.get(key[i], "#b2182b")
        lab = f"{' '.join(key[i])} (cont10 {lp['cont10'][i]:.1f}, cont15 {lp['cont15'][i]:.1f})"
        ax[0].plot(wave[g] / (1 + z), medfilt(flux[i][g], N_MEDIAN), color=c, lw=0.6, label=lab)
        both = g & g0
        diff = np.full(len(wave), np.nan)
        diff[both] = medfilt(flux[i][both] - flux[i0][both], N_MEDIAN)
        ax[1].plot(wave / (1 + z), diff, color=c, lw=0.6)
    ax[1].axhline(0, color="0.3", lw=0.6)
    ax[0].set_title(f"{k}/92   {t}   z {z:.3f}", fontsize=13)
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]", fontsize=13)
    ax[1].set_ylabel("minus SV3 bright", fontsize=13)
    ax[1].set_xlabel(r"rest wavelength [\AA]", fontsize=13)
    ax[0].legend(frameon=False, fontsize=11, loc="upper right")
    fig.savefig(OUT / "pairs" / f"{k:02d}_{t}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
print(f"wrote {len(pairs)} figures to {OUT / 'pairs'}")
