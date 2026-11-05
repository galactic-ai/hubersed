"""Rescore the sky_targets.txt outliers with their bright-sky exposures left out (#17).

QUESTION: which screened outliers stop being outliers when exposures with sky g < 21 are dropped?
HYPOTHESIS: none registered.
INPUTS: results/2026-10-31_screen_all/exposures/<TARGETID>.npz (sky_extract.py), exposure sky from
the QA experiment's sv3_bright_exposures.csv, seed 0 flows and encoders. The coadd follows
spender's steps as 2026-10-31_logp_vs_colour/rule.py; the all-exposure coadd is the check.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_screen_all/sky_rescore.py``.
RESULT: 2026-10-31. Of 20, 12 are no longer common outliers (5 clear in both flows, 7 in one,
some at the threshold), 7 stay, 1 has no clean exposure. All-exposure coadds match the stored log
p within 1.7.
FIGURES: results/2026-10-31_screen_all/sky_rescore.txt.
"""

# %%
import csv
from pathlib import Path

import numpy as np
import torch
from spender import load_model
from spender.data.desi import DESI

from hubersed.detect.flow_model import build_flow
from hubersed.io.desi import load_by_index, tids_to_indices
from hubersed.paths import PATHS

HERE = Path(__file__).resolve().parent
RES = PATHS["RESULTS"]
OUT = RES / "2026-10-31_screen_all"
FLOWS = RES / "noised_cue_meanzero_wide_flow"
SKY = 21.0
TAGS = ("cont10", "cont15")
wave = DESI._wave_obs.numpy()
skyline = DESI._skyline_mask.numpy()
sky = {}
with open(
    RES / "2026-10-31_spectrum_qa_cont_outliers" / "sv3_bright_exposures.csv", newline=""
) as fh:
    for r in csv.DictReader(fh):
        sky[int(r["expid"])] = float(r["sky_mag_g_spec"])


def coadd(d, keep, z):
    """Return the spender-normalised spectrum of the kept exposure rows, or None."""
    if not keep.any():
        return None
    num, den = np.zeros(len(wave)), np.zeros(len(wave))
    for arm in "BRZ":
        wv = d[f"{arm}_wave"]
        i0 = int(np.argmin(np.abs(wave - wv[0])))
        iv = np.where(d[f"{arm}_mask"][keep] == 0, d[f"{arm}_ivar"][keep], 0.0)
        num[i0 : i0 + len(wv)] += (d[f"{arm}_flux"][keep] * iv).sum(0)
        den[i0 : i0 + len(wv)] += iv.sum(0)
    flux = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    w = np.where(den > 1e-6, den, 0.0)
    w[skyline] = 0
    rest = wave / (1 + z)
    sel = (w > 0) & (rest > 5300) & (rest < 5850)
    return (flux / np.median(flux[sel])).astype(np.float32)


inst = DESI()
flows = {}
for tag in TAGS:
    dd = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu")
    nde = build_flow(dd["method"], dd["dim"], dd["hidden"], dd["num_transforms"], dd["num_bins"])
    nde.load_state_dict(dd["state_dict"])
    nde.eval()
    enc = load_model(str(PATHS["DATA"] / "checkpoints" / dd["encoder"]), inst, map_location="cpu",
                     weights_only=False).float()  # fmt: skip
    enc.eval()
    o = torch.load(RES / "wide_flow_corrected" / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt",
                   weights_only=False, map_location="cpu")  # fmt: skip
    stored = dict(zip(np.asarray(o["desi_target_ids"]).tolist(),
                      np.asarray(o["log_p_desi"]).tolist(), strict=True))  # fmt: skip
    flows[tag] = (nde, enc, dd["scaler_mean"], dd["scaler_scale"], float(o["threshold"]), stored)


def score(tag, x):
    """Return the flow log p of one normalised spectrum."""
    nde, enc, mean, scale, *_ = flows[tag]
    with torch.no_grad():
        lat = enc.encode(torch.from_numpy(x[None])).numpy()
        return float(nde.log_prob(torch.from_numpy(((lat - mean) / scale).astype(np.float32)))[0])


# %% per target
lines = [f"Exposures with sky g >= {SKY} kept. log p: stored / all-exposure coadd / clean coadd. "
         "Thresholds cont10 -15.72, cont15 -25.24.",  # fmt: skip
         f"{'TARGETID':>18} {'kept':>5}  {'cont10':>26}  {'cont15':>26}  status"]  # fmt: skip
status = {}
for line in (HERE / "sky_targets.txt").read_text().splitlines():
    t = int(line.split()[0])
    d = dict(np.load(OUT / "exposures" / f"{t}.npz"))
    z = float(load_by_index(int(tids_to_indices(np.array([t]))[0]))[2])
    sk = np.array([sky.get(int(e), np.nan) for e in d["EXPID"]])
    allx, clean = coadd(d, np.ones(len(sk), bool), z), coadd(d, sk >= SKY, z)
    cells, still = [], []
    for tag in TAGS:
        thr, stored = flows[tag][4], flows[tag][5][t]
        a = score(tag, allx)
        c = score(tag, clean) if clean is not None else np.nan
        cells.append(f"{stored:8.2f} {a:8.2f} {c:8.2f}")
        still.append(c <= thr)
    status[t] = ("no clean exposure" if clean is None
                 else "still common outlier" if all(still) else "cleared")  # fmt: skip
    lines.append(
        f"{t:>18} {int((sk >= SKY).sum()):>2}/{len(sk)}  {cells[0]}  {cells[1]}  {status[t]}"
    )
lines.append("")
for s in ("cleared", "still common outlier", "no clean exposure"):
    lines.append(f"{s}: {sum(v == s for v in status.values())}")
text = "\n".join(lines)
(OUT / "sky_rescore.txt").write_text(text + "\n")
print(text)
