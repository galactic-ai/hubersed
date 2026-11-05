"""Flow log p of the PR 2377 recoadd of the first #17 outlier 39627757533007793.

The recoadd (2026-10-29_continuum_recoadd, resolution_matrix_rows_ivar.npz, SV3) goes through
spender's steps as in rule.py: arms joined ivar-weighted on spender's grid, sky-line pixels at zero
weight, median normalised at rest 5300-5850 A. The stored DR1 spender spectrum is scored too, as
the check, and the flux change between the two is reported.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_logp_vs_colour/recoadd_score.py``.
OUTPUT: results/2026-10-31_logp_vs_colour/recoadd_score.txt.
RESULT: 2026-10-31. Normalised flux barely changes (99th percentile 1.2%); log p -21.56 to -21.48
(cont10) and -28.10 to -27.98 (cont15), still flagged. The DR1 ivar bug does not set its flag.
"""

import pickle

import numpy as np
import torch
from spender import load_model
from spender.data.desi import DESI

from hubersed.detect.flow_model import build_flow
from hubersed.io.desi import tids_to_indices
from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-10-31_logp_vs_colour"
FLOWS = RES / "noised_cue_meanzero_wide_flow"
TID = 39627757533007793
wave = DESI._wave_obs.numpy()

# stored DR1 spender spectrum, normalised
g = int(tids_to_indices(np.array([TID]))[0])
with open(PATHS["DATA"] / "desi_spectra" / f"DESIchunk1024_{g // 1024}.pkl", "rb") as f:
    s, w, z, tid, *_ = pickle.load(f)
assert int(tid[g % 1024]) == TID
dr1, w_dr1, z = np.asarray(s[g % 1024], np.float32), np.asarray(w[g % 1024]), float(z[g % 1024])

# recoadd through spender's steps
d = np.load(RES / "2026-10-29_continuum_recoadd" / "sv3" / "resolution_matrix_rows_ivar.npz")
assert int(d["targetid"]) == TID
num, den = np.zeros(len(wave)), np.zeros(len(wave))
for arm in "brz":
    wv = d[f"{arm}_wave"]
    i0 = int(np.argmin(np.abs(wave - wv[0])))
    iv = np.where(d[f"{arm}_mask"] == 0, d[f"{arm}_ivar"], 0.0)
    num[i0 : i0 + len(wv)] += d[f"{arm}_flux"] * iv
    den[i0 : i0 + len(wv)] += iv
flux = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
w_re = np.where(den > 1e-6, den, 0.0)
w_re[DESI._skyline_mask.numpy()] = 0
rest = wave / (1 + z)
sel = (w_re > 0) & (rest > 5300) & (rest < 5850)
recoadd = (flux / np.median(flux[sel])).astype(np.float32)

good = (w_dr1 > 0) & (w_re > 0)
rel = np.abs(recoadd[good] - dr1[good]) / np.maximum(np.abs(dr1[good]), 1e-3)
lines = [
    f"{TID}: normalised flux, recoadd against DR1 on {int(good.sum())} common good pixels: "
    f"median |relative difference| {np.median(rel):.4f}, 99th percentile {np.percentile(rel, 99):.4f}.",
    "ivar ratio recoadd / DR1 (median per arm, observed < 5800 / 5800-7600 / > 7600 A): "
    + " / ".join(
        f"{np.median(w_re[good & (wave >= lo) & (wave < hi)] / w_dr1[good & (wave >= lo) & (wave < hi)]):.3f}"
        for lo, hi in ((0, 5800), (5800, 7600), (7600, 1e5))
    )
    + " (DR1 w is ivar times norm^2, so only the ratio between arms is meaningful).",
    "",
    f"{'flow':>6} {'threshold':>9} {'stored':>8} {'DR1 here':>8} {'recoadd':>8}",
]
inst = DESI()
for tag in ("cont10", "cont15"):
    dd = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu")
    nde = build_flow(dd["method"], dd["dim"], dd["hidden"], dd["num_transforms"], dd["num_bins"])
    nde.load_state_dict(dd["state_dict"])
    nde.eval()
    enc = load_model(
        str(PATHS["DATA"] / "checkpoints" / dd["encoder"]),
        inst,
        map_location="cpu",
        weights_only=False,
    ).float()
    enc.eval()
    o = torch.load(
        RES / "wide_flow_corrected" / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt",
        weights_only=False,
        map_location="cpu",
    )
    j = int(np.flatnonzero(np.asarray(o["desi_target_ids"]) == TID)[0])
    lp = []
    for x in (dr1, recoadd):
        with torch.no_grad():
            lat = enc.encode(torch.from_numpy(x[None])).numpy()
            xs = ((lat - dd["scaler_mean"]) / dd["scaler_scale"]).astype(np.float32)
            lp.append(float(nde.log_prob(torch.from_numpy(xs))[0]))
    lines.append(
        f"{tag:>6} {float(o['threshold']):9.2f} {float(o['log_p_desi'][j]):8.2f} {lp[0]:8.2f} "
        f"{lp[1]:8.2f}"
    )
text = "\n".join(lines)
(OUT / "recoadd_score.txt").write_text(text + "\n")
print(text)
