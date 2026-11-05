"""Draft data-quality rule for the 94 and exposure-cleaned rescoring of the 18 (#17).

R1: dgr (FIBERTOTFLUX) < -0.146, the 1st percentile of all S/N > 3 spectra (cut.py).
R2: Legacy DR9 Tractor fracflux_g > 1 (ls_dr9.tractor on Data Lab, by LS_ID).
R3: for the 18 of exp_targets.txt, coadd only exposures with sky g >= 21, as spender would see it
(ivar-weighted per arm, arms joined on spender's grid, sky-line pixels at zero weight, median
normalised at rest 5300-5850 A), then encode and score with both seed 0 flows. The same coadd of
all exposures is the check against the stored log p.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_logp_vs_colour/rule.py``.
OUTPUT: results/2026-10-31_logp_vs_colour/rule.txt.
RESULT: 2026-10-31. Of the 94, R1 fails 42, R2 15, either 46. The all-exposure coadd reproduces
the stored log p (17 of 18 within 0.02). Without sky g < 21 exposures, 5 of 18 stop being common
outliers, among them 39633433328091196 (cont15 -70.2 to -10.3).
"""

# %%
import csv
import warnings
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from astroquery.utils.tap.core import TapPlus
from spender import load_model
from spender.data.desi import DESI

from hubersed.detect.flow_model import build_flow
from hubersed.io.desi import load_by_index, tids_to_indices
from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-10-31_logp_vs_colour"
QA = RES / "2026-10-31_spectrum_qa_cont_outliers"
FLOWS = RES / "noised_cue_meanzero_wide_flow"
R1, R2, R3 = -0.146, 1.0, 21.0
TAGS = ("cont10", "cont15")
warnings.filterwarnings("ignore", module="astroquery")

# %% R1 and R2 on the 94
lines = QA.joinpath("qa_outliers.txt").read_text().splitlines()
col = {k: i for i, k in enumerate(lines[0].split())}
dgr = {int(p[0]): float(p[col["dgr_tot"]]) for p in (x.split() for x in lines[1:])}
h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)
M = h["METADATA"].data
vrow = {int(t): i for i, t in enumerate(M["TARGETID"])}
lsid = {t: int(M["LS_ID"][vrow[t]]) for t in dgr}
tap = TapPlus(url="https://datalab.noirlab.edu/tap", verbose=False)
ids = ",".join(str(v) for v in lsid.values())
tab = tap.launch_job(f"SELECT ls_id, fracflux_g FROM ls_dr9.tractor WHERE ls_id IN ({ids})")
ff = {int(r["ls_id"]): float(r["fracflux_g"]) for r in tab.get_results()}
fail1 = {t for t, d in dgr.items() if d < R1}
fail2 = {t for t in dgr if ff.get(lsid[t], np.nan) > R2}
out = [
    f"R1 dgr < {R1}: {len(fail1)} of {len(dgr)}. R2 fracflux_g > {R2}: {len(fail2)} "
    f"({len(dgr) - len(ff)} without a Tractor row). Both: {len(fail1 & fail2)}. "
    f"Either: {len(fail1 | fail2)}. Pass both: {len(dgr) - len(fail1 | fail2)}.",
    "",
]

# %% R3, exposure-cleaned coadds of the 18
wave = DESI._wave_obs.numpy()
skyline = DESI._skyline_mask.numpy()
sky = {}
with open(QA / "sv3_bright_exposures.csv", newline="") as fh:
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
                     weights_only=False).float().eval()  # fmt: skip
    o = torch.load(RES / "wide_flow_corrected" / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt",
                   weights_only=False, map_location="cpu")  # fmt: skip
    stored = dict(zip(np.asarray(o["desi_target_ids"]).tolist(), np.asarray(o["log_p_desi"]).tolist(),
                      strict=True))  # fmt: skip
    flows[tag] = (nde, enc, dd["scaler_mean"], dd["scaler_scale"], float(o["threshold"]), stored)


def score(tag, x):
    """Return the flow log p of one normalised spectrum."""
    nde, enc, mean, scale, *_ = flows[tag]
    with torch.no_grad():
        lat = enc.encode(torch.from_numpy(x[None])).numpy()
        return float(nde.log_prob(torch.from_numpy(((lat - mean) / scale).astype(np.float32)))[0])


out.append(f"R3: exposures with sky g >= {R3}. log p: stored / all-exposure coadd / clean coadd "
           "(threshold cont10 -15.72, cont15 -25.24).")  # fmt: skip
out.append(f"{'TARGETID':>18} {'kept':>5}  " + "  ".join(f"{t:>26}" for t in TAGS))
for line in (Path(__file__).resolve().parent / "exp_targets.txt").read_text().splitlines():
    t = int(line.split()[0])
    d = dict(np.load(OUT / "exposures" / f"{t}.npz"))
    z = float(load_by_index(int(tids_to_indices(np.array([t]))[0]))[2])
    skies = np.array([sky.get(int(e), np.nan) for e in d["EXPID"]])
    allx = coadd(d, np.ones(len(skies), bool), z)
    clean = coadd(d, skies >= R3, z)
    cells = []
    for tag in TAGS:
        s = flows[tag][5][t]
        a = score(tag, allx)
        c = score(tag, clean) if clean is not None else np.nan
        cells.append(f"{s:8.2f} {a:8.2f} {c:8.2f}")
    out.append(f"{t:>18} {int((skies >= R3).sum()):>2}/{len(skies)}  " + "  ".join(cells))
text = "\n".join(out)
(OUT / "rule.txt").write_text(text + "\n")
print(text)
