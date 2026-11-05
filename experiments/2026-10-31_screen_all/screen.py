"""Data-driven data-quality screen of the S/N > 3 spectra and its effect on the continuum outliers (#17).

QUESTION: how many flow flags, common outliers and members of the 94 fail a colour or neighbour screen
with thresholds set by the whole sample?
HYPOTHESIS: none registered.
INPUTS: fracflux.csv (fracflux.py), seed 0 flags (results/wide_flow_corrected), FastSpecFit VAC,
sample94.npz. Colour fails below the 1st percentile of dgr (FLUX_SYNTH against FIBERTOTFLUX, as
2026-10-31_logp_vs_colour/analysis.py); neighbour fails above the 99th percentile of fracflux_g.
Missing values pass. The #10 cuts are filters, so screening the 94 equals screening the 512 first.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_screen_all/screen.py``.
RESULT: 2026-10-31. Thresholds dgr < -0.146 and fracflux_g > 1.332. The screen removes 1.8% of
spectra, 15.6% of cont10 and 21.1% of cont15 flags, 102 of the 512 and 46 of the 94 (48 pass).
FIGURES: results/2026-10-31_screen_all/screen.txt and screened94.npz.
"""

# %%
import csv

import numpy as np
import torch
from astropy.io import fits

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-10-31_screen_all"

ff = {}
with open(OUT / "fracflux.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        ff[int(r["targetid"])] = float(r["fracflux_g"]) if r["fracflux_g"] else np.nan
flag = {}
for tag in ("cont10", "cont15"):
    o = torch.load(RES / "wide_flow_corrected" / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt",
                   weights_only=False, map_location="cpu")  # fmt: skip
    flag[tag] = {int(t) for t in o["outlier_target_ids"]}
    alltid = np.asarray(o["desi_target_ids"], np.int64)

h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)
M, S = h["METADATA"].data, h["FASTSPEC"].data
vrow = {int(t): i for i, t in enumerate(M["TARGETID"])}


def dgr(t):
    """Return spectrum-minus-FIBERTOTFLUX g-r, NaN without positive fluxes or outside the VAC."""
    if t not in vrow:
        return np.nan
    i, f = vrow[t], []
    for b in "GR":
        fs, ft = float(S[f"FLUX_SYNTH_{b}"][i]), float(M[f"FIBERTOTFLUX_{b}"][i])
        f.append(fs / ft if fs > 0 and ft > 0 else np.nan)
    return -2.5 * np.log10(f[0] / f[1])


col = {t: dgr(t) for t in alltid.tolist()}
cv = np.array([v for v in col.values() if np.isfinite(v)])
fv = np.array([v for v in ff.values() if np.isfinite(v)])
c_thr, f_thr = float(np.percentile(cv, 1)), float(np.percentile(fv, 99))
fail_c = {t for t, v in col.items() if np.isfinite(v) and v < c_thr}
fail_f = {t for t, v in ff.items() if np.isfinite(v) and v > f_thr}
fail = fail_c | fail_f

s94 = {
    int(t)
    for t in np.load(RES / "2026-10-31_spectrum_qa_cont_outliers" / "sample94.npz")["target_ids"]
}
sets = {
    "all spectra": set(alltid.tolist()),
    "cont10 flags": flag["cont10"],
    "cont15 flags": flag["cont15"],
    "common (512)": flag["cont10"] & flag["cont15"],
    "the 94": s94,
}
lines = [
    f"Thresholds: colour dgr < {c_thr:+.3f} (1st percentile of {len(cv)}), neighbour fracflux_g > "
    f"{f_thr:.3f} (99th percentile of {len(fv)}). Missing values pass.",
    "",
    f"{'set':>14} {'N':>7} {'colour':>14} {'neighbour':>14} {'either':>14} {'pass':>7}",
]


def cell(s, x):
    """Return the count and percentage of set s that is in x."""
    return f"{len(s & x):6d} ({100 * len(s & x) / len(s):4.1f}%)"


for name, s in sets.items():
    lines.append(f"{name:>14} {len(s):7d} {cell(s, fail_c):>14} {cell(s, fail_f):>14} "
                 f"{cell(s, fail):>14} {len(s - fail):7d}")  # fmt: skip
keep94 = sorted(s94 - fail)
named = (39633433328091196, 39627757533007793)
lines += ["", "The two studied outliers: " + ", ".join(
    f"{t} {'passes' if t not in fail else 'fails'} (dgr {col[t]:+.3f}, fracflux_g {ff.get(t, np.nan):.3f})"
    for t in named)]  # fmt: skip
text = "\n".join(lines)
(OUT / "screen.txt").write_text(text + "\n")
np.savez(OUT / "screened94.npz", target_ids=np.array(keep94, np.int64), colour_threshold=c_thr,
         fracflux_threshold=f_thr)  # fmt: skip
print(text)
