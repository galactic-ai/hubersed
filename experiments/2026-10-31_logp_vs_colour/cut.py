"""Count the seed 0 continuum-flow flags that a blue spectrum-minus-photometry colour cut removes (#17).

dgr uses FIBERTOTFLUX as analysis.py. A spectrum fails the cut when dgr is below it. Cuts are fixed
values and percentiles of dgr over all S/N > 3 spectra in the VAC.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_logp_vs_colour/cut.py``.
OUTPUT: results/2026-10-31_logp_vs_colour/cut.txt.
RESULT: 2026-10-31. dgr < -0.1 removes 2.5% of spectra, 20% of cont10 and 22% of cont15 flags,
and 47 of the 94.
"""

import numpy as np
import torch
from astropy.io import fits

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-10-31_logp_vs_colour"

flag, lp_tid = {}, None
for tag in ("cont10", "cont15"):
    o = torch.load(
        RES / "wide_flow_corrected" / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt",
        weights_only=False,
        map_location="cpu",
    )
    flag[tag] = {int(t) for t in o["outlier_target_ids"]}
    lp_tid = np.asarray(o["desi_target_ids"], np.int64)
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


col = {t: dgr(t) for t in lp_tid.tolist()}
allc = np.array([v for v in col.values() if np.isfinite(v)])
s94 = np.load(OUT.parent / "2026-10-31_spectrum_qa_cont_outliers" / "sample94.npz")["target_ids"]
sets = {
    "cont10": flag["cont10"],
    "cont15": flag["cont15"],
    "common": flag["cont10"] & flag["cont15"],
    "the 94": {int(t) for t in s94},
}
cuts = [("-0.10", -0.1), ("-0.20", -0.2)] + [
    (f"pct {p}", float(np.percentile(allc, p))) for p in (1.0, 0.5)
]
lines = [f"{len(allc)} spectra with dgr. Remaining after removing dgr < cut (no colour = kept)."]
lines.append(f"{'cut':>9} {'value':>7} {'all removed':>12}  " + "  ".join(f"{k:>14}" for k in sets))
lines.append(f"{'none':>9} {'':>7} {'':>12}  " + "  ".join(f"{len(v):>14d}" for v in sets.values()))
for name, c in cuts:
    cells = []
    for v in sets.values():
        kept = sum(not (np.isfinite(col.get(t, np.nan)) and col[t] < c) for t in v)
        cells.append(f"{kept:>6d} ({100 * kept / len(v):4.1f}%)")
    lines.append(f"{name:>9} {c:+7.3f} {100 * np.mean(allc < c):11.2f}%  " + "  ".join(cells))
text = "\n".join(lines)
(OUT / "cut.txt").write_text(text + "\n")
print(text)
