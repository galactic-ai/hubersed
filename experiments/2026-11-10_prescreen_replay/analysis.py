"""Where the Tang et al. star-forming galaxies leave our parent sample.

QUESTION: Of the 1,199 DESI x COSMOS2020 star-forming galaxies of Tang et al. (arXiv:2611.02580;
z < 0.25, Halpha S/N > 5, below the Kewley+01 line), how many are in our parent
(prescreened_sample_v1.npz, 104,990), and which step removes the rest?
HYPOTHESIS: none stated before the check.
INPUTS: prescreen_steps.npz (replay.py), tang_zpix.fits (fetch.py), tang_desi_targetid.txt
(from the authors), data/fastspec-iron-sv3-bright.fits.
SEED: none.
COMMAND: ``uv run python experiments/2026-11-10_prescreen_replay/analysis.py``.
RESULT: 503 of 1,199 have an SV3 bright spectrum (the rest are on special, SV1, SV3 dark and main
tiles); 281 are in the parent. Of the 222 others, 210 fail the latent S/N > 3 cut (median per-pixel
S/N 1.5-2.9 for p5-p95, median 2.3), 2 are not in the 254,976-spectrum dataset, and the prescreen
removes 10 (SGA 3, BPT AGN 4, shred 3). Assumption: the S/N in dataset_z_snr.npz is the latent cut
S/N (they agree to 0.002 at the threshold; its writer is not found).
FIGURES: none (table only).
"""

# %%
from collections import Counter

import numpy as np
from astropy.io import fits
from astropy.table import Table

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-10_prescreen_replay"
STEPS = {
    0: "kept",
    1: "not encoded (S/N <= 3)",
    2: "no fastspecfit row",
    3: "z outside 0.01-0.6",
    4: "pipeline z disagrees",
    5: "Gaia star",
    6: "SGA",
    7: "PSF + no gas",
    8: "BPT AGN",
    9: "shred",
}

tang = np.loadtxt(OUT / "tang_desi_targetid.txt", dtype=np.int64, comments="#")
zp = Table.read(OUT / "tang_zpix.fits")
st = np.load(OUT / "prescreen_steps.npz")
meta = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits")["METADATA"].data
lines = [f"Tang targets {tang.size}; DR1 spectra {len(zp)}"]

# %% DR1 spectra by survey and program
for (s, p), n in Counter(zip(zp["survey"], zp["program"], strict=True)).most_common():
    lines.append(f"  spectra {s:8s} {p:7s} {n:5d}")
sv3b = np.isin(tang, np.asarray(meta["TARGETID"]))
lines.append(f"with an SV3 bright spectrum (fastspecfit VAC): {sv3b.sum()}")

# %% step that removes each SV3 bright target
idx = {int(t): j for j, t in enumerate(st["tid"])}
inset = np.array([int(t) in idx for t in tang])
lines.append(f"SV3 bright but not in dataset_z_snr (254,976): {np.sum(sv3b & ~inset)}")
j = np.array([idx[int(t)] for t in tang[sv3b & inset]])
for s, name in STEPS.items():
    lines.append(f"  step {s} {name:24s} {np.sum(st['step'][j] == s):4d}")
low = j[st["step"][j] == 1]
q = np.percentile(st["snr"][low], [5, 50, 95])
qz = np.percentile(st["z"][low], [5, 50, 95])
lines.append(
    f"step 1 targets: S/N p5 {q[0]:.2f} p50 {q[1]:.2f} p95 {q[2]:.2f}; "
    f"z p5 {qz[0]:.3f} p50 {qz[1]:.3f} p95 {qz[2]:.3f}"
)

text = "\n".join(lines)
print(text)
(OUT / "analysis.txt").write_text(text + "\n")
