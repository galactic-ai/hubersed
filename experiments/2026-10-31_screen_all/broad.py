"""FastSpecFit broad Halpha and Hbeta of the 36 follow-up outliers (survivors.txt).

The #10 cuts screen AGN with narrow-line BPT and SIMBAD only. This lists the broad components.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_screen_all/broad.py``.
OUTPUT: results/2026-10-31_screen_all/broad.txt.
RESULT: 2026-10-31. 6 of 36 have broad Halpha with S/N 28-116 and sigma 533-1557 km/s; two more at
z > 0.5 have broad Hbeta with S/N 18-24.
"""

import numpy as np
from astropy.io import fits

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-31_screen_all"
tids = [
    int(x.split()[0])
    for x in (OUT / "survivors.txt").read_text().splitlines()
    if x.split() and x.split()[0].isdigit() and len(x.split()[0]) > 10
]
h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)
M, S = h["METADATA"].data, h["FASTSPEC"].data
vrow = {int(t): i for i, t in enumerate(M["TARGETID"])}


def line(i, name):
    """Return flux S/N, sigma and EW of one line fit."""
    f, iv = float(S[f"{name}_FLUX"][i]), float(S[f"{name}_FLUX_IVAR"][i])
    return (
        f * np.sqrt(iv) if iv > 0 else 0.0,
        float(S[f"{name}_SIGMA"][i]),
        float(S[f"{name}_EW"][i]),
    )


rows = []
for t in tids:
    i = vrow[t]
    ha, hb, nha = line(i, "HALPHA_BROAD"), line(i, "HBETA_BROAD"), line(i, "HALPHA")
    rows.append((t, float(M["Z"][i]), *ha, *hb, nha[0]))
rows.sort(key=lambda r: -r[2])
txt = [f"{len(rows)} follow-up outliers, sorted by broad Halpha S/N.",
       f"{'TARGETID':>18} {'z':>6} {'bHa S/N':>8} {'sigma':>6} {'EW':>6} {'bHb S/N':>8} "
       f"{'sigma':>6} {'nHa S/N':>8}"]  # fmt: skip
txt += [f"{t:>18} {z:6.3f} {a:8.1f} {b:6.0f} {c:6.1f} {d:8.1f} {e:6.0f} {g:8.1f}"
        for t, z, a, b, c, d, e, _, g in rows]  # fmt: skip
text = "\n".join(txt)
(OUT / "broad.txt").write_text(text + "\n")
print(text)
