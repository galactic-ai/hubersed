"""List the screened outliers whose coadd includes a bright-sky exposure, and their iron spectra files.

Targets: screened94.npz members with COADD_NUMEXP >= 2 and a tile exposure with sky g < 21
(qa_outliers.txt columns numexp and sky_min).
COMMAND: ``uv run --no-sync python experiments/2026-10-31_screen_all/sky_select.py``.
OUTPUT: sky_targets.txt and sky_urls.txt next to this script.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits

from hubersed.paths import PATHS

HERE = Path(__file__).resolve().parent
RES = PATHS["RESULTS"]
BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix/sv3/bright"
SKY = 21.0

keep = set(np.load(RES / "2026-10-31_screen_all" / "screened94.npz")["target_ids"].tolist())
lines = (RES / "2026-10-31_spectrum_qa_cont_outliers" / "qa_outliers.txt").read_text().splitlines()
col = {k: i for i, k in enumerate(lines[0].split())}
M = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)["METADATA"].data
vrow = {int(t): i for i, t in enumerate(M["TARGETID"])}
rows = []
for line in lines[1:]:
    p = line.split()
    t, n, sky = int(p[0]), int(float(p[col["numexp"]])), float(p[col["sky_min"]])
    if t in keep and n >= 2 and sky < SKY:
        rows.append((t, int(M["HEALPIX"][vrow[t]]), n, sky))
(HERE / "sky_targets.txt").write_text("".join(f"{t} {hp} {n} {s:.2f}\n" for t, hp, n, s in rows))
hps = sorted({r[1] for r in rows})
(HERE / "sky_urls.txt").write_text(
    "".join(f"{BASE}/{hp // 100}/{hp}/spectra-sv3-bright-{hp}.fits.gz\n" for hp in hps)
)
print(f"{len(rows)} targets in {len(hps)} healpix files")
