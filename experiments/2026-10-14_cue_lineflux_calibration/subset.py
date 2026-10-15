"""Pick the galaxies for the first Cue line-flux fits: 19 at evenly spaced stellar-mass ranks, plus the target.

The ranks are round(linspace(0, N - 1, 19)) of lineflux_table.fits sorted by LOGMSTAR, so the pick
spans the sample's mass range and depends on nothing random.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/subset.py`` from
the repository root.

OUTPUT: subset_20.txt (TARGETID per line) in results/2026-10-14_cue_lineflux_calibration/.
"""

import numpy as np
from astropy.table import Table

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
TID = 39627770174637084
N_PICK = 19

t = Table.read(OUT / "lineflux_table.fits")
t = t[np.argsort(np.asarray(t["LOGMSTAR"]))]
pick = t[np.round(np.linspace(0, len(t) - 1, N_PICK)).astype(int)]
ids = [int(x) for x in pick["TARGETID"]]
if TID not in ids:
    ids.append(TID)
np.savetxt(OUT / "subset_20.txt", ids, fmt="%d")
for tid in ids:
    r = t[t["TARGETID"] == tid][0]
    print(f"{tid} log M* {r['LOGMSTAR']:.2f} z {r['Z']:.3f} EW_HB {r['EW_HB']:.1f}")
