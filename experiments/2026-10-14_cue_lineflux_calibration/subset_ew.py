"""Pick 60 more galaxies stratified by EW(Hbeta) and stellar mass, to test the EW dependence of Cue O/H.

Cells: EW(Hbeta) in EW_BINS (A, FastSpecFit HBETA_EW) times LOGMSTAR in MASS_BINS, N_PER_CELL each,
drawn at random with seed SEED from lineflux_table.fits. Galaxies at z <= Z_MIN are left out,
since a DESI fibre at such low redshift likely sits on an HII region inside a large galaxy and
its FastSpecFit mass does not describe it. Galaxies already in subset_20.txt are left out.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/subset_ew.py`` from
the repository root.

OUTPUT: subset_ew60.txt (TARGETID per line) in results/2026-10-14_cue_lineflux_calibration/.
"""

import numpy as np
from astropy.table import Table

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
EW_BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, np.inf))
MASS_BINS = ((0.0, 9.0), (9.0, 9.75), (9.75, 13.0))
N_PER_CELL = 5
Z_MIN = 0.01
SEED = 1

t = Table.read(OUT / "lineflux_table.fits")
done = np.loadtxt(OUT / "subset_20.txt", dtype=np.int64)
ew, mass = np.asarray(t["EW_HB"], float), np.asarray(t["LOGMSTAR"], float)
ok = (np.asarray(t["Z"]) > Z_MIN) & ~np.isin(np.asarray(t["TARGETID"]), done)
rng = np.random.default_rng(SEED)
ids = []
for elo, ehi in EW_BINS:
    for mlo, mhi in MASS_BINS:
        cell = np.flatnonzero(ok & (ew >= elo) & (ew < ehi) & (mass >= mlo) & (mass < mhi))
        pick = rng.choice(cell, size=min(N_PER_CELL, cell.size), replace=False)
        ids += [int(x) for x in t["TARGETID"][np.sort(pick)]]
        print(
            f"EW {elo:4.0f}-{ehi:4.0f}, log M* {mlo:4.2f}-{mhi:5.2f}: {cell.size} available, {pick.size} picked"
        )
np.savetxt(OUT / "subset_ew60.txt", ids, fmt="%d")
print(f"wrote {len(ids)} TARGETIDs")
