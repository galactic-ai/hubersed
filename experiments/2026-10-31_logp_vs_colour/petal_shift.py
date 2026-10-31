"""Per-exposure colour shift of every fiber in the unzipped exp_stage.sh files, by tile, petal and exposure.

Colour c = -2.5 log10(mean flux 4200-5400 A, b arm / mean flux 5800-7000 A, r arm) over unmasked
pixels. For a target with >= 2 exposures on one tile, shift = c of the exposure minus the mean c
of its exposures. Groups are (tile, petal, exposure); their median and mean shift,
the fraction of fibers below -0.1 and the fiber count are saved.
COMMAND: ``$DESIPY petal_shift.py <unzipped dir> <out dir>`` through petal_shift.slurm.
OUTPUT: <out dir>/petal_shift.csv (groups) and petal_shift_t95.csv (tile 95 fibers).
RESULT: 2026-10-31. Tile 95 petal 5 has no petal-wide shift (mean -0.02 to +0.03, MAD 0.07-0.09);
its three outliers are single-fiber tails. Whole-exposure shifts exist elsewhere, e.g. 83713 on tile
93 (mean -0.12 to -0.18 on three petals) and 87839 on tile 560 (-0.06).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy.io import fits

WIN = {"B": (4200.0, 5400.0), "R": (5800.0, 7000.0)}


def window_mean(h, arm, rows):
    """Return the mean unmasked flux of rows in the arm's window, NaN when no pixel is good."""
    wv = h[f"{arm}_WAVELENGTH"].data
    sel = (wv >= WIN[arm][0]) & (wv < WIN[arm][1])
    f = h[f"{arm}_FLUX"].data[rows][:, sel]
    good = (h[f"{arm}_IVAR"].data[rows][:, sel] > 0) & (h[f"{arm}_MASK"].data[rows][:, sel] == 0)
    n = good.sum(1)
    return np.where(n > 20, np.where(good, f, 0).sum(1) / np.maximum(n, 1), np.nan)


def main(unz, out):
    """Measure every fiber of every file and write the two tables."""
    shift = defaultdict(list)
    t95 = []
    for path in sorted(unz.glob("spectra-sv3-bright-*.fits")):
        with fits.open(path, memmap=True) as h:
            fm = h["FIBERMAP"].data
            rows = np.arange(len(fm))
            c = -2.5 * np.log10(window_mean(h, "B", rows) / window_mean(h, "R", rows))
            tid, tile, pet, exp = (np.asarray(fm[k]) for k in ("TARGETID", "TILEID", "PETAL_LOC",
                                                                "EXPID"))  # fmt: skip
        by = defaultdict(list)
        for k in np.flatnonzero(np.isfinite(c)):
            by[(int(tid[k]), int(tile[k]))].append(k)
        for (t, tl), ks in by.items():
            if len(ks) < 2:
                continue
            ref = np.mean(c[ks])
            for k in ks:
                d = float(c[k] - ref)
                shift[(tl, int(pet[k]), int(exp[k]))].append(d)
                if tl == 95:
                    t95.append((t, int(pet[k]), int(exp[k]), d))
        print(f"{path.name}: {len(fm)} rows", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "petal_shift.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "tileid",
                "petal",
                "expid",
                "n_fibers",
                "median_shift",
                "mean_shift",
                "mad_shift",
                "frac_below_m01",
            ]
        )
        for (tl, p, e), v in sorted(shift.items()):
            v = np.array(v)
            mad = 1.4826 * np.median(np.abs(v - np.median(v)))
            w.writerow(
                [
                    tl,
                    p,
                    e,
                    len(v),
                    f"{np.median(v):.4f}",
                    f"{np.mean(v):.4f}",
                    f"{mad:.4f}",
                    f"{np.mean(v < -0.1):.3f}",
                ]
            )
    with open(out / "petal_shift_t95.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["targetid", "petal", "expid", "shift"])
        w.writerows((t, p, e, f"{d:.4f}") for t, p, e, d in t95)
    print(f"{len(shift)} groups, {len(t95)} tile 95 rows", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
