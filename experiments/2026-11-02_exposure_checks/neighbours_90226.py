"""Fibres around 1309 in exposure 90226: is the star in 39627781742528421's spectrum a neighbour's light?

In 90226 (tile 76, petal 2) the spectrum of 39627781742528421 (fibre 1309) is ~10x brighter than
in its other exposures and shows z = 0 stellar lines. This saves the per-arm flux of fibres 1300 to
1318 from the calibrated frames, with their TARGETID, OBJTYPE and median flux, and the correlation
of each fibre's flux with fibre 1309's after removing 1309's median from both.
COMMAND: on stampede3, with the cframes in <dir>,
``<venv>/bin/python experiments/2026-11-02_exposure_checks/neighbours_90226.py <dir> <out.npz>``.
OUTPUT: <out.npz> and a printed table.
"""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

EXPID, PETAL, FIBER = 90226, 2, 1309
FIBERS = range(1300, 1319)


def main(src, out):
    """Read the three arms, print the table and save the fluxes."""
    save, table = {}, {}
    for arm in "brz":
        with fits.open(Path(src) / f"cframe-{arm}{PETAL}-{EXPID:08d}.fits.gz") as h:
            fm = h["FIBERMAP"].data
            flux, wave = h["FLUX"].data, h["WAVELENGTH"].data
            rows = {int(f): i for i, f in enumerate(fm["FIBER"]) if int(f) in FIBERS}
            save[f"{arm}_wave"] = wave
            ref = flux[rows[FIBER]]
            for f in FIBERS:
                x = flux[rows[f]]
                good = np.isfinite(x) & np.isfinite(ref)
                r = np.corrcoef(x[good], ref[good])[0, 1]
                table.setdefault(f, dict(tid=int(fm["TARGETID"][rows[f]]),
                                         obj=str(fm["OBJTYPE"][rows[f]])))  # fmt: skip
                table[f][f"{arm}_med"] = float(np.median(x[good]))
                table[f][f"{arm}_r"] = float(r)
                save[f"{arm}_{f}"] = x
    print(f"EXPID {EXPID} petal {PETAL}: median flux (1e-17) and correlation with fibre {FIBER}")
    for f, v in table.items():
        print(f"{f:5d} {v['tid']:>18d} {v['obj']:>4} "
              + "  ".join(f"{a} {v[f'{a}_med']:7.2f} r {v[f'{a}_r']:+.2f}" for a in "brz"))  # fmt: skip
    np.savez(out, fibers=np.array(list(FIBERS)), **save)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
