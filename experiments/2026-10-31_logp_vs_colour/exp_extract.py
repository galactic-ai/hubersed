"""Save the single-exposure spectra and per-exposure FIBERSTATUS of the exp_targets.txt galaxies.

Reads each iron spectra file once, keeps the FIBERMAP rows of its targets with their b, r and z flux,
ivar and mask, and decodes FIBERSTATUS with desispec. Run with the DESI environment's python.
COMMAND: ``$DESIPY exp_extract.py <spectra dir> <out dir>`` through exp_extract.slurm.
OUTPUT: <out dir>/<TARGETID>.npz, one per target.
"""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from desispec.maskbits import fibermask

HERE = Path(__file__).resolve().parent
EXTRA = ("EXPID", "NIGHT", "TILEID", "FIBER", "PETAL_LOC", "EXPTIME", "FIBERSTATUS",
         "DELTA_X", "DELTA_Y")  # fmt: skip


def main(spec_dir, out_dir):
    """Extract every target of exp_targets.txt from its healpix spectra file."""
    targets = {}
    for line in (HERE / "exp_targets.txt").read_text().splitlines():
        t, hp, *_ = line.split()
        targets.setdefault(int(hp), []).append(int(t))
    out_dir.mkdir(parents=True, exist_ok=True)
    for hp, tids in sorted(targets.items()):
        path = spec_dir / f"spectra-sv3-bright-{hp}.fits"
        with fits.open(path, memmap=True) as h:
            fm = h["FIBERMAP"].data
            for t in tids:
                rows = np.flatnonzero(fm["TARGETID"] == t)
                assert rows.size, f"{t} not in {path}"
                out = {k: np.asarray(fm[k][rows]) for k in EXTRA if k in fm.columns.names}
                out["fiberstatus_names"] = np.array(
                    [",".join(fibermask.names(int(s))) for s in out["FIBERSTATUS"]]
                )
                for arm in "BRZ":
                    out[f"{arm}_wave"] = np.asarray(h[f"{arm}_WAVELENGTH"].data, float)
                    for kind in ("FLUX", "IVAR", "MASK"):
                        out[f"{arm}_{kind.lower()}"] = np.asarray(h[f"{arm}_{kind}"].data[rows])
                np.savez(out_dir / f"{t}.npz", **out)
                print(f"{t} hp {hp}: EXPID {out['EXPID'].tolist()} "
                      f"FIBERSTATUS {out['fiberstatus_names'].tolist()}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
