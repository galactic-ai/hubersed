"""Save the single-exposure spectra and FIBERSTATUS of each target in targets.txt.

As 2026-10-31_screen_all/sky_extract.py, for targets.txt. Run with the DESI environment.
COMMAND: ``$DESIPY extract.py <unzipped dir> <out dir>`` through extract.slurm.
OUTPUT: <out dir>/<TARGETID>_exposures.npz, one per target.
"""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from desispec.maskbits import fibermask

HERE = Path(__file__).resolve().parent
EXTRA = ("EXPID", "NIGHT", "TILEID", "FIBER", "PETAL_LOC", "EXPTIME", "FIBERSTATUS")


def main(spec_dir, out_dir):
    """Extract every target of targets.txt from its healpix spectra file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for line in (HERE / "targets.txt").read_text().splitlines():
        t, hp = (int(x) for x in line.split())
        with fits.open(spec_dir / f"spectra-sv3-bright-{hp}.fits", memmap=True) as h:
            fm = h["FIBERMAP"].data
            rows = np.flatnonzero(fm["TARGETID"] == t)
            assert rows.size, f"{t} not in healpix {hp}"
            out = {k: np.asarray(fm[k][rows]) for k in EXTRA if k in fm.columns.names}
            out["fiberstatus_names"] = np.array(
                [",".join(fibermask.names(int(s))) for s in out["FIBERSTATUS"]]
            )
            for arm in "BRZ":
                out[f"{arm}_wave"] = np.asarray(h[f"{arm}_WAVELENGTH"].data, float)
                for kind in ("FLUX", "IVAR", "MASK"):
                    out[f"{arm}_{kind.lower()}"] = np.asarray(h[f"{arm}_{kind}"].data[rows])
        np.savez(out_dir / f"{t}_exposures.npz", **out)
        print(f"{t} hp {hp}: EXPID {out['EXPID'].tolist()} "
              f"FIBERSTATUS {out['fiberstatus_names'].tolist()}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
