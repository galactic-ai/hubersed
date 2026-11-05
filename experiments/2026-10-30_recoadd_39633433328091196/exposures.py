"""Save the single-exposure SV3 bright spectra of the continuum outlier 39633433328091196.

The iron spectra file of healpix 10231 holds one row per exposure and fiber, with b, r and z
arms on their own grids. The rows of this TARGETID are saved with their EXPID, and each arm's
flux, ivar and mask, plus the sky-related FIBERMAP columns when present.

COMMAND: on a stampede3 compute node from the repository root, through recoadd.slurm, or
``<venv>/bin/python experiments/2026-10-30_recoadd_39633433328091196/exposures.py <spectra file>``.
OUTPUT: results/2026-10-30_recoadd_39633433328091196/sv3/exposures.npz.
Job 3662886 ran this file before ruff format and this line, so its logged sha256 differs.
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

TID = 39633433328091196
OUT = (
    Path(__file__).resolve().parents[2] / "results" / "2026-10-30_recoadd_39633433328091196" / "sv3"
)
EXTRA = ("EXPID", "NIGHT", "TILEID", "FIBER", "PETAL_LOC", "EXPTIME", "FIBERSTATUS", "FIBER_X",
         "FIBER_Y", "DELTA_X", "DELTA_Y")  # fmt: skip


def main(path):
    """Write exposures.npz for TID from the spectra file at ``path``."""
    with fits.open(path) as h:
        fm = h["FIBERMAP"].data
        rows = np.flatnonzero(fm["TARGETID"] == TID)
        assert rows.size, f"{TID} not in {path}"
        out = {k: np.asarray(fm[k][rows]) for k in EXTRA if k in fm.columns.names}
        for arm in "BRZ":
            out[f"{arm}_wave"] = np.asarray(h[f"{arm}_WAVELENGTH"].data, float)
            for kind in ("FLUX", "IVAR", "MASK"):
                out[f"{arm}_{kind.lower()}"] = np.asarray(h[f"{arm}_{kind}"].data[rows])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "exposures.npz", **out)
    print(f"wrote {OUT / 'exposures.npz'}: rows {rows.tolist()}, EXPID {out['EXPID'].tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spectra", help="iron spectra file of healpix 10231")
    main(ap.parse_args().spectra)
