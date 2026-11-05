"""Save the DR1 resolution matrix of one target of targets.txt.

As 2026-10-29_continuum_39633433328091196_map/resolution_matrix.py, for --tid. The healpix comes from
targets.txt, the coadd from the spender DATA folder on /work. The saved flux and ivar must equal
load_spectrum's on the good pixels. SPEC_DATA names the folder with the staged spender files.
COMMAND: on a stampede3 compute node, through prep.slurm,
``<venv>/bin/python experiments/2026-10-31_poco_robust6/resolution_matrix.py --tid <TARGETID>``.
OUTPUT: results/2026-10-31_poco_robust6/<TARGETID>/resolution_matrix.npz.
"""

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

import hubersed.io.desi as desi_io
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.sps.resolution_matrix import load_resolution_npz

HERE = Path(__file__).resolve().parent
COADD = Path("/work/11006/nikhilgaruda/ls6/research/spender/DATA")
OUT = PATHS["RESULTS"] / HERE.name
if os.environ.get("SPEC_DATA"):
    desi_io.DATA_PATH = Path(os.environ["SPEC_DATA"])

_spec = importlib.util.spec_from_file_location(
    "resolution_check",
    HERE.parent / "2026-10-09_nautilus_c3k_er_broadline" / "resolution_check.py",
)
resolution_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(resolution_check)


def main(tid):
    """Save the matrix of ``tid`` and check it against the fitted spectrum."""
    hp = {
        int(a): int(b)
        for a, b, _ in (x.split() for x in (HERE / "targets.txt").read_text().splitlines())
    }[tid]
    coadd = COADD / str(hp) / f"coadd-sv3-bright-{hp}.fits"
    out = OUT / str(tid)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "resolution_matrix.npz"
    resolution_check.TID = tid  # save_matrix reads the module TID
    resolution_check.save_matrix(coadd, path)
    with fits.open(coadd) as h:
        fm = h["FIBERMAP"].data
        numexp = int(fm["COADD_NUMEXP"][fm["TARGETID"] == tid][0])
    new = load_resolution_npz(path)
    s = load_spectrum(tid)
    flux = np.asarray(s.flux.value, dtype=float)
    ivar = np.asarray(s.uncertainty.array, dtype=float)
    good = np.isfinite(flux) & (ivar > 0) & (new["ivar"] > 0)
    dflux = np.max(np.abs(new["flux"][good] - flux[good]) / np.abs(flux[good]).clip(1e-3))
    divar = np.max(np.abs(new["ivar"][good] / ivar[good] - 1))
    print(f"{tid} healpix {hp} numexp {numexp} ndiag {new['res'].shape[0]} good {good.sum()} "
          f"max rel flux diff {dflux:.2e} max ivar ratio - 1 {divar:.2e}")  # fmt: skip
    assert dflux < 1e-4 and divar < 1e-4, "matrix file differs from load_spectrum"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tid", type=int, required=True)
    main(ap.parse_args().tid)
