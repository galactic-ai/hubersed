"""Save the DR1 resolution matrix of one target of targets.txt.

As 2026-11-01_poco_repeats/resolution_matrix.py. --coadd main saves the main bright coadd instead.
COMMAND: on a compute node, through check.slurm,
``<venv>/bin/python experiments/2026-11-14_poco_psb_sfh/resolution_matrix.py --tid <TARGETID>
[--coadd main]``.
OUTPUT: results/2026-11-14_poco_psb_sfh/<TARGETID>/resolution_matrix[_main].npz.
"""

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

import hubersed.io.desi as desi_io
from hubersed.io.desi import load_resolution_npz, load_spectrum
from hubersed.paths import PATHS

HERE = Path(__file__).resolve().parent
COADD = Path("/work/11006/nikhilgaruda/ls6/research/spender/DATA")
MAIN_DIR = Path(
    "/scratch/11006/nikhilgaruda/hubersed_data/desi_iron_spectra"
)  # 2026-11-13_exposures_gallery105
OUT = PATHS["RESULTS"] / HERE.name
if os.environ.get("SPEC_DATA"):
    desi_io.DATA_PATH = Path(os.environ["SPEC_DATA"])

_spec = importlib.util.spec_from_file_location(
    "resolution_check",
    PATHS["ROOT"] / "experiments" / "2026-10-09_nautilus_c3k_er_broadline" / "resolution_check.py",
)
resolution_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(resolution_check)


def main(tid, which="sv3"):
    """Save the matrix of ``tid`` and, for the SV3 coadd, check it against the fitted spectrum."""
    hp = {
        int(a): int(b)
        for a, b, _ in (x.split() for x in (HERE / "targets.txt").read_text().splitlines())
    }[tid]
    if which == "main":
        coadd = MAIN_DIR / f"coadd-main-bright-{hp}.fits"
    else:
        coadd = COADD / str(hp) / f"coadd-sv3-bright-{hp}.fits"
    out = OUT / str(tid)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("resolution_matrix_main.npz" if which == "main" else "resolution_matrix.npz")
    resolution_check.TID = tid  # save_matrix reads the module TID
    resolution_check.save_matrix(coadd, path)
    with fits.open(coadd) as h:
        fm = h["FIBERMAP"].data
        numexp = int(fm["COADD_NUMEXP"][fm["TARGETID"] == tid][0])
    new = load_resolution_npz(path)
    if which == "main":
        print(f"{tid} main healpix {hp} numexp {numexp} ndiag {new['res'].shape[0]} "
              f"ivar > 0 {np.sum(new['ivar'] > 0)} -> {path}")  # fmt: skip
        return
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
    ap.add_argument("--coadd", choices=("sv3", "main"), default="sv3")
    a = ap.parse_args()
    main(a.tid, a.coadd)
