"""Save the DR1 resolution matrix of the continuum outlier 39627757533007793.

As 2026-10-21_continuum_poco/resolution_matrix.py for one target. The matrix, flux, ivar and mask
come from the DR1 healpix coadd (healpix 25598, from fastspec-iron-sv3-bright METADATA) and are
saved with resolution_check.save_matrix. The script then checks that the saved flux and ivar equal
load_spectrum's on the good pixels, which is what fit.py reads. Stampede3 compute nodes do not
mount Corral, so SPEC_DATA can name a folder with the staged spender files.

COMMAND: on a stampede3 compute node from the repository root,
``SPEC_DATA=/scratch/11006/nikhilgaruda/hubersed_data
PYTHONPATH=experiments/2026-10-09_nautilus_c3k_er_broadline <venv>/bin/python
experiments/2026-10-28_continuum_c3k_er_rmatrix_map/resolution_matrix.py``.

OUTPUT: results/2026-10-28_continuum_c3k_er_rmatrix_map/resolution_matrix.npz.
"""

import os
from pathlib import Path

import numpy as np
import resolution_check
from astropy.io import fits

import hubersed.io.desi as desi_io
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.sps.resolution_matrix import load_resolution_npz

TID, HEALPIX = 39627757533007793, 25598
COADD = Path("/work/11006/nikhilgaruda/ls6/research/spender/DATA") / str(HEALPIX)
OUT = PATHS["RESULTS"] / "2026-10-28_continuum_c3k_er_rmatrix_map"
if os.environ.get("SPEC_DATA"):
    desi_io.DATA_PATH = Path(os.environ["SPEC_DATA"])


def main():
    """Save the matrix and check it against the fitted spectrum."""
    coadd = COADD / f"coadd-sv3-bright-{HEALPIX}.fits"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "resolution_matrix.npz"
    resolution_check.TID = TID  # save_matrix reads the module TID
    resolution_check.save_matrix(coadd, path)

    with fits.open(coadd) as h:
        fm = h["FIBERMAP"].data
        numexp = int(fm["COADD_NUMEXP"][fm["TARGETID"] == TID][0])
    new = load_resolution_npz(path)
    s = load_spectrum(TID)
    flux = np.asarray(s.flux.value, dtype=float)
    ivar = np.asarray(s.uncertainty.array, dtype=float)
    good = np.isfinite(flux) & (ivar > 0) & (new["ivar"] > 0)
    dflux = np.max(np.abs(new["flux"][good] - flux[good]) / np.abs(flux[good]).clip(1e-3))
    divar = np.max(np.abs(new["ivar"][good] / ivar[good] - 1))
    print(
        f"{TID} healpix {HEALPIX} numexp {numexp} ndiag {new['res'].shape[0]} "
        f"good px {good.sum()} max flux rel diff {dflux:.2e} max ivar rel diff {divar:.2e}"
    )
    assert dflux < 1e-4 and divar < 1e-4, "saved spectrum differs from load_spectrum"
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
