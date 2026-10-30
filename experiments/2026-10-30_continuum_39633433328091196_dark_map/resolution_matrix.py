"""Save the SV3 dark spectrum and DR1 resolution matrix of the continuum outlier 39633433328091196.

The spectrum is the SV3 dark DR1 healpix coadd (data/sv3_dark_coadds/coadd-sv3-dark-10231.fits,
1 exposure, EXPID 88050) put through spender DESI.prepare_spectra, as the spender chunk files that
load_spectrum reads were made, with the spender norm undone as hubersed.io.desi.load_by_index does.
The matrix comes from the same coadd through resolution_check.save_matrix, as in
2026-10-29_continuum_39633433328091196_map/resolution_matrix.py, and is checked against the
spectrum on the good pixels.

COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-30_continuum_39633433328091196_dark_map/resolution_matrix.py``.
OUTPUT: results/2026-10-30_continuum_39633433328091196_dark_map/ dark_spectrum.npz and
resolution_matrix.npz.
"""

import importlib.util
from pathlib import Path

import numpy as np
from spender.data.desi import DESI

from hubersed.paths import PATHS
from hubersed.sps.resolution_matrix import load_resolution_npz

TID = 39633433328091196
COADD = PATHS["DATA"] / "sv3_dark_coadds" / "coadd-sv3-dark-10231.fits"
OUT = PATHS["RESULTS"] / "2026-10-30_continuum_39633433328091196_dark_map"

_spec = importlib.util.spec_from_file_location(
    "resolution_check",
    Path(__file__).resolve().parents[1]
    / "2026-10-09_nautilus_c3k_er_broadline"
    / "resolution_check.py",
)
resolution_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(resolution_check)


def main():
    """Save the spender spectrum and the matrix, and check one against the other."""
    OUT.mkdir(parents=True, exist_ok=True)
    spec, w, z, tid, norm, _ = DESI.prepare_spectra(str(COADD))
    i = int(np.flatnonzero(tid.numpy() == TID)[0])
    n = float(norm[i])
    flux, ivar = spec[i].numpy() * n, w[i].numpy() / n**2
    np.savez(OUT / "dark_spectrum.npz", targetid=TID, z=float(z[i]), flux=flux, ivar=ivar)
    path = OUT / "resolution_matrix.npz"
    resolution_check.TID = TID  # save_matrix reads the module TID
    resolution_check.save_matrix(COADD, path)
    new = load_resolution_npz(path)
    good = np.isfinite(flux) & (ivar > 0) & (new["ivar"] > 0)
    dflux = np.max(np.abs(new["flux"][good] - flux[good]) / np.abs(flux[good]).clip(1e-3))
    divar = np.max(np.abs(new["ivar"][good] / ivar[good] - 1))
    print(
        f"{TID} z {float(z[i]):.5f} good px {good.sum()} max flux rel diff {dflux:.2e} "
        f"max ivar rel diff {divar:.2e}"
    )
    assert dflux < 1e-4 and divar < 1e-4, "matrix file differs from the spender spectrum"
    print(f"wrote {OUT / 'dark_spectrum.npz'} and {path}")


if __name__ == "__main__":
    main()
