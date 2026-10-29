"""Build the spender-masked DR1 main-survey spectrum of 39627757533007793 and its resolution matrix.

spender's own DESI.prepare_spectra (spender/data/desi.py) reads a healpix coadd and its redrock
file, merges the arms, zeroes the weights where the DESI mask is set, where ivar <= 1e-6 and
around bright sky lines, and divides by a median norm. Run on the SV3 coadd first, the result must
equal the SV3 spender chunk that load_spectrum reads, which checks the path. Then it runs on the
main coadd (with MAIN_DESI_TARGET aliased to DESI_TARGET, see spender_spectrum), and the flux
and ivar are saved without the norm. The main resolution matrix is saved
with resolution_check.save_matrix, as in 2026-10-28. The fibre correction columns of both coadd
FIBERMAPs are printed.

COMMAND: on a stampede3 compute node from the repository root, through check.slurm.

OUTPUT: results/2026-10-29_continuum_main_survey_map/main_spectrum.npz and resolution_matrix.npz.
"""

import os
from pathlib import Path

import numpy as np
import resolution_check
import spender.data.desi as sdesi
from astropy.io import fits
from spender.data.desi import DESI

import hubersed.io.desi as desi_io
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.sps.resolution_matrix import load_resolution_npz

TID = 39627757533007793
DATA = Path("/work/11006/nikhilgaruda/ls6/research/spender/DATA")
SV3 = DATA / "25598" / "coadd-sv3-bright-25598.fits"
MAIN = DATA / "main_25598" / "coadd-main-bright-25598.fits"
OUT = PATHS["RESULTS"] / "2026-10-29_continuum_main_survey_map"
if os.environ.get("SPEC_DATA"):
    desi_io.DATA_PATH = Path(os.environ["SPEC_DATA"])


def spender_spectrum(coadd, main_alias=False):
    """Return flux, ivar and z of the target from spender's prepare_spectra, without its norm.

    prepare_spectra cuts on the column <SURVEY>_DESI_TARGET, which main-survey coadds call
    DESI_TARGET. With ``main_alias`` the tables it reads get MAIN_DESI_TARGET as a copy of
    DESI_TARGET for that call only.
    """
    read = sdesi.aTable.Table.read

    def read_with_alias(*args, **kwargs):
        t = read(*args, **kwargs)
        if "DESI_TARGET" in t.colnames and "MAIN_DESI_TARGET" not in t.colnames:
            t["MAIN_DESI_TARGET"] = t["DESI_TARGET"]
        return t

    if main_alias:
        sdesi.aTable.Table.read = read_with_alias
    try:
        spec, w, z, tid, norm, _ = DESI.prepare_spectra(str(coadd))
    finally:
        sdesi.aTable.Table.read = read
    tid = np.asarray(tid)
    i = np.where(tid == TID)[0]
    assert i.size == 1, f"target kept {i.size} times by prepare_spectra"
    i = int(i[0])
    n = float(norm[i])
    return spec[i].numpy() * n, w[i].numpy() / n**2, float(z[i])


def fibre_columns(coadd):
    """Print the fibre correction and positioning columns of the target in the coadd FIBERMAP."""
    with fits.open(coadd) as h:
        fm = h["FIBERMAP"].data
        row = fm[fm["TARGETID"] == TID][0]
        keys = [
            k
            for k in fm.columns.names
            if any(
                s in k for s in ("PSF_TO_FIBER", "DELTA", "FIBER_X", "FIBER_Y", "NUMEXP", "EXPTIME")
            )
        ]
        print(coadd.name, {k: row[k] for k in keys})


def main():
    """Check the spender path on SV3, then save the main spectrum and matrix."""
    OUT.mkdir(parents=True, exist_ok=True)
    f3, iv3, z3 = spender_spectrum(SV3)
    s = load_spectrum(TID)
    chunk_f, chunk_iv = np.asarray(s.flux.value, float), np.asarray(s.uncertainty.array, float)
    assert np.isclose(z3, float(s.redshift.value), atol=1e-6), (z3, float(s.redshift.value))
    assert np.array_equal(iv3 > 0, chunk_iv > 0), "SV3 mask differs from the chunk"
    ok = chunk_iv > 0
    df = np.max(np.abs(f3[ok] - chunk_f[ok]) / np.abs(chunk_f[ok]).clip(1e-3))
    di = np.max(np.abs(iv3[ok] / chunk_iv[ok] - 1))
    print(f"SV3 check: same mask, max flux rel diff {df:.2e}, max ivar rel diff {di:.2e}")
    assert df < 1e-5 and di < 1e-5, "prepare_spectra does not reproduce the SV3 chunk"

    fm, ivm, zm = spender_spectrum(MAIN, main_alias=True)
    np.savez(OUT / "main_spectrum.npz", flux=fm, ivar=ivm, z=zm, coadd=str(MAIN))
    print(f"main: z {zm:.6f}, unmasked px {(ivm > 0).sum()} (SV3 {(iv3 > 0).sum()})")

    resolution_check.TID = TID  # save_matrix reads the module TID
    resolution_check.save_matrix(MAIN, OUT / "resolution_matrix.npz")
    new = load_resolution_npz(OUT / "resolution_matrix.npz")
    good = (ivm > 0) & (new["ivar"] > 0)
    dflux = np.max(np.abs(new["flux"][good] - fm[good]) / np.abs(fm[good]).clip(1e-3))
    print(f"main matrix: ndiag {new['res'].shape[0]}, flux rel diff to spender {dflux:.2e}")
    assert dflux < 1e-4, "matrix file differs from the spender spectrum"
    for c in (SV3, MAIN):
        fibre_columns(c)


if __name__ == "__main__":
    main()
