"""Save the DESI resolution matrix of four more line outliers and check their per-exposure S/N.

QUESTION: the corrected calibration ivar of 39627770174637084 changes sigma by 1-4% at per-exposure
pixel S/N 5-20, 6-31% at 20-40 and 17-152% above 40 (ivar_recompute.txt). Which of the four
candidates have enough pixels above S/N 20 to need the recoadd?

Candidates are from the v2 MAP fits of the 310 outliers: 39628357133926597, 39633278973513279,
39632961393395307 and 39633001071509599. The target 39627770174637084 is the reference.

The matrix, flux, ivar and mask are saved with resolution_check.save_matrix, unchanged, so they
merge the arms as the reference does. Per-exposure S/N is the coadd S/N over sqrt(COADD_NUMEXP),
which assumes equal exposures. Line pixels are within 400 km/s of the strong lines. The exposure
list (EXPID, PETAL_LOC, FIBER, CAMERA arms) is written for a later recoadd.

INPUTS: DR1 iron healpix coadds copied from ls6 spender/DATA to data/sv3_bright_coadds/, and
resolution_matrix.npz of the reference. Redshifts from fastspec-iron-sv3-bright METADATA. SEED: none.

COMMAND: from the repository root on the Mac,
``PYTHONPATH=experiments/2026-10-09_nautilus_c3k_er_broadline uv run --no-sync python
experiments/2026-10-18_candidate_resolution_matrix/analysis.py``.

OUTPUT: results/2026-10-18_candidate_resolution_matrix/, one folder per target with
resolution_matrix.npz, and snr_table.txt.
"""

import numpy as np
import resolution_check
from astropy.io import fits

from hubersed.paths import PATHS

REF = 39627770174637084
REF_NUMEXP = 2  # exposures 87113 and 88269, coadd_rows.txt
CANDIDATES = {
    39628357133926597: 28115,
    39633278973513279: 11404,
    39632961393395307: 10512,
    39633001071509599: 10518,
}
COADDS = PATHS["DATA"] / "sv3_bright_coadds"
OUT = PATHS["RESULTS"] / "2026-10-18_candidate_resolution_matrix"
LINES = (3727.09, 3729.88, 4862.68, 4960.30, 5008.24, 6549.86, 6564.61, 6585.27, 6718.29, 6732.67)
DV = 400.0  # km/s
CUTS = (20.0, 40.0)


def redshifts():
    """Return the fastspec redshift of every target."""
    m = fits.getdata(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", "METADATA")
    tids = [REF, *CANDIDATES]
    return {t: float(m["Z"][np.where(m["TARGETID"] == t)[0][0]]) for t in tids}


def exposures(coadd, tid):
    """Return the coadd numexp and the per-exposure rows of ``tid``."""
    with fits.open(coadd) as h:
        fm = h["FIBERMAP"].data
        numexp = int(fm["COADD_NUMEXP"][fm["TARGETID"] == tid][0])
        ex = h["EXP_FIBERMAP"].data
        ex = ex[ex["TARGETID"] == tid]
        rows = [f"{e['EXPID']} petal {e['PETAL_LOC']} fiber {e['FIBER']} night {e['NIGHT']}"
                for e in ex]  # fmt: skip
    return numexp, rows


def snr_row(tid, npz, numexp, z):
    """Return one table line of per-exposure S/N counts for ``tid``."""
    d = np.load(npz)
    parts = []
    for b in "brz":
        wave, flux, ivar, mask = (d[f"{b}_{k}"] for k in ("wave", "flux", "ivar", "mask"))
        good = (mask == 0) & (ivar > 0)
        snr = flux * np.sqrt(np.clip(ivar, 0, None)) / np.sqrt(numexp)
        rest = wave / (1 + z)
        is_line = np.zeros(wave.size, bool)
        for lam in LINES:
            is_line |= np.abs(rest / lam - 1) * 2.998e5 < DV
        for name, sel in (("line", good & is_line), ("cont", good & ~is_line)):
            n = [int(np.sum(snr[sel] > c)) for c in CUTS]
            top = np.max(snr[sel]) if sel.any() else np.nan
            parts.append(f"{b} {name} >20 {n[0]:4d} >40 {n[1]:4d} max {top:6.1f}")
    return f"{tid} numexp {numexp}\n  " + "\n  ".join(parts)


def main():
    """Save the matrices and write the S/N table."""
    z = redshifts()
    lines = [
        "Per-exposure pixel S/N = coadd S/N / sqrt(numexp), unmasked pixels with ivar > 0.",
        f"line: within {DV:g} km/s of {len(LINES)} strong lines; cont: the rest.",
        "",
    ]
    ref_npz = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline" / str(REF)
    lines.append("reference " + snr_row(REF, ref_npz / "resolution_matrix.npz", REF_NUMEXP, z[REF]))
    for tid, hp in CANDIDATES.items():
        coadd = COADDS / f"coadd-sv3-bright-{hp}.fits"
        dest = OUT / str(tid)
        dest.mkdir(parents=True, exist_ok=True)
        resolution_check.TID = tid  # save_matrix reads the module TID
        resolution_check.save_matrix(coadd, dest / "resolution_matrix.npz")
        numexp, rows = exposures(coadd, tid)
        lines += ["", f"healpix {hp}, z {z[tid]:.5f}"]
        lines.append(snr_row(tid, dest / "resolution_matrix.npz", numexp, z[tid]))
        lines += ["  exposure " + r for r in rows]
        print("\n".join(lines[-8:]), flush=True)
    (OUT / "snr_table.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'snr_table.txt'}")


if __name__ == "__main__":
    main()
