"""Save every SV3 and main bright exposure and both coadds of 39627770174637084.

As 2026-11-02_exposure_checks/extract.py, for one target in the SV3 and main bright healpix 26004
spectra files, plus its rows of the two coadd files. Run with the DESI environment.
COMMAND: ``$DESIPY extract.py <spectra dir> <out.npz>`` through extract.slurm.
OUTPUT: <out.npz>. Keys are <survey>_<column> for the exposures, <survey>_<arm>_<flux|ivar|mask>
for the exposures and coadd_<survey>_<arm>_<flux|ivar|mask> for the coadds.
"""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from desispec.maskbits import fibermask

TARGETID = 39627770174637084
HEALPIX = 26004
SURVEYS = ("sv3", "main")
EXTRA = ("EXPID", "NIGHT", "TILEID", "FIBER", "PETAL_LOC", "EXPTIME", "FIBERSTATUS")


def rows_of(h, name):
    """Return the FIBERMAP rows of the target in an open file."""
    rows = np.flatnonzero(h["FIBERMAP"].data["TARGETID"] == TARGETID)
    assert rows.size, f"{TARGETID} not in {name}"
    return rows


def main(spec_dir, out):
    """Extract the exposures and coadd of each survey and save them in one npz."""
    save = {}
    for s in SURVEYS:
        name = f"spectra-{s}-bright-{HEALPIX}.fits"
        with fits.open(spec_dir / name, memmap=True) as h:
            rows = rows_of(h, name)
            fm = h["FIBERMAP"].data
            for k in EXTRA:
                save[f"{s}_{k}"] = np.asarray(fm[k][rows])
            save[f"{s}_fiberstatus_names"] = np.array(
                [",".join(fibermask.names(int(v))) for v in fm["FIBERSTATUS"][rows]]
            )
            for arm in "BRZ":
                save[f"{arm}_wave"] = np.asarray(h[f"{arm}_WAVELENGTH"].data, float)
                for kind in ("FLUX", "IVAR", "MASK"):
                    save[f"{s}_{arm}_{kind.lower()}"] = np.asarray(h[f"{arm}_{kind}"].data[rows])
        print(f"{s}: EXPID {save[f'{s}_EXPID'].tolist()} "
              f"FIBERSTATUS {save[f'{s}_fiberstatus_names'].tolist()}", flush=True)  # fmt: skip
        name = f"coadd-{s}-bright-{HEALPIX}.fits"
        with fits.open(spec_dir / name, memmap=True) as h:
            rows = rows_of(h, name)
            assert rows.size == 1, rows
            fm = h["FIBERMAP"].data
            save[f"coadd_{s}_fiberstatus"] = int(fm["COADD_FIBERSTATUS"][rows[0]])
            save[f"coadd_{s}_numexp"] = int(fm["COADD_NUMEXP"][rows[0]])
            for arm in "BRZ":
                assert np.allclose(h[f"{arm}_WAVELENGTH"].data, save[f"{arm}_wave"])
                for kind in ("FLUX", "IVAR", "MASK"):
                    save[f"coadd_{s}_{arm}_{kind.lower()}"] = np.asarray(
                        h[f"{arm}_{kind}"].data[rows[0]]
                    )
        print(f"coadd {s}: COADD_NUMEXP {save[f'coadd_{s}_numexp']} "
              f"COADD_FIBERSTATUS {save[f'coadd_{s}_fiberstatus']}", flush=True)  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **save)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
