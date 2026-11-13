"""Save every bright exposure and coadd of gallery105 figures 05 and 67 from their healpix files.

Adapted from 2026-11-05_exposures_39627770174637084/extract.py (itself as
2026-11-02_exposure_checks/extract.py) to a list of targets: 05 (39627770317246630, healpix 25944,
SV3 and main bright) and 67 (39633136522363591, healpix 9841, SV3 bright only). For each target and
survey it saves the FIBERMAP columns, the per-arm flux, ivar and mask of every exposure from the
spectra file, and the target's row of the coadd file. Run with the DESI environment.
COMMAND: ``$DESIPY extract.py <spectra dir> <out dir>`` through extract.slurm.
OUTPUT: <out dir>/extract_<nn>.npz. Keys are <survey>_<column> for the exposures,
<survey>_<arm>_<flux|ivar|mask> for the exposures and coadd_<survey>_<arm>_<flux|ivar|mask> for the
coadds.
"""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from desispec.maskbits import fibermask

TARGETS = {
    "05": (39627770317246630, 25944, ("sv3", "main")),
    "67": (39633136522363591, 9841, ("sv3",)),
}
EXTRA = ("EXPID", "NIGHT", "TILEID", "FIBER", "PETAL_LOC", "EXPTIME", "FIBERSTATUS")


def rows_of(h, name, tid):
    """Return the FIBERMAP rows of the target in an open file."""
    rows = np.flatnonzero(h["FIBERMAP"].data["TARGETID"] == tid)
    assert rows.size, f"{tid} not in {name}"
    return rows


def extract(spec_dir, tid, hpx, surveys):
    """Return the exposures and coadds of one target as a dict of arrays."""
    save = {}
    for s in surveys:
        name = f"spectra-{s}-bright-{hpx}.fits"
        with fits.open(spec_dir / name, memmap=True) as h:
            rows = rows_of(h, name, tid)
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
        print(f"{tid} {s}: EXPID {save[f'{s}_EXPID'].tolist()} "
              f"FIBERSTATUS {save[f'{s}_fiberstatus_names'].tolist()}", flush=True)  # fmt: skip
        name = f"coadd-{s}-bright-{hpx}.fits"
        with fits.open(spec_dir / name, memmap=True) as h:
            rows = rows_of(h, name, tid)
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
        print(f"{tid} coadd {s}: COADD_NUMEXP {save[f'coadd_{s}_numexp']} "
              f"COADD_FIBERSTATUS {save[f'coadd_{s}_fiberstatus']}", flush=True)  # fmt: skip
    save["surveys"] = np.array(surveys)
    return save


def main(spec_dir, out_dir):
    """Extract every target and save one npz per target."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for nn, (tid, hpx, surveys) in TARGETS.items():
        np.savez(out_dir / f"extract_{nn}.npz", **extract(spec_dir, tid, hpx, surveys))
        print(f"wrote {out_dir / f'extract_{nn}.npz'}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
