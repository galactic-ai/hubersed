"""Save the target's exposure and the bump ratio of every target fibre in its healpix file.

For each target in targets.txt: its rows of the healpix spectra file, and for every other fibre of the
same EXPID and PETAL_LOC, and for every TGT fibre of the file (all exposures, as a baseline), the
median z-arm flux at observed BAND over the sidebands SIDE.
Run with the DESI environment through neighbours.slurm.
COMMAND: ``$DESIPY neighbours.py <unzipped dir> <out dir>``.
OUTPUT: <out dir>/<TARGETID>_neighbours.npz.
"""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from desispec.maskbits import fibermask

HERE = Path(__file__).resolve().parent
BAND = (8106.0, 8360.0)  # observed A, rest 6400-6600 at z 0.2665
SIDE = ((7854.0, 8043.0), (8486.0, 8676.0))  # observed A, rest 6200-6350 and 6700-6850
EXTRA = ("TARGETID", "EXPID", "NIGHT", "TILEID", "FIBER", "PETAL_LOC", "EXPTIME", "FIBERSTATUS")


def ratio(w, f, iv):
    """Return median flux in BAND over median flux in SIDE, NaN without enough good pixels."""
    ok = iv > 0
    b = ok & (w > BAND[0]) & (w < BAND[1])
    s = ok & (((w > SIDE[0][0]) & (w < SIDE[0][1])) | ((w > SIDE[1][0]) & (w < SIDE[1][1])))
    if b.sum() < 50 or s.sum() < 50 or np.median(f[s]) <= 0:
        return np.nan
    return float(np.median(f[b]) / np.median(f[s]))


def main(spec_dir, out_dir):
    """Write the target exposure and the same-exposure, same-petal ratios."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for line in (HERE / "targets.txt").read_text().splitlines():
        t, hp = (int(x) for x in line.split())
        with fits.open(spec_dir / f"spectra-sv3-bright-{hp}.fits", memmap=True) as h:
            fm = h["FIBERMAP"].data
            rows = np.flatnonzero(fm["TARGETID"] == t)
            assert rows.size, f"{t} not in healpix {hp}"
            out = {k: np.asarray(fm[k][rows]) for k in EXTRA if k in fm.columns.names}
            out["fiberstatus_names"] = np.array([",".join(fibermask.names(int(s)))
                                                 for s in out["FIBERSTATUS"]])  # fmt: skip
            w = np.asarray(h["Z_WAVELENGTH"].data, float)
            zf, zi = h["Z_FLUX"].data, h["Z_IVAR"].data
            for arm in "BRZ":
                out[f"{arm}_wave"] = np.asarray(h[f"{arm}_WAVELENGTH"].data, float)
                for kind in ("FLUX", "IVAR", "MASK"):
                    out[f"{arm}_{kind.lower()}"] = np.asarray(h[f"{arm}_{kind}"].data[rows])
            same = np.flatnonzero(np.isin(fm["EXPID"], out["EXPID"])
                                  & np.isin(fm["PETAL_LOC"], out["PETAL_LOC"]))  # fmt: skip
            out["nb_targetid"] = np.asarray(fm["TARGETID"][same])
            out["nb_fiber"] = np.asarray(fm["FIBER"][same])
            out["nb_objtype"] = np.asarray(fm["OBJTYPE"][same]).astype(str)
            out["nb_ratio"] = np.array([ratio(w, np.asarray(zf[i], float), np.asarray(zi[i], float))
                                        for i in same])  # fmt: skip
            tgt = np.flatnonzero(fm["OBJTYPE"] == "TGT")
            out["all_expid"] = np.asarray(fm["EXPID"][tgt])
            out["all_petal"] = np.asarray(fm["PETAL_LOC"][tgt])
            out["all_fiber"] = np.asarray(fm["FIBER"][tgt])
            out["all_targetid"] = np.asarray(fm["TARGETID"][tgt])
            out["all_ratio"] = np.array([ratio(w, np.asarray(zf[i], float), np.asarray(zi[i], float))
                                         for i in tgt])  # fmt: skip
            out["target_ratio"] = np.array([ratio(w, np.asarray(zf[i], float),
                                                  np.asarray(zi[i], float)) for i in rows])  # fmt: skip
        np.savez(out_dir / f"{t}_neighbours.npz", **out)
        r = out["nb_ratio"]
        print(f"{t} hp {hp}: EXPID {out['EXPID'].tolist()} PETAL {out['PETAL_LOC'].tolist()} "
              f"FIBERSTATUS {out['fiberstatus_names'].tolist()}; target ratio "
              f"{out['target_ratio'].round(3).tolist()}; {np.isfinite(r).sum()} neighbours, median "
              f"{np.nanmedian(r):.3f}, 16-84% {np.nanpercentile(r, [16, 84]).round(3).tolist()}",
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
