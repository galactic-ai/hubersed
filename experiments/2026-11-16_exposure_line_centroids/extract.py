"""Extract the per-exposure b and r rows of the emission-line outliers and the #14 target.

Uses exposure() of 2026-10-09_nautilus_c3k_er_broadline/ivar_recompute.py (iron cframe flux, iron and
recomputed ivar) and the iron and PR 2377 cosmic masks of coadd_rows.py, and keeps each exposure's
header values. Run through extract.slurm, with that folder on PYTHONPATH.
OUTPUT: results/2026-11-16_exposure_line_centroids/rows_<TID>.npz.
"""

import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from coadd_rows import iron_cosmics, pr2377_cosmics
from ivar_recompute import DATA, exposure
from targets import RESULTS, TARGETS

TIDS = (39627818010675704, 39627854832472157, 39633318999755849, 39627806493117457,
        39633127609469239, 39627770174637084)  # fmt: skip
# the results folder that holds the patched fluxcalib
ROOT = Path(os.environ.get("RESULTS_ROOT", RESULTS))
OUT = ROOT / "2026-11-16_exposure_line_centroids"
KEYS = ("NIGHT", "EXPID", "TILEID", "MJD-OBS", "EXPTIME", "HELIOCOR")


def header(night, expid, cam, tid):
    """Return the header values of KEYS and the fiber of ``tid`` in one cframe."""
    with fits.open(DATA / night / f"{expid:08d}" / f"cframe-{cam}-{expid:08d}.fits.gz") as cf:
        heads = [cf[0].header, cf["FLUX"].header]
        vals = [float(next((h[k] for h in heads if k in h), np.nan)) for k in KEYS]
        row = int(np.where(cf["FIBERMAP"].data["TARGETID"] == tid)[0][0])
        fiber = int(cf["FIBERMAP"].data["FIBER"][row])
    return vals, fiber


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for tid in TIDS:
        # targets.py places outputs under its own repository copy; point them at ROOT instead
        os.environ["RECOADD_OUT"] = str(ROOT / TARGETS[tid]["out"].relative_to(RESULTS))
        save = {}
        for arm in "br":
            es, meta, fibers = [], [], []
            for expid, (night, petal, _) in TARGETS[tid]["exposures"].items():
                cam = f"{arm}{petal}"
                es.append(exposure(night, expid, cam, tid))
                vals, fiber = header(night, expid, cam, tid)
                meta.append(vals)
                fibers.append(fiber)
            stack = {k: np.array([e[k] for e in es]) for k in ("flux", "ivar", "ivar_iron", "mask")}
            wave = es[0]["wave"]
            assert all(np.allclose(e["wave"], wave, atol=1e-4) for e in es), "arm grids differ"
            save |= {
                f"{arm}_wave": wave,
                f"{arm}_flux": stack["flux"],
                f"{arm}_ivar_iron": stack["ivar_iron"],
                f"{arm}_ivar_new": stack["ivar"],
                f"{arm}_mask": stack["mask"],
                f"{arm}_cosmic_iron": iron_cosmics(wave, stack["flux"], stack["ivar_iron"], stack["mask"]),
                f"{arm}_cosmic_new": pr2377_cosmics(wave, stack["flux"], stack["ivar"] * (stack["mask"] == 0)),
                f"{arm}_meta": np.array(meta),
                f"{arm}_fiber": np.array(fibers),
            }  # fmt: skip
        np.savez(OUT / f"rows_{tid}.npz", tid=tid, meta_keys=np.array(KEYS), **save)
        print(f"{tid}: {len(es)} exposures, wrote {OUT / f'rows_{tid}.npz'}", flush=True)


if __name__ == "__main__":
    main()
