"""List the exposure-petals that hold our SV3 bright sample, with their calibration quality flags.

QUESTION: which exposure-petals must the flux calibration rerun cover, and which of them cannot be
recalibrated?

Every SV3 bright exposure of iron (exposures-iron.fits) is matched to the sample TARGETIDs
(data/all_target_ids.npy) through the FIBERQA table of its exposure-qa file. A petal is a unit if
it holds at least one sample target. For each unit the script keeps the PETALQA values NSTDSTAR,
STARRMS and NCFRAME, and counts the fibers whose QAFIBERSTATUS has BADPETALPOS (bit 25),
BADPETALSTDSTAR (bit 27) or BADPETALFLUXCAL (bit 28) (desispec maskbits.py). These petal bits are
not in the FIBERSTATUS of the coadd EXP_FIBERMAP (checked for exposure 85617 petal 2 in healpix
26275), and they are not among the bits that drop an exposure from the coadd
(fiberbitmasking.get_all_nonamp_fiberbitmask_val), so the coadds include such petals. With
BADPETALSTDSTAR desispec falls back to a default calibration with 100% uncertainty.

INPUTS: exposures-iron.fits and the exposure-qa files of iron (downloaded into --qa-dir if missing),
data/all_target_ids.npy. SEED: none.

COMMAND: from the repository root, ``uv run python experiments/2026-10-13_fluxcal_sample/make_units.py
<exposures-iron.fits> --qa-dir <dir>``.

OUTPUT: units_all.csv in this folder.
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from hubersed.paths import PATHS

HERE = Path(__file__).resolve().parent
BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/exposures"
BITS = {"BADPETALPOS": 25, "BADPETALSTDSTAR": 27, "BADPETALFLUXCAL": 28}


def qa_file(qa_dir, night, expid):
    """Return the path of an exposure-qa file, downloading it if needed."""
    path = qa_dir / f"exposure-qa-{expid:08d}.fits"
    if not path.exists():
        url = f"{BASE}/{night}/{expid:08d}/exposure-qa-{expid:08d}.fits"
        urllib.request.urlretrieve(url, path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("exposures", help="iron exposures-iron.fits")
    ap.add_argument("--qa-dir", required=True)
    a = ap.parse_args()
    qa_dir = Path(a.qa_dir)
    qa_dir.mkdir(parents=True, exist_ok=True)
    sample = np.load(PATHS["DATA"] / "all_target_ids.npy").astype(np.int64)
    with fits.open(a.exposures) as h:
        ex = h["EXPOSURES"].data
        sel = (np.char.strip(ex["SURVEY"].astype(str)) == "sv3") & (
            np.char.strip(ex["PROGRAM"].astype(str)) == "bright"
        )
        ex = ex[sel]
    rows = []
    for r in ex:
        night, expid, tile = int(r["NIGHT"]), int(r["EXPID"]), int(r["TILEID"])
        with fits.open(qa_file(qa_dir, night, expid)) as h:
            fq, pq = h["FIBERQA"].data, h["PETALQA"].data
            insamp = np.isin(fq["TARGETID"].astype(np.int64), sample)
            status = fq["QAFIBERSTATUS"].astype(np.int64)
            for p in pq:
                pl = int(p["PETAL_LOC"])
                m = fq["PETAL_LOC"] == pl
                if not insamp[m].any():
                    continue
                row = dict(
                    NIGHT=night,
                    EXPID=expid,
                    TILEID=tile,
                    PETAL=pl,
                    NSAMPLE=int(insamp[m].sum()),
                    NSTDSTAR=int(p["NSTDSTAR"]),
                    STARRMS=float(p["STARRMS"]),
                    NCFRAME=int(p["NCFRAME"]),
                )
                for name, bit in BITS.items():
                    row[f"N_{name}"] = int(((status[m] >> bit) & 1).sum())
                rows.append(row)
    t = Table(rows=rows)
    t.write(HERE / "units_all.csv", overwrite=True)
    bad = t["N_BADPETALSTDSTAR"] > 0
    print(
        f"SV3 bright exposures {len(ex)}, units {len(t)}, sample target-exposure pairs {t['NSAMPLE'].sum()}"
    )
    print(f"units with BADPETALSTDSTAR {bad.sum()} (sample pairs {t['NSAMPLE'][bad].sum()}), "
          f"BADPETALPOS {(t['N_BADPETALPOS'] > 0).sum()}, BADPETALFLUXCAL {(t['N_BADPETALFLUXCAL'] > 0).sum()}, "
          f"NCFRAME<3 {(t['NCFRAME'] < 3).sum()}")  # fmt: skip


if __name__ == "__main__":
    main()
