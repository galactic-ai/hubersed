"""Fetch every DESI DR1 coadd spectrum of the sample.py targets from SPARCL.

As experiments/2026-11-10_data_quality_screen/repeats_fetch.py, for the targets of targets.npz
(sample.py). It saves all DESI-DR1 records of each TARGETID (every survey and program) with flux,
ivar and mask on the 7781 pixel DESI grid and their catalogue fields. Records are found in chunks
of 500 target ids. sparclclient is not a project dependency, so this runs in its own uv
environment and does not import hubersed.
COMMAND: from the repository root,
``uv run --no-project --python 3.13 --with sparclclient --with "numpy>=2" --with "pandas>=2.2"
python experiments/2026-11-13_repeat_persistence/fetch.py``.
OUTPUT: results/2026-11-13_repeat_persistence/sparcl_spectra.npz.
"""

from pathlib import Path

import numpy as np
from sparcl.client import SparclClient

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "2026-11-13_repeat_persistence"
FIELDS = ["targetid", "survey", "program", "healpix", "redshift", "redshift_warning", "spectype",
          "coadd_fiberstatus", "coadd_numexp", "tsnr2_bgs", "deltachi2"]  # fmt: skip
ARRAYS = ["flux", "ivar", "mask", "wavelength"]


def main():
    """Find and retrieve the records, check the grid, and save them."""
    s = np.load(OUT / "targets.npz")
    tids = sorted({int(t) for k in s.files for t in s[k]})
    client = SparclClient()
    ids = []
    for i in range(0, len(tids), 500):
        found = client.find(
            outfields=["sparcl_id", "targetid"],
            constraints={"targetid": tids[i : i + 500], "data_release": ["DESI-DR1"]},
            limit=10000,
        )
        ids += [r["sparcl_id"] for r in found.records]
        print(f"found {len(ids)} records after {min(i + 500, len(tids))} targets", flush=True)
    recs = []
    for i in range(0, len(ids), 500):
        got = client.retrieve(uuid_list=ids[i : i + 500], include=FIELDS + ARRAYS,
                              dataset_list=["DESI-DR1"], limit=500)  # fmt: skip
        recs += [dict(r) for r in got.records]
        print(f"retrieved {len(recs)} of {len(ids)}", flush=True)
    assert len(recs) == len(ids), (len(recs), len(ids))
    wave = np.asarray(recs[0]["wavelength"], float)
    assert all(np.array_equal(np.asarray(r["wavelength"], float), wave) for r in recs)
    out = {k: np.array([r.get(k) for r in recs]) for k in FIELDS}
    out.update({k: np.array([np.asarray(r[k]) for r in recs]) for k in ("flux", "ivar", "mask")})
    np.savez(OUT / "sparcl_spectra.npz", wavelength=wave, sparcl_id=np.array(ids), **out)
    missing = sorted(set(tids) - {int(t) for t in out["targetid"]})
    print(f"{len(recs)} spectra of {len(set(out['targetid'].tolist()))} of {len(tids)} targets; "
          f"missing {len(missing)}")  # fmt: skip


if __name__ == "__main__":
    main()
