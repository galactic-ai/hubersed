"""Fetch every DESI DR1 coadd spectrum of the 92 canonical continuum outliers from SPARCL.

Targets are cont of results/prescreened_outliers_postsimbad.npz. All DESI-DR1 SPARCL records of each
TARGETID (every survey and program) are saved with flux, ivar and mask on the 7781 pixel DESI grid
and their catalogue fields. sparclclient is not a project dependency, so this runs in its own uv
environment and does not import hubersed.

COMMAND: from the repository root,
``uv run --no-project --python 3.13 --with sparclclient --with "numpy>=2" --with "pandas>=2.2"
python experiments/2026-11-01_repeats92/fetch.py``.

OUTPUT: results/2026-11-01_repeats92/sparcl_spectra.npz.
"""

from pathlib import Path

import numpy as np
from sparcl.client import SparclClient

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ROOT / "results" / "prescreened_outliers_postsimbad.npz"
OUT = ROOT / "results" / "2026-11-01_repeats92"
FIELDS = ["targetid", "survey", "program", "healpix", "redshift", "redshift_warning", "spectype",
          "coadd_fiberstatus", "coadd_numexp", "tsnr2_bgs", "deltachi2"]  # fmt: skip
ARRAYS = ["flux", "ivar", "mask", "wavelength"]


def main():
    """Find and retrieve the records, check the grid, and save them."""
    tids = [int(t) for t in np.load(TARGETS)["cont"]]
    assert len(tids) == 92, len(tids)
    client = SparclClient()
    found = client.find(
        outfields=["sparcl_id", "targetid"],
        constraints={"targetid": tids, "data_release": ["DESI-DR1"]},
        limit=2000,
    )
    ids = [r["sparcl_id"] for r in found.records]
    got = client.retrieve(uuid_list=ids, include=FIELDS + ARRAYS, dataset_list=["DESI-DR1"],
                          limit=2000)  # fmt: skip
    recs = [dict(r) for r in got.records]
    assert len(recs) == len(ids), (len(recs), len(ids))
    wave = np.asarray(recs[0]["wavelength"], float)
    assert all(np.array_equal(np.asarray(r["wavelength"], float), wave) for r in recs)
    OUT.mkdir(parents=True, exist_ok=True)
    out = {k: np.array([r.get(k) for r in recs]) for k in FIELDS}
    out.update({k: np.array([np.asarray(r[k]) for r in recs]) for k in ("flux", "ivar", "mask")})
    np.savez(OUT / "sparcl_spectra.npz", wavelength=wave, sparcl_id=np.array(ids), **out)
    missing = sorted(set(tids) - {int(t) for t in out["targetid"]})
    print(f"{len(recs)} spectra of {len(set(out['targetid'].tolist()))} targets; missing {missing}")


if __name__ == "__main__":
    main()
