"""Save the DESI DR1 spectra (survey, program, z) of the Tang et al. sample from the Astro Data Lab.

One desi_dr1.zpix row per DR1 spectrum of the TARGETIDs in
results/2026-11-10_prescreen_replay/tang_desi_targetid.txt.
COMMAND: ``uv run --no-sync python experiments/2026-11-10_prescreen_replay/fetch.py``.
OUTPUT: results/2026-11-10_prescreen_replay/tang_zpix.fits.
"""

import warnings

import numpy as np
from astropy.table import vstack
from astroquery.utils.tap.core import TapPlus

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-10_prescreen_replay"


def main():
    """Run the query in four chunks and write the table."""
    warnings.filterwarnings("ignore", module="astroquery")
    tid = np.loadtxt(OUT / "tang_desi_targetid.txt", dtype=np.int64, comments="#")
    tap = TapPlus(url="https://datalab.noirlab.edu/tap", verbose=False)
    parts = []
    for chunk in np.array_split(tid, 4):
        ids = ",".join(map(str, chunk))
        parts.append(
            tap.launch_job(
                "SELECT targetid, survey, program, zcat_primary, z, zwarn "
                f"FROM desi_dr1.zpix WHERE targetid IN ({ids})"
            ).get_results()
        )
    tab = vstack(parts)
    assert np.unique(tab["targetid"]).size == tid.size, "some targets have no DR1 spectrum"
    tab.write(OUT / "tang_zpix.fits", overwrite=True)
    print(f"{len(tab)} spectra of {tid.size} targets")


if __name__ == "__main__":
    main()
