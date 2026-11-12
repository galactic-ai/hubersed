"""Save the DESI DR1 spectra (survey, program, z) of the Tang et al. sample from the Astro Data Lab.

Tang et al. (arXiv:2611.02580) sent their 1,199 TARGETIDs by email; the list is
results/2026-11-10_prescreen_replay/tang_desi_targetid.txt. One row per DR1 spectrum in
desi_dr1.zpix, so a target can appear more than once.
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
