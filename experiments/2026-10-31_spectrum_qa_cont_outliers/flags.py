"""Save the DR1 SV3 bright redshift-catalogue flags and fiber diagnostics of the outliers and controls.

Queries desi_dr1.zpix on the Astro Data Lab TAP service for the 94 outliers and the three control
sets of controls.npz, 200 TARGETIDs per query. Per-exposure FIBERSTATUS bits are not in these
tables. COADD_FIBERSTATUS keeps only the bits set in every exposure.

COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-31_spectrum_qa_cont_outliers/flags.py``.
OUTPUT: results/2026-10-31_spectrum_qa_cont_outliers/zpix_flags.csv.
"""

import warnings

import numpy as np
from astropy.table import vstack
from astroquery.utils.tap.core import TapPlus

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-31_spectrum_qa_cont_outliers"
COLS = (
    "targetid, coadd_fiberstatus, zwarn, deltachi2, coadd_numexp, coadd_numnight, coadd_numtile, "
    "coadd_exptime, mean_delta_x, mean_delta_y, rms_delta_x, rms_delta_y, std_fiber_ra, "
    "std_fiber_dec, mean_psf_to_fiber_specflux, sv_primary, zcat_primary"
)
BATCH = 200


def main():
    """Query zpix in batches and write one CSV row per target."""
    warnings.filterwarnings("ignore", module="astroquery")
    ctl = np.load(OUT / "controls.npz")
    tids = np.unique(
        np.concatenate(
            [ctl["outliers"]] + [ctl[k].ravel() for k in ("matched", "cont10", "cont15")]
        )
    )
    tap = TapPlus(url="https://datalab.noirlab.edu/tap", verbose=False)
    parts = []
    for i in range(0, len(tids), BATCH):
        ids = ",".join(str(int(t)) for t in tids[i : i + BATCH])
        q = (
            f"SELECT {COLS} FROM desi_dr1.zpix WHERE survey = 'sv3' AND program = 'bright' "
            f"AND targetid IN ({ids})"
        )
        parts.append(tap.launch_job(q).get_results())
    tab = vstack(parts)
    if len(tab) != len(tids) or len(np.unique(tab["targetid"])) != len(tids):
        raise RuntimeError(f"{len(tab)} rows for {len(tids)} targets")
    tab.write(OUT / "zpix_flags.csv", format="csv", overwrite=True)
    print(f"{len(tab)} targets -> zpix_flags.csv")


if __name__ == "__main__":
    main()
