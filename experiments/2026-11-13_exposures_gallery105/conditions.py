"""Save sky brightness, seeing and effective time of every exposure in the extract_<nn>.npz files.

One query of desi_dr1.exposure on the Astro Data Lab TAP service, as
2026-10-31_spectrum_qa_cont_outliers/exposures.py, for the EXPIDs of every extract_<nn>.npz.

COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-11-13_exposures_gallery105/conditions.py``.
OUTPUT: results/2026-11-13_exposures_gallery105/conditions.csv.
"""

import warnings

import numpy as np
from astroquery.utils.tap.core import TapPlus

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-13_exposures_gallery105"


def main():
    """Run the query and write the CSV."""
    expids = set()
    for p in sorted(OUT.glob("extract_*.npz")):
        d = np.load(p)
        expids |= {int(e) for s in d["surveys"] for e in d[f"{s}_EXPID"]}
    expids = sorted(expids)
    query = (
        "SELECT expid, tileid, night, survey, program, sky_mag_g_spec, efftime_spec, seeing_gfa "
        f"FROM desi_dr1.exposure WHERE expid IN ({', '.join(map(str, expids))})"
    )
    warnings.filterwarnings("ignore", module="astroquery")
    tab = (
        TapPlus(url="https://datalab.noirlab.edu/tap", verbose=False)
        .launch_job(query)
        .get_results()
    )
    missing = sorted(set(expids) - {int(e) for e in tab["expid"]})
    assert not missing, f"EXPIDs not in desi_dr1.exposure: {missing}"
    tab.write(OUT / "conditions.csv", format="csv", overwrite=True)
    print(f"{len(tab)} exposures -> conditions.csv")


if __name__ == "__main__":
    main()
