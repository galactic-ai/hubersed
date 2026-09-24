"""Screen continuum-flow outlier candidates for Gaia stars and nearby SGA-2020 galaxies.

The candidates are the TARGETIDs flagged by both continuum flows, and the Gaia and SGA results
go to two CSV files. Run it with ``uv run python scripts/contam_screens.py --flow-dir DIR``,
or add ``--self-test`` for the offline geometry check.
"""

import argparse
import csv
import warnings
from pathlib import Path

from hubersed.detect.sky_screens import (
    DL_TAP,
    candidate_pool,
    gaia_screen,
    self_test,
    sga_screen,
)
from hubersed.paths import PATHS


def main(argv=None):
    """Run the Gaia and SGA screens on every candidate and write the two CSV files.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. None reads ``sys.argv``.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--flow-dir",
        required=True,
        help="dir holding desi_outliers_flow_nsf_<tag>_snr3.pt for each tag",
    )
    p.add_argument("--vac", default=str(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits"))
    p.add_argument("--gaia-out", default=str(PATHS["RESULTS"] / "gaia_star_screen.csv"))
    p.add_argument("--sga-out", default=str(PATHS["RESULTS"] / "sga_proximity.csv"))
    p.add_argument("--limit", type=int, default=0, help="screen only the first N candidates")
    p.add_argument("--self-test", action="store_true", help="run the offline geometry check")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()

    warnings.filterwarnings("ignore")
    from astroquery.utils.tap.core import TapPlus

    tap = TapPlus(url=DL_TAP)

    sel, ra, dec = candidate_pool(args.flow_dir, args.vac)
    if args.limit:
        sel, ra, dec = sel[: args.limit], ra[: args.limit], dec[: args.limit]
    print(f"screening {len(sel)} candidates")

    gaia_rows, sga_rows = [], []
    for k, (t, r, d) in enumerate(zip(sel, ra, dec, strict=True), 1):
        gaia_rows.append({"target_id": t, **gaia_screen(r, d)})
        sga_rows.append({"target_id": t, **sga_screen(r, d, tap)})
        if k % 20 == 0 or k == len(sel):
            print(
                f"  {k}/{len(sel)}  gaia flagged {sum(x['flagged'] for x in gaia_rows)}  "
                f"sga flagged {sum(x['flagged'] for x in sga_rows)}"
            )

    for path, rows in ((args.gaia_out, gaia_rows), (args.sga_out, sga_rows)):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out}  n={len(rows)}  flagged={sum(x['flagged'] for x in rows)}")


if __name__ == "__main__":
    main()
