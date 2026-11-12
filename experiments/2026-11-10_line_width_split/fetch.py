"""Save the EmFit line widths of every DR1 SV3 bright spectrum from the Astro Data Lab.

Changed from 2026-11-06_line_ew_prior/fetch_emfit.py: the same chunked synchronous queries of
desi_dr1.emfit (Pucha+2025 method, arXiv:2411.00091), but for the widths. Columns: TARGETID, z,
prob_broad, and for Halpha (narrow, outflow, broad), [OIII]5007 and [SII]6716 (primary, outflow)
the intrinsic sigma in km/s, its flag (0 resolved and corrected, 1 unresolved, -1 undetected) and
the flux.

COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-11-10_line_width_split/fetch.py``.
OUTPUT: results/2026-11-10_line_width_split/emfit_widths_sv3_bright.fits; chunks in
emfit_chunks/ let a rerun resume.
"""

import warnings

import numpy as np
from astropy.table import Table, vstack
from astroquery.utils.tap.core import TapPlus

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-10_line_width_split"
COMPONENTS = ("ha_n", "ha_out", "ha_b", "oiii5007", "oiii5007_out", "sii6716", "sii6716_out")


def main():
    """Run the query and write the table."""
    cols = ["targetid", "z", "prob_broad"] + [
        f"{c}_{k}" for c in COMPONENTS for k in ("sigma", "sigma_flag", "flux")
    ]
    where = "survey = 'sv3' AND program = 'bright'"
    warnings.filterwarnings("ignore", module="astroquery")
    tap = TapPlus(url="https://datalab.noirlab.edu/tap", verbose=False)
    parts = []
    chunks = OUT / "emfit_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0, 12 * 64**2, 49).astype(int)  # nside 64 healpix, 48 chunks
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        path = chunks / f"healpix_{lo:05d}_{hi:05d}.fits"
        if path.exists():  # saved by an earlier run
            parts.append(Table.read(path))
            continue
        cut = f"{where} AND healpix >= {lo} AND healpix < {hi}"
        n = int(
            tap.launch_job(f"SELECT COUNT(*) AS n FROM desi_dr1.emfit WHERE {cut}").get_results()[
                "n"
            ][0]
        )
        part = tap.launch_job(
            f"SELECT TOP 1000000 {', '.join(cols)} FROM desi_dr1.emfit WHERE {cut}"
        ).get_results()
        assert len(part) == n, (lo, hi, len(part), n)
        part.write(path, overwrite=True)
        parts.append(part)
        print(f"chunk {i + 1}/48 healpix {lo}-{hi}: {n} rows", flush=True)
    tab = vstack(parts)
    total = tap.launch_job(f"SELECT COUNT(*) AS n FROM desi_dr1.emfit WHERE {where}").get_results()
    assert len(tab) == int(total["n"][0]), (len(tab), int(total["n"][0]))
    tab.write(OUT / "emfit_widths_sv3_bright.fits", overwrite=True)
    print(f"{len(tab)} rows -> emfit_widths_sv3_bright.fits")


if __name__ == "__main__":
    main()
