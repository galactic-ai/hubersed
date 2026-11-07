"""Save the EmFit line fluxes of every DR1 SV3 bright spectrum from the Astro Data Lab.

Synchronous queries of desi_dr1.emfit (Pucha+2026) for survey sv3, program bright, in HEALPix chunks,
each checked against its row count (the async endpoint returned Bad Request, and sync returns only
2000 rows without TOP): TARGETID, redshift, and the flux of each component (narrow _n, outflow _out, broad Balmer _b) of Hbeta, [OIII]4959,5007,
[NII]6548,6583, Halpha and [SII]6716,6731, in 1e-17 erg/s/cm2.

COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-11-06_line_ew_prior/fetch_emfit.py``.
OUTPUT: results/2026-11-06_line_ew_prior/emfit_sv3_bright.fits; chunks in emfit_chunks/ let a rerun resume.
"""

import warnings

import numpy as np
from astropy.table import Table, vstack
from astroquery.utils.tap.core import TapPlus

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-06_line_ew_prior"
COMPONENTS = {
    "hb": ("hb_n", "hb_out", "hb_b"),
    "oiii4959": ("oiii4959", "oiii4959_out"),
    "oiii5007": ("oiii5007", "oiii5007_out"),
    "nii6548": ("nii6548", "nii6548_out"),
    "nii6583": ("nii6583", "nii6583_out"),
    "ha": ("ha_n", "ha_out", "ha_b"),
    "sii6716": ("sii6716", "sii6716_out"),
    "sii6731": ("sii6731", "sii6731_out"),
}


def main():
    """Run the query and write the table."""
    cols = ["targetid", "z"] + [f"{c}_flux" for comps in COMPONENTS.values() for c in comps]
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
        query = f"SELECT COUNT(*) AS n FROM desi_dr1.emfit WHERE {cut}"
        n = int(tap.launch_job(query).get_results()["n"][0])
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
    OUT.mkdir(parents=True, exist_ok=True)
    tab.write(OUT / "emfit_sv3_bright.fits", overwrite=True)
    print(f"{len(tab)} rows -> emfit_sv3_bright.fits")


if __name__ == "__main__":
    main()
