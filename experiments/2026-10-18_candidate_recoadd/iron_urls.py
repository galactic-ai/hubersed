"""List the iron files that the patched flux calibration and recoadd need for four line outliers.

The steps are those of experiments/2026-10-09_nautilus_c3k_er_broadline for 39627770174637084:
coadd_rows.py needs the healpix spectra file, and fluxcal_rerun.py, ivar_check.py and
ivar_recompute.py need 20 files per exposure and petal. The exposures come from EXP_FIBERMAP of the
DR1 healpix coadds in data/sv3_bright_coadds/.

COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-18_candidate_recoadd/iron_urls.py``.

OUTPUT: iron_urls.txt and exposures.txt next to this script.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits

from hubersed.paths import PATHS

BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron"
CANDIDATES = {
    39628357133926597: 28115,
    39633278973513279: 11404,
    39632961393395307: 10512,
    39633001071509599: 10518,
}
HERE = Path(__file__).parent


def exposure_files(night, expid, petal):
    """Return the 20 file names of one exposure and petal, as in iron_exposure_urls.txt."""
    e = f"{expid:08d}"
    names = [f"calibstars-{e}.csv", f"stdstars-{petal}-{e}.fits.gz"]
    for kind in ("cframe", "fiberflatexp", "fluxcalib", "frame", "sframe", "sky"):
        names += [f"{kind}-{arm}{petal}-{e}.fits.gz" for arm in "brz"]
    return [f"{BASE}/exposures/{night}/{e}/{n}" for n in sorted(names)]


def main():
    """Write the URL list and the exposure table."""
    urls, table = [], []
    for tid, hp in CANDIDATES.items():
        urls.append(f"{BASE}/healpix/sv3/bright/{hp // 100}/{hp}/spectra-sv3-bright-{hp}.fits.gz")
        ex = fits.getdata(PATHS["DATA"] / "sv3_bright_coadds" / f"coadd-sv3-bright-{hp}.fits",
                          "EXP_FIBERMAP")  # fmt: skip
        ex = ex[ex["TARGETID"] == tid]
        keys = sorted({(int(r["NIGHT"]), int(r["EXPID"]), int(r["PETAL_LOC"]), int(r["FIBER"]))
                       for r in ex})  # fmt: skip
        for night, expid, petal, fiber in keys:
            table.append(
                f"{tid} {hp} {night} {expid} petal {petal} fiber {fiber} row {fiber % 500}"
            )
            urls += exposure_files(night, expid, petal)
    urls = list(dict.fromkeys(urls))
    (HERE / "iron_urls.txt").write_text("\n".join(urls) + "\n")
    (HERE / "exposures.txt").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(f"{len(urls)} urls, {len(np.unique([u.rsplit('/', 2)[1] for u in urls]))} folders")


if __name__ == "__main__":
    main()
