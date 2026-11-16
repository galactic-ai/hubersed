"""List the iron files the SV3 recoadds of the six poco_repeats targets need, with destinations.

As 2026-10-18_candidate_recoadd/iron_urls.py, for the fitted targets of targets.txt,
targets_quiescent2.txt and targets_quiescent3.txt.
COMMAND: ``<venv>/bin/python experiments/2026-11-01_poco_repeats/recoadd_urls.py``, on a login node.
OUTPUT: recoadd_urls.txt and recoadd_exposures.txt next to this script.
"""

import importlib.util
from pathlib import Path

from astropy.io import fits

HERE = Path(__file__).resolve().parent
BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron"
COADD = Path("/work/11006/nikhilgaruda/ls6/research/spender/DATA")
TARGETS = {
    39627836721463943: 25977,
    39632961871548329: 9425,
    39633416978694809: 15343,
    39633332836764152: 10157,
    39633399748495657: 10199,
    39633419142956044: 11231,
}

_spec = importlib.util.spec_from_file_location(
    "iron_urls_1018", HERE.parent / "2026-10-18_candidate_recoadd" / "iron_urls.py"
)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


def main():
    """Write the url list and the exposure table."""
    urls, table = [], []
    for tid, hp in TARGETS.items():
        name = f"spectra-sv3-bright-{hp}.fits.gz"
        urls.append(f"{BASE}/healpix/sv3/bright/{hp // 100}/{hp}/{name} desi_iron_spectra/{name}")
        ex = fits.getdata(COADD / str(hp) / f"coadd-sv3-bright-{hp}.fits", "EXP_FIBERMAP")
        ex = ex[ex["TARGETID"] == tid]
        assert len(ex), f"{tid} has no exposures in healpix {hp}"
        keys = sorted({(str(r["NIGHT"]), int(r["EXPID"]), int(r["PETAL_LOC"]), int(r["FIBER"]),
                        int(r["TILEID"])) for r in ex})  # fmt: skip
        for night, expid, petal, fiber, tile in keys:
            table.append(f"{tid} {hp} {night} {expid} {petal} {fiber} {tile}")
            for url in _m.exposure_files(night, expid, petal):
                dest = f"desi_iron_exposures/{night}/{expid:08d}/{url.rsplit('/', 1)[1]}"
                urls.append(f"{url} {dest}")
    urls = list(dict.fromkeys(urls))
    (HERE / "recoadd_urls.txt").write_text("\n".join(urls) + "\n")
    (HERE / "recoadd_exposures.txt").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(f"{len(urls)} files")


if __name__ == "__main__":
    main()
