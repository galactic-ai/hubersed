"""List the iron files the SV3 recoadds of the five emission-line outliers need, with destinations.

As 2026-11-01_poco_repeats/recoadd_urls.py, for the targets of targets_em5.txt. Also prints each
target's exposures in the form of the 2026-10-09_nautilus_c3k_er_broadline targets.py entries.
COMMAND: through recoadd_urls.slurm on a compute node.
OUTPUT: recoadd_urls.txt and recoadd_exposures.txt next to this script.
"""

import importlib.util
from pathlib import Path

from astropy.io import fits

HERE = Path(__file__).resolve().parent
BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron"
COADD = Path("/work/11006/nikhilgaruda/ls6/research/spender/DATA")
TARGETS = {int(t): int(hp) for t, hp, _ in
           (line.split() for line in (HERE / "targets_em5.txt").read_text().splitlines())}  # fmt: skip

_spec = importlib.util.spec_from_file_location(
    "iron_urls_1018", HERE.parent / "2026-10-18_candidate_recoadd" / "iron_urls.py"
)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


def main():
    """Write the url list and the exposure table, and print the targets.py exposures."""
    urls, table = [], []
    for tid, hp in TARGETS.items():
        name = f"spectra-sv3-bright-{hp}.fits.gz"
        urls.append(f"{BASE}/healpix/sv3/bright/{hp // 100}/{hp}/{name} desi_iron_spectra/{name}")
        ex = fits.getdata(COADD / str(hp) / f"coadd-sv3-bright-{hp}.fits", "EXP_FIBERMAP")
        ex = ex[ex["TARGETID"] == tid]
        assert len(ex), f"{tid} has no exposures in healpix {hp}"
        keys = sorted({(str(r["NIGHT"]), int(r["EXPID"]), int(r["PETAL_LOC"]), int(r["FIBER"]),
                        int(r["TILEID"])) for r in ex})  # fmt: skip
        entry = []
        for night, expid, petal, fiber, tile in keys:
            table.append(f"{tid} {hp} {night} {expid} {petal} {fiber} {tile}")
            entry.append(f'{expid}: ("{night}", {petal}, {fiber})')
            for url in _m.exposure_files(night, expid, petal):
                dest = f"desi_iron_exposures/{night}/{expid:08d}/{url.rsplit('/', 1)[1]}"
                urls.append(f"{url} {dest}")
        print(f"{tid} healpix {hp} exposures {{{', '.join(entry)}}}")
    urls = list(dict.fromkeys(urls))
    (HERE / "recoadd_urls.txt").write_text("\n".join(urls) + "\n")
    (HERE / "recoadd_exposures.txt").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(f"{len(urls)} files")


if __name__ == "__main__":
    main()
