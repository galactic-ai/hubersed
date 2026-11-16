"""List the iron files the recoadd of the continuum outlier 39633433328091196 needs, with destinations.

As 2026-10-29_continuum_recoadd/iron_urls.py, for exposures 87837, 87838, 87839 and healpix 10231.
COMMAND: ``uv run --no-sync python experiments/2026-10-30_recoadd_39633433328091196/iron_urls.py``.
OUTPUT: iron_urls.txt next to this script, one "url destination" pair per line.
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).parent
BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron"
HEALPIX = 10231
NIGHT, PETAL = "20210509", 4
EXPIDS = (87837, 87838, 87839)

_spec = importlib.util.spec_from_file_location(
    "iron_urls_1018", HERE.parent / "2026-10-18_candidate_recoadd" / "iron_urls.py"
)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


def main():
    """Write the url and destination of every file."""
    name = f"spectra-sv3-bright-{HEALPIX}.fits.gz"
    lines = [
        f"{BASE}/healpix/sv3/bright/{HEALPIX // 100}/{HEALPIX}/{name} desi_iron_spectra/{name}"
    ]
    for expid in EXPIDS:
        for url in _m.exposure_files(NIGHT, expid, PETAL):
            lines.append(f"{url} desi_iron_exposures/{NIGHT}/{expid:08d}/{url.rsplit('/', 1)[1]}")
    (HERE / "iron_urls.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} files")


if __name__ == "__main__":
    main()
