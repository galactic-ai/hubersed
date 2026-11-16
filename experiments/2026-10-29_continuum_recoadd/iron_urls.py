"""List the iron files the recoadd of the continuum outlier 39627757533007793 needs, with destinations.

COMMAND: from the repository root,
``uv run --no-sync python experiments/2026-10-29_continuum_recoadd/iron_urls.py``.
OUTPUT: iron_urls.txt next to this script, one "url destination" pair per line.
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).parent
BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron"
HEALPIX = 25598
EXPOSURES = {"sv3": (83526, "20210405", 0), "main": (122064, "20220210", 1)}

_spec = importlib.util.spec_from_file_location(
    "iron_urls_1018", HERE.parent / "2026-10-18_candidate_recoadd" / "iron_urls.py"
)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


def main():
    """Write the url and destination of every file."""
    lines = []
    for survey, (expid, night, petal) in EXPOSURES.items():
        name = f"spectra-{survey}-bright-{HEALPIX}.fits.gz"
        url = f"{BASE}/healpix/{survey}/bright/{HEALPIX // 100}/{HEALPIX}/{name}"
        lines.append(f"{url} desi_iron_spectra/{name}")
        for url in _m.exposure_files(night, expid, petal):
            lines.append(f"{url} desi_iron_exposures/{night}/{expid:08d}/{url.rsplit('/', 1)[1]}")
    (HERE / "iron_urls.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} files")


if __name__ == "__main__":
    main()
