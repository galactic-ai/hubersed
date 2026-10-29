"""List the iron files the recoadd of the continuum outlier 39627757533007793 needs, with destinations.

The steps are those of experiments/2026-10-09_nautilus_c3k_er_broadline, as run for the line
outliers in experiments/2026-10-18_candidate_recoadd. coadd_rows.py needs the healpix spectra file,
and fluxcal_rerun.py, ivar_check.py and ivar_recompute.py need the 20 files of each exposure and
petal (exposure_files of 2026-10-18_candidate_recoadd/iron_urls.py). The exposures are SV3 83526
(night 20210405, petal 0, fiber 229) and main 122064 (night 20220210, petal 1, fiber 523), from
desi_dr1.ztile, desi_dr1.exposure and desi_dr1.fiberassign (Astro Data Lab TAP, 2026-10-28).

COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-29_continuum_recoadd/iron_urls.py``.

OUTPUT: iron_urls.txt next to this script, one "url destination" pair per line, with destinations
relative to the Corral data folder.
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
