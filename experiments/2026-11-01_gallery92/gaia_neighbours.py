"""Gaia DR3 stars within 5 arcsec of each of the 92 canonical continuum outliers (#17).

QUESTION: which of the 92 have a Gaia star near enough to light the 1.5 arcsec fibre, missed by the
#10 Gaia screen (G < 16 within 2 arcsec, or on-source within 1 arcsec)?
HYPOTHESIS: none registered.
INPUTS: cont of results/prescreened_outliers_postsimbad.npz, gallery92 plot names,
data/fastspec-iron-sv3-bright.fits, gaiadr3.gaia_source on the ESA Gaia TAP (gea.esac.esa.int).
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-01_gallery92/gaia_neighbours.py``.
RESULT: 2026-11-02. 17 of the 92 have a Gaia star within 5 arcsec, nine within 2 arcsec. Every one
is outside the #10 screen.
Assumptions: Gaia DR3 completeness (G < ~21); 5 sigma parallax or proper motion; seeing not used.
FIGURES: results/2026-11-01_gallery92/gaia_neighbours.txt.
"""

import csv
import io
import time
import urllib.parse
import urllib.request

import numpy as np
from astropy.io import fits

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
GAL = RES / "2026-11-01_gallery92"
TAP = "https://gea.esac.esa.int/tap-server/tap/sync?"
RADIUS = 5.0  # arcsec
SIG = 5.0


def cone(ra, dec):
    """Return the Gaia DR3 rows within RADIUS of (ra, dec), nearest first, retrying twice."""
    q = (f"SELECT phot_g_mean_mag, parallax, parallax_over_error, pm, pmra_error, pmdec_error, "
         f"DISTANCE(POINT(ra, dec), POINT({ra}, {dec})) * 3600 AS sep FROM gaiadr3.gaia_source "
         f"WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {RADIUS / 3600})) ORDER BY sep")  # fmt: skip
    url = TAP + urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q))
    for attempt in range(3):
        try:
            text = urllib.request.urlopen(url, timeout=90).read().decode()
            return list(csv.DictReader(io.StringIO(text)))
        except OSError:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Gaia query failed at {ra} {dec}")


def num(r, k):
    """Return field ``k`` of a CSV row as a float, NaN if empty."""
    return float(r[k]) if r[k] not in ("", None) else np.nan


def main():
    """Query every target and write the table."""
    tids = [int(t) for t in np.load(RES / "prescreened_outliers_postsimbad.npz")["cont"]]
    nn = {int(p.stem.split("_")[1]): p.stem.split("_")[0] for p in GAL.rglob("*_*.png")
          if p.stem.split("_")[0].isdigit()}  # fmt: skip
    folder = {int(p.stem.split("_")[1]): (p.parent.name if p.parent != GAL else "-")
              for p in GAL.rglob("*_*.png") if p.stem.split("_")[0].isdigit()}  # fmt: skip
    md = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits")["METADATA"].data
    pos = {int(t): (float(r), float(d)) for t, r, d in zip(md["TARGETID"], md["RA"], md["DEC"],
                                                           strict=True)}  # fmt: skip
    lines = [f"Gaia DR3 sources within {RADIUS:.0f} arcsec; star = parallax/err > {SIG:.0f} or "
             f"pm/err > {SIG:.0f}. Columns: nn TARGETID folder | each source: sep, G, plx/err, "
             "pm/err (* = star)", ""]  # fmt: skip
    for t in sorted(tids, key=lambda x: nn.get(x, "99")):
        rows = cone(*pos[t])
        cells, star = [], False
        for r in rows:
            pme = np.hypot(num(r, "pmra_error"), num(r, "pmdec_error"))
            pms = num(r, "pm") / pme if pme > 0 else np.nan
            s = num(r, "parallax_over_error") > SIG or pms > SIG
            star |= bool(s)
            cells.append(f"{num(r, 'sep'):.2f}\" G {num(r, 'phot_g_mean_mag'):.2f} "
                         f"plx {num(r, 'parallax_over_error'):.1f} pm {pms:.1f}{'*' if s else ''}")  # fmt: skip
        lines.append(f"{nn.get(t, '??')} {t} {folder.get(t, '?'):>13} {'STAR' if star else '    '} | "
                     + ("; ".join(cells) if cells else "none"))  # fmt: skip
        time.sleep(0.5)
    text = "\n".join(lines)
    (GAL / "gaia_neighbours.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
