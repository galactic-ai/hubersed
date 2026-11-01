"""DESI tile-QA sky check of the five quiescent follow-up outliers before their pocoMC fits (#17).

QUESTION: is any of follow28 12, 15, 20, 22, 27 on a petal whose sky model DESI's tile QA flags,
as 39633325949715572 was (petal 6 of tile 313, 2026-10-31_boss_39633325949715572)?
HYPOTHESIS: none registered.
INPUTS: desi_dr1.ztile (sv3, bright, cumulative) on Astro Data Lab for TILEID and LASTNIGHT; the iron
tile-qa-<TILEID>-thru<LASTNIGHT>.fits files (FIBERQA, PETALQA). Thresholds skythrurms_max 0.05 and
skychi2pdf_max 1.5 from desispec 0.56.2 qa-params.yaml lines 106-107.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_quiescent_qa/analysis.py``.
RESULT: not run yet.
FIGURES: results/2026-10-31_quiescent_qa/ qa.txt, fibres_all.txt (tile-QA files cached in\ndata/iron_tileqa/).
"""

# %%
import urllib.request

import numpy as np
from astropy.io import fits
from astroquery.utils.tap.core import TapPlus

from hubersed.detect.sky_screens import DL_TAP, query
from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-31_quiescent_qa"
CACHE = PATHS["DATA"] / "iron_tileqa"
BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/tiles/cumulative"
FIVE = {12: 39633427376375905, 15: 39627891025118826, 20: 39633301257850206,
        22: 39633166981401035, 27: 39627884985325077}  # fmt: skip
THRU_MAX, CHI2_MAX = 0.05, 1.5


def tiles():
    """Return the sv3/bright cumulative tile rows (targetid, tileid, lastnight) of the five."""
    ids = ",".join(str(t) for t in FIVE.values())
    q = ("SELECT targetid, tileid, lastnight, coadd_numexp FROM desi_dr1.ztile "
         f"WHERE targetid IN ({ids}) AND survey = 'sv3' AND program = 'bright' "
         "AND spgrp = 'cumulative'")  # fmt: skip
    return query(lambda: TapPlus(url=DL_TAP).launch_job(q).get_results())


def tileqa(tile, night):
    """Return the opened tile-QA file of ``tile`` through ``night``, fetched once."""
    name = f"tile-qa-{tile}-thru{night}.fits"
    path = CACHE / str(tile) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{BASE}/{tile}/{night}/{name}", path)
    return fits.open(path)


def main():
    """Write the fibre and petal sky QA of each target's tile."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = tiles()
    out = [f"# tile QA (iron cumulative): fibre QAFIBERSTATUS, petal B/R/Z SKYCHI2PDF and "
           f"SKYTHRURMS; FLAG if chi2pdf > {CHI2_MAX} or thrurms > {THRU_MAX}"]  # fmt: skip
    for k, t in FIVE.items():
        mine = rows[np.asarray(rows["targetid"], np.int64) == t]
        if len(mine) == 0:
            out.append(f"{k:02d} {t}: no sv3/bright cumulative ztile row")
        for r in mine:
            tile, night = int(r["tileid"]), int(r["lastnight"])
            with tileqa(tile, night) as h:
                fq, pq = h["FIBERQA"].data, h["PETALQA"].data
                valid = h["FIBERQA"].header.get("VALID")
                i = np.flatnonzero(fq["TARGETID"] == t)
                if i.size == 0:
                    out.append(f"{k:02d} {t} tile {tile}: TARGETID not in FIBERQA")
                    continue
                fib, pet = int(fq["FIBER"][i[0]]), int(fq["PETAL_LOC"][i[0]])
                p = pq[pq["PETAL_LOC"] == pet][0]
                chi2 = {c: float(p[f"{c}SKYCHI2PDF"]) for c in "BRZ"}
                thru = {c: float(p[f"{c}SKYTHRURMS"]) for c in "BRZ"}
                bad = [c for c in "BRZ" if chi2[c] > CHI2_MAX or thru[c] > THRU_MAX]
                out.append(
                    f"{k:02d} {t} tile {tile} thru {night} (numexp {int(r['coadd_numexp'])}, VALID "
                    f"{valid}): fibre {fib} petal {pet} QAFIBERSTATUS {int(fq['QAFIBERSTATUS'][i[0]])}; "
                    + " ".join(f"{c} {chi2[c]:.2f}/{thru[c]:.4f}" for c in "BRZ")
                    + f"; EFFTIME_SPEC {float(p['EFFTIME_SPEC']):.0f} s; "
                    + (f"FLAG {','.join(bad)}" if bad else "ok")
                )  # fmt: skip
    (OUT / "qa.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


# %%
def fibres_all():
    """Write fibre, petal and QAFIBERSTATUS of every survivors.txt outlier on every sv3/bright tile."""
    ids = []
    for s in (
        (PATHS["RESULTS"] / "2026-10-31_screen_all" / "survivors.txt").read_text().splitlines()
    ):
        p = s.split()
        if p and p[0].isdigit() and len(p[0]) == 17:
            ids.append(int(p[0]))
    q = ("SELECT targetid, tileid, lastnight FROM desi_dr1.ztile "
         f"WHERE targetid IN ({','.join(map(str, ids))}) AND survey = 'sv3' "
         "AND program = 'bright' AND spgrp = 'cumulative'")  # fmt: skip
    rows = query(lambda: TapPlus(url=DL_TAP).launch_job(q).get_results())
    out, fibres = [f"# {len(ids)} survivors.txt outliers, {len(rows)} sv3/bright tile rows"], []
    for r in rows:
        t, tile, night = int(r["targetid"]), int(r["tileid"]), int(r["lastnight"])
        with tileqa(tile, night) as h:
            fq = h["FIBERQA"].data
            i = np.flatnonzero(fq["TARGETID"] == t)
            if i.size == 0:
                out.append(f"{t} tile {tile}: not in FIBERQA")
                continue
            fib = int(fq["FIBER"][i[0]])
            fibres.append(fib)
            out.append(f"{t} tile {tile}: fibre {fib} petal {int(fq['PETAL_LOC'][i[0]])} "
                       f"QAFIBERSTATUS {int(fq['QAFIBERSTATUS'][i[0]])}")  # fmt: skip
    f, n = np.unique(fibres, return_counts=True)
    out.append(
        "fibres used more than once: "
        + ", ".join(f"{a} x{b}" for a, b in zip(f, n, strict=True) if b > 1)
    )
    pet, pn = np.unique(np.array(fibres) // 500, return_counts=True)
    out.append("rows per petal: " + ", ".join(f"{a}: {b}" for a, b in zip(pet, pn, strict=True)))
    (OUT / "fibres_all.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out[-2:]))


if __name__ == "__main__":
    main()
    fibres_all()
