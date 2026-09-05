"""SIMBAD identification screen over the top of the ranked candidate list.

Fourth screen. Covers the one contaminant class the other three cannot reach: an AGN that
is morphologically RESOLVED and has too few forbidden lines to classify on the BPT.
2026-08-27g found three of these in the old top-20 (a gamma-ray BL Lac fit as SER, a QSO
and an AGN both fit as REX); agn_star_screen.py still misses the BL Lac by construction.

Queried on the ranked survivors rather than the whole pool because it costs one network
round-trip each and only the head of the ranking can reach the sample.

MANUAL_EXCLUDE carries findings that are not reducible to a screen -- an individual
redshift refutation. Each entry cites the log entry that established it.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cont_outlier_sample import DEFAULT_FLOW_DIR, TAGS, flow_scores, read_screen  # noqa: E402

from hubersed.paths import PATHS  # noqa: E402

SEP_MAX = 2.0
AGN_TYPES = {"BLL", "QSO", "AGN", "Sy1", "Sy2", "SyG", "QSO_Candidate", "Bla", "LIN", "rG"}
STAR_TYPES = {"Star", "Pe*", "WD*", "HB*", "RGB*", "*", "PM*", "HV*"}

# TARGETID -> (reason, log entry). Not screenable; each was an individual investigation.
MANUAL_EXCLUDE = {
    39633322460057381: ("redshift refuted -- the single feature carrying z=0.5526 cannot "
                        "be [OIII]", "2026-08-27h"),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", default=str(PATHS["RESULTS"] / "simbad_screen.csv"))
    p.add_argument("-n", "--n-query", type=int, default=45,
                   help="how far down the ranked survivor list to query")
    p.add_argument("--flow-dir", default=str(DEFAULT_FLOW_DIR))
    p.add_argument("--vac", default=str(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits"))
    p.add_argument("--screens", nargs="*", default=[
        str(PATHS["RESULTS"] / "gaia_star_screen.csv"),
        str(PATHS["RESULTS"] / "sga_proximity.csv"),
        str(PATHS["RESULTS"] / "agn_star_screen.csv")],
        help="screens already applied; their flags are excluded before ranking")
    args = p.parse_args(argv)

    from astroquery.simbad import Simbad

    lp, pct, out = {}, {}, {}
    for t in TAGS:
        lp[t], pct[t], out[t], _ = flow_scores(t, args.flow_dir)
    common = out[TAGS[0]] & out[TAGS[1]]

    h = fits.open(args.vac)
    md = h["METADATA"].data
    tid = np.asarray(md["TARGETID"]).astype("<i8")
    idx = {int(t): i for i, t in enumerate(tid)}
    ra, dec = np.asarray(md["RA"]).astype("<f8"), np.asarray(md["DEC"]).astype("<f8")

    dropped = set()
    for s in args.screens:
        dropped |= read_screen(s)[0]
    z = md["Z"]
    surv = [t for t in sorted(common) if t in idx and t not in dropped
            and 0.01 <= z[idx[t]] <= 0.6]
    surv.sort(key=lambda t: 0.5 * (pct[TAGS[0]][t] + pct[TAGS[1]][t]))
    todo = surv[:args.n_query]
    print(f"ranked survivors: {len(surv)}; querying the top {len(todo)}")

    s = Simbad()
    for f in ("otype", "rvz_redshift", "sp_type"):
        try:
            s.add_votable_fields(f)
        except Exception:
            pass

    rows = []
    for n, t in enumerate(todo, 1):
        i = idx[t]
        name = otype = sptype = ""
        zext, sep = np.nan, np.nan
        try:
            r = s.query_region(SkyCoord(ra[i], dec[i], unit="deg"), radius=SEP_MAX * u.arcsec)
        except Exception as exc:
            print(f"  {t}: SIMBAD failed: {type(exc).__name__}: {exc}")
            r = None
        if r is not None and len(r):
            row = r[0]
            o = SkyCoord(float(row["ra"]), float(row["dec"]), unit="deg")
            sep = SkyCoord(ra[i], dec[i], unit="deg").separation(o).arcsec
            name, otype = str(row["main_id"]), str(row["otype"]).strip()
            sptype = str(row.get("sp_type", "") or "").strip()
            try:
                zext = float(row["rvz_redshift"])
            except Exception:
                zext = np.nan

        is_agn = otype in AGN_TYPES and sep < SEP_MAX
        is_star = (otype in STAR_TYPES or sptype.startswith("dC")) and sep < SEP_MAX
        man = t in MANUAL_EXCLUDE
        rows.append({"target_id": t, "rank": n, "z_desi": float(z[i]), "simbad": name,
                     "otype": otype, "sp_type": sptype, "sep_arcsec": sep, "z_simbad": zext,
                     "simbad_agn": is_agn, "simbad_star": is_star,
                     "manual_exclude": man,
                     "manual_reason": MANUAL_EXCLUDE[t][0] if man else "",
                     "flagged": bool(is_agn or is_star or man)})
        if is_agn or is_star or man:
            why = MANUAL_EXCLUDE[t][1] if man else f"{otype} {name} at {sep:.2f}\""
            print(f"  rank {n:>3}  {t}  FLAG  {why}")

    outp = Path(args.out)
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    nf = sum(r["flagged"] for r in rows)
    print(f"\n  simbad_agn={sum(r['simbad_agn'] for r in rows)}  "
          f"simbad_star={sum(r['simbad_star'] for r in rows)}  "
          f"manual={sum(r['manual_exclude'] for r in rows)}  flagged={nf}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
