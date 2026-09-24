"""Query SIMBAD around the top-ranked surviving candidates and flag AGN and stars.

Candidates flagged by earlier screen CSVs are dropped before ranking, and targets in
MANUAL_EXCLUDE are flagged too. Run it as ``uv run python scripts/simbad_screen.py``.
"""

import argparse
import csv
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

from hubersed.detect.cont_flow import TAGS, flow_scores, read_screen
from hubersed.paths import PATHS

SEP_MAX = 2.0
AGN_TYPES = {"BLL", "QSO", "AGN", "Sy1", "Sy2", "SyG", "QSO_Candidate", "Bla", "LIN", "rG"}
STAR_TYPES = {"Star", "Pe*", "WD*", "HB*", "RGB*", "*", "PM*", "HV*"}

# Maps TARGETID to (reason, log entry). These cannot be screened automatically.
MANUAL_EXCLUDE = {
    39633322460057381: (
        "redshift refuted -- the single feature carrying z=0.5526 cannot be [OIII]",
        "2026-08-27h",
    ),
}


def main(argv=None):
    """Rank the surviving candidates, query SIMBAD for the top ones and write the CSV.

    Survivors are outliers of both continuum flows that are in the VAC, are not flagged
    by any file in ``--screens`` and have 0.01 <= Z <= 0.6. They are sorted by the mean of
    their two DESI rank percentiles, lowest first. A target is flagged if the first
    SIMBAD match within SEP_MAX arcsec has an AGN or star type, or if it is in
    MANUAL_EXCLUDE. Failed queries are printed and the row is kept unflagged by SIMBAD.

    Parameters
    ----------
    argv : list of str or None, optional
        Command line arguments. None reads ``sys.argv``.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-o", "--out", default=str(PATHS["RESULTS"] / "simbad_screen.csv"))
    p.add_argument(
        "-n",
        "--n-query",
        type=int,
        default=45,
        help="how far down the ranked survivor list to query",
    )
    p.add_argument(
        "--flow-dir",
        required=True,
        help="dir holding desi_outliers_flow_nsf_<tag>_snr3.pt for each tag",
    )
    p.add_argument("--vac", default=str(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits"))
    p.add_argument(
        "--screens",
        nargs="*",
        default=[
            str(PATHS["RESULTS"] / "gaia_star_screen.csv"),
            str(PATHS["RESULTS"] / "sga_proximity.csv"),
            str(PATHS["RESULTS"] / "agn_star_screen.csv"),
        ],
        help="screens already applied; their flags are excluded before ranking",
    )
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
    surv = [t for t in sorted(common) if t in idx and t not in dropped and 0.01 <= z[idx[t]] <= 0.6]
    surv.sort(key=lambda t: 0.5 * (pct[TAGS[0]][t] + pct[TAGS[1]][t]))
    todo = surv[: args.n_query]
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
        rows.append(
            {
                "target_id": t,
                "rank": n,
                "z_desi": float(z[i]),
                "simbad": name,
                "otype": otype,
                "sp_type": sptype,
                "sep_arcsec": sep,
                "z_simbad": zext,
                "simbad_agn": is_agn,
                "simbad_star": is_star,
                "manual_exclude": man,
                "manual_reason": MANUAL_EXCLUDE[t][0] if man else "",
                "flagged": bool(is_agn or is_star or man),
            }
        )
        if is_agn or is_star or man:
            why = MANUAL_EXCLUDE[t][1] if man else f'{otype} {name} at {sep:.2f}"'
            print(f"  rank {n:>3}  {t}  FLAG  {why}")

    outp = Path(args.out)
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    nf = sum(r["flagged"] for r in rows)
    print(
        f"\n  simbad_agn={sum(r['simbad_agn'] for r in rows)}  "
        f"simbad_star={sum(r['simbad_star'] for r in rows)}  "
        f"manual={sum(r['manual_exclude'] for r in rows)}  flagged={nf}"
    )
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
