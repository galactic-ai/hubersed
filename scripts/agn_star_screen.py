"""Flag BPT AGN and PSF-type sources with few forbidden lines among continuum-flow outliers.

The pool is the DESI outliers shared by both continuum flows that are also in the fastspec VAC,
and one row per target goes to a CSV. Run it as
``uv run python scripts/agn_star_screen.py --flow-dir DIR``.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from astropy.io import fits

from hubersed.detect.cont_flow import TAGS, flow_scores
from hubersed.detect.line_screen import (
    NSPECIES_MIN,
    SEP_STAR,
    bpt,
    forbidden,
    read_gaia_screen,
)
from hubersed.paths import PATHS


def main(argv=None):
    """Build the candidate pool, screen each target and write the CSV.

    Parameters
    ----------
    argv : list of str or None, optional
        Command line arguments. None reads ``sys.argv``.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-o", "--out", default=str(PATHS["RESULTS"] / "agn_star_screen.csv"))
    p.add_argument(
        "--flow-dir",
        required=True,
        help="dir holding desi_outliers_flow_nsf_<tag>_snr3.pt for each tag",
    )
    p.add_argument("--vac", default=str(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits"))
    p.add_argument(
        "--morph",
        default=str(PATHS["RESULTS"] / "cand231_morph.csv"),
        help="CSV of target_id and Legacy DR9 type for the candidate pool. Must exist.",
    )
    p.add_argument(
        "--gaia-list",
        default=str(PATHS["RESULTS"] / "gaia_star_screen.csv"),
        help="Gaia screen CSV written by contam_screens, with onsource_star and sep_arcsec",
    )
    args = p.parse_args(argv)

    out = {t: flow_scores(t, args.flow_dir)[2] for t in TAGS}
    common = out[TAGS[0]] & out[TAGS[1]]

    h = fits.open(args.vac)
    d, md = h["FASTSPEC"].data, h["METADATA"].data
    idx = {int(t): i for i, t in enumerate(np.asarray(md["TARGETID"]).astype("<i8"))}

    # Same VAC-membership filter that build_cont_outlier_sample applies.
    pool = sorted(t for t in common if t in idx)
    print(f"common to both continuum flows: {len(common)}  in VAC: {len(pool)}")

    morph = {int(r["target_id"]): r["type"].strip() for r in csv.DictReader(open(args.morph))}
    gaia = read_gaia_screen(args.gaia_list)

    rows = []
    for t in pool:
        i = idx[t]
        cls, n2, o3, s2 = bpt(d, i)
        fs, nsp = forbidden(d, i)
        mt = morph.get(t, "")
        g = gaia.get(t, {})
        onsrc = g.get("onsource_star", "") == "True"
        sep = float(g["sep_arcsec"]) if g.get("sep_arcsec") not in (None, "", "nan") else np.nan

        bpt_agn = cls == "AGN"
        psf_nonstellar = (mt == "PSF") and (nsp < NSPECIES_MIN)
        star_d = psf_nonstellar and onsrc and np.isfinite(sep) and sep < SEP_STAR

        rows.append(
            {
                "target_id": t,
                "bpt_class": cls or "",
                "log_n2_ha": n2,
                "log_o3_hb": o3,
                "log_s2_ha": s2,
                "max_forbidden_snr": fs,
                "n_forbidden_species": nsp,
                "morph": mt,
                "gaia_onsource": onsrc,
                "sep_arcsec": sep,
                "bpt_agn": bpt_agn,
                "psf_nonstellar": psf_nonstellar,
                "star_ruleD": star_d,
                "flagged": bool(bpt_agn or psf_nonstellar),
            }
        )

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def n(k):
        return sum(r[k] for r in rows)

    print(f"  BPT-classifiable : {sum(bool(r['bpt_class']) for r in rows)}")
    print(f"  bpt_agn          : {n('bpt_agn')}")
    print(
        f"  psf_nonstellar   : {n('psf_nonstellar')}   (of {sum(r['morph'] == 'PSF' for r in rows)} PSF)"
    )
    print(f"    of which star_ruleD : {n('star_ruleD')}")
    print(f"  flagged (union)  : {n('flagged')}")
    print(f"wrote {outp}")

    # Print the screen results for four previously confirmed stars.
    known = [39632971447142040, 39627817423472029, 39633339602177656, 39627823475856431]
    by = {r["target_id"]: r for r in rows}
    print("\nself-check, the 4 confirmed stars of 2026-08-27f:")
    for t in known:
        r = by.get(t)
        print(
            f"  {t}  "
            + (
                "NOT IN POOL"
                if r is None
                else f"morph={r['morph']:>4} maxforb={r['max_forbidden_snr']:5.2f} "
                f"nspecies={r['n_forbidden_species']} "
                f"onsrc={str(r['gaia_onsource']):>5} psf_nonstellar={r['psf_nonstellar']}"
            )
        )


if __name__ == "__main__":
    main()
