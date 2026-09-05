import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cont_outlier_sample import DEFAULT_FLOW_DIR, TAGS, flow_scores  # noqa: E402

from hubersed.paths import PATHS  # noqa: E402

SNMIN_BPT = 5.0      # sigma, per line, to attempt a BPT classification
SNMIN_FORB = 3.0     # sigma, above which a forbidden line counts as present
SEP_STAR = 0.5       # arcsec, rule C: the Gaia source must be inside the fibre core
NSPECIES_MIN = 2     # independent forbidden species needed to call it an emission-line galaxy

SINGLETS = ["OIII_5007", "NII_6584", "OI_6300"]
DOUBLETS = [("SII_6716", "SII_6731"), ("OII_3726", "OII_3729")]


def snr(d, i, line):
    f = float(d[f"{line}_FLUX"][i])
    iv = float(d[f"{line}_FLUX_IVAR"][i])
    return f * np.sqrt(iv) if iv > 0 else 0.0


def forbidden(d, i):
    """(strongest forbidden species, how many are present).

    Doublets score on their weaker component; see the COHERENCE GATE note above.
    """
    s = [snr(d, i, ln) for ln in SINGLETS]
    s += [min(snr(d, i, a), snr(d, i, b)) for a, b in DOUBLETS]
    return max(s), sum(x > SNMIN_FORB for x in s)


def bpt(d, i):
    """('AGN'|'composite'|'star-forming'|None, log[NII]/Ha, log[OIII]/Hb, log[SII]/Ha)."""
    need = ["HALPHA", "HBETA", "OIII_5007", "NII_6584"]
    if any(snr(d, i, ln) <= SNMIN_BPT for ln in need):
        return None, np.nan, np.nan, np.nan
    ha, hb = float(d["HALPHA_FLUX"][i]), float(d["HBETA_FLUX"][i])
    o3, n2f = float(d["OIII_5007_FLUX"][i]), float(d["NII_6584_FLUX"][i])
    s2f = float(d["SII_6716_FLUX"][i]) + float(d["SII_6731_FLUX"][i])
    if min(ha, hb, o3, n2f) <= 0:
        return None, np.nan, np.nan, np.nan
    n2, o3r = np.log10(n2f / ha), np.log10(o3 / hb)
    s2 = np.log10(s2f / ha) if s2f > 0 else np.nan

    ke_n = 0.61 / (n2 - 0.47) + 1.19 if n2 < 0.47 else -np.inf
    ke_s = 0.72 / (s2 - 0.32) + 1.30 if np.isfinite(s2) and s2 < 0.32 else -np.inf
    ka_n = 0.61 / (n2 - 0.05) + 1.30 if n2 < 0.05 else -np.inf

    if o3r > ke_n or (np.isfinite(s2) and o3r > ke_s):
        return "AGN", n2, o3r, s2
    if o3r > ka_n:
        return "composite", n2, o3r, s2
    return "star-forming", n2, o3r, s2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", default=str(PATHS["RESULTS"] / "agn_star_screen.csv"))
    p.add_argument("--flow-dir", default=str(DEFAULT_FLOW_DIR))
    p.add_argument("--vac", default=str(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits"))
    p.add_argument("--morph", default=str(PATHS["RESULTS"] / "cand231_morph.csv"),
                   help="LS_ID -> Legacy DR9 type join over the candidate pool; must exist")
    p.add_argument("--gaia-list", default=str(PATHS["RESULTS"] / "gaia_star_screen_v2.csv"),
                   help="v2 CSV, i.e. the one carrying onsource_star")
    args = p.parse_args(argv)

    out = {t: flow_scores(t, args.flow_dir)[2] for t in TAGS}
    common = out[TAGS[0]] & out[TAGS[1]]

    h = fits.open(args.vac)
    d, md = h["FASTSPEC"].data, h["METADATA"].data
    idx = {int(t): i for i, t in enumerate(np.asarray(md["TARGETID"]).astype("<i8"))}

    # Same VAC-membership filter build_cont_outlier_sample.py applies: 512 -> 231.
    pool = sorted(t for t in common if t in idx)
    print(f"common to both continuum flows: {len(common)}  in VAC: {len(pool)}")

    morph = {int(r["target_id"]): r["type"].strip()
             for r in csv.DictReader(open(args.morph))}
    gaia = {int(r["target_id"]): r for r in csv.DictReader(open(args.gaia_list))}

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

        rows.append({"target_id": t, "bpt_class": cls or "", "log_n2_ha": n2,
                     "log_o3_hb": o3, "log_s2_ha": s2, "max_forbidden_snr": fs, "n_forbidden_species": nsp,
                     "morph": mt, "gaia_onsource": onsrc, "sep_arcsec": sep,
                     "bpt_agn": bpt_agn, "psf_nonstellar": psf_nonstellar,
                     "star_ruleD": star_d,
                     "flagged": bool(bpt_agn or psf_nonstellar)})

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = lambda k: sum(r[k] for r in rows)
    print(f"  BPT-classifiable : {sum(bool(r['bpt_class']) for r in rows)}")
    print(f"  bpt_agn          : {n('bpt_agn')}")
    print(f"  psf_nonstellar   : {n('psf_nonstellar')}   (of {sum(r['morph']=='PSF' for r in rows)} PSF)")
    print(f"    of which star_ruleD : {n('star_ruleD')}")
    print(f"  flagged (union)  : {n('flagged')}")
    print(f"wrote {outp}")

    # Self-check against the 4 stars confirmed in 2026-08-27f.
    known = [39632971447142040, 39627817423472029, 39633339602177656, 39627823475856431]
    by = {r["target_id"]: r for r in rows}
    print("\nself-check, the 4 confirmed stars of 2026-08-27f:")
    for t in known:
        r = by.get(t)
        print(f"  {t}  " + ("NOT IN POOL" if r is None else
              f"morph={r['morph']:>4} maxforb={r['max_forbidden_snr']:5.2f} "
              f"nspecies={r['n_forbidden_species']} "
              f"onsrc={str(r['gaia_onsource']):>5} psf_nonstellar={r['psf_nonstellar']}"))


if __name__ == "__main__":
    main()
