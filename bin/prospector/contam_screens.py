import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cont_outlier_sample import DEFAULT_FLOW_DIR, TAGS, flow_scores  # noqa: E402

from hubersed.paths import PATHS  # noqa: E402

GAIA_RADIUS = 2.0        # arcsec, cone radius
GAIA_G_MAX = 16.0        # mag -- applies ONLY to the bright-neighbour (PSF-wing) test
GAIA_PLX_SNR = 5.0       # parallax / parallax_error
GAIA_PM = 3.0            # mas/yr, bare total PM, used only by the wing test
GAIA_ONSRC = 1.0         # arcsec: inside this the Gaia source IS the target
GAIA_PM_SNR = 5.0        # total proper motion / its error, for the on-source test
GAIA_RUWE_MAX = 1.4      # above this the astrometric solution is blended/untrustworthy
SGA_BOX = 0.25           # deg, half-height of the Dec box (RA half-width is this / cos dec)
DL_TAP = "https://datalab.noirlab.edu/tap"


def _col(table, name):
    """First row of `name` as a float, with masked/absent entries becoming NaN."""
    if len(table) == 0 or name not in table.colnames:
        return np.nan
    return float(np.ma.filled(np.ma.asarray(table[name], dtype=float), np.nan)[0])


def query(fn, tries=4):
    """Run a TAP query, retrying transient failures, then raise.

    Never returns a sentinel: a screen that cannot answer must stop the run, not report
    "not flagged".  astroquery raises on VOTable error documents (which TAP serves with
    HTTP 200), so an error body cannot be mistaken for an empty result set.
    """
    for k in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - any TAP failure is retried, then re-raised
            last = exc
            print(f"    TAP attempt {k + 1}/{tries} failed: {type(exc).__name__}: {exc}")
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"TAP query failed {tries}x: {type(last).__name__}: {last}") from last


def gaia_screen(ra, dec):
    """Nearest Gaia DR3 source within GAIA_RADIUS, and whether it flags the target."""
    from astroquery.gaia import Gaia

    # The `AS sep` alias is load-bearing: ORDER BY on the bare expression is rejected.
    q = (f"SELECT TOP 1 parallax,parallax_error,pm,pmra,pmra_error,pmdec,pmdec_error,"
         f"ruwe,phot_g_mean_mag,"
         f"DISTANCE(POINT(ra,dec),POINT({ra},{dec}))*3600 AS sep FROM gaiadr3.gaia_source "
         f"WHERE 1=CONTAINS(POINT(ra,dec),CIRCLE({ra},{dec},{GAIA_RADIUS / 3600})) "
         f"ORDER BY sep ASC")
    t = query(lambda: Gaia.launch_job(q).get_results())

    plx, plx_err = _col(t, "parallax"), _col(t, "parallax_error")
    snr = plx / plx_err if np.isfinite(plx) and np.isfinite(plx_err) and plx_err > 0 else np.nan
    pm, g = _col(t, "pm"), _col(t, "phot_g_mean_mag")
    sep, ruwe = _col(t, "sep"), _col(t, "ruwe")

    # Error on the TOTAL pm, propagated from the components: pm = hypot(pmra, pmdec).
    pmra, pmdec = _col(t, "pmra"), _col(t, "pmdec")
    pmra_e, pmdec_e = _col(t, "pmra_error"), _col(t, "pmdec_error")
    pm_err = np.hypot(pmra * pmra_e, pmdec * pmdec_e) / pm if pm > 0 else np.nan
    pm_snr = pm / pm_err if np.isfinite(pm_err) and pm_err > 0 else np.nan

    onsource = bool(sep < GAIA_ONSRC and ruwe < GAIA_RUWE_MAX
                    and (snr > GAIA_PLX_SNR or pm_snr > GAIA_PM_SNR))
    # NaN comparisons are False, so a 2-parameter solution simply does not flag.
    wing = bool(g < GAIA_G_MAX and (snr > GAIA_PLX_SNR or pm > GAIA_PM))
    return {"sep_arcsec": sep, "parallax": plx, "parallax_over_error": snr,
            "pm": pm, "pm_over_error": pm_snr, "ruwe": ruwe, "phot_g_mean_mag": g,
            "onsource_star": onsource, "wing_star": wing, "flagged": bool(onsource or wing)}


def ellipse_radius(ra, dec, g_ra, g_dec, g_d26, g_pa, g_ba):
    """Normalised elliptical radius of (ra, dec) in each SGA galaxy's own frame.

    r_ell <= 1 means inside the mu = 26 isophote.  Returns (r_ell, sep_arcsec), both arrays.
    """
    da = (ra - g_ra) * np.cos(np.radians(g_dec)) * 3600.0   # arcsec, east positive
    dd = (dec - g_dec) * 3600.0
    p = np.radians(g_pa)
    xp = da * np.sin(p) + dd * np.cos(p)                    # along major axis
    yp = -da * np.cos(p) + dd * np.sin(p)                   # along minor axis
    a = g_d26 / 2.0 * 60.0                                  # arcmin -> arcsec semi-major
    return np.hypot(xp / a, yp / (a * g_ba)), np.hypot(da, dd)


def sga_screen(ra, dec, tap):
    """Closest-in-r_ell SGA-2020 galaxy to the target, and whether it flags it."""
    # Data Lab's ADQL rejects CIRCLE with numeric literals
    # ("function circle(numeric,numeric,numeric) does not exist"), so use a box and do the
    # exact ellipse test below in python.
    dra = SGA_BOX / max(np.cos(np.radians(dec)), 1e-3)
    r0, r1 = ra - dra, ra + dra
    if r0 < 0.0 or r1 >= 360.0:
        rac = f"(ra > {r0 % 360.0} OR ra < {r1 % 360.0})"   # RA wrap at 0/360
    else:
        rac = f"ra BETWEEN {r0} AND {r1}"
    q = (f"SELECT ra,dec,d26,pa,ba,z_leda,galaxy FROM sga2020.ellipse "
         f"WHERE {rac} AND dec BETWEEN {dec - SGA_BOX} AND {dec + SGA_BOX}")
    t = query(lambda: tap.launch_job(q).get_results())

    none = {"r_ell": np.nan, "sep_arcsec": np.nan, "sga_galaxy": "", "d26_arcmin": np.nan,
            "z_leda": np.nan, "flagged": False}
    if len(t) == 0:
        return none
    d26 = np.ma.filled(np.ma.asarray(t["d26"], dtype=float), np.nan)
    ba = np.ma.filled(np.ma.asarray(t["ba"], dtype=float), np.nan)
    pa = np.nan_to_num(np.ma.filled(np.ma.asarray(t["pa"], dtype=float), np.nan))
    ok = np.isfinite(d26) & (d26 > 0) & np.isfinite(ba) & (ba > 0)
    if not ok.any():
        return none

    g_ra = np.ma.filled(np.ma.asarray(t["ra"], dtype=float), np.nan)
    g_dec = np.ma.filled(np.ma.asarray(t["dec"], dtype=float), np.nan)
    r_ell, sep = ellipse_radius(ra, dec, g_ra[ok], g_dec[ok], d26[ok], pa[ok], ba[ok])
    j = int(np.nanargmin(r_ell))
    idx = np.flatnonzero(ok)[j]
    return {"r_ell": float(r_ell[j]), "sep_arcsec": float(sep[j]),
            "sga_galaxy": str(t["galaxy"][idx]), "d26_arcmin": float(d26[idx]),
            "z_leda": float(np.ma.filled(np.ma.asarray(t["z_leda"], dtype=float), np.nan)[idx]),
            "flagged": bool(r_ell[j] <= 1.0)}


def candidate_pool(flow_dir, vac):
    """(sorted TARGETIDs, ra, dec) for the same `sel` build_cont_outlier_sample.py uses."""
    out = {t: flow_scores(t, flow_dir)[2] for t in TAGS}
    common = out[TAGS[0]] & out[TAGS[1]]
    M = fits.open(vac)["METADATA"].data
    iv = {int(t): i for i, t in enumerate(M["TARGETID"])}
    sel = sorted(t for t in common if t in iv)
    I = np.array([iv[t] for t in sel])
    return sel, M["RA"][I].astype(float), M["DEC"][I].astype(float)


def self_test():
    """Offline check of the ellipse geometry; no network."""
    a_arcmin, ba, pa = 2.0, 0.5, 30.0                        # d26 = 2', b/a = 0.5, PA = 30 deg
    g_ra, g_dec = 180.0, 40.0
    a_deg = (a_arcmin / 2.0) / 60.0                          # semi-major in degrees
    cd = np.cos(np.radians(g_dec))
    p = np.radians(pa)
    # A point one semi-major axis along the major axis must land at r_ell = 1.
    ra_maj = g_ra + (a_deg * np.sin(p)) / cd
    dec_maj = g_dec + a_deg * np.cos(p)
    # ... and one semi-minor axis along the minor axis must too.
    ra_min = g_ra - (a_deg * ba * np.cos(p)) / cd
    dec_min = g_dec + a_deg * ba * np.sin(p)
    for name, (ra, dec) in {"major": (ra_maj, dec_maj), "minor": (ra_min, dec_min)}.items():
        r, _ = ellipse_radius(ra, dec, np.array([g_ra]), np.array([g_dec]),
                              np.array([a_arcmin]), np.array([pa]), np.array([ba]))
        assert abs(r[0] - 1.0) < 1e-3, f"{name} axis: r_ell = {r[0]}, expected 1"
    r, sep = ellipse_radius(g_ra, g_dec, np.array([g_ra]), np.array([g_dec]),
                            np.array([a_arcmin]), np.array([pa]), np.array([ba]))
    assert r[0] == 0.0 and sep[0] == 0.0, f"centre: r_ell = {r[0]}, sep = {sep[0]}"
    print("self-test ok: major axis, minor axis and centre all give the expected r_ell")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flow-dir", default=str(DEFAULT_FLOW_DIR))
    p.add_argument("--vac", default=str(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits"))
    p.add_argument("--gaia-out", default=str(PATHS["RESULTS"] / "gaia_star_screen.csv"))
    p.add_argument("--sga-out", default=str(PATHS["RESULTS"] / "sga_proximity.csv"))
    p.add_argument("--limit", type=int, default=0, help="screen only the first N candidates")
    p.add_argument("--self-test", action="store_true", help="run the offline geometry check")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()

    warnings.filterwarnings("ignore")
    from astroquery.utils.tap.core import TapPlus
    tap = TapPlus(url=DL_TAP)

    sel, ra, dec = candidate_pool(args.flow_dir, args.vac)
    if args.limit:
        sel, ra, dec = sel[: args.limit], ra[: args.limit], dec[: args.limit]
    print(f"screening {len(sel)} candidates")

    gaia_rows, sga_rows = [], []
    for k, (t, r, d) in enumerate(zip(sel, ra, dec), 1):
        gaia_rows.append({"target_id": t, **gaia_screen(r, d)})
        sga_rows.append({"target_id": t, **sga_screen(r, d, tap)})
        if k % 20 == 0 or k == len(sel):
            print(f"  {k}/{len(sel)}  gaia flagged {sum(x['flagged'] for x in gaia_rows)}  "
                  f"sga flagged {sum(x['flagged'] for x in sga_rows)}")

    for path, rows in ((args.gaia_out, gaia_rows), (args.sga_out, sga_rows)):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out}  n={len(rows)}  flagged={sum(x['flagged'] for x in rows)}")


if __name__ == "__main__":
    main()
