"""VAC values of one target: FastSpecFit v3.0 (and v2.1), CIGALE DR1 v1.2 and EmFit DR1.

Definitions and units are in UberSED knowledge/vac_reference_fastspecfit_cigale_emfit.md:
- FastSpecFit LOGMSTAR is the surviving mass at h = 1, total aperture.
- FastSpecFit AGE is weighted by mass formed over 5 fixed bins, and its SFR covers 0-30 Myr.
- A_V = 1.086 TAUV.
- CIGALE LOGM comes from photometry only; its LOGSFR is averaged over 10 Myr.
- EmFit fluxes are in 1e-17 erg/s/cm2; an undetected component is 0 with flag -1.
Uncertainties are 1/sqrt(ivar) for FastSpecFit (Monte Carlo, NMONTE 10) and the catalogue errors
for CIGALE and EmFit. EmFit is queried from desi_dr1.emfit on Astro Data Lab.
COMMAND: ``uv run --no-sync python experiments/2026-11-01_poco_repeats/vac_rows.py --tid <TID>``.
OUTPUT: results/2026-11-01_poco_repeats/<TID>/vac_rows.txt.
"""

# %%
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

from hubersed.paths import PATHS

HERE = Path(__file__).resolve().parent
DATA = PATHS["DATA"]
FSF3 = DATA / "fastspecfit_v3.0" / "fastspec-iron-sv3-bright.fits"
FSF2 = DATA / "fastspec-iron-sv3-bright.fits"
CIG = DATA / "cigale_dr1" / "IronPhysProp_v1.2.fits"
EMFIT_LINES = ("hb_n", "oiii5007", "ha_n", "nii6583", "sii6716", "sii6731")


def row(path, ext, tid):
    """Return the single row of ``tid`` in extension ``ext`` of ``path``, reading one column first."""
    with fits.open(path, memmap=True) as h:
        d = h[ext].data
        i = np.flatnonzero(np.asarray(d["TARGETID"]) == tid)
        assert i.size == 1, f"{path.name}[{ext}]: {i.size} rows for {tid}"
        return d[i[0]].array[i[0] : i[0] + 1].copy()[0]


def sigma(ivar):
    """Return 1/sqrt(ivar), or NaN when ivar is not positive."""
    return float(1 / np.sqrt(ivar)) if ivar > 0 else np.nan


def fastspecfit(tid):
    """Return lines of FastSpecFit v3.0 values with errors, and the v2.1 values."""
    s, f = row(FSF3, "SPECPHOT", tid), row(FSF3, "FASTSPEC", tid)
    out = ["FastSpecFit Iron v3.0 (fastspecfit 3.1.5, templates 2.0.0), value +- 1/sqrt(ivar)"]
    for k, unit in (("LOGMSTAR", "log Msun, surviving, h=1"), ("AGE", "Gyr"), ("TAUV", ""),
                    ("ZZSUN", ""), ("SFR", "Msun/yr, 0-30 Myr, h=1"), ("VDISP", "km/s"),
                    ("DN4000", ""), ("DN4000_MODEL", "")):  # fmt: skip
        out.append(f"  {k:14s} {s[k]:12.4f} +- {sigma(s[k + '_IVAR']):9.4f}  {unit}")
    out.append(
        f"  {'A_V':14s} {1.086 * s['TAUV']:12.4f} +- {1.086 * sigma(s['TAUV_IVAR']):9.4f}  mag"
    )
    for k in ("RCHI2_CONT", "RCHI2_PHOT", "APERCORR", "APERCORR_G", "APERCORR_R", "APERCORR_Z"):
        v = s[k] if k in s.array.names else f[k]
        out.append(f"  {k:14s} {v:12.4f}")
    out.append(f"  {'COEFF':14s} " + " ".join(f"{c:.3e}" for c in s["COEFF"]))
    v2 = row(FSF2, "FASTSPEC", tid)
    out.append("FastSpecFit Iron v2.1 (fastspecfit 2.5.1, templates 1.3.0), no errors")
    out.append(
        "  " + ", ".join(f"{k} {v2[k]:.4f}" for k in ("LOGMSTAR", "AGE", "AV", "SFR", "VDISP"))
    )
    return out


def cigale(tid):
    """Return lines of the CIGALE DR1 v1.2 values."""
    c = row(CIG, 1, tid)
    out = ["CIGALE DR1 Iron v1.2 (photometry g r z W1-W4, BC03, Chabrier, solar Z)"]
    out.append(f"  LOGM   {c['LOGM']:.4f} +- {c['LOGM_ERR']:.4f}  log Msun")
    out.append(f"  LOGSFR {c['LOGSFR']:.4f} +- {c['LOGSFR_ERR']:.4f}  log Msun/yr, 10 Myr")
    out.append("  " + ", ".join(f"{k} {c[k]:.4g}" for k in ("Z", "CHI2", "FLAG_MASSPDF",
                                                           "FLAG_SFRPDF", "FLAGOPTICAL",
                                                           "FLAGINFRARED")))  # fmt: skip
    return out


def emfit(tid):
    """Return lines of the EmFit DR1 narrow-line values, queried from Data Lab."""
    from astroquery.utils.tap.core import TapPlus

    from hubersed.detect.sky_screens import DL_TAP, query

    q = f"SELECT * FROM desi_dr1.emfit WHERE targetid = {tid}"
    t = query(lambda: TapPlus(url=DL_TAP).launch_job(q).get_results())
    out = [f"EmFit DR1 ({len(t)} rows; flux 1e-17 erg/s/cm2, sigma km/s, flag -1 = not detected)"]
    for r in t:
        out.append(f"  survey {r['survey']} program {r['program']} specprod {r['specprod']} "
                   f"z {r['z']:.5f} prob_broad {r['prob_broad']}")  # fmt: skip
        for ln in EMFIT_LINES:
            f, e = float(r[f"{ln}_flux"]), float(r[f"{ln}_flux_err"])
            snr = f / e if e > 0 else np.nan
            out.append(f"    {ln:9s} flux {f:9.3f} +- {e:7.3f} (S/N {snr:6.2f})  sigma "
                       f"{float(r[f'{ln}_sigma']):7.1f}  flag {int(r[f'{ln}_sigma_flag'])}")  # fmt: skip
        out.append(
            "    dbl flags: oiii " + str(r["oiii_dbl_flag"]) + ", sii " + str(r["sii_dbl_flag"])
        )
    return out


def emfit_or_note(tid, timeout=120):
    """Return emfit(tid), or a note that Data Lab did not answer within ``timeout`` seconds."""
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as Timeout

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(emfit, tid).result(timeout=timeout)
    except Timeout:
        return [f"EmFit DR1: Data Lab did not answer within {timeout} s; rerun when it is up"]
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def main(tid):
    """Write the VAC values of ``tid``."""
    lines = [f"TARGETID {tid}", ""] + fastspecfit(tid) + [""] + cigale(tid) + [""]
    lines += emfit_or_note(tid)
    path = PATHS["RESULTS"] / HERE.name / str(tid) / "vac_rows.txt"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tid", type=int, required=True)
    main(ap.parse_args().tid)
