"""EmFit DR1 lines of 39633322464249559, robust6 target 5 (#17): is it an AGN?

QUESTION: does the EmFit VAC find a broad Balmer component or AGN-like narrow-line ratios?
HYPOTHESIS: none registered. FastSpecFit gives broad MgII (sigma 2677 km/s) and no broad Halpha.
INPUTS: desi_dr1.emfit on Astro Data Lab TAP (sv3/bright row), as
2026-10-14_cue_lineflux_calibration/data.py; Kewley+01 and Kauffmann+03 [NII] lines (TeX checked
2026-10-31, astro-ph/0106324 and astro-ph/0304239); WHAN of Cid Fernandes+11 (1012.4426, sec 5).
EmFit fits Hbeta, [OIII], [NII], Halpha and [SII] only, so it cannot test MgII.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_emfit_39633322464249559/analysis.py``.
RESULT: EmFit puts almost all Halpha in a broad component, sigma 1651 km/s, flux 167 +- 69 (S/N 2.4),
prob_broad 78.2; narrow Halpha 1.9 +- 14, [NII] 1.5 +- 8.8, Hbeta 2.6 +- 3.2, [OIII] 21 +- 9.
FastSpecFit gives the opposite split (narrow Halpha and [NII] at S/N ~30, no broad), so the
Halpha decomposition is degenerate here (Halpha at ~9100 A observed). No BPT from EmFit.
Assumptions: totals sum narrow, outflow and broad components, as the cue data.py.
FIGURES: results/2026-10-31_emfit_39633322464249559/ emfit_row.fits, emfit.txt.
"""

# %%
import numpy as np
from astropy.table import Table
from astroquery.utils.tap.core import TapPlus

from hubersed.detect.sky_screens import DL_TAP, query
from hubersed.paths import PATHS

TID = 39633322464249559
OUT = PATHS["RESULTS"] / "2026-10-31_emfit_39633322464249559"
LINES = {
    "HB": ("hb_n", "hb_out", "hb_b"),
    "OIII5007": ("oiii5007", "oiii5007_out"),
    "NII6584": ("nii6583", "nii6583_out"),
    "HA": ("ha_n", "ha_out", "ha_b"),
}  # as 2026-10-14_cue_lineflux_calibration/data.py EMFIT


def val(row, name):
    """Return a column of ``row`` as a float, NaN if masked or absent."""
    if name not in row.colnames:
        return np.nan
    v = row[name][0]
    return np.nan if v is np.ma.masked else float(v)


def main():
    """Fetch the EmFit row, write it and a short summary."""
    OUT.mkdir(parents=True, exist_ok=True)
    q = f"SELECT * FROM desi_dr1.emfit WHERE targetid = {TID}"
    em = query(lambda: TapPlus(url=DL_TAP).launch_job(q).get_results())
    em.write(OUT / "emfit_row.fits", overwrite=True)
    print(f"{len(em)} EmFit rows: " + ", ".join(f"{r['survey']}/{r['program']}" for r in em))
    sel = em[(em["survey"] == "sv3") & (em["program"] == "bright")]
    assert len(sel) == 1, f"{len(sel)} sv3/bright rows"
    r = Table(sel)
    lines = [f"# {TID} EmFit DR1 sv3/bright; flux 1e-17 erg/s/cm2, sigma km/s"]
    for k in ("prob_broad", "ha_b_sigma", "oiii_dbl_flag", "sii_dbl_flag"):
        lines.append(f"{k:>14s} {r[k][0]}")
    flux = {}
    for name, comps in LINES.items():
        f = np.nansum([val(r, f"{c}_flux") for c in comps])
        e = np.sqrt(np.nansum([val(r, f"{c}_flux_err") ** 2 for c in comps]))
        flux[name] = (f, e)
        parts = ", ".join(f"{c} {val(r, c + '_flux'):.2f}+-{val(r, c + '_flux_err'):.2f}"
                          for c in comps)  # fmt: skip
        lines.append(f"{name:>14s} total {f:.2f} +- {e:.2f} (S/N {f / e:.1f}); {parts}")
    x = np.log10(flux["NII6584"][0] / flux["HA"][0])
    y = np.log10(flux["OIII5007"][0] / flux["HB"][0])
    kew = 0.61 / (x - 0.47) + 1.19 if x < 0.47 else -np.inf
    kau = 0.61 / (x - 0.05) + 1.3 if x < 0.05 else -np.inf
    lines.append(f"log [NII]/Ha {x:.3f}, log [OIII]/Hb {y:.3f}; Kewley+01 at x {kew:.3f}, "
                 f"Kauffmann+03 at x {kau:.3f}")  # fmt: skip
    (OUT / "emfit.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"columns with 'sigma' or 'ew': {[c for c in r.colnames if 'sigma' in c or 'ew' in c]}")


if __name__ == "__main__":
    main()
