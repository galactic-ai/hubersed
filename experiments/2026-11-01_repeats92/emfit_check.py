"""Check the AGN and emission-line labels of 23 of the 92 continuum outliers with EmFit DR1 (#17).

QUESTION: does the DESI DR1 EmFit VAC confirm our visual and FastSpecFit labels (16 AGN, 5
star-forming, 2 weak AGN by WHAN) for the sv3/bright spectrum, so they can leave the continuum sample?
HYPOTHESIS: none registered.
INPUTS: classes.tsv; desi_dr1.emfit on Astro Data Lab TAP, one ``targetid IN`` query, as
2026-10-31_emfit_39633322464249559. EmFit fits Hbeta, [OIII], [NII], Halpha and [SII] only, z <= 0.45.
Broad Halpha is a BL candidate as Pucha+2026 sec 2 (2606.02699 S02_Data.tex:87): prob_broad >= 80,
S/N of ha_b flux and of ha_b_sigma >= 3, ha_b_amplitude / nii_ha_noise >= 2. Broad sigma < 425 km/s
(FWHM < 1000) is marked, since Pucha+2025 App. B finds many such are missed outflows.
BPT uses narrow Balmer (n + out, no broad) and forbidden lines (primary + out), each at S/N >= 3
(flux_err; a line whose window has no noise is not measurable), classed with line_screen.bpt on the
[NII] curves only, its [SII] curve and S/N 5 gate switched off. EmFit has no EW, so no WHAN.
Verdict: BL candidate or BPT AGN is confirmed AGN, composite is AGN or composite, BPT SF is
confirmed emission-line/SF, else not confirmed. Only the sv3/bright row counts; other rows are listed.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-01_repeats92/emfit_check.py``.
RESULT: 20 of 23 have a row (07, 17, 28 none, z > 0.5), one per target; 03, 16, 36, 38, 82 only
non-sv3/bright. Of the 15 sv3/bright rows, 10 agree (5 broad Ha, 2 BPT AGN, 3 SF), 14 and 24 are
composite, 20, 33, 34 untested (one BPT line S/N < 3), none conflict. Other rows: 03 SF, 36 broad Ha.
Assumptions: EmFit MC flux errors are underestimated (Pucha+2025 erratum), some are ~0.01; Halpha
narrow/broad split is degenerate for broad-line targets; the BL and S/N cuts set the yes/untested split.
FIGURES: results/2026-11-01_repeats92/ emfit_check.txt, emfit_rows.fits (all rows of the query).
"""

# %%
import csv
from pathlib import Path
from unittest import mock

import numpy as np
from astroquery.utils.tap.core import TapPlus

from hubersed.detect import line_screen
from hubersed.detect.sky_screens import DL_TAP, query
from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-01_repeats92"
CLASSES = Path(__file__).resolve().parent / "classes.tsv"
GROUPS = {
    "AGN": [
        "07",
        "08",
        "14",
        "16",
        "17",
        "24",
        "28",
        "29",
        "36",
        "38",
        "43",
        "45",
        "56",
        "70",
        "81",
        "82",
    ],
    "SF": ["03", "33", "46", "61", "77"],
    "WHAN": ["20", "34"],
}
SNR = 3.0
PROB_BROAD = 80.0
AON_BROAD = 2.0
SIGMA_OUTFLOW = 1000.0 / 2.3548  # km/s, FWHM 1000
# line, EmFit components summed for the narrow-line BPT, window whose noise says it was fitted
BPT_LINES = {
    "HBETA": (("hb_n", "hb_out"), "hb"),
    "OIII_5007": (("oiii5007", "oiii5007_out"), "oiii"),
    "NII_6584": (("nii6583", "nii6583_out"), "nii_ha"),
    "HALPHA": (("ha_n", "ha_out"), "nii_ha"),
}


def val(r, name):
    """Return column ``name`` of row ``r`` as a float, NaN if masked."""
    v = r[name]
    return np.nan if np.ma.is_masked(v) else float(v)


def line(r, comps, window):
    """Return the summed flux and error of ``comps``, both NaN if the window was not fitted."""
    if not val(r, f"{window}_noise") > 0:
        return np.nan, np.nan
    f = sum(val(r, f"{c}_flux") for c in comps)
    e = np.sqrt(sum(val(r, f"{c}_flux_err") ** 2 for c in comps))
    return f, e


def broad(r):
    """Return the raw broad Halpha values and whether they pass the BL candidate cuts."""
    f, fe = val(r, "ha_b_flux"), val(r, "ha_b_flux_err")
    s, se = val(r, "ha_b_sigma"), val(r, "ha_b_sigma_err")
    aon = val(r, "ha_b_amplitude") / val(r, "nii_ha_noise")
    sn_f = f / fe if fe > 0 else 0.0
    sn_s = s / se if se > 0 else 0.0
    ok = val(r, "prob_broad") >= PROB_BROAD and sn_f >= SNR and sn_s >= SNR and aon >= AON_BROAD
    return dict(prob=val(r, "prob_broad"), sigma=s, sn_flux=sn_f, sn_sigma=sn_s, aon=aon, bl=ok)


def bpt(r):
    """Return the [NII] BPT class, the line S/N and the log ratios of row ``r``."""
    fl = {k: line(r, *v) for k, v in BPT_LINES.items()}
    sn = {k: f / e if e > 0 else (0.0 if np.isfinite(e) else np.nan) for k, (f, e) in fl.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        n2 = np.log10(fl["NII_6584"][0] / fl["HALPHA"][0])
        o3 = np.log10(fl["OIII_5007"][0] / fl["HBETA"][0])
    if any(np.isnan(v) for v in sn.values()):
        return "not measurable (window not fitted)", sn, n2, o3
    if any(v < SNR for v in sn.values()):
        return "not measurable (S/N < 3)", sn, n2, o3
    d = {f"{k}_FLUX": [f] for k, (f, e) in fl.items()}
    d |= {f"{k}_FLUX_IVAR": [1 / e**2] for k, (f, e) in fl.items()}
    d |= {"SII_6716_FLUX": [0.0], "SII_6731_FLUX": [0.0]}  # [NII] curves only
    with mock.patch.object(line_screen, "SNMIN_BPT", 0.0):  # S/N gate applied above
        cls = line_screen.bpt(d, 0)[0]
    return cls or "not measurable (flux <= 0)", sn, n2, o3


def verdict(b, cls):
    """Return the EmFit verdict from the broad test and the BPT class."""
    if b["bl"]:
        return "confirmed AGN (broad Ha)"
    return {
        "AGN": "confirmed AGN (BPT)",
        "composite": "AGN or composite",
        "star-forming": "confirmed emission-line/SF",
    }.get(cls, "not confirmed")


def agrees(group, v):
    """Return yes, partial, no or untested for a label group and an EmFit verdict."""
    if v == "not confirmed":
        return "untested"
    if v == "AGN or composite":
        return "partial"
    agn = v.startswith("confirmed AGN")
    return "yes" if agn == (group in ("AGN", "WHAN")) else "no"


# %%
def main():
    """Query EmFit for the 23 targets, write the raw rows and the check table."""
    with open(CLASSES, newline="") as fh:
        lab = {r["nn"]: r for r in csv.DictReader(fh, delimiter="\t")}
    nns = [(nn, g) for g, v in GROUPS.items() for nn in v]
    ids = ",".join(lab[nn]["targetid"] for nn, _ in nns)
    q = f"SELECT * FROM desi_dr1.emfit WHERE targetid IN ({ids})"
    em = query(lambda: TapPlus(url=DL_TAP).launch_job(q).get_results())
    OUT.mkdir(parents=True, exist_ok=True)
    em.write(OUT / "emfit_rows.fits", overwrite=True)
    print(f"{len(em)} EmFit rows, {len(em.colnames)} columns:\n{' '.join(em.colnames)}")

    hdr = "nn | TARGETID | label | z | prob_broad, ha_b sigma, S/N flux, S/N sigma, AoN, BL | BPT | verdict | agrees"
    rows, notes, tally = [hdr], [], {}
    for nn, g in sorted(nns):
        tid = int(lab[nn]["targetid"])
        mine = em[np.asarray(em["targetid"], np.int64) == tid]
        sv3 = mine[(mine["survey"] == "sv3") & (mine["program"] == "bright")]
        assert len(sv3) <= 1, f"{tid} has {len(sv3)} sv3/bright rows"
        other = [f"{r['survey']}/{r['program']}" for r in mine]
        for r in mine:
            b, (cls, sn, n2, o3) = broad(r), bpt(r)
            tag = f"{r['survey']}/{r['program']}"
            snt = ", ".join(f"{k} {v:.1f}" for k, v in sn.items())
            warn = (
                " (sigma < 425 km/s, outflow caution)"
                if b["bl"] and b["sigma"] < SIGMA_OUTFLOW
                else ""
            )
            bs = (f"prob_broad {b['prob']:.0f}, ha_b sigma {b['sigma']:.0f}, S/N flux "
                  f"{b['sn_flux']:.2f}, S/N sigma {b['sn_sigma']:.2f}, AoN {b['aon']:.2f}")  # fmt: skip
            notes.append(f"{nn} {tid} {tag} z {val(r, 'z'):.3f}: {bs}; S/N {snt}; log N2/Ha "
                         f"{n2:.2f}, log O3/Hb {o3:.2f}; BPT {cls}; {verdict(b, cls)}{warn}")  # fmt: skip
        if len(sv3) == 0:
            why = (
                "no EmFit row" if len(mine) == 0 else f"no sv3/bright row (only {', '.join(other)})"
            )
            rows.append(f"{nn} | {tid} | {g} | - | - | - | not confirmed ({why}) | untested")
            tally["untested"] = tally.get("untested", 0) + 1
            continue
        r = sv3[0]
        b, (cls, sn, n2, o3) = broad(r), bpt(r)
        v = verdict(b, cls)
        a = agrees(g, v)
        tally[a] = tally.get(a, 0) + 1
        bs = (f"{b['prob']:.0f}, {b['sigma']:.0f}, {b['sn_flux']:.1f}, {b['sn_sigma']:.1f}, "
              f"{b['aon']:.1f}, {'BL' if b['bl'] else 'no'}")  # fmt: skip
        rows.append(f"{nn} | {tid} | {g} | {val(r, 'z'):.3f} | {bs} | {cls} | {v} | {a}")
    rows.append("counts: " + ", ".join(f"{k} {n}" for k, n in sorted(tally.items())))
    rows.append("WHAN: skipped, EmFit gives no EW (its *_continuum is a residual level only).")
    text = "\n".join(["# EmFit DR1 check, sv3/bright row; see emfit_check.py", *rows, "", *notes])
    (OUT / "emfit_check.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
