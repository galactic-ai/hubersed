"""Coverage of direct T_e O/H by the Cue posteriors for each line choice.

QUESTION: dropping [SII] and [OI] widens the Cue O/H posteriors. Do they then contain direct T_e
O/H more often, that is, are they less biased in the sense of coverage?

INPUTS: the saved fit posteriors in results/2026-10-14_cue_lineflux_calibration/fits/ (weighted
nautilus points, read with forced.load) and the direct T_e tables of direct.py r2 (KEEP rows).
The nosii variants use the subset_20 direct table, the semi variant the semi direct table; on the
semi galaxies the all-lines fits that exist are compared on the same galaxies. No new fits.

For each variant: N, the fraction with direct O/H inside the Cue 16-84 interval, the fraction with
|Cue - direct| (medians) within the two half widths in quadrature, the fractions with direct
below 16 and above 84, and the median Cue 16-84 width. For Gaussian, calibrated posteriors the
first two are about 0.68. A bimodal posterior has a wide 16-84 interval that can contain direct
without much posterior mass near it.

RESULT: 2026-10-15. Subset_20 (19): direct inside the Cue 16-84 interval 0.05 with all lines
(width 0.08 dex), 0.16 without [SII], 0.05 with the Nicholls prior, 0.47 without [SII] and [OI]
(width 0.37, direct then mostly below Cue). Semi (433): 0.06, width 0.06; for direct >= 8.4 (153)
0.00, direct above Cue in all, Cue - direct -0.47. Dropping [SII] and [OI] mostly widens the
posteriors; coverage stays below 0.68.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/coverage.py``

OUTPUT: results/2026-10-14_cue_lineflux_calibration/nosii/coverage.txt
"""

import forced
import numpy as np
from astropy.table import Table

OUT = forced.OUT
NOS = OUT / "nosii"
TAIL = "_emfit_mu0.4_zs-0.41"
NO_SII = "strong_noSII6716_noSII6731"
VARIANTS = [
    ("subset_20", "all lines", f"strong{TAIL}"),
    ("subset_20", "no [SII]", f"{NO_SII}{TAIL}"),
    ("subset_20", "no [SII], Nicholls N/O", f"{NO_SII}{TAIL}_pnicholls"),
    ("subset_20", "no [SII] and [OI]", f"{NO_SII}_noOI6300{TAIL}"),
    ("semi_ids", "semi (no [OI], + [SIII], He I)", f"semi_noOI6300{TAIL}"),
    ("semi_ids", "all lines, semi galaxies", f"strong{TAIL}"),
]


def direct(name):
    """Return the KEEP rows of a direct.py r2 table with finite O/H."""
    d = Table.read(OUT / f"direct_wild0.4_campbell_r2_{name}.fits")
    return d[d["KEEP"] & np.isfinite(d["OH_P50"])]


def row(label, d, tag):
    """Return one line of the coverage table, or None if no fit exists."""
    have = [i for i, t in enumerate(d["TARGETID"]) if (OUT / "fits" / f"{t}_{tag}.npz").exists()]
    if not have:
        return None
    d = d[have]
    cue = np.array([forced.load(int(t), tag)["oh"] for t in d["TARGETID"]])
    lo, mid, hi = cue.T
    x = np.asarray(d["OH_P50"])
    dw = 0.5 * (np.asarray(d["OH_P84"]) - np.asarray(d["OH_P16"]))
    inside = (x >= lo) & (x <= hi)
    joint = np.abs(mid - x) <= np.hypot(0.5 * (hi - lo), dw)
    return (
        f"{label:34s} N {len(d):4d}  in 16-84 {inside.mean():.2f}  joint {joint.mean():.2f}  "
        f"direct below {(x < lo).mean():.2f} above {(x > hi).mean():.2f}  "
        f"width {np.median(hi - lo):.2f}  Cue - direct {np.median(mid - x):+.2f}"
    )


def main():
    """Write the coverage table."""
    tables = {name: direct(name) for name in {v[0] for v in VARIANTS}}
    out = []
    for name, label, tag in VARIANTS:
        line = row(label, tables[name], tag)
        out.append(line if line else f"{label:34s} no fits")
    rich = tables["semi_ids"][tables["semi_ids"]["OH_P50"] >= forced.ZMIN_OH]
    out.append(f"direct O/H >= {forced.ZMIN_OH} only")
    for _, label, tag in VARIANTS[4:]:
        line = row(label, rich, tag)
        out.append(line if line else f"{label:34s} no fits")
    text = "\n".join(out)
    print(text)
    (NOS / "coverage.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
