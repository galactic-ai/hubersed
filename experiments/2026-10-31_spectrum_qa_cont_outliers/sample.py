"""Rebuild the 94 common continuum outliers from the seed 0 flows and the 2026-09-04 screen CSVs.

The cuts and their order are those of the issue #10 table of 2026-09-04. A galaxy is a common
outlier when both seed 0 continuum flows flag it. Then, in order: (1) it must be in the
FastSpecFit VAC, (2) VAC z must lie in [0.01, 0.6] and |z_pipeline - z_VAC| / (1 + z_VAC) <= 0.01,
(3) Gaia star, (4) SGA-2020 proximity, (5) SIMBAD AGN or star, (6a) Legacy PSF
with fewer than 2 forbidden species, (6b) BPT AGN, (7) shred. Steps 3 to 7 read the flags of the
*_v2.csv screens in results/. The code that wrote those CSVs and screened_outliers_v4.npz was a
scratch script that no longer exists, so the result is checked against both the #10 counts and
screened_outliers_v4.npz.

COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-31_spectrum_qa_cont_outliers/sample.py``.
OUTPUT: results/2026-10-31_spectrum_qa_cont_outliers/sample94.npz and sample94.txt.
"""

import csv

import h5py
import numpy as np
import torch
from astropy.io import fits

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
FLOWS = RES / "wide_flow_corrected"
OUT = RES / "2026-10-31_spectrum_qa_cont_outliers"
ZMIN, ZMAX, ZBAD = 0.01, 0.6, 0.01
# cont column of the issue #10 table (2026-09-04), after each step
EXPECTED = {
    "raw": 512,
    "1": 231,
    "2": 187,
    "3": 179,
    "4": 157,
    "5": 124,
    "6a": 121,
    "6b": 102,
    "7": 94,
}


def read_csv(name):
    """Return the rows of a results/ screen CSV keyed by TARGETID."""
    with open(RES / name, newline="") as fh:
        return {int(r["target_id"]): r for r in csv.DictReader(fh)}


def screen(keep, rows, cols, name):
    """Drop the targets where any of ``cols`` is "True". Every kept target must have a row."""
    missing = [t for t in keep if t not in rows]
    if missing:
        raise ValueError(f"{name}: {len(missing)} targets have no row, for example {missing[:3]}")
    return [t for t in keep if not any(rows[t][c] == "True" for c in cols)]


def main():
    """Apply the cuts, check the counts and the v4 list, and write the sample."""
    flagged = {}
    for tag in ("cont10", "cont15"):
        o = torch.load(
            FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt",
            weights_only=False,
            map_location="cpu",
        )
        flagged[tag] = {int(t) for t in o["outlier_target_ids"]}
    keep = sorted(flagged["cont10"] & flagged["cont15"])
    counts = {"raw": len(keep)}

    vac = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits")["METADATA"].data
    zvac = {int(t): float(z) for t, z in zip(vac["TARGETID"], vac["Z"], strict=True)}
    keep = [t for t in keep if t in zvac]
    counts["1"] = len(keep)

    with h5py.File(RES / "lineEW_flow" / "desi_lines.h5", "r") as f:
        zpipe = dict(
            zip(f["target_ids"][:].astype(np.int64).tolist(), f["zs"][:].tolist(), strict=True)
        )
    n_zbad = sum(abs(zpipe[t] - zvac[t]) / (1 + zvac[t]) > ZBAD for t in keep)
    keep = [
        t
        for t in keep
        if ZMIN <= zvac[t] <= ZMAX and abs(zpipe[t] - zvac[t]) / (1 + zvac[t]) <= ZBAD
    ]
    counts["2"] = len(keep)

    agn = read_csv("agn_star_screen_v2.csv")
    steps = [
        ("3", read_csv("gaia_star_screen_v2.csv"), ["flagged"], "gaia"),
        ("4", read_csv("sga_proximity_v2.csv"), ["flagged"], "sga"),
        # "manual kept" in #10: the manual exclusion (39633322460057381) is not applied
        ("5", read_csv("simbad_screen_v2.csv"), ["simbad_agn", "simbad_star"], "simbad"),
        ("6a", agn, ["psf_nonstellar"], "psf"),
        ("6b", agn, ["bpt_agn"], "bpt"),
        ("7", read_csv("shred_screen_v2.csv"), ["flagged"], "shred"),
    ]
    for step, rows, cols, name in steps:
        keep = screen(keep, rows, cols, name)
        counts[step] = len(keep)

    v4 = sorted(int(t) for t in np.load(RES / "screened_outliers_v4.npz")["cont"])
    lines = ["step   kept  issue #10"]
    lines += [f"{k:>4} {v:6d} {EXPECTED[k]:10d}" for k, v in counts.items()]
    lines.append(f"zbad flags among the 231: {n_zbad}")
    lines.append(f"equal to screened_outliers_v4.npz cont ({len(v4)}): {keep == v4}")
    text = "\n".join(lines)
    print(text)
    assert counts == EXPECTED, "counts differ from the issue #10 table"
    assert keep == v4, "sample differs from screened_outliers_v4.npz"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample94.txt").write_text(text + "\n")
    np.savez(
        OUT / "sample94.npz",
        target_ids=np.array(keep, np.int64),
        z_vac=np.array([zvac[t] for t in keep]),
    )
    print(f"wrote {OUT / 'sample94.npz'}")


if __name__ == "__main__":
    main()
