"""QUESTION: Which major and minor emission lines of the DESI prospect viewer have no line in Cue's list?
HYPOTHESIS: None registered before this analysis.
METHOD: air entries go to vacuum with _airtovac, as the viewer does. A line is in Cue if a Cue line
lies within 1 A. The fitted range is the rest range of 39633278973513279, 3510-9000 A.
INPUTS: desi_prospect_emission_lines.txt, desihub/prospect py/prospect/data/emission_lines.txt at
699d599b73, and cuejax/data/cue_emlines_info.dat of cue 82ed9dc (uv.lock). MISSING_LINES is not used.
SEED: none.
COMMAND: from the repository root,
uv run python experiments/2026-10-21_desi_line_lists/analysis.py
RESULT: 2026-10-21. All 12 major lines in the fitted range are in Cue within 0.01 A. Of the 17,
only C IV 1549 fails, as DESI lists the doublet blend and Cue has 1548.19 and 1550.77. DESI flags
Mg II as air though 2796.35 and 2803.53 are vacuum, so the match is off by 0.82 A. Of the 37 minor
lines, 10 are not in Cue and only He II 5411 (5413.02 A) is in the fitted range. The other 9 are UV.
FIGURES: none. major_vs_cue.txt and minor_vs_cue.txt in results/2026-10-21_desi_line_lists/.
"""

# %%

import argparse
from importlib.resources import files
from pathlib import Path

import numpy as np

from hubersed.paths import PATHS
from hubersed.sps.utils import _airtovac

HERE = Path(__file__).resolve().parent
OUT = PATHS["RESULTS"] / "2026-10-21_desi_line_lists"
TOL = 1.0  # A, match tolerance
FIT_LO, FIT_HI = 3510.0, 9000.0  # rest A, fitted range of 39633278973513279


def read_desi(path):
    """Return the name, vacuum wavelength and major flag of each DESI prospect line."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, longname, w, vacuum, major = line.strip().split(",")
        w = float(w) if vacuum == "True" else _airtovac(float(w))
        rows.append((longname, w, major == "True"))
    return rows


def table(rows, cue, kind):
    """Return the text table of ``rows`` against the Cue lines."""
    lines = [f"DESI prospect {kind} lines against Cue ({len(cue)} lines), match within {TOL} A", ""]
    lines.append(
        f"{'DESI line':>16} {'vac A':>9} {'nearest Cue line':>26} {'Cue A':>9} {'dA':>6} in_Cue fitted"
    )
    n_miss, n_miss_fit = 0, 0
    for name, w, _ in rows:
        i = int(np.argmin(np.abs(cue["w"] - w)))
        d = cue["w"][i] - w
        ok = abs(d) < TOL
        fit = FIT_LO <= w <= FIT_HI
        n_miss += not ok
        n_miss_fit += (not ok) and fit
        lines.append(
            f"{name:>16} {w:9.2f} {cue['n'][i].strip():>26} {cue['w'][i]:9.2f} {d:6.2f} "
            f"{'yes' if ok else 'NO':>6} {'yes' if fit else 'no':>6}"
        )
    lines += [
        "",
        f"{len(rows)} {kind} lines, {n_miss} not in Cue, {n_miss_fit} of those in the fitted range",
    ]
    return lines


def main():
    """Write the tables of major and minor lines against Cue."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    cue = np.genfromtxt(
        files("cuejax.data").joinpath("cue_emlines_info.dat"),
        dtype=[("w", "f8"), ("n", "<U40")],
        delimiter=",",
    )
    rows = read_desi(HERE / "desi_prospect_emission_lines.txt")
    OUT.mkdir(parents=True, exist_ok=True)
    for kind, flag in (("major", True), ("minor", False)):
        lines = table([r for r in rows if r[2] == flag], cue, kind)
        (OUT / f"{kind}_vs_cue.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
