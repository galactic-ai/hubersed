"""Total metallicity [Z/H] of each MIST2 grid point, from the initial X and Z of its MESA runs.

The MESA work directories of Dotter et al. 2026 (MIST2, arXiv:2602.22012), Zenodo record
15232687, rotating_workdirs.tgz (md5 8c63c7e2f1859c8d1773aaad348fe1d9, vvcrit0.4 as the FSPS
isochrones), hold input_XYZ (initial X, Y, Z) for every grid point and mass. [Z/H] is
log10(Z/X) minus that of [Fe/H] = 0, [alpha/Fe] = 0, so it equals the [Fe/H] label at
[alpha/Fe] = 0, as MILES logzsol does. The script checks that every mass of a grid point has the
same X and Z, and that log10(X_fe56/X_h1) relative to solar reproduces the [Fe/H] label.
Get the inputs with ``tar -xzf rotating_workdirs.tgz -C extracted '*/input_XYZ' '*/input_initial_xa.data'``
in data/mist2_zenodo.
COMMAND: ``uv run --no-sync python experiments/2026-11-01_poco_repeats/mist2_zh.py``.
OUTPUT: results/2026-11-01_poco_repeats/mist2_zh.txt.
"""

# %%
import re
from pathlib import Path

import numpy as np

from hubersed.paths import PATHS

SRC = PATHS["DATA"] / "mist2_zenodo" / "extracted"
OUT = PATHS["RESULTS"] / Path(__file__).resolve().parent.name / "mist2_zh.txt"


def label(s):
    """Return the value of a grid label such as m025 or p4."""
    v = int(s[1:]) / (100 if len(s) == 4 else 10)
    return -v if s[0] == "m" else v


def read_point(folder):
    """Return X, Z and log10(X_fe56/X_h1) of one grid point, checked across its masses."""
    rows = []
    for d in sorted(folder.glob("*M*_dir")):
        x, _, z = np.loadtxt(d / "input_XYZ")
        xa = dict(line.split()[:2] for line in (d / "input_initial_xa.data").read_text().splitlines()
                  if not line.startswith("!"))  # fmt: skip
        rows.append((x, z, np.log10(float(xa["fe56"]) / float(xa["h1"]))))
    rows = np.array(rows)
    assert np.ptp(rows, axis=0).max() < 1e-12, f"{folder.name}: masses differ"
    return rows[0], len(rows)


def table():
    """Return [Fe/H], [alpha/Fe], X, Z, [Z/H] and the Fe check of every grid point."""
    pts = {}
    for folder in sorted(SRC.glob("feh_*_afe_*_vvcrit0.4")):
        feh, afe = (label(s) for s in re.match(r"feh_(\w+)_afe_(\w+)_", folder.name).groups())
        pts[(feh, afe)] = read_point(folder)
    (x0, z0, fe0), _ = pts[(0.0, 0.0)]
    out = []
    for (feh, afe), ((x, z, fe), n) in sorted(pts.items()):
        out.append((feh, afe, x, z, np.log10(z / x) - np.log10(z0 / x0), fe - fe0, n))
    return np.array(out)


def main():
    """Write the [Z/H] table of the MIST2 grid."""
    t = table()
    dfe = np.abs(t[:, 5] - t[:, 0]).max()
    print(f"{len(t)} grid points; max |Fe check - [Fe/H] label| {dfe:.2e} dex")
    assert dfe < 1e-3, "fe56 does not follow the [Fe/H] label"
    lines = [f"{'[Fe/H]':>7} {'[a/Fe]':>7} {'X':>10} {'Z':>12} {'[Z/H]':>8} {'[Z/H]-[Fe/H]':>13} "
             f"{'masses':>6}"]  # fmt: skip
    for feh, afe, x, z, zh, _, n in t:
        lines.append(
            f"{feh:7.2f} {afe:7.2f} {x:10.6f} {z:12.6e} {zh:8.4f} {zh - feh:13.4f} {int(n):6d}"
        )
    lines += ["", "[Z/H] = log10(Z/X) - log10(Z/X) at [Fe/H] = 0, [alpha/Fe] = 0, MIST2 input_XYZ"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
