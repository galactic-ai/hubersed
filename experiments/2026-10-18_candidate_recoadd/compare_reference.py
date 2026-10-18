"""Check that the recoadd outputs of the reference are unchanged after the --tid and cosmic commits.

recoadd.slurm copies the reference outputs of the 2026-10-09 scripts to a backup folder, reruns the
four scripts for 39627770174637084 and calls this script. Text files must match exactly, npz files
must hold the same keys with equal arrays, and the fluxcalib files the same data in every HDU.
The fluxcalib logs hold timestamps and are skipped.

COMMAND: ``.venv/bin/python experiments/2026-10-18_candidate_recoadd/compare_reference.py
<backup folder> <reference folder>``. Exits 1 if anything differs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

TEXT = ("coadd_rows.txt", "fluxcal_rerun.txt", "ivar_check.txt", "ivar_recompute.txt")
NPZ = ("resolution_matrix_rows.npz", "resolution_matrix_rows_ivar.npz")


def same_npz(a, b):
    """Return the keys whose arrays differ, or that only one file has."""
    x, y = np.load(a), np.load(b)
    bad = sorted(set(x.files) ^ set(y.files))
    for k in set(x.files) & set(y.files):
        u, v = x[k], y[k]
        if u.dtype.kind in "fc":
            ok = u.shape == v.shape and np.array_equal(u, v, equal_nan=True)
        else:
            ok = u.shape == v.shape and np.array_equal(u, v)
        if not ok:
            bad.append(k)
    return bad


def same_fits(a, b):
    """Return the HDU names whose data differ."""
    bad = []
    with fits.open(a) as x, fits.open(b) as y:
        if len(x) != len(y):
            return ["number of HDUs"]
        for u, v in zip(x, y, strict=True):
            if u.data is None and v.data is None:
                continue
            if u.data is None or v.data is None or not np.array_equal(u.data, v.data):
                bad.append(u.name)
    return bad


def main(backup, ref):
    """Compare every output and print one line per file."""
    fails = 0
    for name in TEXT:
        ok = (backup / name).read_text() == (ref / name).read_text()
        fails += not ok
        print(f"{'same' if ok else 'DIFFERENT'} {name}")
    for name in NPZ:
        bad = same_npz(backup / name, ref / name)
        fails += bool(bad)
        print(f"{'same' if not bad else 'DIFFERENT'} {name} {bad}")
    olds = sorted((backup / "fluxcalib_rerun").rglob("*.fits.gz"))
    for old in olds:
        rel = old.relative_to(backup)
        bad = same_fits(old, ref / rel)
        fails += bool(bad)
        print(f"{'same' if not bad else 'DIFFERENT'} {rel} {bad}")
    print(f"{len(TEXT) + len(NPZ) + len(olds)} files compared, {fails} different")
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("backup", type=Path)
    ap.add_argument("ref", type=Path)
    args = ap.parse_args()
    sys.exit(1 if main(args.backup, args.ref) else 0)
