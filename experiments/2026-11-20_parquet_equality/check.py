"""Check that every spectrum of the DESI pickle chunks is identical in desi_sv3_bright_v1.

QUESTION: does the Parquet dataset hold every chunk row with the same six columns, so the pickles
can be deleted?
HYPOTHESIS: identical, since chunks_to_parquet.py only copies values (float32 to float32).
INPUTS: data/desi_spectra/DESIchunk*.pkl (250 files), data/datasets/desi_sv3_bright_v1.
SEED: none.
COMMAND: ``uv run python experiments/2026-11-20_parquet_equality/check.py``.
RESULT: identical. All 255,592 rows of the 250 chunks are in the dataset with the same six columns.
Assumptions: equality is exact (np.array_equal, NaN equal to NaN for float columns).
FIGURES: none (check.txt).
"""

import pickle
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

from hubersed.paths import PATHS

CHUNKS = PATHS["DATA"] / "desi_spectra"
DATASET = PATHS["DATA"] / "datasets" / "desi_sv3_bright_v1"
OUT = PATHS["RESULTS"] / "2026-11-20_parquet_equality"
COLUMNS = ("spec", "w", "z", "target_id", "norm", "zerr")


def locate():
    """Return each TARGETID's (file, row group, row) in the dataset and the file list."""
    files = sorted((DATASET / "data").glob("*.parquet"))
    where = {}
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            tid = pf.read_row_group(g, columns=["target_id"])["target_id"].to_numpy()
            for r, t in enumerate(tid.tolist()):
                assert t not in where, f"TARGETID {t} appears twice in the dataset"
                where[t] = (fi, g, r)
    return files, where


def read_rows(files, keys):
    """Return the dataset columns of (file, row group, row) keys, grouped by row group."""
    groups = defaultdict(list)
    for k in keys:
        groups[k[:2]].append(k)
    out = {}
    for (fi, g), ks in groups.items():
        t = pq.ParquetFile(files[fi]).read_row_group(g, columns=list(COLUMNS))
        n = t.num_rows
        cols = {}
        for c in COLUMNS:
            a = t[c].combine_chunks()
            cols[c] = a.values.to_numpy().reshape(n, -1) if c in ("spec", "w") else a.to_numpy()
        for k in ks:
            out[k] = {c: cols[c][k[2]] for c in COLUMNS}
    return out


def main():
    """Compare every chunk row with its dataset row and write check.txt."""
    OUT.mkdir(parents=True, exist_ok=True)
    files, where = locate()
    paths = sorted(CHUNKS.glob("DESIchunk*.pkl"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    n_rows, missing, differ = 0, [], defaultdict(int)
    for p in paths:
        with open(p, "rb") as f:
            parts = [np.asarray(x) for x in pickle.load(f)]
        chunk = dict(zip(COLUMNS, parts, strict=True))
        tids = chunk["target_id"].astype(np.int64)
        keys = [where.get(int(t)) for t in tids]
        missing += [int(t) for t, k in zip(tids, keys, strict=True) if k is None]
        rows = read_rows(files, [k for k in keys if k is not None])
        for i, k in enumerate(keys):
            if k is None:
                continue
            for c in COLUMNS:
                if not np.array_equal(rows[k][c], chunk[c][i], equal_nan=c not in ("target_id",)):
                    differ[c] += 1
        n_rows += len(tids)
    lines = [
        f"chunks {len(paths)}, rows {n_rows}, dataset rows {len(where)}",
        f"rows missing from the dataset: {len(missing)}",
        f"rows differing, per column: {dict(differ) or 'none'}",
        "identical" if not missing and not differ and n_rows == len(where) else "NOT identical",
    ]
    text = "\n".join(lines)
    (OUT / "check.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
