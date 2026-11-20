#!/usr/bin/env python
"""Convert spender pickle chunks into a spender Parquet dataset without changing the data.

Each chunk holds (spec, w, z, target_id, norm, zerr). Chunks are read in the numeric order of the
integer after the last underscore in their file name. Splits are assigned by spender from the
TARGETIDs. Run it as ``uv run python scripts/chunks_to_parquet.py INDIR OUTDIR``.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from spender.data.dataset import get_features, write_dataset

COLUMNS = ("spec", "w", "z", "target_id", "norm", "zerr")


def chunk_index(path):
    """Return the integer after the last underscore of a chunk file name."""
    return int(Path(path).stem.rsplit("_", 1)[1])


def read_chunks(paths):
    """Yield each chunk as a dict of numpy arrays keyed by column name.

    Parameters
    ----------
    paths : list of pathlib.Path
        Chunk pickles, read in the given order.

    Yields
    ------
    dict
        Arrays for ``spec``, ``w``, ``z``, ``target_id``, ``norm`` and ``zerr``.
    """
    for p in paths:
        with open(p, "rb") as f:
            parts = pickle.load(f)
        batch = {c: np.asarray(x) for c, x in zip(COLUMNS, parts, strict=True)}
        print(f"{p.name}: {len(batch['target_id'])} spectra", flush=True)
        yield batch


def main(args):
    """Write every chunk of ``args.indir`` matching ``args.pattern`` to ``args.outdir``."""
    paths = sorted(Path(args.indir).glob(args.pattern), key=chunk_index)
    if not paths:
        raise FileNotFoundError(f"no files match {args.pattern} in {args.indir}")
    with open(paths[0], "rb") as f:
        n_wave = np.asarray(pickle.load(f)[0]).shape[1]
    features = get_features(n_wave, ["target_id"])
    counts = write_dataset(args.outdir, read_chunks(paths), features, ["target_id"])
    print(f"{len(paths)} chunks, {n_wave} pixels per spectrum, split counts {counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert spender pickle chunks to Parquet.")
    ap.add_argument("indir", help="directory with the pickle chunks")
    ap.add_argument("outdir", help="directory of the Parquet dataset to write")
    ap.add_argument("--pattern", default="DESI*.pkl", help="glob for the chunk files")
    main(ap.parse_args())
