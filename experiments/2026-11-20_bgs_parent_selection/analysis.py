"""How the BGS Bright and Faint parent differs from the old parent and the old scored sample.

QUESTION: which old-parent and old-scored galaxies leave the new BGS parent, by which cut, and
which new ones enter?
HYPOTHESIS: none stated before the run.
INPUTS: results/2026-11-20_bgs_parent_selection/parent_bgs.npz (selection.py); the old parent
(data/dataset_z_snr.npz, 254,976 spectra); the old scored sample (results/prescreened_sample_v1.npz,
104,990); data/datasets/desi_sv3_bright_v1 (chunks_to_parquet.py, 255,592).
SEED: none.
COMMAND: ``uv run python experiments/2026-11-20_bgs_parent_selection/analysis.py``.
RESULT: 250,987 in the new parent, all in desi_sv3_bright_v1. Of the old parent 250,388 stay and
4,588 leave (2,547 not BGS Bright or Faint, 1,084 DELTACHI2, 803 z range, 147 fiber status, 7 ZERR);
599 enter, all from chunk 616. Of the old scored sample 104,904 stay and 86 leave (36 fiber status,
25 DELTACHI2, 24 not BGS Bright or Faint, 1 ZERR).
Assumptions: each drop is credited to its first failing cut in selection.py order; fiber status
is the catalog value, which differs from the coadd file for the 147.
FIGURES: none (analysis.txt).
"""

from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-20_bgs_parent_selection"
DATASET = PATHS["DATA"] / "datasets" / "desi_sv3_bright_v1"


def dataset_targetids(path):
    """Return every TARGETID of a spender Parquet dataset."""
    files = sorted((path / "data").glob("*.parquet"))
    return np.concatenate(
        [pq.read_table(f, columns=["target_id"])["target_id"].to_numpy() for f in files]
    )


def reasons(tids, cat_tid, first_fail):
    """Count the first failing catalog cut of each TARGETID, or "not in catalog"."""
    order = np.argsort(cat_tid)
    pos = np.clip(np.searchsorted(cat_tid[order], tids), 0, len(cat_tid) - 1)
    found = cat_tid[order][pos] == tids
    why = np.where(found, first_fail[order][pos], "not in catalog")
    return Counter(why.tolist())


def main():
    """Write the overlap counts and drop reasons to analysis.txt."""
    p = np.load(OUT / "parent_bgs.npz")
    new = p["targetid"]
    cat_tid, first_fail = p["catalog_targetid"], p["catalog_first_fail"]
    old = np.load(PATHS["DATA"] / "dataset_z_snr.npz")["tid"].astype(np.int64)
    scored = np.load(PATHS["RESULTS"] / "prescreened_sample_v1.npz")["keep"].astype(np.int64)
    stored = dataset_targetids(DATASET)

    lines = [
        f"new BGS parent {new.size}; in desi_sv3_bright_v1 {np.isin(new, stored).sum()}",
        "",
        f"old parent {old.size}: kept {np.isin(old, new).sum()}, dropped {(~np.isin(old, new)).sum()}",
    ]
    for why, n in reasons(old[~np.isin(old, new)], cat_tid, first_fail).most_common():
        lines.append(f"  dropped by {why}: {n}")
    added = new[~np.isin(new, old)]
    lines += [
        f"new parent not in old parent: {added.size}, of them in desi_sv3_bright_v1 "
        f"{np.isin(added, stored).sum()}",
        "",
        f"old scored {scored.size}: kept {np.isin(scored, new).sum()}, "
        f"dropped {(~np.isin(scored, new)).sum()}",
    ]
    for why, n in reasons(scored[~np.isin(scored, new)], cat_tid, first_fail).most_common():
        lines.append(f"  dropped by {why}: {n}")
    text = "\n".join(lines)
    (OUT / "analysis.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
