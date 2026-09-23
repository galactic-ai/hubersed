"""Combine flow outlier files from several encoders into a core set of TARGETIDs.

A galaxy is in the core when at least k of the files flag it. Run it with
``python -m hubersed.detect.outliers [DIR]``, which reads three fixed file names from DIR
and writes ``ood_core_3nsf_em_lines.pt`` there.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from hubersed.paths import PATHS

RESULTS_DIR = PATHS["RESULTS"]


def load_flagged(pt_path):
    """Read the flagged TARGETIDs and run metadata from one outlier file.

    Parameters
    ----------
    pt_path : str or Path
        File saved with ``torch.save`` that holds ``outlier_target_ids`` and ``tag``.

    Returns
    -------
    tids : set of int
        Flagged TARGETIDs.
    meta : dict
        ``tag``, ``c2st``, ``n``, ``threshold`` and ``mock_file``. Missing ``c2st`` and
        ``threshold`` become NaN and a missing ``mock_file`` becomes "?".
    """
    d = torch.load(pt_path, weights_only=False)
    tids = set(int(x) for x in np.asarray(d["outlier_target_ids"]))
    meta = {
        "tag": d["tag"],
        "c2st": float(d.get("c2st", np.nan)),
        "n": len(tids),
        "threshold": float(d.get("threshold", np.nan)),
        "mock_file": d.get("mock_file", "?"),
    }
    return tids, meta


def core(pt_paths, k=None):
    """Find the TARGETIDs flagged in at least ``k`` of the outlier files.

    Parameters
    ----------
    pt_paths : list of str or Path
        Outlier files, one per flow.
    k : int, optional
        Minimum number of files that must flag a TARGETID. Defaults to all files.

    Returns
    -------
    core_tids : set of int
        TARGETIDs flagged in at least ``k`` files.
    metas : list of dict
        Metadata of each file from ``load_flagged``.
    counts : dict
        Number of files, ``k``, size of the union, size of the core, flagged count per tag
        and c2st per tag rounded to three decimals.
    """
    sets, metas = [], []
    for p in pt_paths:
        s, m = load_flagged(p)
        sets.append(s)
        metas.append(m)
    N = len(sets)

    # if number of sets to intersect is not specified, use all sets
    if k is None:
        k = N

    votes = Counter()
    for s in sets:
        votes.update(s)

    core_tids = {t for t, v in votes.items() if v >= k}

    union = set().union(*sets)
    counts = {
        "N_flows": N,
        "k": k,
        "union": len(union),
        f"core_{k}of{N}": len(core_tids),
        "per_encoder": {m["tag"]: m["n"] for m in metas},
        "c2st": {m["tag"]: round(m["c2st"], 3) for m in metas},
    }

    return core_tids, metas, counts


if __name__ == "__main__":
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "noised_cue_meanzero_wide_flow"
    paths = [d / f"desi_outliers_flow_nsf_{t}_snr3.pt" for t in ("6latent", "10latent", "15latent")]
    tids, metas, counts = core(paths)
    print(json.dumps(counts, indent=2))

    for m in metas:
        print(
            f"{m['tag']}: {m['n']} flagged, c2st={m['c2st']:.3f}, threshold={m['threshold']:.3f}, mock_file={m['mock_file']}"
        )

    torch.save(
        {
            "core_target_ids": torch.tensor(sorted(tids)),
            "k": 3,
            "encoders": ["6latent", "10latent", "15latent"],
            "source_dir": str(d),
            "per_encoder_c2st": counts["c2st"],
        },
        d / "ood_core_3nsf_em_lines.pt",
    )
