import sys
import json

from itertools import combinations
from pathlib import Path
from collections import Counter

import numpy as np
import torch

from hubersed.paths import PATHS

RESULTS_DIR = PATHS["RESULTS"]

def load_flagged(pt_path):
    """(target_ids, metadata) as the output"""
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
        print(f"{m['tag']}: {m['n']} flagged, c2st={m['c2st']:.3f}, threshold={m['threshold']:.3f}, mock_file={m['mock_file']}")

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
