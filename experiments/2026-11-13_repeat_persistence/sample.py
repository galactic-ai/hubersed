"""Select the B2 sample: cont-flow outliers above S/N 8 and a control matched in S/N and z.

Outliers are DESI targets flagged by cont10 or cont15 (log p at or below the run threshold) with
latent-cut median S/N > 8. The control is three times as many targets flagged by neither flow,
drawn in cells of S/N (edges 8, 9, 10, 11, 12, 13, 15, 18, 22, 30, inf) and z (0.05 bins) with the
outlier counts per cell, or all of the cell when it has fewer.
INPUTS: results/2026-11-09_mocks_v2_outliers/flows/desi_outliers_flow_nsf_cont{10,15}latent_snr3.pt;
data/dataset_z_snr.npz.
SEED: 0 (numpy default_rng).
COMMAND: ``uv run --no-sync python experiments/2026-11-13_repeat_persistence/sample.py``.
OUTPUT: results/2026-11-13_repeat_persistence/targets.npz (outlier, control).
"""

import numpy as np
import torch

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-11-13_repeat_persistence"
FLOWS = RES / "2026-11-09_mocks_v2_outliers" / "flows"
SNR_EDGES = (8, 9, 10, 11, 12, 13, 15, 18, 22, 30, np.inf)
Z_EDGES = np.round(np.arange(0.0, 0.65, 0.05), 2)
N_CONTROL = 3


def main():
    """Write the outlier and control target ids."""
    OUT.mkdir(parents=True, exist_ok=True)
    flagged, scored = set(), None
    for tag in ("cont10", "cont15"):
        o = torch.load(FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt", weights_only=False)
        tids = np.asarray(o["desi_target_ids"], np.int64)
        flagged |= set(tids[np.asarray(o["log_p_desi"]) <= o["threshold"]].tolist())
        scored = set(tids.tolist()) if scored is None else scored & set(tids.tolist())
    d = np.load(PATHS["DATA"] / "dataset_z_snr.npz")
    keep = np.isin(d["tid"], np.fromiter(scored, np.int64)) & (d["snr"] > SNR_EDGES[0])
    tid, z, snr = d["tid"][keep].astype(np.int64), d["z"][keep], d["snr"][keep]
    is_out = np.isin(tid, np.fromiter(flagged, np.int64))
    cell = np.digitize(snr, SNR_EDGES) * 100 + np.digitize(z, Z_EDGES)
    rng = np.random.default_rng(0)
    control = []
    for c in np.unique(cell[is_out]):
        pool = tid[(cell == c) & ~is_out]
        k = min(pool.size, N_CONTROL * int(np.sum(is_out & (cell == c))))
        control.append(rng.choice(pool, k, replace=False))
    control = np.sort(np.concatenate(control))
    np.savez(OUT / "targets.npz", outlier=np.sort(tid[is_out]), control=control)
    print(f"S/N > 8 scored {tid.size}; outliers {is_out.sum()}; control {control.size}")


if __name__ == "__main__":
    main()
