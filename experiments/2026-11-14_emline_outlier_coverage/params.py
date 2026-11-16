"""Do the mocks nearest the emission_line outliers in line space sit at a prior edge?

QUESTION: for each of the 58 outliers, are the parameters of its 20 nearest mocks in EW space piled
at an edge of the v2 prior (prior too narrow) or inside it (region undersampled)?
HYPOTHESIS: none registered. A parameter is at an edge for a class when the share of nearest mocks
with prior rank u < 0.02 or u > 0.98 exceeds 25% (2% expected at each edge by chance).
INPUTS: results/2026-11-14_emline_outlier_coverage/{coverage.csv, ew_desi.npz, ew_mocks.npz},
data/prospector_model/mocks_v2_500000.h5.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-14_emline_outlier_coverage/params.py``.
RESULT: 2026-11-14, params.txt. No parameter is at a prior edge in any class (largest edge share
0.08). Outside and tail classes sit in a joint corner inside the prior.
Assumptions: edges per parameter, not joint; u mixes redshifts; 20 nearest mocks by line chi2 only.
FIGURES: none (params.txt).
"""

import h5py
import numpy as np
import pandas as pd

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-14_emline_outlier_coverage"
MOCKS = PATHS["DATA"] / "prospector_model" / "mocks_v2_500000.h5"
NNEAR = 20
EDGE = 0.02
EDGE_SHARE = 0.25


def main():
    """Write the edge shares per class and parameter and the per-outlier edge parameters."""
    tab = pd.read_csv(OUT / "coverage.csv")
    e = dict(np.load(OUT / "ew_desi.npz"))
    m = dict(np.load(OUT / "ew_mocks.npz"))
    with h5py.File(MOCKS, "r") as h:
        labels = [s.decode() if isinstance(s, bytes) else str(s) for s in h.attrs["theta_labels"]]
        theta = h["priors/theta"][:]
    ok = ~m["failed"].astype(bool)
    u = np.full(theta.shape, np.nan)
    for j in range(theta.shape[1]):
        r = np.argsort(np.argsort(theta[ok, j], kind="stable"), kind="stable")
        u[ok, j] = (r + 0.5) / ok.sum()
    pos = {int(t): i for i, t in enumerate(e["id"])}
    near = {}
    for _, row in tab.iterrows():
        if not isinstance(row.b_lines, str) or not row.b_lines:
            continue
        i = pos[int(row.target_id)]
        ls = row.b_lines.split(",")
        sel = ok.copy()
        for k in ls:
            sel &= np.isfinite(m[k])
        idx = np.flatnonzero(sel)
        c2 = sum(((m[k][idx] - e[k][i]) / e[k + "_err"][i]) ** 2 for k in ls)
        near[int(row.target_id)] = idx[np.argsort(c2)[:NNEAR]]
    lines = [f"{len(near)} outliers, {NNEAR} nearest mocks each; edge = u < {EDGE} or > {1 - EDGE}"]
    per = []
    for cls in ("outside", "tail", "covered"):
        ids = tab.target_id[tab.b_class == cls].astype(int).tolist()
        allidx = np.concatenate([near[t] for t in ids if t in near])
        uu = u[allidx]
        lines.append(f"\n{cls}: {len(ids)} outliers, {allidx.size} nearest mocks")
        for j, lab in enumerate(labels):
            lo, hi = np.mean(uu[:, j] < EDGE), np.mean(uu[:, j] > 1 - EDGE)
            mark = "  <- edge" if max(lo, hi) > EDGE_SHARE else ""
            lines.append(
                f"  {lab:16} median u {np.median(uu[:, j]):.3f}  low edge {lo:.2f}  high edge {hi:.2f}{mark}"
            )
        for t in ids:
            if t not in near:
                continue
            ut = u[near[t]]
            edge = [
                lab
                for j, lab in enumerate(labels)
                if max(np.mean(ut[:, j] < EDGE), np.mean(ut[:, j] > 1 - EDGE)) >= 0.5
            ]
            per.append(
                f"  {cls:8} nn {int(tab.nn[tab.target_id == t].iloc[0]):3d} {t}: {', '.join(edge) or '-'}"
            )
    lines += ["\nper outlier, parameters with half or more of its nearest mocks at one edge:"] + per
    text = "\n".join(lines)
    (OUT / "params.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
