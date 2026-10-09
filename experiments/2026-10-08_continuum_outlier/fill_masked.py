"""Rescore continuum outlier candidates after filling the pixels the training loss ignores.

The spender encoder sees the whole stored spectrum, while the loss gives zero weight to masked
pixels, where the flux can be zero or a sky residual. Here each zero-weight pixel is replaced by
a linear interpolation of its weighted neighbours, the spectrum is encoded again, and both cont
flows score it. The controls are the unflagged analogues of screen_candidates.py.

Run from the repository root with
``uv run python experiments/2026-10-08_continuum_outlier/fill_masked.py``.
"""

import numpy as np
import pandas as pd
from screen_candidates import load_chunk_rows, load_scorers, read_candidates, score

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-08_continuum_outlier" / "screen"


def fill(spec, w):
    """Replace zero-weight pixels by linear interpolation of the weighted ones."""
    good = w > 0
    x = np.arange(spec.size)
    return np.where(good, spec, np.interp(x, x[good], spec[good]))


def main():
    cand, _ = read_candidates()
    ctrl = sorted(set(pd.read_csv(OUT / "control_zones.csv")["control"].astype(int)) - set(cand))
    rows = load_chunk_rows(cand + ctrl)
    tids = [t for t in cand + ctrl if t in rows]
    raw = np.array([rows[t][0] for t in tids])
    filled = np.array([fill(rows[t][0], rows[t][1]) for t in tids])
    table = {"targetid": tids, "candidate": [t in cand for t in tids]}
    for name, s in load_scorers().items():
        table[f"lp_raw_{name}"] = score(s, raw)
        table[f"lp_fill_{name}"] = score(s, filled)
        table[f"thr_{name}"] = s["thr"]
    df = pd.DataFrame(table)
    df.to_csv(OUT / "fill_masked.csv", index=False)
    for name in ["cont10", "cont15"]:
        d = df[f"lp_fill_{name}"] - df[f"lp_raw_{name}"]
        c = ~df["candidate"]
        print(
            f"{name}: controls median shift {d[c].median():+.2f} (16-84 {d[c].quantile(0.16):+.2f} "
            f"{d[c].quantile(0.84):+.2f}), controls flagged after fill "
            f"{(df.loc[c, f'lp_fill_{name}'] <= df[f'thr_{name}'][0]).sum()} of {c.sum()}"
        )
    summary = pd.read_csv(OUT / "screen_summary.csv")[["targetid", "call", "sn_per_obsA"]]
    cd = df[df["candidate"]].merge(summary, on="targetid")
    for name in ["cont10", "cont15"]:
        cd[f"still_{name}"] = cd[f"lp_fill_{name}"] <= cd[f"thr_{name}"]
    cols = ["targetid", "call", "sn_per_obsA"]
    cols += [f"{k}_{n}" for n in ["cont10", "cont15"] for k in ["lp_raw", "lp_fill", "still"]]
    print(
        cd[cols]
        .sort_values(["call", "sn_per_obsA"], ascending=[True, False])
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
