"""Are the 58 emission_line outliers in a sparse tail of the v2 mocks or outside what they cover?

QUESTION: (A) how far is each of the 58 from the v2 mocks in each flow's latent space, against
held-out mocks? (B) How many v2 mocks fall inside each outlier's 2 sigma EW error ellipse?
HYPOTHESIS: none registered. (A) far when the mean distance to the 10 nearest training mocks
exceeds the held-out p99.9 in every flagging flow, else tail-adjacent. (B) 0 mocks outside, 1-50
tail, > 50 covered.
INPUTS: data/latents/mocks_v2/prospector_noise_spec_cuev2_<tag>latent_snr3.h5,
data/latents/spender_spec_<tag>latent_snr3.h5, results/2026-11-09_mocks_v2_outliers/flows,
results/2026-11-14_emline_outlier_coverage/{ew_desi.npz, ew_mocks.npz} (lines.py).
SEED: none (the split uses the flows' seed 0).
COMMAND: ``uv run --no-sync python experiments/2026-11-14_emline_outlier_coverage/analysis.py``.
RESULT: 2026-11-14, coverage.txt. (A) 56 tail-adjacent, 2 far. (B) 21 outside, 17 tail, 20 covered.
Outside and tail ones fail on the line combination, not on any single line.
Assumptions: raw EWs, DESI-error 2 sigma ellipse, 500k mocks in up to 6 dimensions, no z matching.
FIGURES: none (coverage.txt, coverage.csv).
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
from sklearn.neighbors import NearestNeighbors

from hubersed.io.latents import load_latents
from hubersed.paths import PATHS

RES, DATA = PATHS["RESULTS"], PATHS["DATA"]
OUT = RES / "2026-11-14_emline_outlier_coverage"
FLOWS = RES / "2026-11-09_mocks_v2_outliers" / "flows"
TAGS = ("6", "10", "15", "cont10", "cont15")
LINES = ("oii", "hb", "oiii", "ha", "nii", "sii")
K = 10


def part_a(ids):
    """Return per tag the log p, flag, kNN distance and its held-out mock percentile of each id."""
    rows = {}
    for tag in TAGS:
        mock, _, _ = load_latents(
            DATA / "latents" / "mocks_v2" / f"prospector_noise_spec_cuev2_{tag}latent_snr3.h5"
        )
        desi, dtid, _ = load_latents(DATA / "latents" / f"spender_spec_{tag}latent_snr3.h5")
        ck = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu")
        res = torch.load(FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt", weights_only=False)
        sc = lambda x, mu=ck["scaler_mean"], sd=ck["scaler_scale"]: (x - mu) / sd  # noqa: E731
        perm = np.random.default_rng(0).permutation(mock.shape[0])
        nval = mock.shape[0] // 10
        val, tr = perm[:nval], perm[nval:]
        nn = NearestNeighbors(n_neighbors=K).fit(sc(mock[tr]))
        ref = nn.kneighbors(sc(mock[val]))[0].mean(1)
        pos = {int(t): i for i, t in enumerate(dtid)}
        have = np.array([int(t) in pos for t in ids])
        d = np.full(ids.size, np.nan)
        d[have] = nn.kneighbors(sc(desi[[pos[int(t)] for t in ids[have]]]))[0].mean(1)
        lpd = dict(zip(np.asarray(res["desi_target_ids"]).tolist(), np.asarray(res["log_p_desi"], float),
                       strict=True))  # fmt: skip
        lp = np.array([lpd.get(int(t), np.nan) for t in ids])
        rows[tag] = {
            "logp": lp,
            "flag": lp <= res["threshold"],
            "knn": d,
            "knn_pct": np.array([np.mean(ref < x) * 100 if np.isfinite(x) else np.nan for x in d]),
            "ref": np.percentile(ref, [50, 99, 99.9]),
        }
        print(f"A {tag}: done", flush=True)
    return rows


def part_b(e, m):
    """Return the number of mocks inside each outlier's 2 sigma ellipse and the lines used."""
    ok_m = ~m["failed"].astype(bool)
    counts, used, best = [], [], []
    for i in range(e["id"].size):
        ls = [
            k
            for k in LINES
            if np.isfinite(e[k][i]) and np.isfinite(e[k + "_err"][i]) and e[k + "_err"][i] > 0
        ]
        if not ls:
            counts.append(-1)
            used.append("")
            best.append(np.nan)
            continue
        sel = ok_m.copy()
        for k in ls:
            sel &= np.isfinite(m[k])
        c2 = sum(((m[k][sel] - e[k][i]) / e[k + "_err"][i]) ** 2 for k in ls)
        counts.append(int(np.sum(c2 < chi2.ppf(0.9545, len(ls)))))
        used.append(",".join(ls))
        best.append(float(np.min(c2)) if c2.size else np.nan)
    return np.array(counts), np.array(used), np.array(best)


def main():
    """Write the per-outlier table and the summary."""
    e = dict(np.load(OUT / "ew_desi.npz"))
    m = dict(np.load(OUT / "ew_mocks.npz"))
    ids = e["id"].astype(np.int64)
    a = part_a(ids)
    n_in, used, best = part_b(e, m)
    tab = pd.DataFrame({"nn": e["nn"], "target_id": ids, "branch": e["branch"], "z": e["z"]})
    for k in LINES:
        tab[f"ew_{k}"], tab[f"ew_{k}_err"] = e[k], e[k + "_err"]
    far = np.ones(ids.size, bool)
    anyflag = np.zeros(ids.size, bool)
    for tag, r in a.items():
        tab[f"logp_{tag}"], tab[f"flag_{tag}"] = r["logp"], r["flag"]
        tab[f"knn_{tag}"], tab[f"knn_pct_{tag}"] = r["knn"], r["knn_pct"]
        beyond = r["knn"] > r["ref"][2]
        far &= np.where(r["flag"], beyond, True)
        anyflag |= r["flag"]
    tab["a_class"] = np.where(~anyflag, "not flagged", np.where(far, "far", "tail-adjacent"))
    tab["b_mocks_in_2sigma"], tab["b_lines"], tab["b_min_chi2"] = n_in, used, best
    tab["b_class"] = np.select(
        [n_in < 0, n_in == 0, n_in <= 50], ["no lines", "outside", "tail"], "covered"
    )
    tab.to_csv(OUT / "coverage.csv", index=False)
    lines = [f"{ids.size} emission_line outliers"]
    lines.append("A: held-out mock kNN distance p50/p99/p99.9 per flow: " + "; ".join(
        f"{t} {' '.join(f'{v:.2f}' for v in r['ref'])}" for t, r in a.items()))  # fmt: skip
    for t, r in a.items():
        f = r["flag"]
        lines.append(f"  {t}: flags {f.sum()}; their kNN percentile among held-out mocks median "
                     f"{np.nanmedian(r['knn_pct'][f]) if f.any() else np.nan:.2f}, beyond p99.9 "
                     f"{np.sum(r['knn'][f] > r['ref'][2])}")  # fmt: skip
    lines.append("A class: " + ", ".join(f"{k} {v}" for k, v in tab.a_class.value_counts().items()))
    lines.append("B class: " + ", ".join(f"{k} {v}" for k, v in tab.b_class.value_counts().items()))
    lines.append("A x B:\n" + pd.crosstab(tab.a_class, tab.b_class, margins=True).to_string())
    lines.append(
        f"B mocks in 2 sigma, 10/50/90: {' '.join(f'{v:.0f}' for v in np.percentile(n_in[n_in >= 0], [10, 50, 90]))}"
    )
    text = "\n".join(lines)
    (OUT / "coverage.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
