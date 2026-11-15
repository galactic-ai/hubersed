"""Where do the 58 emission_line outliers sit in the log p of flows 6, 10 and 15?

QUESTION: Against all DESI and the held-out mocks, how far below the flag threshold are the 58
gallery105 emission_line outliers in each full flow, and where do the 12 cont-only ones sit?
HYPOTHESIS: none registered. Plan ~/.claude/plans/logp_hist_and_effective_volume_2026-11-14.md.
The 58 were picked by these flags, so the 46 full-branch ones are below threshold by construction.
INPUTS: results/2026-11-09_mocks_v2_outliers/flows (flow_nsf_<tag>latent.pt, desi_outliers_flow_nsf_
<tag>latent_snr3.pt), the mock and DESI latent files named there, and
results/2026-11-14_emline_outlier_coverage/coverage.csv (the 58, branch).
METHOD: mock log p recomputed with the saved flow and scaler; held-out mocks are the first tenth of
the default_rng(0) permutation (get_outliers_flow.py). Checks first: recomputed log p of the 58
within 1e-3 of the saved log_p_desi (float32 rounding), and the recomputed 0.1% quantile of all
mock log p within 1e-3 of the saved threshold. Percentiles are the share of DESI or held-out mocks with log p at or below the value.
SEED: none (the split uses the flows' seed 0).
COMMAND: ``uv run --no-sync python experiments/2026-11-14_logp_emline_histogram/analysis.py``.
RESULT: 2026-11-14, logp.txt, logp.csv. Checks pass (max |diff| 3.0e-4, thresholds exact). The 46
full ones sit just below threshold, not far out: median log p -18.8/-21.3/-28.5 against threshold
-15.0/-19.1/-25.0 (flows 6/10/15), held-out mock percentile 0.08-0.11%, DESI 0.4-0.9%; lowest
-51.9 (flow 6). The 12 cont ones sit in the full-flow mock bulk (1/0/5 below threshold, mock
percentile 0.3-0.8%). DESI below threshold 1.65/0.64/1.75% against held-out mocks 0.13/0.12/0.15%:
the DESI low-log-p tail is heavier than the mock one. Assumptions: saved flows and seed-0 split;
flow log p is least reliable in its own tail.
FIGURES: logp_hist.png. One row, flows 6, 10, 15. Density histograms (log y, full range, 100 shared
bins) of all DESI and held-out mocks, saved threshold dashed, the 58 as ticks at the panel bottom
coloured by branch (full 46, cont 12).
"""

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hubersed.detect.flow_model import build_flow
from hubersed.io.latents import load_latents
from hubersed.paths import PATHS

RES, ROOT = PATHS["RESULTS"], PATHS["ROOT"]
OUT = RES / "2026-11-14_logp_emline_histogram"
FLOWS = RES / "2026-11-09_mocks_v2_outliers" / "flows"
TAGS = ("6", "10", "15")
COLOURS = {"full": "firebrick", "cont": "royalblue"}
NBINS = 100


def log_prob(g, x):
    """Return log p of raw latents x under the saved flow checkpoint g."""
    m = build_flow(g["method"], g["dim"], g["hidden"], g["num_transforms"], g["num_bins"])
    m.load_state_dict(g["state_dict"])
    m.eval()
    xs = ((x - g["scaler_mean"]) / g["scaler_scale"]).astype(np.float32)
    with torch.no_grad():
        return np.concatenate(
            [
                m.log_prob(torch.from_numpy(xs[i : i + 65536])).numpy()
                for i in range(0, len(xs), 65536)
            ]
        )


def pct(ref, x):
    """Return the percentage of ref at or below each value of x."""
    s = np.sort(ref)
    return 100 * np.searchsorted(s, x, side="right") / s.size


def main():
    """Run the checks, write logp.txt and draw logp_hist.png."""
    OUT.mkdir(parents=True, exist_ok=True)
    tab = pd.read_csv(RES / "2026-11-14_emline_outlier_coverage" / "coverage.csv")
    ids, branch = tab.target_id.to_numpy(np.int64), tab.branch.to_numpy()
    lines = [
        f"58 emission_line outliers: full {np.sum(branch == 'full')}, cont {np.sum(branch == 'cont')}"
    ]
    rows = {"target_id": ids, "branch": branch}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, tag in zip(axes, TAGS, strict=True):
        g = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu")
        r = torch.load(FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt", weights_only=False)
        mock, _, _ = load_latents(ROOT / r["mock_file"])
        desi, dtid, _ = load_latents(ROOT / r["desi_file"])
        lp_mock = log_prob(g, mock)
        val = np.random.default_rng(0).permutation(mock.shape[0])[: mock.shape[0] // 10]
        lp_val = lp_mock[val]
        lp_desi = np.asarray(r["log_p_desi"], float)
        saved = dict(zip(np.asarray(r["desi_target_ids"]).tolist(), lp_desi, strict=True))
        pos = {int(t): i for i, t in enumerate(dtid)}
        lp58 = log_prob(g, desi[[pos[int(t)] for t in ids]])
        diff = np.max(np.abs(lp58 - np.array([saved[int(t)] for t in ids])))
        q = np.quantile(lp_mock, 0.001)
        thr = float(r["threshold"])
        lines.append(f"\n{tag}latent check: 58 recomputed vs saved max |diff| {diff:.2e}; "
                     f"mock 0.1% quantile {q:.3f} vs saved threshold {thr:.3f}")  # fmt: skip
        if diff > 1e-3 or abs(q - thr) > 1e-3:
            raise SystemExit("\n".join(lines) + "\ncheck failed, stopping")
        p_d, p_m = pct(lp_desi, lp58), pct(lp_val, lp58)
        rows[f"logp_{tag}"], rows[f"desi_pct_{tag}"], rows[f"mock_pct_{tag}"] = lp58, p_d, p_m
        lines.append(
            f"  DESI {lp_desi.size}, held-out mocks {lp_val.size}, threshold {thr:.2f}; below "
            f"threshold: DESI {100 * np.mean(lp_desi <= thr):.2f}%, held-out mocks "
            f"{100 * np.mean(lp_val <= thr):.2f}%"
        )
        for b in COLOURS:
            s = branch == b
            lines.append(
                f"  {b} ({s.sum()}): below threshold {np.sum(lp58[s] <= thr)}; log p median "
                f"{np.median(lp58[s]):.1f} (min {lp58[s].min():.1f}, max {lp58[s].max():.1f}); "
                f"DESI pct median {np.median(p_d[s]):.3f}; held-out mock pct median {np.median(p_m[s]):.3f}"
            )

        lo = min(lp_desi.min(), lp_val.min(), lp58.min())
        hi = max(lp_desi.max(), lp_val.max(), lp58.max())
        bins = np.linspace(lo, hi, NBINS + 1)
        ax.hist(lp_desi, bins=bins, density=True, histtype="step", color="seagreen", lw=1.3,
                label=f"DESI ({lp_desi.size})")  # fmt: skip
        ax.hist(lp_val, bins=bins, density=True, histtype="step", color="0.4", lw=1.3,
                label=f"held-out mocks ({lp_val.size})")  # fmt: skip
        ax.axvline(thr, color="k", ls="--", lw=1, label="threshold (mock 0.1%)")
        tr = ax.get_xaxis_transform()
        for b, col in COLOURS.items():
            s = branch == b
            for i, x in enumerate(lp58[s]):
                ax.plot([x, x], [0, 0.08], color=col, lw=1, transform=tr,
                        label=f"{b} outliers ({s.sum()})" if i == 0 else None)  # fmt: skip
        ax.set_yscale("log")
        ax.set_xlabel("flow log p")
        ax.set_title(f"{tag}latent: {np.sum(lp58 <= thr)} of 58 below threshold")
        print(f"{tag} done", flush=True)
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "logp_hist.png", dpi=150)
    pd.DataFrame(rows).to_csv(OUT / "logp.csv", index=False)
    text = "\n".join(lines)
    (OUT / "logp.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
