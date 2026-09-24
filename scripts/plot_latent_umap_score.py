"""Plot a 2D UMAP or PCA projection of DESI latents coloured by flow log p.

Flow outliers are circled in red and the figure is saved as a PNG. Run it as
``uv run python scripts/plot_latent_umap_score.py``.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
import umap
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hubersed.io.latents import load_latents
from hubersed.paths import PATHS

DATA, RES = PATHS["DATA"], PATHS["RESULTS"]


def main():
    """Join flow scores to latents by TARGETID, project to 2D and save the figure.

    Latents are standardised with the scaler stored in ``flow_nsf_10latent.pt`` under the
    results directory. Points with log p at or below the stored threshold are circled.

    Raises
    ------
    SystemExit
        If the output file exists and ``--force`` is not given.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--latents", type=Path, default=DATA / "latents" / "spender_spec_10latent_snr3.h5"
    )
    ap.add_argument("--scores", type=Path, default=RES / "desi_outliers_flow_nsf_10latent_snr3.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proj", default="umap", choices=["umap", "pca"])
    ap.add_argument("--n_neighbors", type=int, default=15)
    ap.add_argument("--min_dist", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=RES / "latent_umap_logp_10latent.png")
    ap.add_argument("-f", "--force", action="store_true")
    args = ap.parse_args()
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists, pass --force to overwrite")

    lat, tid, _ = load_latents(args.latents)
    scores = torch.load(args.scores, weights_only=False)
    lp, score_tid, thr = scores["log_p_desi"], scores["desi_target_ids"], scores["threshold"]

    # Explicit TARGETID join, since the latent h5 and score .pt may differ in order and length.
    tid_to_row = {int(t): i for i, t in enumerate(tid)}
    keep = np.array([t in tid_to_row for t in score_tid])
    rows = np.array([tid_to_row[int(t)] for t in score_tid[keep]])
    lat_m, lp_m = lat[rows], lp[keep]
    print(f"matched {keep.sum()}/{len(score_tid)} scored TARGETIDs to latent rows")

    # standardize with the SAME scaler the flow was trained/scored with
    flow_ckpt = torch.load(RES / "flow_nsf_10latent.pt", weights_only=False)
    lat_s = (lat_m - flow_ckpt["scaler_mean"]) / flow_ckpt["scaler_scale"]

    if args.proj == "umap":
        emb = umap.UMAP(
            n_components=2,
            random_state=args.seed,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        ).fit_transform(lat_s)
        xlab, ylab = "UMAP 1", "UMAP 2"
    else:
        emb = PCA(n_components=2, random_state=args.seed).fit_transform(lat_s)
        xlab, ylab = "PC 1", "PC 2"

    out_mask = lp_m <= thr
    # robust vmin/vmax so a few extreme-log-p points don't wash out the color scale
    vmin, vmax = np.percentile(lp_m, [1, 99])
    order = np.argsort(lp_m)[::-1]  # plot high-density (typical) points first, tail on top

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        emb[order, 0],
        emb[order, 1],
        c=lp_m[order],
        cmap="cividis",
        vmin=vmin,
        vmax=vmax,
        s=3,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        emb[out_mask, 0],
        emb[out_mask, 1],
        facecolors="none",
        edgecolors="firebrick",
        linewidths=0.4,
        s=6,
        alpha=0.5,
        label=f"flow outliers (n={out_mask.sum()})",
        rasterized=True,
    )
    fig.colorbar(sc, ax=ax, label="flow log p")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f"DESI DR1 BGS SV3, 10D SPENDER latent, n={len(lp_m)} ({args.proj})")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=250)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
