"""Figure of where the 58 emission_line outliers sit in latent UMAP, kNN distances and line ratios.

The top left panel is a UMAP (n_neighbors 15, min_dist 0.1, seed 0, as
scripts/plot_latent_umap_score.py) of the standardised flow-15 latents of 20,000 random training
mocks, 20,000 random DESI galaxies and the 58, fitted on all of them, with the 58 coloured by
analysis.py B class. UMAP does not keep distances. The top right panel shows log
EW([NII]6585)/EW(Halpha) against log EW([OIII]5008)/EW(Hbeta), raw EWs of lines.py, only where all
four are above 1 A, as mock density (log counts, hexbin) and the 58 by class. The bottom panels
show, per flow, the mean distance to the 10 nearest training mocks of the held-out mocks
(histogram, density) and of the 58 (ticks, flagged by that flow in colour, others grey), with a
dashed line at the held-out p99.9. Splits and scalers as analysis.py.
INPUTS: as analysis.py, plus results/2026-11-14_emline_outlier_coverage/coverage.csv.
SEED: 0 (subsamples and UMAP).
COMMAND: ``uv run --no-sync python experiments/2026-11-14_emline_outlier_coverage/figure.py``.
OUTPUT: results/2026-11-14_emline_outlier_coverage/coverage_figure.png.
"""

import matplotlib
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hubersed.io.latents import load_latents
from hubersed.paths import PATHS

RES, DATA = PATHS["RESULTS"], PATHS["DATA"]
OUT = RES / "2026-11-14_emline_outlier_coverage"
FLOWS = RES / "2026-11-09_mocks_v2_outliers" / "flows"
TAGS = ("6", "10", "15", "cont10", "cont15")
COLOURS = {"outside": "firebrick", "tail": "darkorange", "covered": "royalblue"}
NSUB = 20000
EWMIN = 1.0


def latents(tag):
    """Return standardised training mocks, held-out mocks, DESI latents and DESI ids of ``tag``."""
    mock, _, _ = load_latents(
        DATA / "latents" / "mocks_v2" / f"prospector_noise_spec_cuev2_{tag}latent_snr3.h5"
    )
    desi, dtid, _ = load_latents(DATA / "latents" / f"spender_spec_{tag}latent_snr3.h5")
    ck = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu")
    mu, sd = ck["scaler_mean"], ck["scaler_scale"]
    perm = np.random.default_rng(0).permutation(mock.shape[0])
    nval = mock.shape[0] // 10
    return (
        (mock[perm[nval:]] - mu) / sd,
        (mock[perm[:nval]] - mu) / sd,
        (desi - mu) / sd,
        np.asarray(dtid),
    )


def main():
    """Draw and save the figure."""
    tab = pd.read_csv(OUT / "coverage.csv")
    ids = tab.target_id.to_numpy(np.int64)
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 5, height_ratios=(1.6, 1))

    tr, _, desi, dtid = latents("15")
    pos = {int(t): i for i, t in enumerate(dtid)}
    out = desi[[pos[int(t)] for t in ids]]
    sm, sd_ = (
        tr[rng.choice(len(tr), NSUB, replace=False)],
        desi[rng.choice(len(desi), NSUB, replace=False)],
    )
    emb = umap.UMAP(n_components=2, random_state=0, n_neighbors=15, min_dist=0.1).fit_transform(
        np.vstack([sm, sd_, out])
    )
    ax = fig.add_subplot(gs[0, :3])
    ax.scatter(
        *emb[:NSUB].T, s=2, c="0.6", alpha=0.3, linewidths=0, rasterized=True, label="v2 mocks"
    )
    ax.scatter(*emb[NSUB : 2 * NSUB].T, s=2, c="seagreen", alpha=0.25, linewidths=0, rasterized=True,
               label="DESI")  # fmt: skip
    eo = emb[2 * NSUB :]
    for cls, col in COLOURS.items():
        s = (tab.b_class == cls).to_numpy()
        ax.scatter(
            *eo[s].T, s=28, c=col, edgecolors="k", linewidths=0.4, label=f"{cls} ({s.sum()})"
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"flow-15 latent UMAP: {NSUB} mocks, {NSUB} DESI, 58 emission_line outliers")
    ax.legend(markerscale=2, fontsize=8)

    m = np.load(OUT / "ew_mocks.npz")
    ok = ~m["failed"].astype(bool)
    for k in ("ha", "nii", "hb", "oiii"):
        ok &= np.isfinite(m[k]) & (m[k] > EWMIN)
    ax = fig.add_subplot(gs[0, 3:])
    x, y = np.log10(m["nii"][ok] / m["ha"][ok]), np.log10(m["oiii"][ok] / m["hb"][ok])
    hb = ax.hexbin(x, y, gridsize=70, bins="log", cmap="Greys", mincnt=1)
    fig.colorbar(hb, ax=ax, label="mocks per cell (log)")
    for cls, col in COLOURS.items():
        s = tab[(tab.b_class == cls)]
        g = (s[["ew_ha", "ew_nii", "ew_hb", "ew_oiii"]] > EWMIN).all(axis=1)
        ax.scatter(np.log10(s.ew_nii[g] / s.ew_ha[g]), np.log10(s.ew_oiii[g] / s.ew_hb[g]), s=28, c=col,
                   edgecolors="k", linewidths=0.4, label=f"{cls} ({g.sum()} of {len(s)})")  # fmt: skip
    ax.set_xlabel("log EW([NII]6585) / EW(Halpha)")
    ax.set_ylabel("log EW([OIII]5008) / EW(Hbeta)")
    ax.set_title(f"raw EWs, all four > {EWMIN:.0f} A ({ok.sum()} mocks)")
    ax.legend(fontsize=8)

    for j, tag in enumerate(TAGS):
        tr, val, desi, dtid = latents(tag)
        nn = NearestNeighbors(n_neighbors=10).fit(tr)
        ref = nn.kneighbors(val)[0].mean(1)
        d, f = tab[f"knn_{tag}"].to_numpy(), tab[f"flag_{tag}"].to_numpy(bool)
        ax = fig.add_subplot(gs[1, j])
        hi = max(np.nanmax(d), np.percentile(ref, 99.99)) * 1.05
        ax.hist(ref, bins=np.linspace(0, hi, 80), density=True, color="0.6", label="held-out mocks")
        ax.axvline(np.percentile(ref, 99.9), color="k", ls="--", lw=1, label="mock p99.9")
        top = ax.get_ylim()[1]
        cols = [COLOURS[c] for c in tab.b_class]
        for xi, fi, ci in zip(d, f, cols, strict=True):
            ax.plot([xi, xi], [0, 0.25 * top], color=ci if fi else "0.75", lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("mean distance to 10 nearest mocks")
        ax.set_title(f"{tag}latent: {f.sum()} of 58 flagged")
        if j == 0:
            ax.set_ylabel("density")
            ax.legend(fontsize=7)
        print(f"{tag} done", flush=True)
    fig.tight_layout()
    fig.savefig(OUT / "coverage_figure.png", dpi=150)
    print(OUT / "coverage_figure.png")


if __name__ == "__main__":
    main()
