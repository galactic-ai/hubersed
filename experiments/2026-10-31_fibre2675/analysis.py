"""Are galaxies observed on fibre 2675 (petal 5) flagged by the continuum flows more often (#17)?

QUESTION: 5 of the 36 follow-up outliers sit on fibre 2675 (2026-10-31_quiescent_qa, fibres_all.txt).
Is the outlier rate, or the log p distribution, of all sample galaxies on that fibre different from
other fibres?
HYPOTHESIS: none registered.
INPUTS: iron tiles-iron.csv and the tile-qa FIBERQA (TARGETID, FIBER) of every sv3/bright tile, cached
in data/iron_tileqa/; log p and thresholds of results/wide_flow_corrected/
desi_outliers_flow_nsf_cont{10,15}latent_snr3.pt (seed 0), as 2026-10-31_logp_vs_colour/analysis.py.
A galaxy counts for a fibre if any of its sv3/bright tiles put it there.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_fibre2675/analysis.py``.
RESULT: fibre 2675 holds 48 sample galaxies, 7 both-flow outliers (14.6% against 0.45% overall);
binomial P 2.2e-9, 9.9e-6 after 4452 fibres; its median cont10 log p is -8.06 against -4.57
(Mann-Whitney p 1.8e-7), so the whole fibre scores low, not only its outliers. Neighbour 2676: 2/45,
median -6.81; 2666: 1/46, -4.70. Rank 2 of 4452 fibres (rank 1 is 2 of 10).
Assumptions: fibre 2675 was picked after seeing the outliers (look-elsewhere only through the 4452
factor); seed-0 flows; a galaxy counts for every fibre it was observed on.
FIGURES: results/2026-10-31_fibre2675/ fibre_rates.txt, fibre_rates.png.
"""

# %%
import csv
import io
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

OUT = PATHS["RESULTS"] / "2026-10-31_fibre2675"
CACHE = PATHS["DATA"] / "iron_tileqa"
IRON = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron"
TAGS = ("cont10", "cont15")
WATCH = (2666, 2675, 2676)


def sv3_bright_tiles():
    """Return (TILEID, LASTNIGHT) of every sv3/bright tile in tiles-iron.csv."""
    path = PATHS["DATA"] / "iron_tiles" / "tiles-iron.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{IRON}/tiles-iron.csv", path)
    rows = csv.DictReader(io.StringIO(path.read_text()))
    return [(int(r["TILEID"]), int(r["LASTNIGHT"])) for r in rows
            if r["SURVEY"] == "sv3" and r["PROGRAM"] == "bright"]  # fmt: skip


def fibre_map():
    """Return TARGETID and FIBER arrays over all sv3/bright tile-QA files (one row per tile)."""
    tids, fibs = [], []
    for tile, night in sv3_bright_tiles():
        name = f"tile-qa-{tile}-thru{night}.fits"
        path = CACHE / str(tile) / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{IRON}/tiles/cumulative/{tile}/{night}/{name}", path)
        q = fits.getdata(path, "FIBERQA")
        tids.append(np.asarray(q["TARGETID"], np.int64))
        fibs.append(np.asarray(q["FIBER"], int))
    return np.concatenate(tids), np.concatenate(fibs)


def flows():
    """Return sorted TARGETIDs, log p per flow and the both-flow outlier mask."""
    lp, out = {}, None
    for tag in TAGS:
        o = torch.load(PATHS["RESULTS"] / "wide_flow_corrected" / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt",
                       weights_only=False, map_location="cpu")  # fmt: skip
        tid = np.asarray(o["desi_target_ids"], np.int64)
        k = np.argsort(tid)
        lp[tag] = np.asarray(o["log_p_desi"], float)[k]
        flag = lp[tag] < float(o["threshold"])
        out = flag if out is None else out & flag
        ids = tid[k]
    return ids, lp, out


def main():
    """Rate of both-flow outliers per fibre, and fibre 2675 against the rest."""
    use_apj_style()
    OUT.mkdir(parents=True, exist_ok=True)
    ids, lp, out = flows()
    qt, qf = fibre_map()
    keep = np.isin(qt, ids)
    qt, qf = qt[keep], qf[keep]
    pos = np.searchsorted(ids, qt)
    n = np.bincount(qf, minlength=5000)  # sample galaxies per fibre (tile rows)
    k = np.bincount(qf, weights=out[pos].astype(float), minlength=5000)
    p0 = out.mean()
    lines = [f"# {ids.size} scored galaxies, {out.sum()} both-flow outliers (rate {p0:.4f}); "
             f"{qt.size} sample tile rows on {np.unique(qt).size} galaxies"]  # fmt: skip
    for f in WATCH:
        m = np.unique(qt[qf == f])
        r = out[np.searchsorted(ids, m)]
        lps = lp["cont10"][np.searchsorted(ids, m)]
        lines.append(f"fibre {f}: {m.size} galaxies, {r.sum()} outliers (rate {r.mean():.3f}); "
                     f"median cont10 log p {np.median(lps):.2f} (all galaxies {np.median(lp['cont10']):.2f})")  # fmt: skip
    ok = n >= 10
    rate = np.where(ok, k / np.maximum(n, 1), np.nan)
    order = np.argsort(-np.nan_to_num(rate, nan=-1))[:10]
    lines.append("highest outlier rates (fibres with >= 10 galaxies): "
                 + ", ".join(f"{f} {k[f]:.0f}/{n[f]}" for f in order))  # fmt: skip
    rank = int(np.sum(np.nan_to_num(rate, nan=-1) >= rate[2675]))
    lines.append(f"fibre 2675 rate rank {rank} of {ok.sum()} fibres with >= 10 galaxies")
    from scipy.stats import binom, mannwhitneyu

    m = np.unique(qt[qf == 2675])
    pb = float(binom.sf(int(out[np.searchsorted(ids, m)].sum()) - 1, m.size, p0))
    u = mannwhitneyu(lp["cont10"][np.searchsorted(ids, m)], lp["cont10"], alternative="less")
    lines.append(
        f"fibre 2675: binomial P(>= observed outliers) {pb:.2e}, times {ok.sum()} fibres "
        f"{min(1.0, pb * ok.sum()):.2e}; Mann-Whitney cont10 log p lower, p {u.pvalue:.2e}"
    )
    (OUT / "fibre_rates.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(np.arange(5000)[ok], rate[ok], ".", ms=2, color="0.5")
    for f in WATCH:
        ax.plot(f, rate[f], "o", ms=6, color="#b2182b")
    ax.axhline(p0, color="#2166ac", lw=0.8, label="sample rate")
    ax.set_xlabel("FIBER")
    ax.set_ylabel("both-flow outlier fraction")
    ax.legend(frameon=False)
    fig.savefig(OUT / "fibre_rates.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fibre_rates.png'}")


if __name__ == "__main__":
    main()
