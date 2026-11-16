"""Do DESI galaxies and v2 mocks mix in each flow's latent space, or sit apart?

QUESTION: in each flow's standardised latent space, what fraction of the k nearest neighbours of a
DESI galaxy are mocks, and of a mock are DESI, with and without S/N and z matching?
HYPOTHESIS: none registered. With equal numbers and full mixing the other-set fraction is about 0.5;
fully separate sets give 0.
INPUTS: data/latents/mocks_v2/prospector_noise_spec_cuev2_<tag>latent_snr3.h5,
data/latents/spender_spec_<tag>latent_snr3.h5, data/dataset_z_snr.npz,
results/2026-11-09_mocks_v2_outliers/flows/flow_nsf_<tag>latent.pt,
results/2026-11-14_emline_outlier_coverage/coverage.csv.
SEED: 0.
COMMAND: ``uv run --no-sync python experiments/2026-11-14_latent_domain_gap/analysis.py``.
RESULT: 2026-11-14, mixing.txt. DESI and mocks barely mix in any flow. Mean other-set fraction
0.003-0.016 for DESI and 0.008-0.026 for mocks against 0.5, unchanged by S/N and z matching.
Assumptions: Euclidean k = 15 in the flow-standardised space; 20,000 of each; cause not tested.
FIGURES: none (mixing.txt).
"""

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from hubersed.io.latents import load_latents
from hubersed.paths import PATHS

RES, DATA = PATHS["RESULTS"], PATHS["DATA"]
OUT = RES / "2026-11-14_latent_domain_gap"
FLOWS = RES / "2026-11-09_mocks_v2_outliers" / "flows"
TAGS = ("6", "10", "15", "cont10", "cont15")
N, K = 20000, 15
SNR_EDGES = (3, 4, 5, 6, 8, 10, 15, 25, np.inf)
Z_EDGES = np.round(np.arange(0.0, 0.65, 0.05), 2)


def matched(rng, m_cell, d_cell, n):
    """Return mock indices drawn to the cell counts of n random DESI cells."""
    want = pd.Series(d_cell).value_counts()
    out = []
    for c, k in want.items():
        pool = np.flatnonzero(m_cell == c)
        if pool.size:
            out.append(rng.choice(pool, k, replace=pool.size < k))
    idx = np.concatenate(out)
    return idx[:n] if idx.size >= n else idx


def mixing(x_mock, x_desi, x_query):
    """Return other-set neighbour fractions of DESI, of mocks, and the mock fraction of queries."""
    pool = np.vstack([x_mock, x_desi])
    is_mock = np.r_[np.ones(len(x_mock), bool), np.zeros(len(x_desi), bool)]
    nn = NearestNeighbors(n_neighbors=K + 1).fit(pool)
    idx = nn.kneighbors(pool)[1][:, 1:]
    frac_mock = is_mock[idx].mean(1)
    q = is_mock[nn.kneighbors(x_query, n_neighbors=K)[1]].mean(1)
    return frac_mock[~is_mock], 1 - frac_mock[is_mock], q


def pct(x):
    """Return the mean and the 10/50/90 percentiles as text."""
    return f"mean {x.mean():.3f}, 10/50/90 {' '.join(f'{v:.2f}' for v in np.percentile(x, [10, 50, 90]))}"


def main():
    """Write the mixing fractions per flow."""
    OUT.mkdir(parents=True, exist_ok=True)
    zs = np.load(DATA / "dataset_z_snr.npz")
    snr_of = dict(zip(zs["tid"].tolist(), zs["snr"].tolist(), strict=True))
    ids58 = pd.read_csv(
        RES / "2026-11-14_emline_outlier_coverage" / "coverage.csv"
    ).target_id.to_numpy(np.int64)
    lines = [
        f"N {N} DESI and {N} mocks, k {K}; values are fractions of neighbours from the other set"
    ]
    for tag in TAGS:
        path = DATA / "latents" / "mocks_v2" / f"prospector_noise_spec_cuev2_{tag}latent_snr3.h5"
        mock, _, _ = load_latents(path)
        with h5py.File(path, "r") as h:
            m_snr, m_z = h["snr_med"][:], h["zs"][:]
        desi, dtid, _ = load_latents(DATA / "latents" / f"spender_spec_{tag}latent_snr3.h5")
        with h5py.File(DATA / "latents" / f"spender_spec_{tag}latent_snr3.h5", "r") as h:
            d_z = h["zs"][:]
        d_snr = np.array([snr_of.get(int(t), np.nan) for t in dtid])
        ck = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu")
        mu, sd = ck["scaler_mean"], ck["scaler_scale"]
        perm = np.random.default_rng(0).permutation(mock.shape[0])
        tr = perm[mock.shape[0] // 10 :]
        rng = np.random.default_rng(0)
        di = rng.choice(np.flatnonzero(np.isfinite(d_snr)), N, replace=False)
        cell = lambda s, z: np.digitize(s, SNR_EDGES) * 100 + np.digitize(z, Z_EDGES)  # noqa: E731
        m_cell, d_cell = cell(m_snr[tr], m_z[tr]), cell(d_snr[di], d_z[di])
        pos = {int(t): i for i, t in enumerate(dtid)}
        q = (desi[[pos[int(t)] for t in ids58 if int(t) in pos]] - mu) / sd
        xd = (desi[di] - mu) / sd
        lines.append(f"\n{tag}latent")
        for lab, mi in (("unmatched", rng.choice(tr.size, N, replace=False)),
                        ("matched S/N and z", matched(rng, m_cell, d_cell, N))):  # fmt: skip
            xm = (mock[tr[mi]] - mu) / sd
            fd, fm, fq = mixing(xm, xd, q)
            lines.append(f"  {lab} ({len(xm)} mocks): DESI {pct(fd)}; mocks {pct(fm)}")
            lines.append(f"    58 outliers, mock fraction of neighbours: {pct(fq)}")
        print(f"{tag} done", flush=True)
    text = "\n".join(lines)
    (OUT / "mixing.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
