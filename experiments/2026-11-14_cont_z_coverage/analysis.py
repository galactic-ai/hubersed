"""Is the rise of cont flags with redshift a mock coverage effect, and why is 99 flagged?

QUESTION: (1) Are the cont flows' training mocks past both cont thresholds as often at z 0.45-0.60 as
DESI is (thin flow density there), or do high-z mocks score normally while DESI does not? (2) Is
gallery105 99 (z 0.015) flagged because mocks are sparse near its z, or for a reason of its own?
HYPOTHESIS: none registered. Plan ~/.claude/plans/z_coverage_and_99_2026-11-14.md. Held-out
mocks at z 0.45-0.60 flagged at >= 5x their overall 0.1% means thin coverage; within 2x while DESI is
20x+ means a DESI-mock difference at high z. For 99, under 20 mocks within |z - 0.015| <= 0.01, or
its kNN distance above the 99.9 percentile of same-z held-out mocks while those mocks are not
flagged, means coverage; else specific to 99.
INPUTS: results/2026-11-09_mocks_v2_outliers/flows (flow_nsf_cont{10,15}latent.pt; desi_outliers_flow_
nsf_cont{10,15}latent_snr3.pt for DESI log p, thresholds and the latent files), data/dataset_z_snr.npz
(DESI z and median S/N, the latent-cut definition as the mocks' snr_med, 2026-11-13_mock_noise_snr_check),
results/prescreened_sample_v1.npz (keep).
METHOD: mock log p recomputed with the saved flows; held-out mocks are the first tenth of the
default_rng(0) permutation. Flag = at or below both saved cont thresholds; held-out mocks are flagged
in both flows on the same rows (mock files share target_ids, checked). Per z bin (0.05, plus 0-0.03)
it gives counts, median S/N, flagged share with binomial error and median cont10 log p, for
held-out mocks and DESI (prescreen keep); then z >= 0.45 split in S/N bins 3-4, 4-5, 5-8, >= 8.
For 99, the mean Euclidean distance to the 10 nearest training mocks in each standardised cont
latent, against held-out mocks within |z - 0.015| <= 0.01.
SEED: none (the split uses the flows' seed 0).
COMMAND: ``uv run --no-sync python experiments/2026-11-14_cont_z_coverage/analysis.py``.
RESULT: 2026-11-14, coverage.txt. Not coverage: no held-out mock is flagged in any z bin (0 of 2,065
at z 0.45-0.60; median cont10 log p -5.4 to -6.8 vs -4.0 to -5.0 below z 0.45), while DESI rises to
2.5, 5.5 and 14% at z 0.45-0.50, 0.50-0.55, 0.55-0.60 (median log p -12.4 to -16.0). At z >= 0.45 and
matched S/N 3-4, mocks 0% (N 385) vs DESI 2.5 +- 0.6%; DESI rises with S/N there (6, 19, 40%). So high-z
DESI differs from the mocks at the same z and S/N. DESI at z < 0.03 is also high (0.66 +- 0.27%). 99:
10,023 mocks within z +- 0.01, none flagged, so not sparse sampling; its kNN distance is beyond every
same-z held-out mock (2.40 vs p99.9 1.49 in cont10; 3.68 vs 2.12 in cont15). The plan's 99 rule calls
that coverage, but it does not separate a mock gap from a peculiar spectrum; what makes 99 differ is
not tested. Assumptions: held-out tenth only; joint flag of both flows; cause of the DESI-mock high-z
difference not tested.
FIGURES: none (coverage.txt).
"""

import h5py
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from hubersed.detect.flow_model import build_flow
from hubersed.io.latents import load_latents
from hubersed.paths import PATHS

RES, ROOT, DATA = PATHS["RESULTS"], PATHS["ROOT"], PATHS["DATA"]
OUT = RES / "2026-11-14_cont_z_coverage"
FLOWS = RES / "2026-11-09_mocks_v2_outliers" / "flows"
TAGS = ("cont10", "cont15")
Z_EDGES = (0.0, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
SNR_EDGES = (3, 4, 5, 8, np.inf)
TID_99, Z_99, DZ_99, K = 39633421919584782, 0.0151, 0.01, 10


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


def row(flag, lp, snr, sel):
    """Return N, median S/N, flagged share with error and median log p of ``sel`` as text."""
    n = int(sel.sum())
    if n == 0:
        return f"N {0:6d}"
    p = flag[sel].mean()
    return (f"N {n:6d}, S/N {np.median(snr[sel]):5.1f}, flagged {100 * p:6.2f} +- "
            f"{100 * np.sqrt(p * (1 - p) / n):5.2f}%, cont10 log p {np.median(lp[sel]):7.2f}")  # fmt: skip


mock, flows, desi, thr = {}, {}, {}, {}
for tag in TAGS:
    flows[tag] = torch.load(
        FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu"
    )
    r = torch.load(FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt", weights_only=False)
    thr[tag] = float(r["threshold"])
    mock[tag] = load_latents(ROOT / r["mock_file"])[:2]
    desi[tag] = (load_latents(ROOT / r["desi_file"])[:2], np.asarray(r["log_p_desi"], float),
                 np.asarray(r["desi_target_ids"], np.int64))  # fmt: skip
assert np.array_equal(mock["cont10"][1], mock["cont15"][1]), "mock files differ in rows"
with h5py.File(ROOT / torch.load(FLOWS / "desi_outliers_flow_nsf_cont10latent_snr3.pt",
                                 weights_only=False)["mock_file"], "r") as h:  # fmt: skip
    mz, msnr = h["zs"][:], h["snr_med"][:]
n_mock = len(mz)
perm = np.random.default_rng(0).permutation(n_mock)
val, tr = perm[: n_mock // 10], perm[n_mock // 10 :]
mlp = {tag: log_prob(flows[tag], mock[tag][0][val]) for tag in TAGS}
mflag = (mlp["cont10"] <= thr["cont10"]) & (mlp["cont15"] <= thr["cont15"])
vz, vsnr = mz[val], msnr[val]

zs = np.load(DATA / "dataset_z_snr.npz")
zof = dict(zip(zs["tid"].tolist(), zs["z"].tolist(), strict=True))
sof = dict(zip(zs["tid"].tolist(), zs["snr"].tolist(), strict=True))
keep = set(np.load(RES / "prescreened_sample_v1.npz")["keep"].astype(np.int64).tolist())
lp10 = dict(zip(desi["cont10"][2].tolist(), desi["cont10"][1], strict=True))
lp15 = dict(zip(desi["cont15"][2].tolist(), desi["cont15"][1], strict=True))
dt = np.array([t for t in desi["cont10"][2].tolist() if t in keep and t in lp15 and t in zof])
dz = np.array([zof[t] for t in dt])
dsnr = np.array([sof[t] for t in dt])
dlp = np.array([lp10[t] for t in dt])
dflag = (dlp <= thr["cont10"]) & (np.array([lp15[t] for t in dt]) <= thr["cont15"])

lines = [
    f"held-out mocks {val.size} (of {n_mock}); DESI prescreen keep {dt.size}; thresholds cont10 "
    f"{thr['cont10']:.2f}, cont15 {thr['cont15']:.2f}",
    f"overall: mocks {row(mflag, mlp['cont10'], vsnr, np.ones(val.size, bool))}",
    f"         DESI  {row(dflag, dlp, dsnr, np.ones(dt.size, bool))}",
    "",
    "per z bin: held-out mocks | DESI",
]
for lo, hi in zip(Z_EDGES[:-1], Z_EDGES[1:], strict=True):
    ms, ds = (vz >= lo) & (vz < hi), (dz >= lo) & (dz < hi)
    lines.append(f"  z {lo:.2f}-{hi:.2f}: {row(mflag, mlp['cont10'], vsnr, ms)} | "
                 f"{row(dflag, dlp, dsnr, ds)}")  # fmt: skip
lines += ["", "z >= 0.45 per S/N bin: held-out mocks | DESI"]
for lo, hi in zip(SNR_EDGES[:-1], SNR_EDGES[1:], strict=True):
    ms = (vz >= 0.45) & (vsnr >= lo) & (vsnr < hi)
    ds = (dz >= 0.45) & (dsnr >= lo) & (dsnr < hi)
    lines.append(f"  S/N {lo}-{hi}: {row(mflag, mlp['cont10'], vsnr, ms)} | "
                 f"{row(dflag, dlp, dsnr, ds)}")  # fmt: skip

lines += ["", f"99 ({TID_99}), z {Z_99}:"]
near_all = np.abs(mz - Z_99) <= DZ_99
near_val = np.abs(vz - Z_99) <= DZ_99
lines.append(f"  mocks within |z - {Z_99}| <= {DZ_99}: {near_all.sum()} of all, {near_val.sum()} held-out; "
             f"held-out there {row(mflag, mlp['cont10'], vsnr, near_val)}")  # fmt: skip
for tag in TAGS:
    g = flows[tag]
    sc = lambda x, g=g: (x - g["scaler_mean"]) / g["scaler_scale"]  # noqa: E731
    nn = NearestNeighbors(n_neighbors=K).fit(sc(mock[tag][0][tr]))
    ref = nn.kneighbors(sc(mock[tag][0][val[near_val]]))[0].mean(1)
    (dlat, dtid), dlog, lptid = desi[tag]
    i = int(np.flatnonzero(dtid == TID_99)[0])
    d99 = nn.kneighbors(sc(dlat[i : i + 1]))[0].mean()
    lines.append(f"  {tag}: 99 kNN distance {d99:.2f}; same-z held-out mocks p50/p99/p99.9 "
                 f"{' '.join(f'{v:.2f}' for v in np.percentile(ref, [50, 99, 99.9]))}; percentile "
                 f"{100 * np.mean(ref < d99):.1f}; 99 log p {dlog[lptid == TID_99][0]:.2f}")  # fmt: skip
OUT.mkdir(parents=True, exist_ok=True)
text = "\n".join(lines)
(OUT / "coverage.txt").write_text(text + "\n")
print(text)
