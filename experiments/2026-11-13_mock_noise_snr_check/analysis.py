"""Do the noised v2 mocks the flows trained on have the S/N of DESI, and does S/N drive the outliers?

QUESTION: (1) How do the median per-pixel S/N distributions of the flow training mocks (noised v2,
S/N > 3) and the DESI spectra the flows scored compare, overall and per redshift? (2) What sets the
mock S/N, redshift, mass formed or the normalisation A? (3) Per S/N bin, what are the DESI outlier rate
and median log p of each flow, and the median log p of the mocks?
HYPOTHESIS: none registered. A first look (2026-11-13_psb_fraction_like_for_like snr step) found
two thirds of the S/N > 3 mocks above S/N 8, against 13.5% of the DESI parent.
INPUTS: data/latents/mocks_v2/prospector_noise_spec_cuev2_<tag>latent_snr3.h5 (latents, snr_med, A,
target_ids = mock index, zs); results/2026-11-09_mocks_v2_outliers/flows/flow_nsf_<tag>latent.pt and
desi_outliers_flow_nsf_<tag>latent_snr3.pt (log_p_desi, desi_target_ids, threshold);
data/dataset_z_snr.npz (DESI median S/N, the latent-cut S/N); data/latents/spender_spec_cont10latent_
snr3.h5 (DESI A); data/prospector_model/mocks_v2_500000.h5 (logmass); 2026-11-13_gallery105 review.tsv
and gallery105.tsv (the 105 new outliers).
METHOD: mock log p from each saved flow (build_flow, the saved scaler) on the mock latents. S/N bins
3-5, 5-8, 8-15, >15; z bins of 0.1. A is compared across samples only through its quantiles (its
units in the two loaders are not checked here).
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-13_mock_noise_snr_check/analysis.py``.
RESULT: 2026-11-13, snr_check.txt. (1) Mock S/N median 11.9 vs DESI 4.6; S/N > 15: 40% vs 3.6%; mocks
are higher at every z (z 0.5-0.6: 6.4 vs 3.5). (2) Mock S/N rises with mass formed (median 5.7 at
log M 7-9, 14.5 at 11-12; 88k of the 158k mocks have log M 11-12) and follows A (r 0.70; DESI 0.84);
mock A median 16 vs DESI 3.6. (3) DESI outlier rate rises with S/N in every flow (cont10 1.6% at S/N
3-5 to 11.9% above 15; cont15 0.35% to 5.1%); mocks stay near 0.1% below threshold in every bin. The
105 new outliers have median S/N 4.6, as the parent. Assumptions: A in the two loaders is in the same
units (not checked); S/N is the latent-cut definition for both.
FIGURES: none (snr_check.txt).
"""

import h5py
import numpy as np
import pandas as pd
import torch

from hubersed.detect.flow_model import build_flow
from hubersed.paths import PATHS

RES, DATA = PATHS["RESULTS"], PATHS["DATA"]
OUT = RES / "2026-11-13_mock_noise_snr_check"
FLOWS = RES / "2026-11-09_mocks_v2_outliers" / "flows"
TAGS = ("6", "10", "15", "cont10", "cont15")
SNR_BINS = (3, 5, 8, 15, np.inf)
ZBINS = np.round(np.arange(0.0, 0.61, 0.1), 2)


def mock_logp(tag, latents):
    """Return the log p of each mock latent under the saved flow of ``tag``."""
    ck = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False)
    nde = build_flow(ck["method"], ck["dim"], ck["hidden"], ck["num_transforms"], ck["num_bins"])
    nde.load_state_dict(ck["state_dict"])
    nde.eval()
    x = ((latents - ck["scaler_mean"]) / ck["scaler_scale"]).astype(np.float32)
    with torch.no_grad():
        return np.concatenate([nde.log_prob(torch.from_numpy(x[k : k + 20000])).numpy()
                               for k in range(0, len(x), 20000)])  # fmt: skip


def snr_bins(snr):
    """Return the S/N bin label of each value as a pandas Series."""
    labels = [f"{a}-{b}" for a, b in zip(SNR_BINS[:-1], SNR_BINS[1:], strict=True)]
    return pd.Series(pd.cut(np.asarray(snr), SNR_BINS, labels=labels))


def pct(x):
    """Return the 10/50/90 percentiles as text."""
    x = x[np.isfinite(x)]
    return " ".join(f"{v:7.2f}" for v in np.percentile(x, [10, 50, 90])) if x.size else "-"


def main():
    """Write the three parts of the check."""
    OUT.mkdir(parents=True, exist_ok=True)
    zs = np.load(DATA / "dataset_z_snr.npz")
    desi_snr = dict(zip(zs["tid"].tolist(), zs["snr"].tolist(), strict=True))
    with h5py.File(DATA / "prospector_model" / "mocks_v2_500000.h5", "r") as h:
        labels = [s.decode() if isinstance(s, bytes) else str(s) for s in h.attrs["theta_labels"]]
        logmass = h["priors/theta"][:, labels.index("logmass")]
    with h5py.File(
        DATA / "latents" / "mocks_v2" / "prospector_noise_spec_cuev2_cont10latent_snr3.h5", "r"
    ) as h:
        m_snr, m_z, m_a, m_id = h["snr_med"][:], h["zs"][:], h["A"][:, 0], h["target_ids"][:]
    with h5py.File(DATA / "latents" / "spender_spec_cont10latent_snr3.h5", "r") as h:
        d_tid, d_z, d_a = h["target_ids"][:], h["zs"][:], h["A"][:, 0]
    d_snr = np.array([desi_snr.get(int(t), np.nan) for t in d_tid])
    lines = ["(1) median per-pixel S/N, 10/50/90 percentiles and fraction per S/N bin",
             f"  mocks (flow training, N {m_snr.size}): {pct(m_snr)}",
             f"  DESI (scored, N {d_snr.size}): {pct(d_snr)}"]  # fmt: skip
    for name, s in (("mocks", m_snr), ("DESI", d_snr)):
        f = snr_bins(s).value_counts(normalize=True).sort_index()
        lines.append(f"  {name} fraction: " + ", ".join(f"{k} {v:.3f}" for k, v in f.items()))
    lines.append("  per z bin, S/N 10/50/90 (mocks | DESI) and N")
    for lo, hi in zip(ZBINS[:-1], ZBINS[1:], strict=True):
        a, b = (m_z >= lo) & (m_z < hi), (d_z >= lo) & (d_z < hi)
        lines.append(
            f"    z {lo:.1f}-{hi:.1f}: {pct(m_snr[a])} (N {a.sum():6d}) | {pct(d_snr[b])} (N {b.sum():6d})"
        )

    lines += ["", "(2) what sets the mock S/N", f"  A 10/50/90: mocks {pct(m_a)} | DESI {pct(d_a)}"]
    m_lm = logmass[m_id.astype(int)]
    for lo, hi in ((7, 9), (9, 10), (10, 10.5), (10.5, 11), (11, 12)):
        s = (m_lm >= lo) & (m_lm < hi)
        lines.append(f"  mocks log M_formed {lo}-{hi}: S/N {pct(m_snr[s])} (N {s.sum()})")
    for name, s, a in (("mocks", m_snr, m_a), ("DESI", d_snr, d_a)):
        ok = np.isfinite(s) & (a > 0)
        r = np.corrcoef(np.log10(a[ok]), np.log10(s[ok]))[0, 1]
        lines.append(f"  {name}: correlation of log A and log S/N {r:.2f}")

    lines += [
        "",
        "(3) per S/N bin: DESI N, outlier rate, median log p | mocks N, median log p, below threshold",
    ]
    for tag in TAGS:
        with h5py.File(
            DATA / "latents" / "mocks_v2" / f"prospector_noise_spec_cuev2_{tag}latent_snr3.h5", "r"
        ) as h:
            lat, snr_t = h["latents"][:], h["snr_med"][:]
        lp_m = mock_logp(tag, lat)
        res = torch.load(FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt", weights_only=False)
        lp_d, tid_d, thr = (
            np.asarray(res["log_p_desi"]),
            np.asarray(res["desi_target_ids"]),
            res["threshold"],
        )
        s_d = np.array([desi_snr.get(int(t), np.nan) for t in tid_d])
        lines.append(f"  {tag}latent (threshold {thr:.2f}; mocks below {np.mean(lp_m <= thr):.4f})")
        bd, bm = snr_bins(s_d), snr_bins(snr_t)
        for lab in bd.cat.categories:
            a, b = np.asarray(bd == lab), np.asarray(bm == lab)
            lines.append(
                f"    S/N {lab:>7}: DESI {a.sum():6d} outliers {np.mean(lp_d[a] <= thr):.4f} median log p "
                f"{np.median(lp_d[a]):7.2f} | mocks {b.sum():6d} median log p {np.median(lp_m[b]):7.2f} "
                f"below {np.mean(lp_m[b] <= thr):.4f}"
            )

    rev = pd.read_csv(
        PATHS["ROOT"] / "experiments" / "2026-11-13_gallery105" / "review.tsv", sep="\t"
    )
    tab = pd.read_csv(RES / "2026-11-13_gallery105" / "gallery105.tsv", sep="\t")
    g = rev.merge(tab[["nn", "target_id"]], on="nn")
    g_snr = np.array([desi_snr.get(int(t), np.nan) for t in g["target_id"]])
    lines += ["", f"gallery105 outliers: S/N {pct(g_snr)} (N {len(g)}); by class:"]
    for cls, sub in g.assign(snr=g_snr).groupby("review_class"):
        lines.append(f"  {cls:14} N {len(sub):3d} median S/N {np.nanmedian(sub['snr']):6.2f}")
    text = "\n".join(lines)
    (OUT / "snr_check.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
