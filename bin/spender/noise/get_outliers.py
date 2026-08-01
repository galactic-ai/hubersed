import argparse
import numpy as np
import h5py
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from hubersed.paths import PATHS

DATA_PATH = PATHS["DATA"]
RESULTS_PATH = PATHS["RESULTS"]

torch.set_default_dtype(torch.float32)


def load_latents(path):
    with h5py.File(path, "r") as f:
        lat = np.asarray(f["latents"], dtype=np.float32)
        if "target_ids" not in f:
            raise KeyError(
                f"{path} has no 'target_ids' -- re-encode with the fixed "
                "get_latent_space.py (this file predates the TARGETID fix)."
            )
        tid = np.asarray(f["target_ids"], dtype=np.int64)
        snr_min = f.attrs.get("snr_min", None)
    return lat, tid, snr_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--desi",
        default="spender_spec_6latent_snr3.h5",
        help="DESI latent file name (in DATA dir)",
    )
    ap.add_argument(
        "--mock",
        default="prospector_noise_spec_cue_6latent_snr3.h5",
        help="mock latent file name (in DATA dir)",
    )
    ap.add_argument(
        "--out",
        default="desi_outliers_cue_snr3.pt",
        help="output filename (in RESULTS dir)",
    )
    ap.add_argument(
        "--quantile",
        type=float,
        default=0.001,
        help="outlier threshold = this quantile of the MOCK score distribution",
    )
    args = ap.parse_args()

    desi_latents, desi_tid, _ = load_latents(DATA_PATH / args.desi)
    p_l, _, mock_snr_min = load_latents(DATA_PATH / args.mock)
    print(
        f"DESI {tuple(desi_latents.shape)} from {args.desi}  |  mock {tuple(p_l.shape)} from {args.mock}"
    )
    assert desi_latents.shape[1] == p_l.shape[1], (
        "latent dim mismatch between DESI and mock files!"
    )

    # IsolationForest: fit on mocks (normal), score DESI
    scaler = StandardScaler()
    p_l_scaled = scaler.fit_transform(p_l)
    s_l_scaled = scaler.transform(desi_latents)

    iso = IsolationForest(
        n_estimators=300,
        max_samples=2048,
        contamination="auto",
        n_jobs=-1,
        random_state=0,
    )
    iso.fit(p_l_scaled)
    scores_desi = iso.decision_function(s_l_scaled)
    scores_mock = iso.decision_function(p_l_scaled)

    threshold = torch.quantile(torch.tensor(scores_mock), args.quantile)
    outlier_pos = torch.where(torch.tensor(scores_desi) <= threshold)[0]
    outlier_tid = desi_tid[outlier_pos]

    torch.save(
        {
            "outlier_target_ids": outlier_tid,
            "scores_desi": torch.tensor(scores_desi),
            "desi_target_ids": desi_tid,
            "threshold": float(threshold),
            "n_latent": int(desi_latents.shape[1]),
            "desi_file": args.desi,
            "mock_file": args.mock,
            "snr_min": mock_snr_min,
        },
        RESULTS_PATH / args.out,
    )
    print(
        f"{len(outlier_tid)} outliers -> {args.out}  (TARGETIDs; threshold {float(threshold):.4f}; "
        f"DESI rate {100 * len(outlier_tid) / desi_latents.shape[0]:.3f}%)"
    )


if __name__ == "__main__":
    main()
