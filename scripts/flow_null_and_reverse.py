#!/usr/bin/env python
"""Train flows on half of one latent set, then score the held-out half and the other set.

With ``--mode selfdist`` the flows train on half the mocks and score the other half and DESI.
With ``--mode reverse`` they train on half of DESI and score the other half and the mocks.
Run it as ``uv run python scripts/flow_null_and_reverse.py --mode selfdist --tag cont10latent``.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from hubersed.detect.ensemble_flow_seeds import paths_for, score, train_one, validate
from hubersed.detect.get_outliers_flow import load_h5
from hubersed.paths import PATHS

RES = PATHS["RESULTS"]


def halves(n, seed):
    """Split range(n) into two disjoint random halves.

    Use the same seed for every ensemble member so that all members share one
    held-out half.

    Parameters
    ----------
    n : int
        Number of rows to split.
    seed : int
        Seed for the permutation.

    Returns
    -------
    a : numpy.ndarray
        The first n // 2 indices of the permutation.
    b : numpy.ndarray
        The remaining indices.
    """
    idx = np.random.default_rng(seed).permutation(n)
    a, b = idx[: n // 2], idx[n // 2 :]
    assert not (set(a.tolist()) & set(b.tolist())), "A/B overlap"
    assert len(a) + len(b) == n
    return a, b


def main():
    """Parse arguments, train one flow per seed and save the scores to an npz file.

    The npz is rewritten after every seed, so a partial run keeps the finished seeds.

    Raises
    ------
    SystemExit
        If the mock and DESI latents come from different encoders, or if the output
        file exists and ``--force`` is not given.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", required=True, choices=["selfdist", "reverse"])
    p.add_argument(
        "--tag", required=True, help="6latent/10latent/15latent/cont10latent/cont15latent"
    )
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument(
        "--split-seed",
        type=int,
        default=777,
        help="A/B split of the training dataset; fixed across ensemble members",
    )
    p.add_argument("--method", default="nsf", choices=["maf", "nsf"])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--num_transforms", type=int, default=10)
    p.add_argument("--num_bins", type=int, default=10)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--c2st-n", type=int, default=20000)
    p.add_argument("--outdir", type=Path, default=RES / "flow_calibration")
    p.add_argument(
        "--dry-run", action="store_true", help="load, split, print shapes, train nothing"
    )
    p.add_argument("-f", "--force", action="store_true", help="overwrite existing npz")
    a = p.parse_args()

    dev = torch.device(a.device)
    mock_f, desi_f = paths_for(a.tag)
    mock, _, mock_ckpt = load_h5(mock_f)
    desi, desi_tid, desi_ckpt = load_h5(desi_f)
    if mock_ckpt != desi_ckpt and "unknown" not in (mock_ckpt, desi_ckpt):
        raise SystemExit(f"encoder mismatch: mock={mock_ckpt} vs DESI={desi_ckpt}")

    # Both modes split the TRAINING dataset in half so there is always a genuine
    # same-distribution null (half B) to read the other dataset's rate against.
    # Only which dataset trains the flow differs.
    src, tgt, names = (
        (mock, desi, ("mockA", "mockB", "desi"))
        if a.mode == "selfdist"
        else (desi, mock, ("desiA", "desiB", "mock"))
    )
    ia, ib = halves(src.shape[0], a.split_seed)
    train_pool, held, other = src[ia], src[ib], tgt

    # Scaler fits the TRAINING pool only. Fitting on everything would leak the
    # held-out half's moments into the standardisation and shrink the null.
    scaler = StandardScaler().fit(train_pool)
    Xtrain = scaler.transform(train_pool).astype(np.float32)
    Xheld = scaler.transform(held).astype(np.float32)
    Xother = scaler.transform(other).astype(np.float32)

    print(
        f"mode={a.mode} tag={a.tag} dim={mock.shape[1]} encoder={mock_ckpt}\n"
        f"  train[{names[0]}]={Xtrain.shape[0]}  held[{names[1]}]={Xheld.shape[0]}  "
        f"other[{names[2]}]={Xother.shape[0]}  device={dev}",
        flush=True,
    )
    if a.dry_run:
        return

    S = a.seeds
    seeds = np.arange(a.seed0, a.seed0 + S)
    lp_train = np.empty((S, Xtrain.shape[0]), np.float32)
    lp_held = np.empty((S, Xheld.shape[0]), np.float32)
    lp_other = np.empty((S, Xother.shape[0]), np.float32)
    thr = np.empty(S)
    c2st = np.empty(S)
    ks_max = np.empty(S)
    ks_med = np.empty(S)
    nll_tr = np.empty(S)
    nll_val = np.empty(S)
    a.outdir.mkdir(parents=True, exist_ok=True)
    out_f = a.outdir / f"cal_{a.mode}_{a.method}_{a.tag}.npz"
    if out_f.exists() and not a.force:
        raise SystemExit(f"refusing to overwrite {out_f}\npass --force, or --outdir elsewhere")
    vrng = np.random.default_rng(12345)

    for i, sd in enumerate(seeds):
        t0 = time.time()
        # train_one does its own internal 90/10 split of whatever pool it is given,
        # so val_idx here indexes the TRAINING pool, not the held-out half.
        nde, val_idx, nll_tr[i], nll_val[i] = train_one(int(sd), Xtrain, dev, a)
        nde.eval()
        lp_train[i] = score(nde, Xtrain, dev)
        lp_held[i] = score(nde, Xheld, dev)
        lp_other[i] = score(nde, Xother, dev)
        # Same rule as production: 0.1% quantile over the whole training pool.
        # The training pool is half the dataset, so this threshold is noisier.
        thr[i] = float(np.quantile(lp_train[i], 0.001))
        ks_max[i], ks_med[i], c2st[i] = validate(nde, Xtrain, val_idx, dev, a.c2st_n, vrng)

        # The two numbers this script exists to produce.
        fpr = float((lp_held[i] <= thr[i]).mean())  # nominal 0.001 under the null
        rate_other = float((lp_other[i] <= thr[i]).mean())
        print(
            f"  seed {sd:2d}  NLL {nll_tr[i]:7.3f}/{nll_val[i]:7.3f}  thr {thr[i]:8.3f}  "
            f"{names[1]}<=thr {100 * fpr:.3f}%  {names[2]}<=thr {100 * rate_other:.3f}%  "
            f"KS {ks_max[i]:.3f}  C2ST {c2st[i]:.3f}  [{time.time() - t0:.0f}s]",
            flush=True,
        )
        if not np.isnan(c2st[i]) and c2st[i] > 0.6:
            print(
                f"    WARNING C-2ST {c2st[i]:.3f} > 0.6: flow does not reproduce its own "
                f"training set. 2 of 100 flows failed this way in the 2026-08-27o ensemble.",
                flush=True,
            )
        del nde
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        np.savez_compressed(
            out_f,
            mode=a.mode,
            seeds=seeds[: i + 1],
            lp_train=lp_train[: i + 1],
            lp_held=lp_held[: i + 1],
            lp_other=lp_other[: i + 1],
            # split_idx_* index the TRAINING dataset, which is the mocks under
            # selfdist and DESI under reverse. Read them together with `mode`.
            names=np.array(names),
            split_idx_a=ia,
            split_idx_b=ib,
            desi_tid=desi_tid,
            thr=thr[: i + 1],
            c2st=c2st[: i + 1],
            ks_max=ks_max[: i + 1],
            ks_med=ks_med[: i + 1],
            nll_train=nll_tr[: i + 1],
            nll_valid=nll_val[: i + 1],
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            tag=a.tag,
            method=a.method,
            encoder=mock_ckpt,
            split_seed=a.split_seed,
            mock_file=str(mock_f),
            desi_file=str(desi_f),
            hyper=json.dumps(
                {
                    k: getattr(a, k)
                    for k in (
                        "epochs",
                        "batch",
                        "num_transforms",
                        "num_bins",
                        "hidden",
                        "lr",
                        "device",
                        "c2st_n",
                    )
                }
            ),
        )
    print(f"wrote {out_f}  ({out_f.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
