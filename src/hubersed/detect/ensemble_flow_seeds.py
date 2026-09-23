#!/usr/bin/env python
"""Train the NSF flow S times per latent set (one seed each) and keep every log p.

Motivation: seed 0 and seed 1 give different DESI outlier sets at the same nominal
0.1% mock threshold. Without knowing how much of an outlier set is seed noise, no
single-seed outlier list can be trusted, and the 20-galaxy sample was built from one.

Architecture, optimiser, LR schedule and threshold rule are IDENTICAL to
get_outliers_flow.py -- build_flow and load_h5 are imported from it rather than
copied, so the two cannot drift. The seed is the only thing that changes between
ensemble members, and it enters in exactly the two places it does there:
torch.manual_seed (flow init) and np.random.default_rng (train/valid split +
batch order).

Writes one npz per tag holding log p for every mock and every DESI object for
every seed. Stats live in ensemble_flow_stats.py so re-analysis needs no retrain.

  python -m hubersed.detect.ensemble_flow_seeds --tag cont10latent --seeds 20 \
      --device cuda:0 --outdir results/flow_ensemble
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from hubersed.detect.get_outliers_flow import build_flow, load_h5
from hubersed.paths import PATHS

DATA, RES = PATHS["DATA"], PATHS["RESULTS"]
# The mock set behind results/wide_flow_corrected, i.e. the one the 20-galaxy sample
# came from. cont15latent exists ONLY here, not in noised_cue_meanzero.
MOCK_DIR = DATA / "latents" / "noised_cue_meanzero_wide"


def paths_for(tag):
    return (
        MOCK_DIR / f"prospector_noise_spec_cue_{tag}_snr3.h5",
        DATA / "latents" / f"spender_spec_{tag}_snr3.h5",
    )


def score(nde, X, dev, chunk=200_000):
    """log p in chunks; the full mock array does not need to sit on the GPU at once."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            out.append(nde.log_prob(torch.from_numpy(X[i : i + chunk]).to(dev)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def train_one(seed, mock_s, dev, a):
    """One ensemble member. Returns (flow, val_idx, nll_train, nll_valid)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    perm = rng.permutation(mock_s.shape[0])
    nval = mock_s.shape[0] // 10
    val_idx, tr_idx = perm[:nval], perm[nval:]
    Xtr = torch.from_numpy(mock_s[tr_idx]).to(dev)
    Xval = torch.from_numpy(mock_s[val_idx]).to(dev)

    nde = build_flow(a.method, mock_s.shape[1], a.hidden, a.num_transforms, a.num_bins).to(dev)
    opt = torch.optim.Adam(nde.parameters(), lr=a.lr)
    steps = max(1, (Xtr.shape[0] + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, steps_per_epoch=steps, epochs=a.epochs
    )

    tr_last = vl = np.nan
    for ep in range(a.epochs):
        nde.train()
        tl = []
        for b in torch.split(torch.from_numpy(rng.permutation(Xtr.shape[0])), a.batch):
            opt.zero_grad()
            loss = -nde.log_prob(Xtr[b.to(dev)]).mean()
            loss.backward()
            opt.step()
            sched.step()
            tl.append(loss.item())
        nde.eval()
        with torch.no_grad():
            vl = -nde.log_prob(Xval).mean().item()
        tr_last = float(np.mean(tl))
    del Xtr, Xval
    return nde, val_idx, tr_last, vl


def validate(nde, mock_s, val_idx, dev, n_c2st, rng):
    """KS per dim + C-2ST of flow samples against held-out mocks. n_c2st=0 skips C-2ST."""
    n = min(20000, len(val_idx))
    with torch.no_grad():
        samp = nde.sample(n).cpu().numpy()
    real = mock_s[val_idx][:n]
    ks = [ks_2samp(samp[:, j], real[:, j]).statistic for j in range(mock_s.shape[1])]
    c2st = np.nan
    if n_c2st:
        k = min(n_c2st, n)
        i1, i2 = rng.choice(n, k, replace=False), rng.choice(n, k, replace=False)
        Xc = np.vstack([samp[i1], real[i2]])
        yc = np.r_[np.zeros(k), np.ones(k)]
        c2st = cross_val_score(
            HistGradientBoostingClassifier(max_iter=120, random_state=0),
            Xc,
            yc,
            cv=3,
            scoring="accuracy",
        ).mean()
    return float(max(ks)), float(np.median(ks)), float(c2st)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--tag", required=True, help="6latent/10latent/15latent/cont10latent/cont15latent"
    )
    p.add_argument("--seeds", type=int, default=20, help="number of ensemble members")
    p.add_argument("--seed0", type=int, default=0, help="first seed; members are seed0..seed0+S-1")
    p.add_argument("--method", default="nsf", choices=["maf", "nsf"])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--num_transforms", type=int, default=10)
    p.add_argument("--num_bins", type=int, default=10)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--c2st-n", type=int, default=20000, help="0 to skip the C-2ST")
    p.add_argument("--outdir", type=Path, default=RES / "flow_ensemble")
    a = p.parse_args()

    dev = torch.device(a.device)
    mock_f, desi_f = paths_for(a.tag)
    mock, _, mock_ckpt = load_h5(mock_f)
    desi, desi_tid, desi_ckpt = load_h5(desi_f)
    if mock_ckpt != desi_ckpt and "unknown" not in (mock_ckpt, desi_ckpt):
        raise SystemExit(f"encoder mismatch: mock={mock_ckpt} vs DESI={desi_ckpt}")

    # Fit on ALL mocks, so the scaler is seed-independent: every member sees the same
    # input space and only the flow differs.
    scaler = StandardScaler().fit(mock)
    mock_s = scaler.transform(mock).astype(np.float32)
    desi_s = scaler.transform(desi).astype(np.float32)
    S = a.seeds
    print(
        f"tag={a.tag} dim={mock.shape[1]} mock={mock.shape[0]} desi={desi.shape[0]} "
        f"seeds={a.seed0}..{a.seed0 + S - 1} device={dev} encoder={mock_ckpt}",
        flush=True,
    )

    lp_mock = np.empty((S, mock_s.shape[0]), np.float32)
    lp_desi = np.empty((S, desi_s.shape[0]), np.float32)
    thr = np.empty(S)
    thr_ho = np.empty(S)
    c2st = np.empty(S)
    ks_max = np.empty(S)
    ks_med = np.empty(S)
    nll_tr = np.empty(S)
    nll_val = np.empty(S)
    seeds = np.arange(a.seed0, a.seed0 + S)
    a.outdir.mkdir(parents=True, exist_ok=True)
    out_f = a.outdir / f"ens_{a.method}_{a.tag}.npz"
    vrng = np.random.default_rng(12345)  # validation subsampling only, not a member seed

    for i, sd in enumerate(seeds):
        t0 = time.time()
        nde, val_idx, nll_tr[i], nll_val[i] = train_one(int(sd), mock_s, dev, a)
        nde.eval()
        lp_mock[i] = score(nde, mock_s, dev)
        lp_desi[i] = score(nde, desi_s, dev)
        # Same rule as get_outliers_flow.py: 0.1% quantile over ALL mocks.
        thr[i] = float(np.quantile(lp_mock[i], 0.001))
        thr_ho[i] = float(np.quantile(lp_mock[i][val_idx], 0.001))
        ks_max[i], ks_med[i], c2st[i] = validate(nde, mock_s, val_idx, dev, a.c2st_n, vrng)
        n_out = int((lp_desi[i] <= thr[i]).sum())
        print(
            f"  seed {sd:2d}  NLL {nll_tr[i]:7.3f}/{nll_val[i]:7.3f}  thr {thr[i]:8.3f}  "
            f"outliers {n_out:5d} ({100 * n_out / len(desi_tid):.3f}%)  "
            f"KS {ks_max[i]:.3f}  C2ST {c2st[i]:.3f}  [{time.time() - t0:.0f}s]",
            flush=True,
        )
        del nde
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        # Checkpoint every member: a 20-seed run is ~an hour and a crash at member 19
        # should not cost the first 18.
        np.savez_compressed(
            out_f,
            seeds=seeds[: i + 1],
            lp_mock=lp_mock[: i + 1],
            lp_desi=lp_desi[: i + 1],
            desi_tid=desi_tid,
            thr=thr[: i + 1],
            thr_heldout=thr_ho[: i + 1],
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
