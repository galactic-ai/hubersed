import argparse
from pathlib import Path
import numpy as np
import torch
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import (
    CompositeTransform,
    RandomPermutation,
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from hubersed.paths import PATHS

DATA, RES = PATHS["DATA"], PATHS["RESULTS"]


def build_flow(method, dim, hidden, n_transforms, num_bins=8, tail_bound=10.0):
    """Autoregressive flow over standardized latents. method='maf' (affine) or
    'nsf' (rational-quadratic spline; more flexible for non-Gaussian/multimodal)."""
    ts = []
    for _ in range(n_transforms):
        if method == "nsf":
            ts.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim,
                    hidden_features=hidden,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=tail_bound,
                )
            )
        else:
            ts.append(
                MaskedAffineAutoregressiveTransform(
                    features=dim, hidden_features=hidden
                )
            )
        ts.append(RandomPermutation(features=dim))
    return Flow(CompositeTransform(ts), StandardNormal([dim]))


def load_h5(path):
    """Return (latents, target_ids). Catalogue key is TARGETID, never a position.

    Takes a full path: two different mock sets (noised_cue_meanzero vs
    ..._wide) use the SAME filename in different directories, so resolving a
    bare name against a single root silently picks the wrong one.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"no latent file at {path}")
    with h5py.File(path, "r") as f:
        lat = np.asarray(f["latents"], np.float32)
        if "target_ids" not in f:
            raise KeyError(
                f"{path.name} has no 'target_ids' -- re-encode with the fixed get_latent_space.py"
            )
        tid = np.asarray(f["target_ids"], np.int64)
        ckpt = f.attrs.get("checkpoint", "unknown")
    return lat, tid, str(ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tag", default="10latent", help="6latent/10latent/15latent/cont10latent"
    )
    ap.add_argument("--method", default="maf", choices=["maf", "nsf"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--num_transforms", type=int, default=8)
    ap.add_argument("--num_bins", type=int, default=8, help="spline bins (nsf only)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--device", default="cpu"
    )  # flow is tiny; cpu avoids mps nflows gaps
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--desi", type=Path, required=True, help="DESI latent h5")
    ap.add_argument("--mock", type=Path, required=True, help="mock latent h5")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="where to write flow / outliers / pdf (default: results/)",
    )
    ap.add_argument("-f", "--force", action="store_true",
                    help="overwrite existing outputs")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    mock, _, mock_ckpt = load_h5(args.mock)
    desi, desi_tid, desi_ckpt = load_h5(args.desi)
    D = mock.shape[1]
    print(f"tag={args.tag}  dim={D}  mock {mock.shape}  DESI {desi.shape}")
    print(f"  mock {args.mock}  (encoder {mock_ckpt})")
    print(f"  DESI {args.desi}  (encoder {desi_ckpt})")
    if mock_ckpt != desi_ckpt and "unknown" not in (mock_ckpt, desi_ckpt):
        raise SystemExit(
            f"encoder mismatch: mock={mock_ckpt} vs DESI={desi_ckpt}.\n"
            "Latents from different encoders are not comparable."
        )

    # NB not RES = RES / ... -- assigning a module global inside a function makes
    # it local for the whole function, so the RHS raises UnboundLocalError.
    out_dir = args.outdir or RES
    out_dir.mkdir(parents=True, exist_ok=True)

    # standardize on mock
    scaler = StandardScaler().fit(mock)
    mock_s = scaler.transform(mock).astype(np.float32)
    desi_s = scaler.transform(desi).astype(np.float32)

    # train / valid split on mocks
    perm = rng.permutation(mock_s.shape[0])
    nval = mock_s.shape[0] // 10
    val_idx, tr_idx = perm[:nval], perm[nval:]
    Xtr = torch.from_numpy(mock_s[tr_idx]).to(dev)
    Xval = torch.from_numpy(mock_s[val_idx]).to(dev)

    nde = build_flow(
        args.method, D, args.hidden, args.num_transforms, args.num_bins
    ).to(dev)
    print(
        f"flow: method={args.method} transforms={args.num_transforms} hidden={args.hidden}"
        + (f" bins={args.num_bins}" if args.method == "nsf" else "")
    )
    opt = torch.optim.Adam(nde.parameters(), lr=args.lr)
    steps = max(
        1, (Xtr.shape[0] + args.batch - 1) // args.batch
    )  # ceil: torch.split yields a partial last batch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, steps_per_epoch=steps, epochs=args.epochs
    )

    hist = {"train": [], "valid": []}
    for ep in range(args.epochs):
        nde.train()
        tl = []
        for b in torch.split(
            torch.from_numpy(rng.permutation(Xtr.shape[0])), args.batch
        ):
            opt.zero_grad()
            loss = -nde.log_prob(Xtr[b.to(dev)]).mean()
            loss.backward()
            opt.step()
            sched.step()
            tl.append(loss.item())
        nde.eval()
        with torch.no_grad():
            vl = -nde.log_prob(Xval).mean().item()
        hist["train"].append(float(np.mean(tl)))
        hist["valid"].append(vl)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(
                f"  epoch {ep:3d}  train NLL {hist['train'][-1]:.3f}  valid NLL {vl:.3f}"
            )

    # ---- VALIDATION: did it train correctly? ----
    nde.eval()
    with torch.no_grad():
        samp = nde.sample(min(20000, Xval.shape[0])).cpu().numpy()
    real = mock_s[val_idx][: samp.shape[0]]
    ks = [ks_2samp(samp[:, j], real[:, j]).statistic for j in range(D)]
    # C-2ST: flow samples vs real mock (held-out). ~0.5 => flow matches the data.
    Xc = np.vstack([samp, real])
    yc = np.r_[np.zeros(len(samp)), np.ones(len(real))]
    c2st = cross_val_score(
        HistGradientBoostingClassifier(max_iter=120, random_state=0),
        Xc,
        yc,
        cv=3,
        scoring="accuracy",
    ).mean()
    print(f"\nVALIDATION  dim={D}")
    print(
        f"  final train/valid NLL = {hist['train'][-1]:.3f} / {hist['valid'][-1]:.3f}"
        f"  (gap {hist['valid'][-1] - hist['train'][-1]:+.3f})"
    )
    print(f"  sample-vs-mock per-dim KS: max {max(ks):.3f} median {np.median(ks):.3f}")
    print(
        f"  sample-vs-mock C-2ST accuracy = {c2st:.3f}  (0.5 = flow reproduces mocks)"
    )

    # ---- SCORE DESI ----
    with torch.no_grad():
        lp_mock = nde.log_prob(torch.from_numpy(mock_s).to(dev)).cpu().numpy()
        lp_desi = nde.log_prob(torch.from_numpy(desi_s).to(dev)).cpu().numpy()
    # Threshold from the HELD-OUT mocks: the flow was fit on tr_idx, so those points
    # carry inflated log_p, which pushes the 0.1% quantile up and over-flags DESI.
    # thr_all is reported alongside so the size of that bias is visible.
    thr = float(np.quantile(lp_mock[val_idx], 0.001))
    thr_all = float(np.quantile(lp_mock, 0.001))
    out_mask = lp_desi <= thr
    out_tid = desi_tid[out_mask]  # -> TARGETIDs
    print(f"\nSCORE  threshold (0.1% mock log p, held-out) = {thr:.2f}")
    print(f"  (same quantile over ALL mocks incl. training = {thr_all:.2f})")
    print(f"  DESI outliers: {out_mask.sum()}  ({100 * out_mask.mean():.3f}%)")
    print(f"  using thr_all instead would give: {int((lp_desi <= thr_all).sum())}")

    # overlap vs IsoForest for this tag (by TARGETID). Stays at the results/ root --
    # IsoForest outputs are not written per-outdir.
    iso_file = RES / (
        "desi_outliers_cue_snr3.pt"
        if args.tag == "6latent"
        else f"desi_outliers_{args.tag}_snr3.pt"
    )
    if iso_file.exists():
        iso = set(
            int(x)
            for x in torch.load(iso_file, weights_only=False)["outlier_target_ids"]
        )
        fl = set(int(x) for x in out_tid)
        print(
            f"  IsoForest outliers: {len(iso)}  | flow∩iso = {len(fl & iso)}  "
            f"({100 * len(fl & iso) / max(len(fl), 1):.0f}% of flow)"
        )

    mtag = f"{args.method}_{args.tag}"
    outputs = [
        out_dir / f"desi_outliers_flow_{mtag}_snr3.pt",
        out_dir / f"flow_{mtag}.pt",
        out_dir / f"flow_{mtag}_validation.pdf",
    ]
    clash = [p for p in outputs if p.exists()]
    if clash and not args.force:
        raise SystemExit(
            "refusing to overwrite:\n  "
            + "\n  ".join(str(p) for p in clash)
            + "\npass --force, or --outdir to write elsewhere"
        )

    torch.save(
        {
            "outlier_target_ids": torch.tensor(out_tid),
            "threshold": thr,
            "threshold_all_mocks": thr_all,
            "log_p_desi": lp_desi,
            "desi_target_ids": desi_tid,
            "tag": args.tag,
            "method": args.method,
            "c2st": float(c2st),
            # which inputs produced this. Two mock sets share a filename
            # (noised_cue_meanzero vs ..._wide), so the paths are the only record.
            "mock_file": str(args.mock),
            "desi_file": str(args.desi),
            "encoder": mock_ckpt,
            "seed": args.seed,
        },
        out_dir / f"desi_outliers_flow_{mtag}_snr3.pt",
    )
    torch.save(
        {
            "state_dict": nde.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "hist": hist,
            "dim": D,
            "method": args.method,
            "num_transforms": args.num_transforms,
            "num_bins": args.num_bins,
            "hidden": args.hidden,
            "mock_file": str(args.mock),
            "encoder": mock_ckpt,
            "seed": args.seed,
        },
        out_dir / f"flow_{mtag}.pt",
    )

    # ---- plots ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(hist["train"], label="train")
    ax[0].plot(hist["valid"], label="valid")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("NLL")
    ax[0].legend()
    ax[0].set_title("flow training")
    j = int(np.argmax(ks))
    ax[1].hist(real[:, j], bins=60, density=True, alpha=0.5, label="mock")
    ax[1].hist(samp[:, j], bins=60, density=True, alpha=0.5, label="flow samples")
    ax[1].set_title(f"worst dim {j} (KS={ks[j]:.3f})")
    ax[1].legend()
    bins = np.linspace(
        np.percentile(np.r_[lp_mock, lp_desi], 0.5), np.percentile(lp_mock, 99.5), 80
    )
    ax[2].hist(lp_mock, bins=bins, density=True, alpha=0.5, label="mock")
    ax[2].hist(lp_desi, bins=bins, density=True, alpha=0.5, label="DESI")
    ax[2].axvline(thr, color="k", ls=":")
    ax[2].set_xlabel("log p")
    ax[2].legend()
    ax[2].set_title(f"{mtag}: DESI outliers {out_mask.sum()}")
    fig.tight_layout()
    fig.savefig(out_dir / f"flow_{mtag}_validation.pdf", bbox_inches="tight")
    print(f"\nsaved -> {out_dir}/")
    for p in outputs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
