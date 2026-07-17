"""
spender reconstruction-error test: are the Cue mocks out-of-distribution to the
DESI-trained encoder?

spender was trained on DESI only. If mocks are OOD inputs, the encoder pushes them
to a different latent region and the latent-space C-2ST (A=0.94) is an encoder
artifact, NOT a statement about Cue physics. Direct check: decode the stored
latents back through the decoder, compare the reconstruction to the stored input
spectrum, and contrast the per-spectrum reconstruction error of mocks vs DESI.

  mock recon  >>  DESI recon  -> mocks OOD to encoder; latent positions unreliable;
                                 A=0.94 is mostly encoder domain shift.
  mock recon  ~=  DESI recon  -> encoder reconstructs mocks as well as data;
                                 the latent offset is a REAL (modest) mock-data diff.

Uses stored (latents, specs, zs) directly via model._forward(s=latents) -- no
re-encode. Reconstruction error = median over valid pixels of |spec-recon|/|spec|.

Usage (from bin/prospector/):
    python recon_error.py --checkpoint ../../../spender/spender_asc_run_6latent_zmax_em_mask.pt --n 3000
(point --checkpoint at the SAME 6-latent checkpoint used to make the latent files)
"""

import argparse
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spender import load_model
from spender.data import desi

from hubersed.paths import PATHS

DATA, RES = PATHS["DATA"], PATHS["RESULTS"]


def frac_resid(model, inst, latents, specs, zs, device, batch=256):
    """Per-spectrum median |spec - recon| / |spec| over valid pixels."""
    out = []
    n = latents.shape[0]
    for i in range(0, n, batch):
        s = latents[i : i + batch].float().to(device)
        y = specs[i : i + batch].float().to(device)
        z = zs[i : i + batch].float().to(device)
        with torch.no_grad():
            _, _, recon, valid = model._forward(
                y, instrument=inst, z=z, s=s, normalize=True, weights=None
            )
        v = valid.bool() & torch.isfinite(recon) & torch.isfinite(y) & (y.abs() > 0)
        fr = torch.where(v, (y - recon).abs() / y.abs(), torch.nan)
        med = torch.nanmedian(fr, dim=1).values
        out.append(med.cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("device", device)

    inst = desi.DESI()
    model = (
        load_model(
            args.checkpoint, inst, map_location="cpu", weights_only=False, mmap=True
        )
        .float()
        .to(device)
        .eval()
    )

    d = torch.load(DATA / "spender_spec_6latent", map_location="cpu", mmap=True)
    m = torch.load(
        DATA / "prospector_noise_spec_6latent_cue", map_location="cpu", mmap=True
    )

    def sub(f):
        ntot = f["latents"].shape[0]
        idx = np.sort(rng.choice(ntot, min(args.n, ntot), replace=False))
        return (f["latents"][idx], f["specs"][idx], f["zs"][idx])

    dl, dsp, dz = sub(d)
    ml, msp, mz = sub(m)

    dfr = frac_resid(model, inst, dl, dsp, dz, device)
    mfr = frac_resid(model, inst, ml, msp, mz, device)

    def stat(x):
        return np.round(np.nanpercentile(x, [16, 50, 84]), 4)

    print(f"\nrecon frac-resid (median per spectrum):")
    print(f"  DESI  p16,50,84 = {stat(dfr)}   n={np.isfinite(dfr).sum()}")
    print(f"  mock  p16,50,84 = {stat(mfr)}   n={np.isfinite(mfr).sum()}")
    rd, rm = np.nanmedian(dfr), np.nanmedian(mfr)
    print(f"  ratio mock/DESI (median) = {rm / rd:.2f}")
    print(
        "  -> >~1.5 suggests mocks OOD to encoder (A=0.94 = artifact);"
        " ~1.0 means encoder fits mocks fine (offset is real)"
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, np.nanpercentile(np.r_[dfr, mfr], 99), 60)
    ax.hist(dfr, bins=bins, density=True, alpha=0.5, label=f"DESI (med {rd:.3f})")
    ax.hist(mfr, bins=bins, density=True, alpha=0.5, label=f"mock (med {rm:.3f})")
    ax.set_xlabel("median |spec-recon|/|spec| per spectrum")
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title("spender reconstruction error: mock vs DESI")
    fig.tight_layout()
    fig.savefig(RES / "recon_error.pdf", bbox_inches="tight")
    np.savez(RES / "recon_error_summary.npz", desi=dfr, mock=mfr)
    print(f"\nsaved -> {RES / 'recon_error.pdf'}")


if __name__ == "__main__":
    main()
