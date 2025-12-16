#!/usr/bin/env python

import argparse
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from spender import SpectrumAutoencoder
from spender.data import desi
from spender import load_model
from contextlib import nullcontext

def process_loader(
    model: SpectrumAutoencoder,
    loader: DataLoader,
    device: torch.device,
    compute_snr: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
]:
    """
    Iterate through `loader`, encode either raw spectra or SNR, and collect outputs.

    Returns:
        latents, A, specs_or_none, snrs_or_none, zs
    """
    model.eval()
    n_total = len(loader.dataset)

    # infer dims
    first = next(iter(loader))
    spec0, w0, z0, target_id0, norm0, zerr0 = first
    L = spec0.shape[1]

    amp_ctx = (
        torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()

    )

    with torch.inference_mode(), amp_ctx:
        spec0 = spec0.to(device)
        w0 = w0.to(device)
        x0 = spec0 * torch.sqrt(w0) if compute_snr else spec0
        latent_dim = model.encode(x0[:1].float()).shape[1]
    
    latents = torch.empty((n_total, latent_dim), dtype=torch.float32, device="cpu")
    A = torch.empty((n_total, 1), dtype=torch.float32, device="cpu")
    zs = torch.empty((n_total, ), dtype=torch.float32, device="cpu")
    
    specs = torch.empty((n_total, L), dtype=torch.float32, device="cpu")
    snrs = torch.empty((n_total, L), dtype=torch.float32, device="cpu")

    idx = 0

    with torch.inference_mode(), amp_ctx:
        for i, batch in enumerate(loader):
            spec, w, z, target_id, norm, zerr = batch
            bsz = spec.shape[0]

            specs[idx:idx+bsz] = spec

            spec_gpu = spec.to(device, non_blocking=True)
            w_gpu    = w.to(device, non_blocking=True)
            z_gpu    = z.to(device, non_blocking=True)

            if compute_snr:
                snr_gpu = spec_gpu * torch.sqrt(w_gpu)
                to_encode = snr_gpu
                snrs[idx:idx+bsz] = snr_gpu.detach().cpu()
            else: 
                to_encode = spec_gpu
                snrs[idx:idx+bsz].zero_()
            
            s = model.encode(to_encode.float())
            
            latents[idx:idx+bsz] = s.detach().cpu()
            A[idx:idx+bsz]       = norm.unsqueeze(1).cpu()
            zs[idx:idx+bsz]      = z_gpu.detach().cpu().view(-1)

            idx += bsz

            if (i + 1) % 50 == 0:
                print(f"Processed {idx} spectra")

    return latents, A, specs, snrs, zs


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    inst = desi.DESI()

    # Build wave_rest according to selected mode
    model = load_model(args.checkpoint, inst, map_location=device, weights_only=False)
    
    # data loader: allow tag override like original spec script
    tag = args.tag or "chunk1024"
    loader = inst.get_data_loader(
        args.datadir,
        tag=tag,
        which="all",
        batch_size=args.batch_size,
        shuffle=False,
        shuffle_instance=False,
    )

    # Decide whether to compute SNRs; for 'noise' mode we compute them by default
    compute_snr = args.compute_snr or (args.mode == "noise")

    latents, A, specs, snrs, zs = process_loader(
        model.to(device), loader, device, compute_snr=compute_snr
    )

    print("Latents shape:", latents.shape)
    print("A shape:", A.shape)

    out = {
        "latents": latents,
        "A": A,
        "zs": zs,
        # keep both keys for backward compatibility; caller can choose which to use
        "specs": specs,
        "snrs": snrs,
        "meta": {
            "mode": args.mode,
            "zmax": args.zmax,
            "tag": tag,
        },
    }

    torch.save(out, args.outfile)
    print(f"Saved latents to {args.outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate latent space (spectra or noise) from trained autoencoder"
    )
    parser.add_argument(
        "datadir", type=str, help="Directory containing DESI training data"
    )
    parser.add_argument(
        "checkpoint", type=str, help="Path to the trained autoencoder checkpoint"
    )
    parser.add_argument(
        "outfile", type=str, help="Output file to save the latent representations"
    ) 
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for processing spectra",
    )
    parser.add_argument(
        "--zmax",
        type=float,
        default=0.0,
        help="Maximum redshift for rest-frame wavelength calculation",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="chunk1024",
        help="Data tag to load from DESI data directory",
    )
    parser.add_argument(
        "--mode",
        choices=("spec", "noise"),
        default="noise",
        help="Mode to run: 'spec' for rest-frame spectra encoding, 'noise' for noise/SNR encoding",
    )
    parser.add_argument(
        "--compute_snr",
        action="store_true",
        help="Compute SNRs (spec * sqrt(w)). Always enabled for --mode noise.",
    )

    args = parser.parse_args()
    main(args)
