#!/usr/bin/env python

import argparse
from typing import Optional, Tuple

import torch
from spender import SpectrumAutoencoder
from spender.data import desi
from spender import load_model

def process_loader(
    model: SpectrumAutoencoder,
    loader,
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
    all_latents = []
    all_A = []
    all_specs = []
    all_snrs = []
    all_z = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            spec, w, z, target_id, norm, zerr = batch

            spec = spec.to(device)
            w = w.to(device)
            z = z.to(device)

            snr = None
            if compute_snr:
                snr = spec * torch.sqrt(w)

            # Decide what to encode: prefer SNR if computed, otherwise raw spectrum
            to_encode = snr if (snr is not None) else spec

            s = model.encode(to_encode.float())
            all_latents.append(s.cpu())
            all_A.append(norm.unsqueeze(1).cpu())
            all_z.append(z.cpu())

            # store both representations so outputs are uniform
            all_specs.append(spec.cpu())
            all_snrs.append(
                snr.cpu() if snr is not None else torch.zeros_like(spec.cpu())
            )

            if (i + 1) % 50 == 0:
                print(f"Processed {(i + 1) * loader.batch_size} spectra")

    latents = torch.cat(all_latents, dim=0)
    A = torch.cat(all_A, dim=0)
    specs = torch.cat(all_specs, dim=0)
    snrs = torch.cat(all_snrs, dim=0)
    zs = torch.cat(all_z, dim=0)

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
