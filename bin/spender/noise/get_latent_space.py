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
    snr_min: float = 0.0,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """
    Iterate through `loader`, encode either raw spectra or SNR, and collect outputs.

    If ``snr_min > 0`` keep only spectra whose per-pixel S/N (= spec*sqrt(w),
    median over valid pixels w>0) exceeds ``snr_min``. The encode INPUT is still
    chosen by mode (spec vs snr) -- the S/N cut is decoupled from what is encoded.

    Because rows are dropped, position no longer equals the global index. The
    running global index (chunk*1024+row order, loader shuffle=False) of every
    KEPT spectrum is tracked and returned so downstream code (outliers, map_chi2)
    can still map back to the original catalogue.

    Returns:
        latents, A, specs, snrs_or_none, zs, indices
    """
    all_latents, all_A, all_specs, all_snrs, all_z, all_idx = [], [], [], [], [], []
    offset = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            spec, w, z, target_id, norm, zerr = batch   # CPU tensors from loader
            B = spec.shape[0]
            g = torch.arange(offset, offset + B)         # global indices for this batch
            offset += B

            # per-pixel S/N on CPU (norm cancels); needed for the cut and/or storage
            snr_cpu = spec * torch.sqrt(w) if (compute_snr or snr_min > 0) else None

            # S/N cut on CPU (torch.nanmedian unimplemented on MPS), BEFORE encode.
            # Filtering pre-encode is identical to post-encode (encoder is per-spectrum)
            # but skips encoding discarded rows and avoids a mid-loop GPU->CPU sync.
            if snr_min > 0:
                snr_pix = torch.where(w > 0, snr_cpu, torch.nan)
                med_snr = torch.nanmedian(snr_pix, dim=1).values
                keep = med_snr > snr_min
            else:
                keep = torch.ones(B, dtype=torch.bool)

            if not keep.any():
                if (i + 1) % 50 == 0:
                    print(f"Processed {offset} spectra (kept {sum(int(x.shape[0]) for x in all_idx)})",
                          end="\r", flush=True)
                continue

            # keep-only, then move just survivors to device and encode
            spec_k = spec[keep].float().to(device)
            
            # encode input is mode-driven: snr only when compute_snr (noise mode)
            to_encode = (snr_cpu[keep].float().to(device)) if compute_snr else spec_k
            s = model.encode(to_encode)

            all_latents.append(s.cpu())
            all_A.append(norm[keep].unsqueeze(1))
            all_z.append(z[keep])
            all_specs.append(spec[keep].half())
            all_idx.append(g[keep])
            if compute_snr:
                all_snrs.append(snr_cpu[keep])

            if (i + 1) % 50 == 0:
                kept = sum(int(x.shape[0]) for x in all_idx)
                print(f"Processed {offset} spectra (kept {kept})", end="\r", flush=True)

    latents = torch.cat(all_latents, dim=0)
    A = torch.cat(all_A, dim=0)
    specs = torch.cat(all_specs, dim=0)
    snrs = torch.cat(all_snrs, dim=0) if all_snrs else None
    zs = torch.cat(all_z, dim=0)
    indices = torch.cat(all_idx, dim=0)

    return latents, A, specs, snrs, zs, indices

def save_output(out, path):
    if path.startswith("hf://"):
        from huggingface_hub import hffs
        # hf://buckets/nikhil0504/... → buckets/nikhil0504/...
        with hffs.open(path.replace("hf://", ""), 'wb') as f:
            torch.save(out, f)
    else:
        torch.save(out, path)

def main(args: argparse.Namespace) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    inst = desi.DESI()

    # Build wave_rest according to selected mode
    model = load_model(args.checkpoint, inst, map_location="cpu", weights_only=False, mmap=True).float().to(device)
    
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

    latents, A, specs, snrs, zs, indices = process_loader(
        model, loader, device, compute_snr=compute_snr, snr_min=args.snr_min
    )

    print("Latents shape:", latents.shape, "(kept after S/N cut)" if args.snr_min > 0 else "")
    print("A shape:", A.shape)

    out = {
        "latents": latents,
        "A": A,
        "zs": zs,
        # keep both keys for backward compatibility; caller can choose which to use
        "specs": specs,
        "snrs": snrs,
        "indices": indices,   # global catalogue index of each kept spectrum
        "meta": {
            "mode": args.mode,
            "zmax": args.zmax,
            "tag": tag,
            "snr_min": args.snr_min,
        },
    }

    save_output(out, args.outfile)
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
    parser.add_argument(
        "--snr_min",
        type=float,
        default=0.0,
        help="Keep only spectra with median per-pixel S/N > this (0 = no cut). "
             "Stores 'indices' = global catalogue index of each kept spectrum.",
    )

    args = parser.parse_args()
    main(args)
