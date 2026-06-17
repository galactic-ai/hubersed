#!/usr/bin/env python
import argparse

import h5py
import torch
from spender.data import desi
from spender import load_model


def process_loader_h5(model, loader, device, outfile, compute_snr=False,
                      snr_min=0.0, meta=None):
    f = h5py.File(outfile, 'w')
    d = {}
    seen = 0
    kept = 0

    def append(name, arr):
        ds = d[name]; n = ds.shape[0]
        ds.resize(n + arr.shape[0], axis=0); ds[n:] = arr

    with torch.no_grad():
        for i, batch in enumerate(loader):
            spec, w, z, target_id, norm, zerr = batch     # CPU tensors
            B = spec.shape[0]; seen += B

            snr_cpu = spec * torch.sqrt(w) if (compute_snr or snr_min > 0) else None
            if snr_min > 0:
                med = torch.nanmedian(torch.where(w > 0, snr_cpu, torch.nan), dim=1).values
                keep = med > snr_min
            else:
                keep = torch.ones(B, dtype=torch.bool)
            if not keep.any():
                if (i + 1) % 50 == 0:
                    print(f"Processed {seen} (kept {kept})", end="\r", flush=True)
                continue

            spec_k = spec[keep].float().to(device)
            to_encode = (snr_cpu[keep].float().to(device)) if compute_snr else spec_k
            lat = model.encode(to_encode).cpu().numpy().astype('float32')

            zk = z[keep].numpy().astype('float32').reshape(-1)
            Ak = norm[keep].unsqueeze(1).numpy().astype('float32')
            tk = target_id[keep].numpy().astype('int64')
            spk = spec[keep].half().numpy()
            snk = snr_cpu[keep].numpy().astype('float32') if compute_snr else None

            if not d:                                       # lazily create datasets
                nlat, L = lat.shape[1], spk.shape[1]
                mk = lambda nm, c, dt: f.create_dataset(nm, shape=(0,) + c, maxshape=(None,) + c, dtype=dt, chunks=True)
                d['latents'] = mk('latents', (nlat,), 'float32')
                d['zs'] = mk('zs', (), 'float32')
                d['A'] = mk('A', (1,), 'float32')
                d['target_ids'] = mk('target_ids', (), 'int64')
                d['specs'] = mk('specs', (L,), 'float16')
                if compute_snr:
                    d['snrs'] = mk('snrs', (L,), 'float32')

            append('latents', lat); append('zs', zk); append('A', Ak)
            append('target_ids', tk); append('specs', spk)

            if compute_snr:
                append('snrs', snk)
            kept += lat.shape[0]

            if device.type == 'mps' and (i + 1) % 20 == 0:
                torch.mps.empty_cache()

            if (i + 1) % 50 == 0:
                print(f"Processed {seen} (kept {kept})", end="\r", flush=True)

    for k, v in (meta or {}).items():
        f.attrs[k] = 'None' if v is None else v
    f.close()
    return kept

def main(args: argparse.Namespace) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    inst = desi.DESI()

    model = load_model(args.checkpoint, inst, map_location="cpu", weights_only=False, mmap=True).float().to(device)
    model.eval()

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

    if not str(args.outfile).endswith('.h5'):
        raise RuntimeError(f"warning: output is HDF5 format; '{args.outfile}' does not end with .h5")
    meta = {"mode": args.mode, "zmax": args.zmax, "tag": tag, "snr_min": args.snr_min}
    n = process_loader_h5(model, loader, device, args.outfile,
                          compute_snr=compute_snr, snr_min=args.snr_min, meta=meta)
    print(f"\nSaved {n} spectra (streamed HDF5) to {args.outfile}")


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
             "Each kept spectrum is tagged with its TARGETID (target_ids) for catalogue matching.",
    )

    args = parser.parse_args()
    main(args)
