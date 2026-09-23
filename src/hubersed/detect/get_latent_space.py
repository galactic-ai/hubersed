#!/usr/bin/env python
"""Encode DESI spectra with a trained spender autoencoder and stream the latents to HDF5.

Run it as ``python -m hubersed.detect.get_latent_space DATADIR CHECKPOINT OUTFILE.h5``.
"""

import argparse
from pathlib import Path

import h5py
import torch
from spender import load_model
from spender.data import desi


def process_loader_h5(
    model,
    loader,
    device,
    outfile,
    encode_snr=False,
    store_snr=False,
    snr_min=0.0,
    meta=None,
):
    """Encode every batch from a loader and append the results to an HDF5 file.

    The per-pixel S/N is spec times sqrt(w). Its median over pixels with w > 0 is
    stored as ``snr_med`` whenever S/N is computed. Every kept spectrum is saved with
    its TARGETID in ``target_ids``. Match rows to a catalogue by TARGETID, never by
    row position, because the loader order is not the numeric chunk and row order.

    Parameters
    ----------
    model : torch.nn.Module
        Spender model with an ``encode`` method.
    loader : iterable
        Yields batches of (spec, w, z, target_id, norm, zerr) CPU tensors.
    device : torch.device
        Device the encoder runs on.
    outfile : str or pathlib.Path
        HDF5 file to create. An existing file is overwritten.
    encode_snr : bool, optional
        If True, feed the S/N array to the encoder instead of the spectrum.
    store_snr : bool, optional
        If True, also save the per-pixel S/N arrays as ``snrs``. This is
        independent of ``encode_snr``.
    snr_min : float, optional
        Keep only spectra whose median S/N is above this value. 0 keeps all.
    meta : dict or None, optional
        Written as file attributes. None values are stored as the string "None".

    Returns
    -------
    int
        Number of spectra written.
    """
    f = h5py.File(outfile, "w")
    d = {}
    seen = 0
    kept = 0

    def append(name, arr):
        """Grow dataset ``name`` along axis 0 and write ``arr`` at the end."""
        ds = d[name]
        n = ds.shape[0]
        ds.resize(n + arr.shape[0], axis=0)
        ds[n:] = arr

    with torch.no_grad():
        for i, batch in enumerate(loader):
            spec, w, z, target_id, norm, zerr = batch  # CPU tensors
            B = spec.shape[0]
            seen += B

            need_snr = encode_snr or store_snr or snr_min > 0
            snr_cpu = spec * torch.sqrt(w) if need_snr else None
            # Median per-pixel S/N over good pixels.
            med = (
                torch.nanmedian(torch.where(w > 0, snr_cpu, torch.nan), dim=1).values
                if snr_cpu is not None
                else None
            )
            keep = (med > snr_min) if snr_min > 0 else torch.ones(B, dtype=torch.bool)
            if not keep.any():
                if (i + 1) % 50 == 0:
                    print(f"Processed {seen} (kept {kept})", end="\r", flush=True)
                continue

            spec_k = spec[keep].float().to(device)
            to_encode = (snr_cpu[keep].float().to(device)) if encode_snr else spec_k
            lat = model.encode(to_encode).cpu().numpy().astype("float32")

            zk = z[keep].numpy().astype("float32").reshape(-1)
            Ak = norm[keep].unsqueeze(1).numpy().astype("float32")
            tk = target_id[keep].numpy().astype("int64")
            spk = spec[keep].half().numpy()
            snk = snr_cpu[keep].numpy().astype("float32") if store_snr else None

            if not d:  # lazily create datasets
                nlat, L = lat.shape[1], spk.shape[1]

                def mk(nm, c, dt):
                    return f.create_dataset(
                        nm, shape=(0,) + c, maxshape=(None,) + c, dtype=dt, chunks=True
                    )

                d["latents"] = mk("latents", (nlat,), "float32")
                d["zs"] = mk("zs", (), "float32")
                d["A"] = mk("A", (1,), "float32")
                d["target_ids"] = mk("target_ids", (), "int64")
                d["specs"] = mk("specs", (L,), "float16")
                if med is not None:
                    # one float per spectrum: makes "compare the S/N distribution of
                    # retained mocks vs retained DESI" a read, not a re-encode.
                    d["snr_med"] = mk("snr_med", (), "float32")
                if store_snr:
                    d["snrs"] = mk("snrs", (L,), "float32")

            append("latents", lat)
            append("zs", zk)
            append("A", Ak)
            append("target_ids", tk)
            append("specs", spk)

            if med is not None:
                append("snr_med", med[keep].numpy().astype("float32"))
            if store_snr:
                append("snrs", snk)
            kept += lat.shape[0]

            if device.type == "mps" and (i + 1) % 20 == 0:
                torch.mps.empty_cache()

            if (i + 1) % 50 == 0:
                print(f"Processed {seen} (kept {kept})", end="\r", flush=True)

    for k, v in (meta or {}).items():
        f.attrs[k] = "None" if v is None else v
    f.close()
    return kept


def main(args: argparse.Namespace) -> None:
    """Load the model and DESI data, then write latents to ``args.outfile``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Raises
    ------
    RuntimeError
        If ``args.outfile`` does not end with ".h5".
    """
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    inst = desi.DESI()

    model = (
        load_model(args.checkpoint, inst, map_location="cpu", weights_only=False, mmap=True)
        .float()
        .to(device)
    )
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

    # Two independent settings, deliberately not one flag. encode_snr feeds the S/N
    # array to the encoder instead of the spectrum and is only for mode noise.
    # store_snr also saves the per-pixel S/N arrays alongside the latents.
    encode_snr = args.mode == "noise"
    store_snr = args.compute_snr or encode_snr

    if not str(args.outfile).endswith(".h5"):
        raise RuntimeError(
            f"warning: output is HDF5 format; '{args.outfile}' does not end with .h5"
        )
    meta = {
        "mode": args.mode,
        "zmax": args.zmax,
        "tag": tag,
        "snr_min": args.snr_min,
        # Records which encoder produced these latents, since latents from
        # different encoders are not interchangeable.
        "checkpoint": Path(args.checkpoint).name,
    }
    n = process_loader_h5(
        model,
        loader,
        device,
        args.outfile,
        encode_snr=encode_snr,
        store_snr=store_snr,
        snr_min=args.snr_min,
        meta=meta,
    )
    print(f"\nSaved {n} spectra (streamed HDF5) to {args.outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate latent space (spectra or noise) from trained autoencoder"
    )
    parser.add_argument("datadir", type=str, help="Directory containing DESI training data")
    parser.add_argument("checkpoint", type=str, help="Path to the trained autoencoder checkpoint")
    parser.add_argument("outfile", type=str, help="Output file to save the latent representations")
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
        default="spec",
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
