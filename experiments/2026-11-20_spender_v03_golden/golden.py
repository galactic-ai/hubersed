"""Run fixed spectra through two spender checkpoints and save the outputs for comparison.

All autoencoders share one class, so one spectrum encoder tests the code. The noise spender is
also run because the noise flow needs its latents. Run it once with the pinned spender and
once with the spender merged with upstream v0.3, then compare with analysis.py. It imports only
spender, so it runs in an environment without hubersed. Run from the repository root.
COMMAND: ``uv run python experiments/2026-11-20_spender_v03_golden/golden.py --tag pinned`` and
``<merged-env>/bin/python -I experiments/2026-11-20_spender_v03_golden/golden.py --tag merged``.
OUTPUT: results/2026-11-20_spender_v03_golden/golden_<tag>.npz.
"""

import argparse
import importlib.metadata
import pickle
from pathlib import Path

import numpy as np
import torch
from spender import load_model
from spender.data.desi import DESI
from spender.flow import NeuralDensityEstimator

OUT = Path("results/2026-11-20_spender_v03_golden")
CKPT = Path("data/checkpoints")
AUTOENCODERS = ["spender_asc_run_10latent_zmax", "desi_noise_spender_10latent"]
NOISE = "desi_noise_spender_10latent"
BATCHES = {
    "desi": ["data/desi_spectra/DESIchunk1024_0.pkl"],
    "mock": ["data/prospector_model/noised_cuev2/DESIcuev2_0.pkl"],
}
N_SPECTRA = 256  # first 256 spectra of each batch
N_SPECTRA_OUT = 64  # decode and forward outputs are saved for the first 64 only


def load_batch(paths):
    """Return spec, w, z and norm stacked from spender pickle chunks."""
    parts = [pickle.load(open(p, "rb")) for p in paths]
    spec, w, z = (torch.cat([torch.as_tensor(b[k]) for b in parts]).float() for k in (0, 1, 2))
    norm = torch.cat([torch.as_tensor(b[4]) for b in parts]).float()
    k = slice(0, N_SPECTRA)
    return spec[k], w[k], z[k], norm[k]


def noise_flow():
    """Build the noise flow as hubersed.detect.utils does and load its weights."""
    nde = NeuralDensityEstimator(
        dim=10,
        initial_pos={"bounds": [[0, 0]] * 10, "std": [0.5] * 10},
        hidden_features=64,
        num_transforms=5,
        context_features=1,
    )
    nde.load_state_dict(
        torch.load(CKPT / f"{NOISE}_flow.pt", map_location="cpu", weights_only=False)
    )
    return nde.eval()


def tensors(x):
    """Return the tensors in a model output as a list of float64 arrays."""
    items = x if isinstance(x, (tuple, list)) else [x]
    return [t.detach().double().numpy() for t in items if torch.is_tensor(t)]


def main():
    """Encode, decode and forward each batch with each checkpoint and save everything."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)

    inst = DESI()
    out = {"spender_version": importlib.metadata.version("spender"), "torch": torch.__version__}
    data = {k: load_batch(v) for k, v in BATCHES.items()}
    nde = noise_flow()

    with torch.no_grad():
        for name in AUTOENCODERS:
            model = load_model(
                str(CKPT / f"{name}.pt"), inst, map_location="cpu", weights_only=False
            )
            model = model.float().eval()
            for b, (spec, w, z, norm) in data.items():
                x = spec * torch.sqrt(w) if name == NOISE else spec
                lat = model.encode(x)
                out[f"{name}/{b}/latent"] = lat.double().numpy()
                k = slice(0, N_SPECTRA_OUT)
                out[f"{name}/{b}/decode"] = model.decode(lat[k]).double().numpy()
                for i, t in enumerate(tensors(model(x[k], instrument=inst, z=z[k]))):
                    out[f"{name}/{b}/forward{i}"] = t
                if name == NOISE:
                    logp = nde.log_prob(lat.float(), context=norm.unsqueeze(1))
                    out[f"flow/{b}/logp"] = logp.double().numpy()

    np.savez(OUT / f"golden_{args.tag}.npz", **out)
    print(f"spender {out['spender_version']}, torch {out['torch']}, {len(out) - 2} arrays")


if __name__ == "__main__":
    main()
