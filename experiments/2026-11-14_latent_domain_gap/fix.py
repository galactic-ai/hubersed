"""Do the non-physical mock input differences cause the DESI-mock separation in latent space?

QUESTION: if 20,000 mocks get noise at the sky-line mask, the missing pixel and DESI-like zero
pixels, does their neighbour mixing with DESI rise from ~0.01 toward 0.5?
HYPOTHESIS: none registered. A large rise means preprocessing drives the separation.
INPUTS: data/prospector_model/noised_cuev2/DESIcuev2_<k>.pkl, data/dataset_z_snr.npz,
data/prospector_model/mocks_v2_500000.h5, data/latents/spender_spec_<tag>latent_snr3.h5,
data/latents/mocks_v2/prospector_noise_spec_cuev2_<tag>latent_snr3.h5, data/checkpoints.
SEED: 0 (samples), 1 (noise and zero patterns).
COMMAND: ``uv run --no-sync python experiments/2026-11-14_latent_domain_gap/fix.py``.
RESULT: 2026-11-14, fix.txt. The fixes barely change mixing. DESI other-set fraction stored
0.003-0.013, arm 1 0.005-0.015, arm 2 0.005-0.017.
Assumptions: interpolated sky-pixel sigma (no DESI sky-line excess); zero patterns from random DESI.
FIGURES: none (fix.txt).
"""

import pickle

import astropy.units as u
import h5py
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from spender import load_model
from spender.data.desi import DESI

from hubersed.conversion import to_flambda
from hubersed.io.latents import load_latents
from hubersed.paths import PATHS

RES, DATA = PATHS["RESULTS"], PATHS["DATA"]
OUT = RES / "2026-11-14_latent_domain_gap"
FLOWS = RES / "2026-11-09_mocks_v2_outliers" / "flows"
NOISED = DATA / "prospector_model" / "noised_cuev2"
MOCKS = DATA / "prospector_model" / "mocks_v2_500000.h5"
TAGS = ("6", "10", "15", "cont10", "cont15")
N, K = 20000, 15
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def mock_inputs(idx):
    """Return the pickled spectra, ivar and norms of mock indices ``idx`` (sorted)."""
    spec, ivar, norm = [], [], []
    for f in np.unique(idx // 1024):
        with open(NOISED / f"DESIcuev2_{f}.pkl", "rb") as fh:
            s, iv, _, tid, nm, _ = pickle.load(fh)
        rows = idx[idx // 1024 == f] - f * 1024
        assert np.array_equal(np.asarray(tid)[rows], idx[idx // 1024 == f])
        spec.append(np.asarray(s)[rows])
        ivar.append(np.asarray(iv)[rows])
        norm.append(np.asarray(nm)[rows])
    return np.concatenate(spec), np.concatenate(ivar), np.concatenate(norm)


def sky_sigma(ivar, sky):
    """Return sigma per pixel with sky-mask pixels interpolated from the other pixels."""
    sig = np.where(ivar > 0, 1 / np.sqrt(np.clip(ivar, 1e-30, None)), 0.0)
    pix = np.arange(sig.shape[1])
    for i in range(sig.shape[0]):
        ok = (~sky) & (sig[i] > 0)
        if ok.sum() > 1:
            sig[i, sky] = np.interp(pix[sky], pix[ok], sig[i, ok])
    return sig


def encode(ckpt, x):
    """Return the latents of spectra ``x`` with encoder ``ckpt``."""
    enc = load_model(
        str(DATA / "checkpoints" / ckpt), DESI(), map_location="cpu", weights_only=False
    )
    enc = enc.float().eval().to(DEVICE)
    out = []
    with torch.no_grad():
        for k in range(0, len(x), 512):
            out.append(
                enc.encode(torch.from_numpy(x[k : k + 512].astype(np.float32)).to(DEVICE))
                .cpu()
                .numpy()
            )
    return np.concatenate(out)


def mixing(xm, xd):
    """Return the mean other-set neighbour fraction of DESI and of mocks."""
    pool = np.vstack([xm, xd])
    is_mock = np.r_[np.ones(len(xm), bool), np.zeros(len(xd), bool)]
    nb = NearestNeighbors(n_neighbors=K + 1).fit(pool).kneighbors(pool)[1][:, 1:]
    f = is_mock[nb].mean(1)
    return f[~is_mock].mean(), 1 - f[is_mock].mean()


def main():
    """Build the arms, check the null, write the mixing per flow."""
    OUT.mkdir(parents=True, exist_ok=True)
    sky = DESI._skyline_mask.numpy()
    _, mtid, _ = load_latents(
        DATA / "latents" / "mocks_v2" / "prospector_noise_spec_cuev2_cont10latent_snr3.h5"
    )
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(np.asarray(mtid, np.int64), N, replace=False))
    spec, ivar, norm = mock_inputs(pick)
    print(f"loaded {len(pick)} mock inputs", flush=True)
    sig = sky_sigma(ivar, sky[: spec.shape[1]])
    rng1 = np.random.default_rng(1)
    noise = rng1.normal(0.0, 1.0, spec.shape) * sig
    arm1 = spec.copy()
    s7780 = sky[: spec.shape[1]]
    arm1[:, s7780] += noise[:, s7780]
    wave = DESI._wave_obs.numpy().astype(float)
    with h5py.File(MOCKS, "r") as h:
        last = h["fluxes"][pick, -1].astype(float)
    last = to_flambda(wave[-1:] * u.AA, last[:, None] * u.mgy).value[:, 0] / norm
    last = last + rng1.normal(0.0, 1.0, last.size) * sig[:, -1]
    arm2 = np.hstack([arm1, last[:, None]])
    with h5py.File(DATA / "latents" / "spender_spec_cont10latent_snr3.h5", "r") as h:
        nd = h["specs"].shape[0]
        rows = np.sort(rng1.choice(nd, N, replace=False))
        zmask = h["specs"][rows] == 0
    arm2[zmask[rng1.permutation(N)]] = 0.0
    zs = np.load(DATA / "dataset_z_snr.npz")
    snr_of = dict(zip(zs["tid"].tolist(), zs["snr"].tolist(), strict=True))
    lines = [
        f"{N} mocks; sky-mask pixels {s7780.sum()}; arm 2 zero pixels per spectrum mean {zmask.sum(1).mean():.1f}"
    ]
    for tag in TAGS:
        ck = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu")
        mu, sd = ck["scaler_mean"], ck["scaler_scale"]
        stored, stid, _ = load_latents(
            DATA / "latents" / "mocks_v2" / f"prospector_noise_spec_cuev2_{tag}latent_snr3.h5"
        )
        pos = {int(t): i for i, t in enumerate(stid)}
        st = stored[[pos[int(t)] for t in pick]]
        desi, dtid, _ = load_latents(DATA / "latents" / f"spender_spec_{tag}latent_snr3.h5")
        d_snr = np.array([snr_of.get(int(t), np.nan) for t in dtid])
        di = np.random.default_rng(0).choice(np.flatnonzero(np.isfinite(d_snr)), N, replace=False)
        xd = (desi[di] - mu) / sd
        base = encode(ck["encoder"], spec)
        null = np.linalg.norm((base - st) / sd, axis=1)
        lines.append(
            f"\n{tag}latent: null |re-encoded - stored| (scaler units) median {np.median(null):.4f}, max {null.max():.4f}"
        )
        for lab, lat in (("stored", st), ("arm 1 sky noise", encode(ck["encoder"], arm1)),
                         ("arm 2 + last pixel + zeros", encode(ck["encoder"], arm2))):  # fmt: skip
            fd, fm = mixing((lat - mu) / sd, xd)
            lines.append(f"  {lab:28}: other-set fraction DESI {fd:.3f}, mocks {fm:.3f}")
        print(f"{tag} done", flush=True)
    text = "\n".join(lines)
    (OUT / "fix.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
