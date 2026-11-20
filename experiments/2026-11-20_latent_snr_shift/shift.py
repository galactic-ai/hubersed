"""Noise the same mocks at fixed S/N levels and save their cont10 latents and outlier-flow log p.

Changed from scripts/make_prospector_noisy_sed.py: each mock keeps one S/N curve drawn from the
noise model, rescaled to a median S/N of 3, 5, 10 and 20, with 5 noise draws per level. The rest
of the noising (sky pixels, sigma cap, Gaussian noise) follows that script.
COMMAND: ``uv run python experiments/2026-11-20_latent_snr_shift/shift.py``.
OUTPUT: results/2026-11-20_latent_snr_shift/shift.npz.
"""

import astropy.units as u
import h5py
import numpy as np
import torch
from spender import load_model
from spender.data.desi import DESI
from spender.instrument import get_skyline_mask

from hubersed.conversion import to_flambda
from hubersed.detect.flow_model import build_flow
from hubersed.detect.quantities import normalize_spectra
from hubersed.detect.utils import load_models
from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-20_latent_snr_shift"
MOCKS = PATHS["DATA"] / "prospector_model" / "mocks_v2_500000.h5"
CKPT = PATHS["DATA"] / "checkpoints"
FLOW = PATHS["RESULTS"] / "noised_cue_meanzero_wide_flow" / "flow_nsf_cont10latent.pt"
N_MOCKS = 500
LEVELS = [3.0, 5.0, 10.0, 20.0]
N_DRAWS = 5
SEED = 0


def load_flow():
    """Return the cont10 outlier flow, its scaler and the path of the mocks it was fit to."""
    b = torch.load(FLOW, map_location="cpu", weights_only=False)
    nde = build_flow(b["method"], b["dim"], b["hidden"], b["num_transforms"], b["num_bins"])
    nde.load_state_dict(b["state_dict"])
    return nde.eval(), np.asarray(b["scaler_mean"]), np.asarray(b["scaler_scale"]), b["mock_file"]


def log_p(nde, mean, scale, lat):
    """Return the flow log p of latents after the flow's standardization."""
    x = torch.from_numpy(((lat - mean) / scale).astype(np.float32))
    with torch.no_grad():
        return nde.log_prob(x).numpy()


def main():
    """Noise, encode and score the mocks at each S/N level and save everything."""
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)

    with h5py.File(MOCKS, "r") as h:
        ok = np.flatnonzero(~h["failed"][:])
        idx = np.sort(rng.choice(ok, N_MOCKS, replace=False))
        flux = h["fluxes"][idx]
        wave = h["wavelength"][:]
        z = torch.tensor(h["priors/redshifts"][idx])
    flam = torch.from_numpy(to_flambda(wave * u.AA, flux * u.mgy).value)
    flam_norm, norms, good = normalize_spectra(flam, z, torch.tensor(wave), inplace=False)
    flam_norm = flam_norm[:, :-1].float()

    inst = DESI()
    sky = get_skyline_mask(inst._wave_obs)[:-1]
    nde_noise, noise_spender = load_models(
        CKPT / "desi_noise_spender_10latent_flow.pt",
        str(CKPT / "desi_noise_spender_10latent.pt"),
        flow_latent=10,
        instrument=inst,
        map_location="cpu",
        weights_only=False,
    )
    with torch.no_grad():
        s = nde_noise.sample(1, context=norms.unsqueeze(1).float()).permute(1, 0, 2).float()
        snr = noise_spender.eval().decode(s).squeeze(0).clone()
    snr[:, sky] = float("nan")
    snr[snr <= 0] = float("nan")
    snr_med0 = torch.nanmedian(snr, dim=1).values

    enc = load_model(
        str(CKPT / "spender_desi_cont_10latent_zmax.pt"),
        inst,
        map_location="cpu",
        weights_only=False,
    )
    enc = enc.float().eval()
    nde, mean, scale, mock_file = load_flow()

    lat = np.zeros((len(LEVELS), N_DRAWS, N_MOCKS, 10), np.float32)
    snr_meas = np.zeros((len(LEVELS), N_DRAWS, N_MOCKS), np.float32)
    for i, level in enumerate(LEVELS):
        snr_l = snr * (level / snr_med0)[:, None]
        sigma = torch.nan_to_num((flam_norm / snr_l).abs(), nan=0.0, posinf=0.0, neginf=0.0)
        cap = torch.nanquantile(sigma, 0.995, dim=1, keepdim=True)
        sigma = torch.minimum(sigma, cap)
        w = torch.where(sigma > 0, 1.0 / sigma**2, torch.zeros_like(sigma))
        for k in range(N_DRAWS):
            f = flam_norm + torch.normal(mean=0.0, std=sigma, generator=gen)
            m = f * torch.sqrt(w)
            snr_meas[i, k] = torch.nanmedian(torch.where(w > 0, m, torch.nan), dim=1).values.numpy()
            with torch.no_grad():
                lat[i, k] = enc.encode(f).numpy()
            print(f"S/N {level:g} draw {k}: median measured S/N {np.median(snr_meas[i, k]):.2f}")

    logp = np.stack(
        [[log_p(nde, mean, scale, lat[i, k]) for k in range(N_DRAWS)] for i in range(len(LEVELS))]
    )
    with h5py.File(PATHS["ROOT"] / mock_file, "r") as h:
        lp_ref = log_p(nde, mean, scale, h["latents"][:])

    np.savez(
        OUT / "shift.npz",
        levels=np.array(LEVELS),
        mock_index=idx,
        good_norm=good.numpy(),
        latents=lat,
        logp=logp,
        snr_measured=snr_meas,
        scaler_mean=mean,
        scaler_scale=scale,
        logp_ref_mocks=lp_ref,
    )
    print(f"saved {OUT / 'shift.npz'}")


if __name__ == "__main__":
    main()
