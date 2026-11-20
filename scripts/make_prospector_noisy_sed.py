"""Add DESI-like noise to Prospector mock SEDs and write them as a spender Parquet dataset.

Each mock gets an S/N spectrum drawn from the noise flow and noise spender, conditioned on its
normalization. Run it as ``uv run python scripts/make_prospector_noisy_sed.py --input MOCKS.h5
--outdir data/datasets/<name>``.
"""

import argparse
from pathlib import Path

import astropy.units as u
import h5py
import torch
from spender.data import desi
from spender.data.dataset import get_features, write_dataset
from spender.instrument import get_skyline_mask

from hubersed.conversion import to_flambda
from hubersed.detect.quantities import normalize_spectra
from hubersed.detect.utils import load_models
from hubersed.paths import PATHS

DATA_PATH = PATHS["DATA"]

# CUE: noise the Cue mocks into a separate dataset (don't touch the FSPS chain)
CUE = True
OUTDIR = DATA_PATH / "datasets" / ("mocks_cue_noised" if CUE else "mocks_fsps_noised")

# set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# set generator
generator = torch.Generator()
generator.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

flow_file = str(DATA_PATH / "checkpoints" / "desi_noise_spender_10latent_flow.pt")
spender_file = str(DATA_PATH / "checkpoints" / "desi_noise_spender_10latent.pt")
prospector_sed_file = (
    DATA_PATH
    / "prospector_model"
    / (
        "prospector_stochastic_model_seds_cue_500000.h5"
        if CUE
        else "prospector_stochastic_model_seds_500000.h5"
    )
)

# The defaults above stay; the arguments let other mock sets use the same noise.
ap = argparse.ArgumentParser(description="Add DESI-like noise to Prospector mock SEDs.")
ap.add_argument("--input", type=Path, default=prospector_sed_file, help="mock h5 file")
ap.add_argument("--outdir", type=Path, default=OUTDIR, help="Parquet dataset to write")
ap.add_argument("--device", default=str(device), help="cpu or cuda")
args = ap.parse_args()
prospector_sed_file, OUTDIR = args.input, args.outdir
device = torch.device(args.device)
print(f"input {prospector_sed_file}, outdir {OUTDIR}, device {device}")

flow_latent = 10
instrument = desi.DESI()

wave_obs = instrument._wave_obs
sky_mask = get_skyline_mask(wave_obs)

# loading flow model and spender model for noise
NDE_theta, model_spender = load_models(
    flow_file=flow_file,
    spender_file=spender_file,
    flow_latent=flow_latent,
    instrument=instrument,
    map_location=device,
    weights_only=False,
)

hdu = h5py.File(prospector_sed_file, "r")
fluxes = hdu["fluxes"]  # lazy load, in maggies
wavelength = torch.tensor(hdu["wavelength"][:])
redshifts = torch.tensor(hdu["priors/redshifts"][:])

batch_size = 1024
total_samples = redshifts.shape[0]


def noisy_batches():
    """Yield each batch of noisy mocks as a dict of the spender dataset columns."""
    for i in range(0, total_samples, batch_size):
        fluxes_batch = fluxes[i : i + batch_size]  # in maggies
        # DESI units, 1e-17 erg/s/cm2/Ang
        flambda = torch.from_numpy(
            to_flambda(wavelength.numpy() * u.AA, fluxes_batch * u.mgy).value
        )

        redshifts_batch = redshifts[i : i + batch_size]
        redshifts_err_batch = torch.zeros_like(redshifts_batch)
        target_id_batch = torch.arange(i, min(i + batch_size, total_samples))

        flambda_norm, norms, good_mask = normalize_spectra(
            flambda, redshifts_batch, wavelength, inplace=False
        )
        flambda_norm = flambda_norm[:, :-1]  # remove last point to match instrument wave grid

        print(
            f"Processing batch {i} - {i + batch_size}, "
            f"normalized {good_mask.sum().item()} / {good_mask.shape[0]} spectra, "
        )

        with torch.no_grad():
            # shape is n_sims, n_samples, n_latent
            samples = NDE_theta.sample(1, context=norms.unsqueeze(1).float())

            # make it (n_samples, n_sims, n_latent)
            samples = samples.permute(1, 0, 2).float()

            model_spender.eval()
            instrument.eval()

            snr_sample_maf = model_spender.decode(samples)
            snr_sample_maf = snr_sample_maf.squeeze(0)  # (n_samples, n_wave)

        snr_batch = snr_sample_maf.clone()  # (1024, 7780), last batch could be smaller
        snr_batch[:, sky_mask[:-1]] = float("nan")
        snr_batch[snr_batch <= 0] = float("nan")

        sigma_batch = (flambda_norm / snr_batch).abs()
        sigma_batch = torch.nan_to_num(sigma_batch, nan=0.0, posinf=0.0, neginf=0.0)

        # clamp sigma batch to be 99.5 percentile
        sigma_cap = torch.nanquantile(sigma_batch, 0.995, dim=1, keepdim=True)
        sigma_batch = torch.minimum(sigma_batch, sigma_cap)

        ivar_batch = 1.0 / (sigma_batch**2)
        ivar_batch = torch.nan_to_num(ivar_batch, nan=0.0, posinf=0.0, neginf=0.0)
        ivar_batch[sigma_batch == 0] = 0.0

        noise = torch.normal(mean=0.0, std=sigma_batch, generator=generator)
        f_noisy = flambda_norm + noise

        yield dict(
            spec=f_noisy,
            w=ivar_batch,
            z=redshifts_batch,
            zerr=redshifts_err_batch,
            norm=norms,
            target_id=target_id_batch,
        )


features = get_features(wavelength.shape[0] - 1, ["target_id"])
counts = write_dataset(str(OUTDIR), noisy_batches(), features, ["target_id"])
print(f"wrote {sum(counts.values())} mocks to {OUTDIR}, split counts {counts}")
