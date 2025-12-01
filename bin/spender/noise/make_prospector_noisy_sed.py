import h5py
import torch

from hubersed.spender.utils import load_models
from hubersed.spender.quantities import normalize_spectra, compute_ivar
from hubersed.conversion import maggies_to_flambda
from hubersed.paths import PATHS
from hubersed.utils import nanstd

from spender.data import desi
from spender.instrument import get_skyline_mask

import pickle

DATA_PATH = PATHS['DATA']
RESULTS_PATH = PATHS['RESULTS']

# set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# set generator
generator = torch.Generator()
generator.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

flow_file = str(DATA_PATH / 'desi_noise_spender_10latent_flow.pt')
spender_file = str(DATA_PATH / 'desi_noise_spender_10latent.pt')
prospector_sed_file = DATA_PATH / 'prospector_stochastic_model_seds_500000.h5'

flow_latent = 10
instrument = desi.DESI()

wave_rest = instrument._wave_obs
sky_mask = get_skyline_mask(wave_rest)

# loading flow model and spender model for noise
NDE_theta, model_spender = load_models(flow_file=flow_file, 
                                       spender_file=spender_file,
                                       flow_latent=flow_latent,
                                       instrument=instrument,
                                       map_location=device,
                                       weights_only=False)

hdu = h5py.File(prospector_sed_file, 'r')
fluxes = hdu['fluxes'] # lazy load, in maggies
wavelength = torch.tensor(hdu['wavelength'][:])
redshifts = torch.tensor(hdu['priors/redshifts'][:])

batch_size = 1024
total_samples = redshifts.shape[0]

idx = 0
for i in range(0, total_samples, batch_size):
    fluxes_batch = fluxes[i:i+batch_size] # in maggies
    flambda = maggies_to_flambda(wavelength, fluxes_batch) # in flambda
    # to desi like units (they are in 1e-17 erg/s/cm2/Ang)
    flambda = flambda / 1e-17 

    redshifts_batch = redshifts[i:i+batch_size]
    redshifts_err_batch = torch.zeros_like(redshifts_batch)
    target_id_batch = torch.arange(i, min(i+batch_size, total_samples))

    flambda_norm, norms, good_mask = normalize_spectra(flambda, redshifts_batch, wavelength, inplace=False)
    flambda_norm = flambda_norm[:, :-1]  # remove last point to match instrument wave grid

    print(f'Processing batch {i} - {i+batch_size}, normalized {good_mask.sum().item()} / {good_mask.shape[0]} spectra')

    with torch.no_grad():
        samples = NDE_theta.sample(1, context=norms.unsqueeze(1).float())# (n_sims, n_samples, n_latent)

        # make it (n_samples, n_sims, n_latent)
        samples = samples.permute(1, 0, 2).float()

        model_spender.eval()
        instrument.eval()

        snr_sample_maf = model_spender.decode(samples)
        snr_sample_maf = snr_sample_maf.squeeze(0) # (n_samples, n_wave)


    snr_batch = snr_sample_maf.clone() # (1024, 7780)
    snr_batch[:, sky_mask[:-1]] = float('nan')
    snr_batch[snr_batch <= 0] = float('nan')

    # median and std along wavelength for each object
    median_snr = torch.nanmedian(snr_batch, dim=1).values     # (1024,)
    std_snr    = nanstd(snr_batch, dim=1)                     # (1024,)

    ivar_batch = compute_ivar(flambda_norm, snr_batch)
    ivar_batch = torch.nan_to_num(ivar_batch, nan=0.0, posinf=0.0, neginf=0.0)
    
    sigma_batch = flambda_norm / snr_batch
    sigma_batch = torch.nan_to_num(sigma_batch, nan=0.0, posinf=0.0, neginf=0.0)

    # clamp sigma batch to be 5 * std of the sigma
    sigma_batch = torch.clamp(sigma_batch, max=5.0 * sigma_batch.std(dim=1, keepdim=True))

    noise = torch.normal(mean=0.0, std=sigma_batch, generator=generator)
    
    f_noisy = flambda_norm + noise

    # save to a pickle file (DESIprospector1024_idx.pkl)
    # such that when I open I can upack it as
    # spec, w, z, target_id,  norm, zerr

    save_dict = [f_noisy,
                 ivar_batch,
                 redshifts_batch,
                 target_id_batch,
                norms,
                redshifts_err_batch,
    ]
    save_path = DATA_PATH / f'DESIprospector1024_{idx}.pkl'

    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)
    
    idx += 1