from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import torch
from spender.data import desi
from spender import load_model

from hubersed.paths import PATHS
DATA_PATH = PATHS['DATA']
RESULTS_PATH = PATHS['RESULTS']

torch.set_default_dtype(torch.float32)


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

inst = desi.DESI().float().to(device)

# CUE: use the Cue-mock latents + write a separate outlier file (keep the FSPS 352 intact)
CUE = True
# S/N>3 run: point at the snr3 latent files and write a separate outlier file so the
# old (no-cut) 280/352 outlier sets stay intact for comparison. Set SNR_TAG='' for old behaviour.
SNR_TAG = '_snr3'

desi_file = DATA_PATH / f'spender_spec_6latent{SNR_TAG}'
_prospector_latents = f'prospector_noise_spec_6latent_cue{SNR_TAG}' if CUE else f'prospector_noise_spec_6latent{SNR_TAG}'

desi_blob = torch.load(desi_file, map_location='cpu', mmap=True)
desi_latents = desi_blob['latents'].float().to(device)
# global catalogue index of each kept DESI spectrum (position != global after the S/N cut)
desi_index = (desi_blob['indices'].cpu().long() if 'indices' in desi_blob
              else torch.arange(desi_latents.shape[0]))
print(f"DESI S/N-cut latents: {desi_latents.shape[0]} (from file {desi_file.name})")

prospector_spec = torch.load(DATA_PATH / _prospector_latents, map_location='cpu', mmap=True)
p_l    = prospector_spec['latents'].to(device='cpu', dtype=torch.float32)
print(f"mock S/N-cut latents: {p_l.shape[0]}")


# Isolation Forest for outlier detection
scaler = StandardScaler()
p_l_scaled = scaler.fit_transform(p_l)
s_l_scaled = scaler.transform(desi_latents.cpu())

iso = IsolationForest(
    n_estimators=300,
    max_samples=2048,     # subsampling makes it fast and robust
    contamination="auto",
    n_jobs=-1,
    random_state=0,
)
iso.fit(p_l_scaled)

scores_desi = iso.decision_function(s_l_scaled)
scores_prospector = iso.decision_function(p_l_scaled)


# threshold for outliers, worse than 0.1% in the prospector distribution
threshold = torch.quantile(torch.tensor(scores_prospector), 0.001)
outlier_mask = torch.tensor(scores_desi) <= threshold
outlier_pos = torch.where(outlier_mask)[0]            # position in the S/N-cut DESI array
outlier_idx = desi_index[outlier_pos]                  # -> global catalogue index

# save outlier indices for later analysis (outlier_indices are GLOBAL catalogue indices)
_outfile = f"desi_outliers_cue{SNR_TAG}.pt" if CUE else f"desi_outliers{SNR_TAG}.pt"
torch.save({"outlier_indices": outlier_idx,
            "outlier_pos": outlier_pos,
            "desi_index": desi_index,
            "threshold": float(threshold),
            "snr_min": prospector_spec.get('meta', {}).get('snr_min', None)},
           RESULTS_PATH / _outfile)
print(f"{len(outlier_idx)} outliers -> {_outfile}  (global indices; threshold {float(threshold):.4f})")