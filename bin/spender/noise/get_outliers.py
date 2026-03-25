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

# Build wave_rest according to selected mode
model = load_model(str(DATA_PATH / 'spender_asc_run_6latent_zmax.pt'), inst, map_location='cpu', weights_only=False).float()
model = model.to(device)
model.eval()

loader = inst.get_data_loader(
        str(DATA_PATH),
        tag='chunk1024',
        which="all",
        batch_size=1024,
        shuffle=False,
        shuffle_instance=False,
    )

amount = len(loader.dataset)
print(amount)

desi_latents = torch.empty((amount*1024, 6), dtype=torch.float32, device=device)

prospector_spec = torch.load(DATA_PATH / 'prospector_noise_spec_6latent', map_location='cpu')
p_l    = prospector_spec['latents'].to(device='cpu', dtype=torch.float32)

with torch.no_grad():
    for i, batch in enumerate(loader):
        s, *_ = batch
        s = s.float().to(device)
        l = model.encode(s)
        desi_latents[i*1024:(i+1)*1024] = l

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
outlier_idx = torch.where(outlier_mask)[0]

# save outlier indices for later analysis
torch.save({"outlier_indices": outlier_idx}, RESULTS_PATH / "desi_outliers.pt")