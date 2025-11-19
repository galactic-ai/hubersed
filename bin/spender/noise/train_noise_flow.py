#!/usr/bin/env python
# coding: utf-8

import sys

import numpy as np
import torch
from spender.flow import NeuralDensityEstimator
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------
# Latents from single file
# ----------------------------------------------------------------------
blob = torch.load('../desi_noise_spender_10latent_space.pt', map_location='cpu')
theta = blob['latents'].float()   # [N, 10]
A = blob['A'].float() # [N, 1]

print("latents shape:", theta.shape, theta.device)
print("A shape:", A.shape, A.device)

N, n_latent = theta.shape

# train/valid split
perm = torch.randperm(N)
n_train = int(0.85 * N)
train_theta = theta[perm[:n_train]]
valid_theta = theta[perm[n_train:]]
train_A = A[perm[:n_train]]
valid_A = A[perm[n_train:]]

print("N_train:", n_train, "N_valid:", N - n_train)

batch_size = 10_000

train_ds = TensorDataset(train_theta, train_A)
valid_ds = TensorDataset(valid_theta, valid_A)

data_loader = DataLoader(train_ds, batch_size=batch_size,
                         shuffle=True, drop_last=True)
valid_data_loader = DataLoader(valid_ds, batch_size=batch_size,
                               shuffle=False)

# ----------------------------------------------------------------------
# Build / load flow (your class)
# ----------------------------------------------------------------------
flow_file = sys.argv[1]
print("flow_file:", flow_file)

# use empirical mean/std as initial_pos
mu = train_theta.mean(0)
sigma = train_theta.std(0)
initial_pos = {
    "bounds": [[m.item(), m.item()] for m in mu],
    "std": [s.item() for s in sigma],
}

context_dim = A.shape[1]

if "new" in sys.argv:
    NDE_theta = NeuralDensityEstimator(
        dim=n_latent,
        initial_pos=initial_pos,
        hidden_features=64,
        num_transforms=5,
        context_features=context_dim,
    ).to(device)
else:
    # re-create same architecture then load state_dict
    NDE_theta = NeuralDensityEstimator(
        dim=n_latent,
        initial_pos=initial_pos,
        hidden_features=64,
        num_transforms=5,
        context_features=context_dim,
    )
    NDE_theta.load_state_dict(torch.load(flow_file, map_location=device))
    NDE_theta.to(device)

# keep your attributes
NDE_theta.optimizer = torch.optim.Adam(NDE_theta.parameters(), lr=1e-3)
NDE_theta.train_loss_history = []
NDE_theta.valid_loss_history = []

n_epoch = 150
n_steps = 20

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    NDE_theta.optimizer,
    max_lr=2e-3,
    steps_per_epoch=n_steps,
    epochs=n_epoch,
)

# ----------------------------------------------------------------------
# Training loop (same structure as your original code)
# ----------------------------------------------------------------------
for epoch in trange(n_epoch, desc='Training NDE', unit='epochs'):
    print('    Epoch', epoch)
    print('    lr:', NDE_theta.optimizer.param_groups[0]['lr'])

    # ----------------- train -----------------
    train_loss = []
    for k, batch in enumerate(data_loader):
        latent_batch, A_batch = [b.to(device) for b in batch]          # <- exactly your pattern

        # call log_prob on the flow itself (no .net)
        NDE_theta.optimizer.zero_grad()
        loss = -NDE_theta.log_prob(latent_batch, context=A_batch).mean()
        loss.backward()
        NDE_theta.optimizer.step()

        train_loss.append(loss.item())
        if k >= n_steps:
            break

    train_loss = float(np.mean(train_loss))
    NDE_theta.train_loss_history.append(train_loss)

    # ----------------- valid -----------------
    valid_loss = []
    with torch.no_grad():
        for k, batch in enumerate(valid_data_loader):
            latent_batch, A_batch = [b.to(device) for b in batch]
            loss = -NDE_theta.log_prob(latent_batch, context=A_batch).mean()
            valid_loss.append(loss.item())

    valid_loss = float(np.mean(valid_loss))
    NDE_theta.valid_loss_history.append(valid_loss)

    print(f'Loss = {train_loss:.3f} (train), {valid_loss:.3f} (valid)')

    scheduler.step()

    # save like you did before
    if epoch % 10 == 0 or epoch == n_epoch - 1:
        torch.save(NDE_theta.state_dict(), flow_file)
        print("saved", flow_file)