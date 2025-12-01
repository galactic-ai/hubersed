import torch

from spender.flow import NeuralDensityEstimator
from spender import load_model

def _load_flow_model(filename, n_latent, **kwargs):
    NDE_theta = NeuralDensityEstimator(
        dim=n_latent,
        initial_pos={"bounds": [[0, 0]] * n_latent, "std": [0.5] * n_latent},
        hidden_features=64,
        num_transforms=5,
        context_features=1,
    )

    state_dict = torch.load(filename, **kwargs)
    NDE_theta.load_state_dict(state_dict)

    return NDE_theta


def _load_spender_model(filename, instrument, **kwargs):
    return load_model(filename, instrument, **kwargs)


def load_models(flow_file, spender_file, flow_latent, instrument, **kwargs):
    NDE_theta = _load_flow_model(flow_file, n_latent=flow_latent, **kwargs)
    spender_model = _load_spender_model(spender_file, instrument, **kwargs)
    return NDE_theta, spender_model