import pickle
import sys
import time
import copy
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.utils.obsutils import fix_obs
from prospect.models import priors, transforms
from prospect.models.sedmodel import HyperSpecModel
from prospect.models.templates import TemplateLibrary, adjust_stochastic_params
from prospect.sources import FastStepBasis
from prospect import prospect_args
import numpy as np
import torch

from hubersed.prospector.utils import make_stochastic_agebins
from hubersed.paths import PATHS
from hubersed.prospector.utils import load_lines

from huggingface_hub import hffs

if not hasattr(np, "infty"):
    np.infty = np.inf  # compatibility shim for older code

DATA_PATH = PATHS['DATA']
RESULTS_PATH = PATHS['RESULTS']

WAVE_OBS = np.linspace(3600.0, 9824.0, 7781, dtype=np.float32)
OUTLIERS_IDX = torch.load(RESULTS_PATH / "desi_outliers.pt", weights_only=False)["outlier_indices"]
    
C_CGS = 2.99792458e10          # cm/s
FNU_PER_MAGGIE = 3631e-23       # erg/s/cm^2/Hz

EM_LINES_A = load_lines()['emission']['wave_vac']

# build sps
def build_sps():
    return FastStepBasis()

# DESI Spectra
def get_outlier_info(idx, streaming=True):
    """
    Get information about a specific outlier.

    Parameters    
    ----------
    idx : int
        Index of the outlier to retrieve (0 to len(OUTLIERS_IDX)-1)
    
    Returns
    -------
    spec : np.ndarray
        The observed spectrum for the outlier, in the original units (flambda).
    unc : np.ndarray
        The uncertainty on the observed spectrum, in the original units (flambda).
    redshift : float
        The redshift of the outlier.
    mask : np.ndarray
        A boolean array indicating which pixels are valid (True) or should be masked (False).
    id : int
        The original ID of the spectrum in the DESI dataset, for reference.
    """

    chunk_size = 1024
    chunk_indices = OUTLIERS_IDX // chunk_size
    chunk_files = [DATA_PATH / f"DESIchunk1024_{i}.pkl" for i in chunk_indices]
    chunk_indices_in_chunk = OUTLIERS_IDX % chunk_size

    chunk_file = chunk_files[idx]
    idx_in_chunk = chunk_indices_in_chunk[idx]    

    if not streaming:
        with open(chunk_file, "rb") as f:
            s, w, z, id, norm, *_ = pickle.load(f)
    else:
        with hffs.open(f"buckets/nikhil0504/hubersed-data/{chunk_file.name}", "rb") as f:
            s, w, z, id, norm, *_ = pickle.load(f)
    
    # correct for normalization
    s = s * norm[:, None]
    w = w / norm[:, None]**2
    mask = np.isfinite(s) & np.isfinite(w) & (w > 0)

    # convert to numpy arrays if not already from torch tensors
    if isinstance(s, torch.Tensor):
        s = s.cpu().numpy()
    if isinstance(w, torch.Tensor):
        w = w.cpu().numpy()
    if isinstance(z, torch.Tensor):
        z = z.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(id, torch.Tensor):
        id = id.cpu().numpy()



    return s[idx_in_chunk], w[idx_in_chunk], z[idx_in_chunk], mask[idx_in_chunk], id[idx_in_chunk]


def flambda_to_maggies(wave_A, flambda):
    # flambda: erg/s/cm^2/Å ; wave_A: Å
    flambda_cgs = flambda * 1e-17 # DESI spectra are in 1e-17 erg/s/cm^2/Å, convert to cgs
    return flambda_cgs * (wave_A**2) * 1e-8 / C_CGS / FNU_PER_MAGGIE

def ivar_flambda_to_ivar_maggies(wave_A, ivar_flambda):
    # maggies = flambda * K  => ivar_maggies = ivar_flambda / K^2
    # because DESI spectra are in 1e-17 erg/s/cm^2/Å, 
    # we need to include that factor in the conversion from flambda to maggies
    K = (wave_A**2) * 1e-8 * 1e-17 / C_CGS / FNU_PER_MAGGIE 
    return ivar_flambda / (K**2)

def mask_spectral_lines(wave_obs, mask, z, line_waves=EM_LINES_A, halfwidth_kms=500.0):
    """Mask spectral lines using a velocity-based window.
    
    Parameters
    ----------
    wave_obs : np.ndarray
        Observed wavelength array [Å].
    mask : np.ndarray
        Boolean array; True = good pixel, False = masked.
    z : float
        Redshift of the object.
    line_waves : np.ndarray
        Rest-frame vacuum wavelengths of lines to mask [Å].
    halfwidth_kms : float, optional
        Half-width of mask window in km/s. Default is 500 km/s.
    
    Returns
    -------
    np.ndarray
        Updated boolean mask.
    """
    c_kms = 299792.458
    mask = mask.copy()
    wave_rest = wave_obs / (1.0 + z)
    
    for line in line_waves:
        dwave = line * halfwidth_kms / c_kms
        mask &= np.abs(wave_rest - line) > dwave
    
    return mask
