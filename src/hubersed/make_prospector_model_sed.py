#!/usr/bin/env python
# Prospector v1.4 | SpecModel | Non-parametric SFH | DESI-like grid (noise-free)

import os 
import sys
import copy

import numpy as np
import pandas as pd

from astropy.cosmology import Planck18 as cosmo
from scipy.stats import qmc

import astropy.units as u

from prospect.models.sedmodel import SpecModel
from prospect.models.templates import TemplateLibrary, adjust_continuity_agebins
from prospect.sources import FastStepBasis

from hubersed.paths import PATHS

# Random seed
SEED = 42
rng = np.random.default_rng(SEED)

# Paths
DATA_DIR = PATHS["DATA"]
OUTDIR = DATA_DIR / sys.argv[1]

if not OUTDIR.exists():
    OUTDIR.mkdir(parents=True, exist_ok=True)

MASS = 1e10  # for normalization
N_BINS = 8   # number of continuity SFH bins
N_SFH  = 2   # SFH draws per parameter

# using Sobol sequence for better space-filling
# Parameter bounds (low/high) in *physical* space
bounds = np.array([
    [-1.5,  0.3],   # logzsol
    [ 0.0,  2.0],   # dust2
    [-3.6, -1.4],   # gas_logu
    [ 0.0,  0.6],   # zred  (BGS-focused)
], float)

d = bounds.shape[0]
n = 2**21   # total number of samples 

sobol = qmc.Sobol(d, scramble=True, rng=rng)
samp_sobol = sobol.random_base2(int(np.log2(n)))      # n must be 2^m for balance
X = qmc.scale(samp_sobol, bounds[:,0], bounds[:,1])   # map to physical ranges

# DESI observed-frame wavelength grid (vacuum AA)
WAVE_OBS = np.arange(3600.0, 9824.1, 0.8)

# Toggle a quick variable-R smoothing (approximate)
DO_DESI_SMOOTH = False
DESI_ARMS = [(3600, 5550, 2600.0), (5550, 6560, 3600.0), (6560, 9824, 4500.0)]


# Helpers
def universe_age_gyr(z):
    return cosmo.age(z).to_value(u.Gyr)

def make_agebins_for_z(z, n_bins=N_BINS, t_min_myr=10.0):
    tuniv = universe_age_gyr(z) # Gyr
    tmin = max(t_min_myr / 1e3, 1e-3) # Gyr
    tmax = max(min(tuniv, 13.5), tmin * 1.01) # > tmin
    edges = np.geomspace(tmin, tmax, n_bins+1)  # Gyr
    e = np.log10(edges*1e9).astype(float)
    return np.column_stack([e[:-1], e[1:]])  # (N,2) log10(yr)

def draw_dirichlet_sfh(n_bins, alpha=1.0):
    if rng is None:
        raise ValueError("Pass an RNG!!")
    return rng.dirichlet([alpha]*n_bins).astype(float)  # sums to 1

def massfrac_to_logsfr_ratios(frac):
    """
    Continuity SFH samples logsfr_ratios[j] = ln(SFR_j / SFR_{j+1}), length = N_BINS-1.
    Here we use Dirichlet draw as proportional to SFR in each bin, then take ratios.
    """
    # Avoid zeros
    f = np.asarray(frac, float)
    eps = 1e-30
    s = np.clip(f, eps, None)
    return np.log(s[:-1] / s[1:])


def smooth_to_desi(wave, flux, arms):
    """
    Quick variable-R Gaussian smoothing (approx DESI LSF per arm).
    For exact comparisons later, prefer desispec Resolution matrices.
    """
    if not DO_DESI_SMOOTH:
        return flux
    out = np.zeros_like(flux)
    for lo, hi, R in arms:
        m = (wave>=lo) & (wave<hi)
        if not np.any(m): continue
        lam = wave[m]
        dlam = np.median(np.diff(lam))
        sigma = np.median(lam/(R*2.355))  # Å
        if not np.isfinite(sigma) or sigma<=0:
            out[m] = flux[m]; continue
        halfw = int(max(3, np.ceil(5*sigma/dlam)))
        x = np.arange(-halfw, halfw+1)*dlam
        k = np.exp(-0.5*(x/sigma)**2); k /= k.sum()
        out[m] = np.convolve(flux[m], k, mode="same")
    return out

def set_theta(model, **updates):
    """Convenience: set named parameters into a theta vector using model.theta_index."""
    theta = model.theta.copy()
    for name, val in updates.items():
        sl = model.theta_index[name]
        val = np.atleast_1d(val)
        if val.size != (sl.stop - sl.start):
            raise ValueError(f"Size mismatch for {name}: got {val.size}, expected {(sl.stop-sl.start)}")
        theta[sl] = val
    return theta


# SPS engine (once)
sps = FastStepBasis(zcontinuous=1)
sps.params["imf_type"] = 1             # Chabrier
sps.params["dust_type"] = 4            # Charlot & Fall
sps.params["add_neb_emission"] = True
sps.params["add_neb_continuum"] = True


AGEBINS = make_agebins_for_z(0.0, n_bins=N_BINS)  # max-age bins

# make base model
base = copy.deepcopy(TemplateLibrary["continuity_sfh"])
base.update(TemplateLibrary["nebular"])
base.update(TemplateLibrary["dust_emission"])

base = adjust_continuity_agebins(base, nbins=N_BINS)
base["agebins"]["isfree"] = False
base["agebins"]["init"]   = AGEBINS

model = SpecModel(base)


# init fix parameters we vary externally
for p in ("mass", "logzsol", "dust2", "zred", "gas_logz", "gas_logu"):
    if p in base:
        base[p]["isfree"] = False
    else:
        print(f"Warning: {p} not in base model params.")

# Observed-frame wavelength grid (SpecModel will return on this grid)
obs = {"wavelength": WAVE_OBS, "filters": []}

rows, count = [], 0

# Main loop
for logz, dust2, logu, zred in X:
    base["zred"]["init"] = float(zred)
    base["mass"]["init"] = float(MASS)
    base["logzsol"]["init"] = float(logz)
    base["dust2"]["init"] = float(dust2)
    base["gas_logz"]["init"] = float(logz)
    base["gas_logu"]["init"] = float(logu)

    for k in range(N_SFH):
        frac = draw_dirichlet_sfh(N_BINS, alpha=1.0)
        logsfr = massfrac_to_logsfr_ratios(frac)

        if "logsfr_ratios" not in model.theta_index:
            raise RuntimeError("Model missing 'logsfr_ratios' parameter for continuity SFH.")
        # build theta with fixed params + this SFH
        theta = set_theta(
            model, 
            logsfr_ratios=logsfr)

        # Evaluate (maggies on WAVE_OBS), theta may be empty since all fixed
        spec_maggies, phot, extras = model.predict(theta, obs=obs, sps=sps)

        # Optional DESI-like smoothing (fast approximation)
        spec_maggies = smooth_to_desi(WAVE_OBS, spec_maggies, arms=DESI_ARMS)

        # Save
        tag = f"z{zred:.3f}_logz{logz:+.1f}_d2{dust2:.1f}_logu{logu:+.1f}_sfh{k:03d}"
        fpath = os.path.join(OUTDIR, f"spec_{tag}.npz")
        meta = dict(
            zred=float(zred), logzsol=float(logz),
            dust2=float(dust2), mass=float(MASS),
            gas_logu=float(logu), n_bins=int(N_BINS),
            # store the continuity parameters we actually used
            logsfr_ratios=logsfr.tolist(),
            # keep a normalized SFR proxy for convenience (your original fractions)
            sfr_fraction=frac.tolist(),
            units="maggies", wave_units="Angstrom",
            sfh_model="continuity_sfh"
        )
        np.savez_compressed(fpath, wave=WAVE_OBS, flux=spec_maggies, meta=np.asarray(meta, dtype=object))

        rows.append({"file": fpath, "zred": zred, "logzsol": logz,
                    "dust2": dust2, "gas_logu": logu, "sfh_id": k})
        count += 1

print(f"Wrote {count} spectra → {OUTDIR}")
pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "index.csv"), index=False)
print("Index written:", os.path.join(OUTDIR, "index.csv"))
