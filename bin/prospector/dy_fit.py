import parameter_file as P
import numpy as np
import copy
import sys

from astropy.cosmology import Planck18 as cosmo
from prospect.models.sedmodel import HyperSpecModel
from prospect.models.templates import TemplateLibrary, adjust_stochastic_params
from prospect.models.transforms import dustratio_to_dust1
from prospect.models.priors import TopHat, ClippedNormal, LogUniform, Uniform
from prospect.fitting import fit_model
import prospect.io.write_results as writer
from hubersed.prospector.utils import make_stochastic_agebins

# ── Build EVERYTHING on all processes ──
spec, unc, redshift, mask, id = P.get_outlier_info(0)
mask = (spec > 0) & np.isfinite(spec) & np.isfinite(unc)

assert id == 39632941227181576, f"Expected ID 39632941227181576, but got {id}"

wave_A = P.WAVE_OBS
spec_maggies = P.flambda_to_maggies(wave_A, spec)
ivar_maggies = P.ivar_flambda_to_ivar_maggies(wave_A, unc)
sigma_maggies = 1 / np.sqrt(np.where(ivar_maggies > 0, ivar_maggies, np.inf))
mask_em = P.mask_em_lines(mask, redshift)

set_vals = {
    "logmass": 8.7,
    "logzsol": -0.1,
    "sigma_reg": 0.17,
    "tau_eq": 2.5, 
    "tau_in": cosmo.age(redshift).value,
    "sigma_dyn": 0.005,
    "tau_dyn": 0.025,
    "dust_index": 0.0,
    "dust2": 0.1,
    "dust_ratio": 1.0,
}
set_priors = {
    "logmass": Uniform(mini=7.0, maxi=12.0),
    "logzsol": Uniform(mini=-1.0, maxi=0.19),
    "sigma_reg": LogUniform(mini=0.1, maxi=10.0),
    "tau_eq": Uniform(mini=0.01, maxi=cosmo.age(redshift).value),
    "tau_in": Uniform(mini=0.01, maxi=cosmo.age(redshift).value),
    "sigma_dyn": LogUniform(mini=0.001, maxi=0.1),
    "tau_dyn": ClippedNormal(mean=0.01, sigma=0.02, mini=0.005, maxi=0.2),
    "dust_index": TopHat(mini=-1.0, maxi=0.4),
    "dust2": ClippedNormal(mean=0.3, sigma=1.0, mini=0.0, maxi=4.0),
    "dust_ratio": ClippedNormal(mean=1.0, sigma=0.3, mini=0.0, maxi=2.0)
    
}

template = copy.deepcopy(TemplateLibrary['stochastic_sfh'])
dust_template = copy.deepcopy(TemplateLibrary["dust_emission"])
nebular_template = copy.deepcopy(TemplateLibrary["nebular"])

template.update(dust_template)
template.update(nebular_template)

template['zred']['init'] = redshift
template['zred']['isfree'] = False

# ratio prior (tau_1/tau_2)
template["dust_ratio"] = {
    "N": 1,
    "isfree": True,
    "init": 1.0,
    "units": "ratio of birth-cloud to diffuse dust",
    "prior": set_priors["dust_ratio"],
}

# dust index prior
template["dust_index"] = {
    "N": 1,
    "isfree": True,
    "init": np.float64(0.0),
    "units": "power-law multiplication of Calzetti",
    "prior": set_priors["dust_index"],
}

template["dust1"] = {
    "N": 1,
    "isfree": False,
    "depends_on": dustratio_to_dust1,
    "init": 0.0,
    "units": "optical depth towards young stars",
}

for key in set_vals:
    if key in template:
        print(f"  Setting {key} = {set_vals[key]} in template")
        template[key]['init'] = set_vals[key]
        template[key]['prior'] = set_priors[key]
        if key in ['tau_eq', 'tau_in', 'sigma_dyn', 'tau_dyn', 'sigma_reg']:
            # The continuum is dominated by old stars that accumulated over billions of years 
            # — it cares about total mass at broad age ranges, not fine-grained SFH variations. 
            # The hyperparameters control bin-to-bin correlations and burstiness, which barely 
            # affect the cumulative light. Emission lines are the opposite — they trace stars 
            # younger than ~10 Myr, where the exact recent SFH shape 
            # (controlled by the hyperparameters) matters enormously.
            template[key]['isfree'] = False
        else:
            template[key]['isfree'] = True
    else:
        print(f"  Warning: {key} not found in template parameters")

template["agebins"] = {
    "N": 10,
    "isfree": False,
    "init": make_stochastic_agebins(redshift),
    "units": "log10(yr)"
}

# Charlot & Fall dust model
template["dust_type"]["init"] = 0


template = adjust_stochastic_params(template)

model =  HyperSpecModel(template)
obs = P.build_obs(spec=spec_maggies, unc=sigma_maggies, mask=mask_em)
sps = P.build_sps()
theta_init = model.theta.copy()
index = model.theta_index

# ── NOW set up MPI pool ──
import mpi4py
from mpi4py import MPI
from schwimmbad import MPIPool

mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False

comm = MPI.COMM_WORLD
size = comm.Get_size()

withmpi = comm.Get_size() > 1

# ── Only master runs from here ──
print("Starting dynesty fit...")
print(f"Free params: {model.free_params}")
print(f"N free: {len(model.theta)}")

if (withmpi) & ('logzsol' in model.free_params):
    dummy_obs = dict(filters=None, wavelength=None)

    logzsol_prior = model.config_dict["logzsol"]['prior']
    lo, hi = logzsol_prior.range
    logzsol_grid = np.around(np.arange(lo, hi, step=0.1), decimals=2)

    sps.update(**model.params)  # make sure we are caching the correct IMF / SFH / etc
    for logzsol in logzsol_grid:
        model.params["logzsol"] = np.array([logzsol])
        _ = model.predict(model.theta, obs=dummy_obs, sps=sps)

# ensure that each processor runs its own version of FSPS
# this ensures no cross-over memory usage
from prospect.fitting import lnprobfn
from functools import partial
lnprobfn_fixed = partial(lnprobfn, sps=sps)

if withmpi:
    with MPIPool() as pool:
        # The dependent processes will run up to this point in the code
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        nprocs = pool.size
        # The parent process will oversee the fitting
        results = fit_model(obs, model, sps, pool=pool, 
                           queue_size=nprocs, lnprobfn=lnprobfn_fixed, 
                           nlive=100, dynesty=True, nested_sample='rwalk',
                           print_progress=True)
else:
    # without MPI we don't pass the pool
    raise NotImplementedError("This script is designed to run with MPI. Please run with mpirun or mpiexec.")

writer.write_hdf5("continuum_fit.h5", {}, model, obs,
                   results["sampling"][0], results["optimization"][0], 
                   tsample=results["sampling"][1],
                   toptimize=results["optimization"][1],
                   sps=sps)
print("Done! Saved to continuum_fit.h5")