from prospect.models import priors, transforms
from prospect.models.sedmodel import HyperSpecModel
from prospect.models.templates import TemplateLibrary, adjust_stochastic_params

from tqdm.auto import tqdm

from hubersed.prospector.utils import make_stochastic_agebins
from hubersed.paths import PATHS

import numpy as np
import h5py

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

import os
import copy

# export OMP_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# to make it faster / avoid oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

DATA_PATH = PATHS['DATA']

# load priors (hard coded for now)
priors_npz = np.load(f'{DATA_PATH}/stochastic_priors_sample_100000.npz', allow_pickle=True)
priors_dict = {k: priors_npz[k] for k in priors_npz.files}

DESI_WAV = np.linspace(3600.0, 9824.0, 7781, dtype=np.float64)

n_spectra = len(priors_dict['redshifts'])
n_wave = DESI_WAV.size
n_workers = 8
chunk_size = 500

def build_base_template():
    base_template = copy.deepcopy(TemplateLibrary["stochastic_sfh"])
    dust_template = copy.deepcopy(TemplateLibrary["dust_emission"])
    nebular_template = copy.deepcopy(TemplateLibrary["nebular"])

    # merge templates
    base_template.update(dust_template)
    base_template.update(nebular_template)

    # Charlot & Fall dust model
    base_template["dust_type"]["init"] = 0

    # mass init (not super-important; we overwrite logmass below)
    base_template["mass"]["init"] = 10 ** 10.7

    # dust1 (birth-cloud) derived from dust_ratio
    base_template["dust1"] = {
        "N": 1,
        "isfree": False,
        "depends_on": transforms.dustratio_to_dust1,
        "init": 0.0,
        "units": "optical depth towards young stars",
    }

    # ratio prior (tau_1/tau_2)
    base_template["dust_ratio"] = {
        "N": 1,
        "isfree": True,
        "init": 1.0,
        "units": "ratio of birth-cloud to diffuse dust",
        "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3),
    }

    # dust index prior
    base_template["dust_index"] = {
        "N": 1,
        "isfree": True,
        "init": np.float64(0.0),
        "units": "power-law multiplication of Calzetti",
        "prior": priors.TopHat(mini=-1.0, maxi=0.4),
    }


    return base_template

BASE_TEMPLATE = build_base_template()

def build_parset_for_index(i):
    base_template = copy.deepcopy(BASE_TEMPLATE)

    # set the *initial values* for this specific prior sample
    base_template["logmass"]["init"] = priors_dict["stellar_masses"][i]
    base_template["logzsol"]["init"] = priors_dict["stellar_metallicities"][i]

    # tau_dust_1s used as dust_ratio init
    base_template["dust_index"]["init"] = priors_dict["ns"][i]
    base_template["dust_ratio"]["init"] = priors_dict["tau_dust_1s"][i]
    base_template["dust2"]["init"] = priors_dict["tau_dust_2s"][i]

    # dust emission (Draine & Li 2007) parameters
    base_template["duste_umin"]["init"] = priors_dict["u_mins"][i]
    base_template["duste_qpah"]["init"] = priors_dict["q_pahs"][i]
    base_template["duste_gamma"]["init"] = priors_dict["gamma_es"][i]

    # gas / nebular parameters
    base_template["gas_logz"]["init"] = priors_dict["gas_metallicities"][i]
    base_template["gas_logu"]["init"] = priors_dict["gas_ionization_parameters"][i]

    # redshift & agebins
    z = priors_dict["redshifts"][i]
    base_template["zred"]["init"] = z
    base_template["agebins"]["init"] = make_stochastic_agebins(z=z)

    # stochastic SFH hyperparameters
    base_template["sigma_reg"]["init"] = priors_dict["sigma_regs"][i]
    base_template["tau_eq"]["init"] = priors_dict["tau_eqs"][i]
    base_template["tau_in"]["init"] = priors_dict["tau_ins"][i]
    base_template["sigma_dyn"]["init"] = priors_dict["sigma_dyns"][i]
    base_template["tau_dyn"]["init"] = priors_dict["tau_dyns"][i]

    # adjust stochastic parameters (same call as in your example)
    base_template = adjust_stochastic_params(base_template)

    return base_template


@lru_cache(maxsize=None)
def _get_sps():
    from prospect.sources import FastStepBasis
    return FastStepBasis()


def worker_block(start, stop):
    """
    Compute spectra[start:stop] in this process and return a 2D block.
    """

    sps = _get_sps()
    obs = {"wavelength": DESI_WAV, "filters": None}

    block = np.empty((stop - start, n_wave), dtype=np.float32)

    for j, i in enumerate(range(start, stop)):
        parset = build_parset_for_index(i)
        model = HyperSpecModel(configuration=parset)
        spec, _, _ = model.predict(model.theta, obs, sps=sps)
        block[j, :] = spec.astype(np.float32)

    return start, stop, block

# store in hdf5

def main():
    output_file = DATA_PATH / "prospector_stochastic_model_seds.h5"

    with h5py.File(output_file, "w") as hf:
        # store wavs
        hf.create_dataset("wavelength", data=DESI_WAV, compression="gzip")

        # store fluxes
        flux_dset = hf.create_dataset(
            "fluxes",
            shape=(n_spectra, n_wave),
            dtype=np.float32,
            compression="gzip",
        )

        # store priors
        for key, arr in priors_dict.items():
            hf.create_dataset(f"priors/{key}", data=arr, compression="gzip")
        
        # prepare block ranges
        blocks = []
        for start in range(0, n_spectra, chunk_size):
            stop = min(start + chunk_size, n_spectra)
            blocks.append((start, stop))

        with ProcessPoolExecutor(max_workers=n_workers) as executor, \
             tqdm(total=n_spectra, desc="Spectra") as pbar:

            futures = {executor.submit(worker_block, start, stop): (start, stop)
                       for (start, stop) in blocks}

            for fut in as_completed(futures):
                start, stop, block = fut.result()
                flux_dset[start:stop, :] = block
                pbar.update(stop - start)

    print(f"Saved model SEDs to {output_file}")


if __name__ == "__main__":
    main()