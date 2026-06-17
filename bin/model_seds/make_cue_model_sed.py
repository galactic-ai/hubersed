"""
Cue (Li+24) variant of make_prospector_model_sed.py.

Same stochastic SFH + Charlot&Fall dust + Draine-Li dust emission as the FSPS
generator, but the nebular emission is computed by the Cue emulator (NebStepBasis +
cue_stellar_nebular) with FREE N/O, C/O, nH. Reads the cue priors npz, writes a
separate h5. Lines injected via the SpecModel eline path (nebemlineinspec=False +
Spectrum obs); DESI LSF applied externally via R_mat; stores dust-attenuated Cue
line luminosities + the Cue line wavelengths.

Usage:  python bin/model_seds/make_cue_model_sed.py [SAMPLE_SIZE]   (default 500000)
"""
import sys

from prospect.models import priors, transforms
from prospect.models.sedmodel import HyperSpecModel
from prospect.models.templates import TemplateLibrary, adjust_stochastic_params
from prospect.observation import Spectrum

from tqdm.auto import tqdm

from hubersed.prospector.utils import make_stochastic_agebins
from hubersed.paths import PATHS
from hubersed.prospector.lsf import build_desi_resolution_matrix, DESI_WAV

import numpy as np
import h5py

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

import os
import copy
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# single-thread the numerics (JAX/cuejax + numpy) to avoid oversubscription across workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import glob

DATA_PATH = PATHS['DATA'] / "prospector_model"


def _resolve_sample_size():
    # CLI arg if it's a valid int (robust to Jupyter import, where argv[1] is the kernel json)
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            pass
    # else auto-detect from an existing cue priors file
    files = sorted(glob.glob(str(DATA_PATH / "stochastic_priors_sample_cue_*.npz")))
    if files:
        return int(files[-1].split("_")[-1].split(".")[0])
    return 500_000


SAMPLE_SIZE = _resolve_sample_size()

# Cue priors (separate file written by get_stochastic_priors.py with CUE=True)
priors_npz = np.load(f'{DATA_PATH}/stochastic_priors_sample_cue_{SAMPLE_SIZE}.npz', allow_pickle=True)
priors_dict = {k: priors_npz[k] for k in priors_npz.files}
for need in ("gas_lognHs", "gas_lognos", "gas_logcos"):
    assert need in priors_dict, f"missing Cue prior {need}; regenerate priors with CUE=True"

n_spectra = len(priors_dict['redshifts'])
n_wave = DESI_WAV.size
n_workers = 8
chunk_size = 500


def build_base_template():
    base_template = copy.deepcopy(TemplateLibrary["stochastic_sfh"])
    base_template.update(copy.deepcopy(TemplateLibrary["dust_emission"]))
    # Cue nebular (ionizing spectrum tied to FSPS stars; free gas_logz/u/nH/no/co)
    base_template.update(copy.deepcopy(TemplateLibrary["cue_stellar_nebular"]))

    # lines added via the SpecModel eline path (honors eline_sigma), not in-spectrum
    base_template["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}

    # Charlot & Fall dust
    base_template["dust_type"]["init"] = 0
    base_template["mass"]["init"] = 10 ** 10.7
    base_template["dust1"] = {
        "N": 1, "isfree": False, "depends_on": transforms.dustratio_to_dust1,
        "init": 0.0, "units": "optical depth towards young stars",
    }
    base_template["dust_ratio"] = {
        "N": 1, "isfree": True, "init": 1.0, "units": "ratio of birth-cloud to diffuse dust",
        "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3),
    }
    base_template["dust_index"] = {
        "N": 1, "isfree": True, "init": np.float64(0.0),
        "units": "power-law multiplication of Calzetti", "prior": priors.TopHat(mini=-1.0, maxi=0.4),
    }

    # velocity dispersion
    base_template["sigma_smooth"] = {"N": 1, "isfree": False, "init": 200.0, "units": "km/s"}
    base_template["smoothtype"] = {"N": 1, "isfree": False, "init": "vel"}
    base_template["fftsmooth"] = {"N": 1, "isfree": False, "init": True}
    base_template["eline_sigma"] = {"N": 1, "isfree": False, "init": 100.0, "units": "km/s"}
    return base_template


BASE_TEMPLATE = build_base_template()


def build_parset_for_index(i):
    t = copy.deepcopy(BASE_TEMPLATE)

    t["logmass"]["init"] = priors_dict["stellar_masses"][i]
    t["logzsol"]["init"] = priors_dict["stellar_metallicities"][i]

    t["dust_index"]["init"] = priors_dict["ns"][i]
    t["dust_ratio"]["init"] = priors_dict["tau_dust_1s"][i]
    t["dust2"]["init"] = priors_dict["tau_dust_2s"][i]

    t["duste_umin"]["init"] = priors_dict["u_mins"][i]
    t["duste_qpah"]["init"] = priors_dict["q_pahs"][i]
    t["duste_gamma"]["init"] = priors_dict["gamma_es"][i]

    # Cue nebular gas params
    t["gas_logz"]["init"] = priors_dict["gas_metallicities"][i]
    t["gas_logu"]["init"] = priors_dict["gas_ionization_parameters"][i]
    t["gas_lognH"]["init"] = priors_dict["gas_lognHs"][i]
    t["gas_logno"]["init"] = priors_dict["gas_lognos"][i]
    t["gas_logco"]["init"] = priors_dict["gas_logcos"][i]

    z = priors_dict["redshifts"][i]
    t["zred"]["init"] = z
    t["agebins"]["init"] = make_stochastic_agebins(z=z)

    t["sigma_reg"]["init"] = priors_dict["sigma_regs"][i]
    t["tau_eq"]["init"] = priors_dict["tau_eqs"][i]
    t["tau_in"]["init"] = priors_dict["tau_ins"][i]
    t["sigma_dyn"]["init"] = priors_dict["sigma_dyns"][i]
    t["tau_dyn"]["init"] = priors_dict["tau_dyns"][i]

    t["sigma_smooth"]["init"] = priors_dict["sigma_smooths"][i]
    t["eline_sigma"]["init"] = priors_dict["sigma_gass"][i]

    t = adjust_stochastic_params(t)
    ratios = t['logsfr_ratios']['prior'].sample()
    t['logsfr_ratios']['init'] = ratios
    return t, ratios


@lru_cache(maxsize=None)
def _get_sps():
    from prospect.sources import NebStepBasis
    return NebStepBasis()


@lru_cache(maxsize=None)
def _get_resolution_matrix():
    return build_desi_resolution_matrix(DESI_WAV)


def _make_obs():
    obs = Spectrum(
        wavelength=np.asarray(DESI_WAV, dtype=np.float64),
        flux=np.ones(n_wave, dtype=np.float64),
        uncertainty=np.ones(n_wave, dtype=np.float64),
        mask=np.ones(n_wave, dtype=bool),
    )
    obs.rectify()
    return obs


def _get_line_wave():
    """Rest-frame wavelengths (AA) of the Cue nebular lines."""
    parset, _ = build_parset_for_index(0)
    m = HyperSpecModel(configuration=parset)
    m.predict(m.theta, [_make_obs()], sps=_get_sps())
    return np.asarray(m._eline_wave, dtype=np.float64)


def worker_block(start, stop):
    sps = _get_sps()
    R_mat = _get_resolution_matrix()
    obs = _make_obs()

    block = np.empty((stop - start, n_wave), dtype=np.float32)
    ratios_block = np.empty((stop - start, 9), dtype=np.float32)
    lum_block = None

    for j, i in enumerate(range(start, stop)):
        parset, ratios = build_parset_for_index(i)
        ratios_block[j, :] = ratios.astype(np.float32)
        model = HyperSpecModel(configuration=parset)
        preds, _ = model.predict(model.theta, [obs], sps=sps)
        spec = R_mat.dot(preds[0])
        block[j, :] = spec.astype(np.float32)
        lum = np.asarray(model._eline_lum, dtype=np.float32)  # dust-attenuated Cue line lums
        if lum_block is None:
            lum_block = np.empty((stop - start, lum.size), dtype=np.float32)
        lum_block[j, :] = lum

    return start, stop, block, ratios_block, lum_block


def main():
    output_file = DATA_PATH / f"prospector_stochastic_model_seds_cue_{SAMPLE_SIZE}.h5"

    with h5py.File(output_file, "w") as hf:
        hf.create_dataset("wavelength", data=DESI_WAV, compression="gzip")
        flux_dset = hf.create_dataset("fluxes", shape=(n_spectra, n_wave), dtype=np.float32, compression="gzip")
        ratios_dset = hf.create_dataset("priors/logsfr_ratios", shape=(n_spectra, 9), dtype=np.float32, compression="gzip")

        line_wave = _get_line_wave()
        n_lines = line_wave.size
        hf.create_dataset("line_wave", data=line_wave, compression="gzip")
        lum_dset = hf.create_dataset("priors/line_lum", shape=(n_spectra, n_lines), dtype=np.float32, compression="gzip")
        print(f"Cue line list: {n_lines} lines")

        for key, arr in priors_dict.items():
            hf.create_dataset(f"priors/{key}", data=arr, compression="gzip")

        blocks = [(s, min(s + chunk_size, n_spectra)) for s in range(0, n_spectra, chunk_size)]

        # spawn context: JAX/cuejax + fork can deadlock
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex, \
             tqdm(total=n_spectra, desc="Cue spectra") as pbar:
            futures = {ex.submit(worker_block, s, e): (s, e) for (s, e) in blocks}
            for fut in as_completed(futures):
                start, stop, block, ratios_block, lum_block = fut.result()
                flux_dset[start:stop, :] = block
                ratios_dset[start:stop, :] = ratios_block
                lum_dset[start:stop, :] = lum_block
                pbar.update(stop - start)

    print(f"Saved Cue model SEDs to {output_file}")


if __name__ == "__main__":
    main()
