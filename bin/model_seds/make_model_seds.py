import argparse
import copy
import glob
import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from pathlib import Path

import h5py
import numpy as np
import scipy.stats
from prospect.models import priors, transforms
from prospect.models.sedmodel import HyperSpecModel
from prospect.models.templates import TemplateLibrary, adjust_stochastic_params
from prospect.observation import Spectrum
from tqdm.auto import tqdm

from hubersed.paths import PATHS
from hubersed.prospector.lsf import DESI_WAV, build_desi_resolution_matrix
from hubersed.prospector.utils import make_stochastic_agebins

# ignore warnings from zero ivar
warnings.filterwarnings("ignore", category=RuntimeWarning)

# single-thread the numerics (numpy, and JAX/cuejax on the Cue path) to avoid
# oversubscription across workers. Inherited by spawn children via os.environ.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

DATA_PATH = PATHS["DATA"] / "prospector_model"

N_RATIOS = 9  # 9 logsfr_ratios for 10 age bins

# per-worker state, populated by _init_worker(). Never touched at module level.
_S = {}


def priors_path(nebular, sample_size):
    stem = "stochastic_priors_sample_cue" if nebular == "cue" else "stochastic_priors_sample"
    return DATA_PATH / f"{stem}_{sample_size}.npz"


def out_path(nebular, sample_size):
    stem = (
        "prospector_stochastic_model_seds_cue"
        if nebular == "cue"
        else "prospector_stochastic_model_seds"
    )
    return DATA_PATH / f"{stem}_{sample_size}.h5"


def resolve_sample_size(nebular):
    """
    Largest N with a priors npz on disk. NUMERIC max, not sorted()[-1].
    """
    pat = str(priors_path(nebular, "*"))
    sizes = []
    for f in glob.glob(pat):
        tail = Path(f).name.removesuffix(".npz").split("_")[-1]
        if tail.isdigit():
            sizes.append(int(tail))
    if not sizes:
        raise SystemExit(f"no priors npz matching {pat}; pass -n explicitly")
    return max(sizes)


def build_base_template(nebular):
    t = copy.deepcopy(TemplateLibrary["stochastic_sfh"])
    t.update(copy.deepcopy(TemplateLibrary["dust_emission"]))

    if nebular == "cue":
        # Cue (Li+24): ionizing spectrum tied to FSPS stars; free gas_logz/u/nH/no/co
        t.update(copy.deepcopy(TemplateLibrary["cue_stellar_nebular"]))
    else:
        # FSPS / Byler+2017 CLOUDY grids; free gas_logz/u only
        t.update(copy.deepcopy(TemplateLibrary["nebular"]))


    t["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}

    t["dust_type"]["init"] = 4
    t["mass"]["init"] = 10**10.7
    t["dust1"] = {
        "N": 1,
        "isfree": False,
        "depends_on": transforms.dustratio_to_dust1,
        "init": 0.0,
        "units": "optical depth towards young stars",
    }
    t["dust_ratio"] = {
        "N": 1,
        "isfree": True,
        "init": 1.0,
        "units": "ratio of birth-cloud to diffuse dust",
        "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3),
    }
    t["dust_index"] = {
        "N": 1,
        "isfree": True,
        "init": np.float64(0.0),
        "units": "power-law multiplication of Calzetti",
        "prior": priors.TopHat(mini=-1.0, maxi=0.4),
    }

    # velocity dispersion
    t["sigma_smooth"] = {"N": 1, "isfree": False, "init": 200.0, "units": "km/s"}
    t["smoothtype"] = {"N": 1, "isfree": False, "init": "vel"}
    t["fftsmooth"] = {"N": 1, "isfree": False, "init": True}
    t["eline_sigma"] = {"N": 1, "isfree": False, "init": 100.0, "units": "km/s"}
    return t


def load_priors(nebular, sample_size):
    path = priors_path(nebular, sample_size)
    if not path.exists():
        raise SystemExit(f"no priors at {path}; run get_stochastic_priors.py first")
    npz = np.load(path, allow_pickle=True)
    d = {k: npz[k] for k in npz.files}

    if nebular == "cue":
        for need in ("gas_lognHs", "gas_lognos", "gas_logcos"):
            if need not in d:
                raise SystemExit(
                    f"missing Cue prior {need} in {path.name}; regenerate with: "
                    f"python bin/model_seds/get_stochastic_priors.py --cue -n {sample_size}"
                )
    return d


def sample_logsfr_ratios(parset, index, seed):
    """
    Draw logsfr_ratios ~ MVN(mean, Sigma) reproducibly.
    """
    p = parset["logsfr_ratios"]["prior"]
    rng = np.random.default_rng([seed, index])
    dist = scipy.stats.multivariate_normal(mean=p.loc, cov=p.scale)
    return np.asarray(dist.rvs(random_state=rng), dtype=float).reshape(-1)


def build_parset_for_index(i):
    nebular = _S["nebular"]
    priors_dict = _S["priors"]
    t = copy.deepcopy(_S["base_template"])

    t["logmass"]["init"] = priors_dict["stellar_masses"][i]
    t["logzsol"]["init"] = priors_dict["stellar_metallicities"][i]

    # tau_dust_1s is used as the dust_ratio init
    t["dust_index"]["init"] = priors_dict["ns"][i]
    t["dust_ratio"]["init"] = priors_dict["tau_dust_1s"][i]
    t["dust2"]["init"] = priors_dict["tau_dust_2s"][i]

    # dust emission (Draine & Li 2007)
    t["duste_umin"]["init"] = priors_dict["u_mins"][i]
    t["duste_qpah"]["init"] = priors_dict["q_pahs"][i]
    t["duste_gamma"]["init"] = priors_dict["gamma_es"][i]

    # nebular gas
    t["gas_logz"]["init"] = priors_dict["gas_metallicities"][i]
    t["gas_logu"]["init"] = priors_dict["gas_ionization_parameters"][i]
    if nebular == "cue":
        t["gas_lognH"]["init"] = priors_dict["gas_lognHs"][i]
        t["gas_logno"]["init"] = priors_dict["gas_lognos"][i]
        t["gas_logco"]["init"] = priors_dict["gas_logcos"][i]

    # redshift & agebins
    z = priors_dict["redshifts"][i]
    t["zred"]["init"] = z
    t["agebins"]["init"] = make_stochastic_agebins(z=z)

    # stochastic SFH hyperparameters
    t["sigma_reg"]["init"] = priors_dict["sigma_regs"][i]
    t["tau_eq"]["init"] = priors_dict["tau_eqs"][i]
    t["tau_in"]["init"] = priors_dict["tau_ins"][i]
    t["sigma_dyn"]["init"] = priors_dict["sigma_dyns"][i]
    t["tau_dyn"]["init"] = priors_dict["tau_dyns"][i]

    t["sigma_smooth"]["init"] = priors_dict["sigma_smooths"][i]
    t["eline_sigma"]["init"] = priors_dict["sigma_gass"][i]

    # builds the MVN(0, Sigma_ACF) prior object on logsfr_ratios
    t = adjust_stochastic_params(t)

    ratios = sample_logsfr_ratios(t, i, _S["seed"])
    t["logsfr_ratios"]["init"] = ratios
    return t, ratios


@cache
def _get_sps(nebular):
    if nebular == "cue":
        from prospect.sources import NebStepBasis

        return NebStepBasis()
    from prospect.sources import FastStepBasis

    return FastStepBasis()


@cache
def _get_resolution_matrix():
    return build_desi_resolution_matrix(DESI_WAV)


def make_obs(n_wave):
    obs = Spectrum(
        wavelength=np.asarray(DESI_WAV, dtype=np.float64),
        flux=np.ones(n_wave, dtype=np.float64),
        uncertainty=np.ones(n_wave, dtype=np.float64),
        mask=np.ones(n_wave, dtype=bool),
    )
    obs.rectify()
    return obs


def _init_worker(nebular, sample_size, seed):
    """Runs once per worker process. Replaces the old module-level globals."""
    _S["nebular"] = nebular
    _S["seed"] = seed
    _S["priors"] = load_priors(nebular, sample_size)
    _S["base_template"] = build_base_template(nebular)


def worker_block(start, stop):
    """Compute spectra[start:stop] in this process and return a 2D block."""
    nebular = _S["nebular"]
    n_wave = DESI_WAV.size

    sps = _get_sps(nebular)
    R_mat = _get_resolution_matrix()
    obs = make_obs(n_wave)

    block = np.empty((stop - start, n_wave), dtype=np.float32)
    ratios_block = np.empty((stop - start, N_RATIOS), dtype=np.float32)
    lum_block = None  # (stop-start, n_lines), allocated once n_lines is known

    for j, i in enumerate(range(start, stop)):
        parset, ratios = build_parset_for_index(i)
        ratios_block[j, :] = ratios.astype(np.float32)
        model = HyperSpecModel(configuration=parset)
        # sigma_smooth applied by Prospector; lines injected because obs carries flux
        preds, _ = model.predict(model.theta, [obs], sps=sps)
        # DESI instrumental LSF applied externally: MILES (sigma ~35-64 km/s) is
        # coarser than DESI (17-57), so obs.resolution trips prospect's
        # sqrt(sigma_inst^2 - sigma_lib^2) assert.
        spec = R_mat.dot(preds[0])
        block[j, :] = spec.astype(np.float32)
        # dust-ATTENUATED nebular line luminosities, erg/s (corr with dust2 ~0.83)
        lum = np.asarray(model._eline_lum, dtype=np.float32)
        if lum_block is None:
            lum_block = np.empty((stop - start, lum.size), dtype=np.float32)
        lum_block[j, :] = lum

    return start, stop, block, ratios_block, lum_block

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate mock DESI spectra from the stochastic-SFH prior sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nebular", choices=["cue", "fsps"], default="cue")
    p.add_argument("-n", "--sample-size", type=int, default=None,
                   help="number of spectra; default = largest N with a priors npz on disk")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for the logsfr_ratios draw. Keyed per-index, so it is "
                        "independent of --workers and completion order. Recorded in h5 attrs.")
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("-f", "--force", action="store_true",
                   help="overwrite the output h5 if it exists")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=500)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    nebular = args.nebular
    n = args.sample_size if args.sample_size is not None else resolve_sample_size(nebular)

    output_file = args.out or out_path(nebular, n)
    if output_file.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite {output_file}\n"
            f"  {output_file.stat().st_size / 1e9:.1f} GB\n"
            f"  pass --force if you mean it"
        )

    # parent needs the priors too: to size the datasets and to copy them into the h5
    _init_worker(nebular, n, args.seed)
    priors_dict = _S["priors"]
    n_spectra = len(priors_dict["redshifts"])
    n_wave = DESI_WAV.size

    print(
        f"nebular={nebular}  n={n_spectra}  seed={args.seed}  "
        f"priors={priors_path(nebular, n).name}"
    )

    with h5py.File(output_file, "w") as hf:
        hf.create_dataset("wavelength", data=DESI_WAV, compression="gzip")
        flux_dset = hf.create_dataset(
            "fluxes", shape=(n_spectra, n_wave), dtype=np.float32, compression="gzip"
        )
        ratios_dset = hf.create_dataset(
            "priors/logsfr_ratios",
            shape=(n_spectra, N_RATIOS),
            dtype=np.float32,
            compression="gzip",
        )

        line_wave = _get_line_wave(nebular, n_wave)
        n_lines = line_wave.size
        hf.create_dataset("line_wave", data=line_wave, compression="gzip")
        lum_dset = hf.create_dataset(
            "priors/line_lum",
            shape=(n_spectra, n_lines),
            dtype=np.float32,
            compression="gzip",
        )
        print(f"{nebular} line list: {n_lines} lines")

        for key, arr in priors_dict.items():
            # get_stochastic_priors.py writes provenance as 0-d scalars (_seed,
            # _sample_size, _cue, _git_sha). h5py rejects chunk/filter options on
            # scalar datasets, and they are metadata anyway -> carry them as attrs.
            if np.ndim(arr) == 0:
                hf.attrs[f"priors{key}" if key.startswith("_") else f"priors_{key}"] = (
                    arr.item()
                )
            else:
                hf.create_dataset(f"priors/{key}", data=arr, compression="gzip")

        hf.attrs["nebular"] = nebular
        hf.attrs["sample_size"] = n_spectra
        hf.attrs["seed"] = args.seed
        hf.attrs["sfh_mean"] = "zero"  # no alpha tilt; see wip/alpha-tilt
        hf.attrs["priors_file"] = priors_path(nebular, n).name

        blocks = [
            (s, min(s + args.chunk_size, n_spectra))
            for s in range(0, n_spectra, args.chunk_size)
        ]

        # spawn: fork deadlocks with JAX/cuejax
        ctx = mp.get_context("spawn")
        with (
            ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(nebular, n, args.seed),
            ) as ex,
            tqdm(total=n_spectra, desc=f"{nebular} spectra") as pbar,
        ):
            futures = {ex.submit(worker_block, s, e): (s, e) for (s, e) in blocks}
            for fut in as_completed(futures):
                start, stop, block, ratios_block, lum_block = fut.result()
                flux_dset[start:stop, :] = block
                ratios_dset[start:stop, :] = ratios_block
                lum_dset[start:stop, :] = lum_block
                pbar.update(stop - start)

    print(f"Saved {nebular} model SEDs to {output_file}")
    return 0


def _get_line_wave(nebular, n_wave):
    """Rest-frame wavelengths (AA) of the nebular lines, for labelling line_lum."""
    parset, _ = build_parset_for_index(0)
    m = HyperSpecModel(configuration=parset)
    m.predict(m.theta, [make_obs(n_wave)], sps=_get_sps(nebular))
    return np.asarray(m._eline_wave, dtype=np.float64)


if __name__ == "__main__":
    sys.exit(main())
