"""pocoMC fits of gallery105 22, 51 and 69 with the current SFH (model A) and a PSB SFH (model B).

Changed from 2026-11-01_poco_repeats/fit.py are prospect's pocoMC fit_model, the ladder node pool,
ResolutionMatrixSpectrum, the SSP cache, models B (PSB SFH), AC and BC (Cue), AC0, _main, L and
L_recoadd (model L on the SV3 recoadd).
COMMAND: through check.slurm and prod.slurm, for example
``ibrun <venv>/bin/python experiments/2026-11-14_poco_psb_sfh/fit.py run --tid <TID> --model B --mpi``.
OUTPUT: results/2026-11-14_poco_psb_sfh/<TID>/<model>/ (pmc_<i>.state, fit.h5).
"""

import argparse
import copy
import multiprocessing as mp
import os
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import astropy.units as u
import fsps
import numpy as np
from fit_fork import jax_backend_started
from poco import NodePool, latest_state, serve
from prospect.models.priors import LogUniform, TopHat

from hubersed.conversion import DESI_FLAM, ivar_to_maggies, to_maggies
from hubersed.io import desi as desi_io
from hubersed.io.desi import load_resolution_npz, load_spectrum
from hubersed.paths import PATHS
from hubersed.sps.config import build_continuum_model, library_sigma_floor
from hubersed.sps.parameter_file import WAVE_OBS, build_cue_sps, build_sps
from hubersed.sps.utils import universe_age_gyr

HERE = Path(__file__).resolve().parent
OUT = PATHS["RESULTS"] / HERE.name
REPEATS = PATHS["RESULTS"] / "2026-11-01_poco_repeats"
TID_REPRO = 39633332836764152
Z_MAIN = {
    39633421919584782: 0.01507489
}  # main bright redrock z (SPARCL, 2026-11-10 sparcl_spectra.npz)
CACHE = Path(
    os.environ.get(
        "SSP_CACHE", "/scratch/11006/nikhilgaruda/hubersed_data/ssp_cache/c3k_er_afe_v3.npz"
    )
)
REST_MIN, REST_MAX = 3320.0, 9000.0
POCO = dict(n_effective=1024, n_active=512, n_max_steps=None, n_total=4096, n_evidence=4096)
# the #14 priors and starting values, copied from 2026-10-31_poco_c3k_er_dark_39633433328091196/fit.py
HYPER_PRIORS = {
    "sigma_dyn": (LogUniform(mini=0.01, maxi=1.0), 0.05),
    "tau_dyn": (LogUniform(mini=0.001, maxi=0.02), 0.005),
    "sigma_reg": (LogUniform(mini=0.1, maxi=3.0), None),
    "tau_eq": (LogUniform(mini=0.01, maxi=1.0), 0.1),
}
STOCHASTIC = ("agebins", "mass", "logsfr_ratios", "sfh", "sigma_reg", "tau_eq", "tau_in",
              "sigma_dyn", "tau_dyn")  # fmt: skip
PSB = ("sfh", "tflex", "nflex", "nfixed", "tlast", "logsfr_ratio_young", "logsfr_ratio_old",
       "logsfr_ratios", "mass", "agebins")  # fmt: skip
GIB_WORKER = 6.5  # commit per worker after the speclib free, as rebuild.py of the mock PSB run
GIB_MARGIN = 8.0
TORCH_THREADS = 16  # pocoMC's flow runs in the parent; it keeps this many cores, workers the rest
_LOADED_KEY = {}

if os.environ.get("SPEC_DATA"):  # staged spender files; Corral is not mounted on compute nodes
    desi_io.DATA_PATH = Path(os.environ["SPEC_DATA"])


def split_name(name):
    """Return the model part and the coadd ("sv3", "main" or "recoadd") of a name such as AC_main."""
    base, _, coadd = name.partition("_")
    return base, coadd or "sv3"


def matrix_path(tid, coadd="sv3"):
    """Return the resolution matrix file of ``tid`` for its SV3 coadd, main coadd or SV3 recoadd."""
    if coadd == "main":
        return OUT / str(tid) / "resolution_matrix_main.npz"
    if coadd == "recoadd":
        return OUT / str(tid) / "recoadd" / "resolution_matrix_rows_ivar.npz"
    if tid == TID_REPRO:
        return REPEATS / str(tid) / "resolution_matrix.npz"
    return OUT / str(tid) / "resolution_matrix.npz"


def load_data(tid, coadd="sv3"):
    """Return z, flux and uncertainty in maggies, fitted pixels and the R file, as poco_repeats.

    Parameters
    ----------
    coadd : str
        "sv3" for the SV3 coadd and R file, "main" for the main coadd R file at the main redrock z,
        "recoadd" for the SV3 recoadd (row-weighted R, flux-calibration ivar) on spender's good
        pixels, as 2026-10-29_continuum_recoadd.
    """
    wave = WAVE_OBS * u.AA
    if coadd == "recoadd":
        s = load_spectrum(tid)
        z = float(s.redshift.value)
        r = load_resolution_npz(matrix_path(tid, "recoadd"))
        assert np.allclose(r["wave"], WAVE_OBS, rtol=0, atol=1e-3), "R grid differs from WAVE_OBS"
        flux = to_maggies(wave, r["flux"] * DESI_FLAM).value
        iv = ivar_to_maggies(wave, r["ivar"] * DESI_FLAM**-2).value
        rest = WAVE_OBS / (1 + z)
        good = (~s.mask & np.isfinite(flux) & np.isfinite(iv) & (iv > 0)
                & (rest >= REST_MIN) & (rest < REST_MAX))  # fmt: skip
        unc = np.where(good, 1 / np.sqrt(np.where(good, iv, 1.0)), np.inf)
        return z, np.where(good, flux, 0.0), unc, good, r
    if coadd == "main":
        z = Z_MAIN[tid]
        r = load_resolution_npz(matrix_path(tid, "main"))
        assert np.allclose(r["wave"], WAVE_OBS, rtol=0, atol=1e-3), "R grid differs from WAVE_OBS"
        rest = WAVE_OBS / (1 + z)
        # spender's pixel cuts, as in the SV3 chunk files: ivar > 1e-6 and its sky-line mask
        from spender.data.desi import DESI

        sky = np.asarray(DESI._skyline_mask, bool)
        assert sky.shape == WAVE_OBS.shape, "sky-line mask grid differs from WAVE_OBS"
        good = (r["ivar"] > 1e-6) & ~sky & np.isfinite(r["flux"])
        good &= (rest >= REST_MIN) & (rest < REST_MAX)
        flux = to_maggies(wave, r["flux"] * DESI_FLAM).value
        iv = ivar_to_maggies(wave, np.where(good, r["ivar"], 1.0) * DESI_FLAM**-2).value
        unc = np.where(good, 1 / np.sqrt(iv), np.inf)
        return z, np.where(good, flux, 0.0), unc, good, r
    s = load_spectrum(tid)
    z = float(s.redshift.value)
    flux = to_maggies(wave, s.flux).value
    iv = ivar_to_maggies(wave, s.uncertainty.quantity).value
    rest = WAVE_OBS / (1 + z)
    r = load_resolution_npz(matrix_path(tid))
    assert np.allclose(r["wave"], WAVE_OBS, rtol=0, atol=1e-3), "R grid differs from WAVE_OBS"
    good = (
        ~s.mask
        & np.isfinite(flux)
        & np.isfinite(iv)
        & (iv > 0)
        & (r["ivar"] > 0)
        & (rest >= REST_MIN)
        & (rest < REST_MAX)
    )
    check = to_maggies(wave, r["flux"] * DESI_FLAM).value
    assert np.allclose(check[good], flux[good], rtol=1e-5, atol=0), "R file flux differs"
    unc = np.where(good, 1 / np.sqrt(np.where(good, iv, 1.0)), np.inf)
    return z, np.where(good, flux, 0.0), unc, good, r


def params_a(z):
    """Return the parameter template of model A at redshift ``z``."""
    _, t = build_continuum_model(z)
    t["add_neb_emission"] = {"N": 1, "isfree": False, "init": False}
    t["tau_in"]["isfree"] = False
    t["tau_in"]["init"] = universe_age_gyr(z)
    for name, (prior, init) in HYPER_PRIORS.items():
        t[name]["isfree"] = True
        t[name]["prior"] = prior
        if init is not None:
            t[name]["init"] = init
    t["afe"] = {"N": 1, "isfree": True, "init": 0.0, "prior": TopHat(mini=-0.2, maxi=0.6)}
    return t


def params_b(z):
    """Return the parameter template of model B, model A with the PSB SFH of prospect."""
    from prospect.models.templates import TemplateLibrary

    t = params_a(z)
    for k in STOCHASTIC:
        t.pop(k)
    psb = copy.deepcopy(TemplateLibrary["continuity_psb_sfh"])
    for k in PSB:
        t[k] = psb[k]
    tflex, nflex = float(t["tflex"]["init"]), int(t["nflex"]["init"])
    young = float(10 ** psb["agebins"]["init"][0, 1])  # 0.2 Gyr, the tlast init
    edges = np.r_[1.0, young, np.linspace(young + 0.2e9, tflex * 1e9, nflex),
                  np.linspace(tflex * 1e9, universe_age_gyr(z) * 1e9, 4)[1:]]  # fmt: skip
    t["agebins"]["init"] = np.log10(np.array([edges[:-1], edges[1:]]).T)
    return t


def add_cue(t):
    """Return template ``t`` with the Cue nebular block and a free narrow line width."""
    from prospect.models.templates import TemplateLibrary

    t = copy.deepcopy(t)
    t.update(copy.deepcopy(TemplateLibrary["cue_stellar_nebular"]))
    t["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}
    t["eline_sigma"] = {"N": 1, "isfree": True, "init": 100.0, "units": "km/s",
                        "prior": TopHat(mini=10.0, maxi=250.0)}  # fmt: skip
    return t


def ladder_module():
    """Return the 2026-11-07_emline_ladder fit.py module, loaded by path once."""
    import importlib.util

    if "fit_emline_ladder" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "fit_emline_ladder",
            PATHS["ROOT"] / "experiments" / "2026-11-07_emline_ladder_39627770174637084" / "fit.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["fit_emline_ladder"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["fit_emline_ladder"]


def model_l(z, floor):
    """Return model L, ladder step 7 (stochastic SFH, Cue, free and broad lines, ratio widths).

    Parameters
    ----------
    floor : float
        The sigma_smooth floor (library resolution).
    """
    import json

    from prospect.models.sedmodel import HyperSpecModel

    from hubersed.sps.line_columns import build_columns

    lad = ladder_module()
    params = lad.make_params(z, None, lines=True, floor=floor, widths="ratio")
    names = HyperSpecModel(params).emline_info["name"]
    columns = build_columns(names, json.loads(lad.LIMITS.read_text()))
    return HyperSpecModel(params, line_columns=columns)


def make_model(name, z, floor=None):
    """Return model ``name`` (A, B, AC, BC, AC0 or L, "_main" suffix ignored) at redshift ``z``."""
    from prospect.models.sedmodel import HyperSpecModel, SpecModel

    base, _ = split_name(name)
    if base == "L":
        assert floor is not None, "model L needs the sigma_smooth floor"
        return model_l(z, floor)
    t = params_a(z) if base[0] == "A" else params_b(z)
    if "C" in base:
        t = add_cue(t)
    if base.endswith("0"):
        t["afe"] = {"N": 1, "isfree": False, "init": 0.0}
    return HyperSpecModel(t) if base[0] == "A" else SpecModel(t)


def get_sps(cue=False):
    """Return the continuum or Cue source, checked to be C3K ER with five afe slots."""
    sps = build_cue_sps() if cue else build_sps(zcontinuous=1, zero_library_resolution=False)
    assert sps.ssp.libraries[1] in ("c3k_er", b"c3k_er"), sps.ssp.libraries
    assert fsps.fsps.driver.get_nafe() == 5, "FSPS was built without AFE_FLAG"
    return sps


def build(tid, name):
    """Return ([observation], model, sps) of ``tid`` and model ``name``, the likelihood builder."""
    from prospect.observation import ResolutionMatrixSpectrum

    base, coadd = split_name(name)
    z, flux, unc, good, r = load_data(tid, coadd)
    obs = ResolutionMatrixSpectrum(
        wavelength=r["wave"], resolution_matrix=r["res"], flux=flux, uncertainty=unc, mask=good
    )
    obs.rectify()
    sps = get_sps(cue="C" in base or base == "L")
    floor = library_sigma_floor(sps, REST_MIN, REST_MAX) if base == "L" else None
    return [obs], make_model(base, z, floor), sps


_PROBE = {}


def probe(args):
    """Return lnL at theta in this process, whether the SSP cache key held, and the call time."""
    from prospect.fitting import lnprobfn

    tid, name, theta = args
    if (tid, name) not in _PROBE:
        _PROBE[(tid, name)] = build(tid, name)
    observations, model, sps = _PROBE[(tid, name)]
    t = time.time()
    lnl = lnprobfn(np.asarray(theta), model=model, observations=observations, sps=sps, nested=True)
    return lnl, fsps.fsps._ssp_cache_key == _LOADED_KEY.get("key"), time.time() - t


def repro(pool):
    """Check model A data and lnlike against the poco_repeats max-L point of TID_REPRO."""
    d = np.load(REPEATS / str(TID_REPRO) / f"{TID_REPRO}_poco_maxl.npz")
    z, flux, unc, good, _ = load_data(TID_REPRO)
    model = make_model("A", z)
    labels = list(model.theta_labels())
    print(f"repro {TID_REPRO}: z {z:.6f} vs {float(d['z']):.6f}; labels equal "
          f"{labels == [str(x) for x in d['labels']]}; good equal {np.array_equal(good, d['good'])}; "
          f"max |flux diff| {np.max(np.abs(flux - d['flux'])):.2e}; unc equal on good "
          f"{np.allclose(unc[good], d['unc'][good], rtol=1e-12, atol=0)}")  # fmt: skip
    assert labels == [str(x) for x in d["labels"]] and np.array_equal(good, d["good"])
    out = pool.map(probe, [(TID_REPRO, "A", d["best"])] * 2)
    main = probe((TID_REPRO, "A", d["best"]))
    observations, model, sps = _PROBE[(TID_REPRO, "A")]
    spec = np.asarray(model.predict(d["best"], observations, sps=sps)[0][0], float)
    dmod = np.abs(spec - d["sp_best"])[good] / unc[good]
    print(f"repro lnlike: saved {float(d['lnlike']):.6f}, parent {main[0]:.6f}, pool "
          f"{[round(o[0], 6) for o in out]}; key held {main[1]} {[o[1] for o in out]}; "
          f"model vs saved sp_best max |d|/sigma {dmod.max():.4f}, rms {np.sqrt(np.mean(dmod**2)):.4f}")  # fmt: skip
    # tolerances of 2026-11-06_rmatrix_port_check (old hubersed vs new prospect R path)
    assert abs(main[0] - float(d["lnlike"])) < 1, "model A does not reproduce poco_repeats lnlike"
    assert dmod.max() < 0.1, "model A does not reproduce the poco_repeats max-L spectrum"
    assert main[1] and all(o[1] for o in out), "a likelihood call changed the SSP cache key"
    print("repro done")


def check(pool, tid, name):
    """Print the setup, compare parent and pool lnL at prior draws, check the key, one batch."""
    from prospect.fitting import fit_model

    observations, model, _ = build(tid, name)
    obs = observations[0]
    print(f"{tid} model {name}: {model.ndim} free, n_pix {int(obs.mask.sum())}")
    for k in model.free_params:
        print(f"  {k:20s} {model.config_dict[k]['prior']}")
    if name == "B":
        print(f"  agebins init (Gyr): {np.round(10 ** model.params['agebins'] / 1e9, 3).tolist()}")
    rng = np.random.default_rng(0)
    thetas, main, rejected = [], [], 0
    while len(thetas) < 3 and rejected < 50:
        th = model.prior_transform(rng.uniform(size=model.ndim))
        m = probe((tid, name, th))
        if np.isfinite(m[0]):
            thetas.append(th)
            main.append(m)
        else:
            rejected += 1
    print(f"prior draws: {rejected} with lnL -inf before {len(thetas)} finite")
    assert len(thetas) == 3, "too many non-finite lnL at prior draws"
    work = pool.map(probe, [(tid, name, th) for th in thetas])
    print(f"lnL parent {[round(m[0], 4) for m in main]}, pool {[round(w[0], 4) for w in work]}; "
          f"call s parent {[round(m[2], 2) for m in main]}, pool {[round(w[2], 2) for w in work]}")  # fmt: skip
    assert all(m[0] == w[0] for m, w in zip(main, work, strict=True)), "parent and pool differ"
    assert all(m[1] for m in main) and all(w[1] for w in work), "a call changed the SSP cache key"

    class OneBatch(Exception):
        """Raised after the first batch."""

    class FirstBatchPool:
        """Run one batch on the pool, then stop the sampler."""

        def map(self, f, tasks):
            t0 = time.time()
            res = pool.map(f, list(tasks))
            print(f"first pocoMC batch: {len(res)} calls in {time.time() - t0:.0f} s, "
                  f"{np.isfinite(res).mean():.3f} finite", flush=True)  # fmt: skip
            raise OneBatch

    try:
        fit_model(observations, model, None, likelihood_builder=partial(build, tid, name),
                  nested_sampler="pocomc", nested_target_n_effective=POCO["n_total"],
                  pool=FirstBatchPool(), random_state=0,
                  **{k: POCO[k] for k in ("n_effective", "n_active", "n_max_steps")})  # fmt: skip
    except OneBatch:
        print("check done")


def run(pool, tid, name):
    """Run or resume pocoMC through fit_model and write the result."""
    from prospect.fitting import fit_model
    from prospect.io.write_results import write_hdf5

    observations, model, _ = build(tid, name)
    folder = OUT / str(tid) / name
    folder.mkdir(parents=True, exist_ok=True)
    resume = latest_state(folder)
    print(f"{tid} model {name}: {model.ndim} parameters, resume from {resume}", flush=True)
    run_params = dict(
        nested_sampler="pocomc",
        nested_target_n_effective=POCO["n_total"],
        n_effective=POCO["n_effective"],
        n_active=POCO["n_active"],
        n_max_steps=POCO["n_max_steps"],
        n_evidence=POCO["n_evidence"],
        random_state=0,
        save_every=1,
        output_dir=str(folder),
        output_label="pmc",
        pytorch_threads=TORCH_THREADS,
        verbose=True,  # pocoMC progress bar in the log: iteration, beta, calls, ESS, log Z
    )
    out = fit_model(observations, model, None, likelihood_builder=partial(build, tid, name),
                    pool=pool, resume_state_path=None if resume is None else str(resume),
                    **run_params)  # fmt: skip
    sampling = out["sampling"]
    print(f"done in {sampling['duration']:.0f} s", flush=True)
    write_hdf5(str(folder / "fit.h5"), run_params, model, observations, sampling, None)


def commit_room():
    """Return CommitLimit minus Committed_AS of this node in GiB."""
    with open("/proc/meminfo") as f:
        info = {line.split(":")[0]: int(line.split()[1]) for line in f}
    return (info["CommitLimit"] - info["Committed_AS"]) / 2**20


def pin_to_core(counter, cores):
    """Pin this pool worker to its own core, skipping the first TORCH_THREADS cores."""
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    os.sched_setaffinity(0, {cores[TORCH_THREADS + i % (len(cores) - TORCH_THREADS)]})


def cached_pool(npool, gib_worker):
    """Load the SSP cache, free speclib, then fork this node's pinned pool."""
    import fsps_mem

    from hubersed.sps.ssp_cache import load_ssp_cache

    if not CACHE.is_file():
        raise SystemExit(f"no SSP cache at {CACHE}; this script never builds SSPs")
    t = time.time()
    sp = fsps.StellarPopulation(zcontinuous=1)
    for k in ("add_igm_absorption", "add_dust_emission", "add_neb_emission", "nebemlineinspec"):
        sp.params[k] = False
    sp.params["sfh"] = 0
    nslot = load_ssp_cache(sp, CACHE)
    _LOADED_KEY["key"] = fsps.fsps._ssp_cache_key
    freed = fsps_mem.free_speclib()
    room = commit_room()
    cores = sorted(os.sched_getaffinity(0))
    npool = npool or max(1, min(len(cores) - TORCH_THREADS, int((room - GIB_MARGIN) // gib_worker)))
    print(f"{nslot} SSP slots loaded from {CACHE} in {time.time() - t:.0f} s, speclib freed "
          f"{freed}, commit room {room:.1f} GiB, {len(cores)} cores, pool {npool}", flush=True)  # fmt: skip
    assert not jax_backend_started(), "the JAX backend started before the fork"
    ctx = mp.get_context("fork")
    counter = ctx.Value("i", 0)
    pool = ctx.Pool(npool, initializer=pin_to_core, initargs=(counter, cores))
    os.sched_setaffinity(0, set(cores[:TORCH_THREADS]))
    return pool


def main():
    """Load the SSPs, fork the pool, then repro, check or run (one MPI rank per node with --mpi)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("repro", "check", "run"))
    ap.add_argument("--tid", type=int, default=TID_REPRO)
    ap.add_argument(
        "--model",
        choices=("A", "B", "AC", "BC", "AC0", "AC_main", "BC_main", "L", "L_recoadd"),
        default="A",
    )
    ap.add_argument("--pool", type=int, default=0, help="0 sizes it from the commit room")
    ap.add_argument("--gib-worker", type=float, default=GIB_WORKER, help="commit per worker")
    ap.add_argument("--mpi", action="store_true", help="one MPI rank per node, see NodePool")
    args = ap.parse_args()
    if args.action == "repro":
        action = repro
    else:
        action = partial({"check": check, "run": run}[args.action], tid=args.tid, name=args.model)
    with cached_pool(args.pool, args.gib_worker) as pool:
        if not args.mpi:
            action(pool)
            return
        from mpi4py import MPI  # initialises MPI, so only after the fork

        comm = MPI.COMM_WORLD
        print(
            f"rank {comm.Get_rank()} of {comm.Get_size()} on {MPI.Get_processor_name()}", flush=True
        )
        if comm.Get_rank() > 0:
            serve(comm, pool)
            return
        nodes = NodePool(comm, pool)
        try:
            action(nodes)
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            comm.Abort(1)
        nodes.close()


if __name__ == "__main__":
    main()
