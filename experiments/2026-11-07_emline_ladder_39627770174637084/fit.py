"""Ladder steps 3-8 of 39627770174637084 (``--step``, default 5). Step 5 is described here.

Steps 7 and 8 (2026-11-10) are steps 5 and 6 with the broad widths set as a ratio of the narrow
width, r ~ TopHat(1.2, 40), and the narrow width TopHat(10, 450) km/s (add_line_params widths
"ratio", bounds from 2026-11-10_line_width_split), instead of the 250 km/s split.

Step 3: Cue narrow lines, one width, median DESI Gaussian LSF (map_fits.LSF). Step 4: as 3 with
the own DR1 R instead. Step 5: as 4 with the free line columns below. Same data, pixels, priors
and sampler in all three; steps 3 and 4 keep Cue's eline_sigma prior TopHat(10, 250).


Changed from step 4 (ladder plan): weak ions ([SII], [OI], [ArIII], auroral, [NeIII], [SIII])
replace their Cue lines with free amplitudes, and H, [OII], [OIII], [NII], [SII], [OI] get free
broad components, all under a uniform prior in equivalent width up to W (ion_limits.json of
2026-11-06_line_ew_prior), marginalized by prospect (line_columns). Narrow width TopHat(10, 250),
broad widths TopHat(250, 800), broad offset TopHat(-300, 300) km/s. Data: DR1 coadd from
load_spectrum (spender mask), rest 3320-9000 A, own R from the 10-09 resolution_matrix.npz. C3K ER
with AFE_FLAG at afe 0; sigma_smooth floor = library resolution.

pocoMC through prospect's fit_model with likelihood_builder=build, so each pool worker builds the
data, model and Cue once (Cue cannot be pickled). The pool forks after the FSPS setup and SSP build
and before JAX starts; workers are pinned to one core. With ``--mpi`` each node runs one rank with
its own pool and rank 0 splits each batch (NodePool of 2026-10-15 poco.py).

COMMAND: on the Mac, from the repository root,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-15_broadline_free_amplitude
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-11-07_emline_ladder_39627770174637084/fit.py check --pool 2``.
Use ``run`` instead of ``check`` for the fit.

OUTPUT: results/2026-11-07_emline_ladder_39627770174637084/<STEMS[step]>/ (pmc_<i>.state, .h5).
"""

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import resource
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import astropy.units as u
import fsps
import numpy as np
from fit_fork import jax_backend_started, pin_worker
from poco import NodePool, latest_state, serve

from hubersed.conversion import DESI_FLAM, ivar_to_maggies, to_maggies
from hubersed.io import desi as desi_io
from hubersed.io.desi import load_resolution_npz, load_spectrum
from hubersed.paths import PATHS
from hubersed.sps.config import build_continuum_model, build_full_cue_model, library_sigma_floor
from hubersed.sps.line_columns import add_line_params, build_columns
from hubersed.sps.parameter_file import WAVE_OBS, build_cue_sps
from hubersed.sps.utils import universe_age_gyr

TID = 39627770174637084
REST_MIN, REST_MAX = 3320.0, 9000.0
RECOADD_NPZ = (  # row-weighted recoadd with corrected calibration and ivar (10-09 ivar_recompute.py)
    PATHS["RESULTS"]
    / "2026-10-09_nautilus_c3k_er_broadline"
    / str(TID)
    / "resolution_matrix_rows_ivar.npz"
)
DR1_NPZ = (
    PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline" / str(TID) / "resolution_matrix.npz"
)
LIMITS = PATHS["RESULTS"] / "2026-11-06_line_ew_prior" / "ion_limits.json"
OUT = PATHS["RESULTS"] / "2026-11-07_emline_ladder_39627770174637084"
POCO = dict(n_effective=1024, n_active=512, n_max_steps=300, n_total=4096, n_evidence=4096)
STEMS = {
    1: f"step1_dr1_miles_zeroed_cue_neff{POCO['n_effective']}",
    2: f"step2_dr1_c3khr_degraded_cue_neff{POCO['n_effective']}",
    3: f"step3_dr1_median_cue_neff{POCO['n_effective']}",
    4: f"step4_dr1_rmatrix_cue_neff{POCO['n_effective']}",
    5: f"step5_dr1_rmatrix_cols_neff{POCO['n_effective']}",
    6: f"step6_recoadd_rmatrix_cols_neff{POCO['n_effective']}",
    7: f"step7_dr1_rmatrix_cols_ratio_neff{POCO['n_effective']}",
    8: f"step8_recoadd_rmatrix_cols_ratio_neff{POCO['n_effective']}",
}
RECOADD_STEPS = (6, 8)  # steps fitted to the recoadd instead of the DR1 coadd
if os.environ.get("SPEC_DATA"):  # staged spender files; Corral is not mounted on compute nodes
    desi_io.DATA_PATH = Path(os.environ["SPEC_DATA"])


def load_data(recoadd=False):
    """Return the redshift, flux and uncertainty in maggies, fitted pixels and the R matrix.

    DR1 (default): flux and ivar from load_spectrum, checked equal to the DR1 npz flux on the fitted
    pixels, with the DR1 R. Recoadd (step 6): flux, ivar and R from the recoadd npz on the DR1
    fitted pixels where its ivar is positive, as 2026-10-13 model_setup, without its missing-line
    masks. Pixels outside ``good`` have zero flux and infinite uncertainty.
    """
    s = load_spectrum(TID)
    z = float(s.redshift.value)
    wave = WAVE_OBS * u.AA
    flux = to_maggies(wave, s.flux).value
    ivar = ivar_to_maggies(wave, s.uncertainty.quantity).value
    rest = WAVE_OBS / (1 + z)
    good = (
        np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0) & (rest >= REST_MIN) & (rest < REST_MAX)
    )
    r = load_resolution_npz(DR1_NPZ)
    assert np.allclose(r["wave"], WAVE_OBS, rtol=0, atol=1e-3), "R grid differs from WAVE_OBS"
    check = to_maggies(wave, r["flux"] * DESI_FLAM).value
    assert np.allclose(check[good], flux[good], rtol=1e-5, atol=0), "npz flux differs from DR1"
    if recoadd:
        r = load_resolution_npz(RECOADD_NPZ)
        assert np.allclose(r["wave"], WAVE_OBS, rtol=0, atol=1e-3), "R grid differs from WAVE_OBS"
        good &= r["ivar"] > 0
        flux = to_maggies(wave, r["flux"] * DESI_FLAM).value
        ivar = ivar_to_maggies(wave, np.where(good, r["ivar"], 1.0) * DESI_FLAM**-2).value
    unc = np.where(good, 1 / np.sqrt(np.where(good, ivar, 1.0)), np.inf)
    return z, np.where(good, flux, 0.0), unc, good, r


def make_params(z, cue, lines=True, floor=None, afe_zero=False, widths="ratio"):
    """Return the parameter specification: Cue model, fixed tau_in, 10-13 hyper priors.

    With ``lines`` the line widths and broad offset are added (add_line_params, ``widths``
    "split" for steps 5-6, "ratio" for steps 7-8 and the mocks). ``floor``
    is the sigma_smooth prior floor, by default the library resolution over rest 3320-9000 A.
    ``afe_zero`` adds afe fixed at 0, as the 11-01 c3k_hr step.
    """
    from prospect.models.priors import LogUniform

    if floor is None:
        floor = library_sigma_floor(cue, REST_MIN, REST_MAX)
    cont_model, cont_tmpl = build_continuum_model(z, sigma_floor=floor)
    _, tmpl = build_full_cue_model(cont_tmpl, cont_model.theta, cont_model)
    tmpl["tau_in"]["isfree"] = False
    tmpl["tau_in"]["init"] = universe_age_gyr(z)
    tmpl["sigma_dyn"]["prior"] = LogUniform(mini=0.01, maxi=1.0)
    tmpl["tau_dyn"]["prior"] = LogUniform(mini=0.001, maxi=0.02)
    tmpl["sigma_reg"]["prior"] = LogUniform(mini=0.1, maxi=3.0)
    tmpl["tau_eq"]["prior"] = LogUniform(mini=0.01, maxi=1.0)
    tmpl["sigma_dyn"]["init"] = 0.05
    tmpl["tau_dyn"]["init"] = 0.005
    tmpl["tau_eq"]["init"] = 0.1
    if afe_zero:
        tmpl["afe"] = {"N": 1, "isfree": False, "init": 0.0}
    return add_line_params(tmpl, widths) if lines else tmpl


def f08():
    """Return the 2026-10-08_continuum_outlier fit.py module, loaded by path."""
    path = Path(__file__).resolve().parents[1] / "2026-10-08_continuum_outlier" / "fit.py"
    spec = importlib.util.spec_from_file_location("fit_1008", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_library_step(step):
    """Return ([observation], model, cue) of step 1 (MILES) or 2 (C3K HR), with 10-08's data code.

    Step 1: MILES rest window 3601.8-7400.8 A, data not degraded, library resolution zeroed (10-08
    make_sps), median DESI LSF, sigma_smooth floor 10 km/s (as 09-24). Step 2: whole DESI range in
    the C3K HR window, data degraded to C3K HR + 10 km/s where DESI is sharper, afe fixed at 0,
    sigma_smooth floor = C3K HR resolution over its window.
    """
    from prospect.models.sedmodel import HyperSpecModel

    m = f08()
    lib, window, degrade = ("miles", "miles", False) if step == 1 else ("c3k_hr", "full", True)
    cue = m.make_sps(True, degrade)
    assert cue.ssp.libraries[1] in (lib, lib.encode()), cue.ssp.libraries
    z, flux, unc, good, target_kms = m.load_data(TID, lib, window, degrade)
    if step == 1:
        floor = 10.0
    else:
        floor = library_sigma_floor(cue, *m.rebin.LIBRARIES["c3k_hr"]["window"])
    params = make_params(z, cue, lines=False, floor=floor, afe_zero=step == 2)
    obs = m.make_obs(z, np.where(good, flux, 0.0), unc, good, target_kms, cue)
    return obs, HyperSpecModel(params), cue


def build(step=5):
    """Return ([observation], model, cue) of ``step``. The likelihood builder of fit_model."""
    from prospect.models.sedmodel import HyperSpecModel
    from prospect.observation import ResolutionMatrixSpectrum

    if step in (1, 2):
        return build_library_step(step)

    cue = build_cue_sps()
    assert cue.ssp.libraries[1] in ("c3k_er", b"c3k_er"), cue.ssp.libraries
    z, flux, unc, good, r = load_data(recoadd=step in RECOADD_STEPS)
    params = make_params(z, cue, lines=step >= 5, widths="split" if step in (5, 6) else "ratio")
    if step >= 5:
        names = HyperSpecModel(params).emline_info["name"]
        columns = build_columns(names, json.loads(LIMITS.read_text()))
        model = HyperSpecModel(params, line_columns=columns)
    else:
        model = HyperSpecModel(params)
    if step == 3:
        from hubersed.fitting.map_fits import LSF
        from hubersed.sps.parameter_file import build_obs

        return build_obs(flux, unc, good, resolution=LSF, wavelength=WAVE_OBS), model, cue
    obs = ResolutionMatrixSpectrum(
        wavelength=r["wave"], resolution_matrix=r["res"], flux=flux, uncertainty=unc, mask=good
    )
    obs.rectify()
    return [obs], model, cue


def check(pool, step):
    """Print the setup, compare main and pool likelihoods, time a call, send one pocoMC batch."""
    from prospect.fitting import ProcessLikelihood, fit_model, lnprobfn

    builder = partial(build, step=step)
    t = time.time()
    observations, model, cue = builder()
    print(f"step {step}", flush=True)
    obs = observations[0]
    print(f"build {time.time() - t:.0f} s; {model.ndim} free: {model.theta_labels()}")
    print(
        f"n_pix {int(obs.mask.sum())}; sigma_smooth prior {model.config_dict['sigma_smooth']['prior']}"
    )
    columns = getattr(model, "line_columns", None) or []
    print("columns: " + ", ".join(f"{c.name} (W {c.ew_max:.1f})" for c in columns))
    if hasattr(fsps.fsps, "_ssp_cache_key"):
        same = fsps.fsps._ssp_cache_inputs(cue.ssp.params) == fsps.fsps._ssp_cache_key
        print(f"Cue SSP inputs match the prebuilt SSPs: {same}")
    theta = model.theta.copy()
    main = lnprobfn(theta, model=model, observations=observations, sps=cue, nested=True)
    lik = ProcessLikelihood(builder, lnprobfn, labels=model.theta_labels(), nested=True)
    workers = pool.map(lik, [theta] * 2)
    print(f"lnL at init: main {main:.6f}, pool {workers}; equal {all(w == main for w in workers)}")
    times = []
    for _ in range(20):
        t = time.perf_counter()
        lnprobfn(theta, model=model, observations=observations, sps=cue, nested=True)
        times.append(time.perf_counter() - t)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"median call {1e3 * np.median(times):.1f} ms; main max RSS {rss} (KiB Linux, B macOS)")

    class OneBatch(Exception):
        """Raised by FirstBatchPool after the first batch."""

    class FirstBatchPool:
        """Run one batch on the pool, then stop the sampler."""

        def map(self, f, tasks):
            t0 = time.time()
            out = pool.map(f, list(tasks))
            print(
                f"first pocoMC batch: {len(out)} calls in {time.time() - t0:.0f} s, "
                f"{np.isfinite(out).mean():.3f} finite; commit room {commit_room():.1f} GiB",
                flush=True,
            )
            raise OneBatch

    try:
        fit_model(
            observations,
            model,
            None,
            likelihood_builder=builder,
            nested_sampler="pocomc",
            nested_target_n_effective=POCO["n_total"],
            pool=FirstBatchPool(),
            random_state=0,
            **{k: POCO[k] for k in ("n_effective", "n_active", "n_max_steps")},
        )
    except OneBatch:
        print("check done")


N_DRAWS = 100
_PREDICT = {}


def predict_one(theta, step):
    """Return the model spectrum at ``theta`` and, for step 5, the free-column EWs in Angstrom.

    The EW of a column is its truncated-mean amplitude (box prior respected) times W / upper.
    Columns without fitted pixels get NaN.
    """
    from prospect.models import line_marginalisation as lm

    if step not in _PREDICT:
        _PREDICT[step] = build(step)
    observations, model, cue = _PREDICT[step]
    spec = model.predict(theta, observations, sps=cue)[0][0]
    if step < 5:
        return np.asarray(spec), None
    ew = np.full(len(model.line_columns), np.nan)
    res, used = model._line_marginal, model._line_column_index
    if res is not None and len(used):
        upper = model._line_column_upper
        amp = lm.truncated_mean(res.amplitude, res.covariance, upper, res.blocks)
        fitted = np.zeros(len(used), bool)
        for blk in res.blocks:
            fitted[blk] = True
        w = np.array([model.line_columns[j].ew_max for j in used])
        ew[np.asarray(used)[fitted]] = (amp * w / upper)[fitted]
    return np.asarray(spec), ew


def predict(pool, step):
    """Write predict.npz: data, the max-likelihood model, percentiles of 100 posterior draws, chi.

    Draws are taken by weight from the step's .h5 (seed 0); step 5 also gets the free-column EWs.
    """
    from prospect.io.read_results import results_from

    folder = OUT / STEMS[step]
    res, _, _ = results_from(str(folder / f"{STEMS[step]}.h5"))
    chain, weights, lnlike = res["unstructured_chain"], res["weights"], res["lnlike"]
    ml = int(np.argmax(lnlike))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(chain), size=N_DRAWS, p=weights / weights.sum())
    thetas = [chain[ml]] + [chain[i] for i in pick]
    out = pool.map(partial(predict_one, step=step), thetas)
    observations, model, _ = build(step)
    obs = observations[0]
    spec = np.array([o[0] for o in out])
    flux, unc, mask = np.asarray(obs.flux), np.asarray(obs.uncertainty), np.asarray(obs.mask, bool)
    save = dict(
        wave=np.asarray(obs.wavelength), flux=flux, unc=unc, mask=mask, labels=model.theta_labels(),
        theta_ml=chain[ml], lnlike_ml=lnlike[ml], spec_ml=spec[0],
        spec_p16=np.percentile(spec[1:], 16, axis=0), spec_p50=np.percentile(spec[1:], 50, axis=0),
        spec_p84=np.percentile(spec[1:], 84, axis=0),
        chi_ml=np.where(mask, (flux - spec[0]) / unc, np.nan),
        chi2_ml=float(np.nansum(np.where(mask, ((flux - spec[0]) / unc) ** 2, np.nan))),
        n_pix=int(mask.sum()),
    )  # fmt: skip
    if step >= 5:
        ew = np.array([o[1] for o in out])
        save.update(
            column_names=[c.name for c in model.line_columns],
            column_wmax=[c.ew_max for c in model.line_columns],
            ew_ml=ew[0], ew_p16=np.nanpercentile(ew[1:], 16, axis=0),
            ew_p50=np.nanpercentile(ew[1:], 50, axis=0), ew_p84=np.nanpercentile(ew[1:], 84, axis=0),
        )  # fmt: skip
    np.savez(folder / "predict.npz", **save)
    print(
        f"step {step}: chi2_ml {save['chi2_ml']:.1f} on {save['n_pix']} pixels, wrote predict.npz"
    )


def run(pool, step):
    """Run or resume pocoMC through fit_model and write the result."""
    from prospect.fitting import fit_model
    from prospect.io.write_results import write_hdf5

    builder = partial(build, step=step)
    observations, model, _ = builder()
    folder = OUT / STEMS[step]
    folder.mkdir(parents=True, exist_ok=True)
    resume = latest_state(folder)
    print(f"{model.ndim} parameters, resume from {resume}", flush=True)
    run_params = dict(
        nested_sampler="pocomc",
        nested_target_n_effective=POCO["n_total"],
        n_effective=POCO["n_effective"],
        n_active=POCO["n_active"],
        n_max_steps=POCO["n_max_steps"],
        n_evidence=POCO["n_evidence"],
        random_state=0,
        save_every=2,
        output_dir=str(folder),
        output_label="pmc",
        pytorch_threads=TORCH_THREADS,
    )
    out = fit_model(
        observations,
        model,
        None,
        likelihood_builder=builder,
        pool=pool,
        resume_state_path=None if resume is None else str(resume),
        **run_params,
    )
    sampling = out["sampling"]
    print(f"done in {sampling['duration']:.0f} s", flush=True)
    write_hdf5(str(folder / f"{STEMS[step]}.h5"), run_params, model, observations, sampling, None)


GIB_HELPER = 15.0  # commit of one build_ssps helper (10-31 job 3664748)
GIB_WORKER = 8.0  # commit of one worker after the free: 5.44 GiB without Cue (job 3664817) + Cue
GIB_MARGIN = 8.0  # commit left free on each node
TORCH_THREADS = 16  # pocoMC's flow runs in the parent; it keeps this many cores, workers the rest


def commit_room():
    """Return CommitLimit minus Committed_AS of this node in GiB, NaN where unknown."""
    try:
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) for line in f}
        return (info["CommitLimit"] - info["Committed_AS"]) / 2**20
    except FileNotFoundError:  # macOS
        return float("nan")


def pin_to_core(counter, cores):
    """Pin this pool worker to its own core, skipping the first TORCH_THREADS cores of the parent."""
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    os.sched_setaffinity(0, {cores[TORCH_THREADS + i % (len(cores) - TORCH_THREADS)]})


def parent_pool(ctx, npool, cores):
    """Fork the pinned pool, then keep the parent on the first TORCH_THREADS cores."""
    counter = ctx.Value("i", 0)
    pool = ctx.Pool(npool, initializer=pin_to_core, initargs=(counter, cores))
    os.sched_setaffinity(0, set(cores[:TORCH_THREADS]))
    return pool


def node_pool(npool, gib_worker=GIB_WORKER, free=True):
    """Return this node's forked pool, after the SSP build and the speclib free on TACC builds.

    Stampede3 forbids memory overcommit, so each fork is charged all of its parent's private
    memory. On the TACC FSPS build the parent builds every SSP slot with the flags Cue's source
    uses, frees speclib (fsps_mem of 2026-10-31_poco_c3k_er_dark), and sizes the pool from the
    node's commit room when ``npool`` is 0. Cue is built only after the fork, in each process.
    """
    ctx = mp.get_context("fork")
    if not hasattr(fsps.StellarPopulation, "build_ssps"):  # no prebuild: macOS or an older build
        assert not jax_backend_started(), "the JAX backend started before the fork"
        room = commit_room()
        if np.isnan(room):  # macOS
            return ctx.Pool(npool or 2, initializer=pin_worker)
        cores = sorted(os.sched_getaffinity(0))
        npool = npool or max(
            1, min(len(cores) - TORCH_THREADS, int((room - GIB_MARGIN) // gib_worker))
        )
        print(f"no SSP prebuild; commit room {room:.1f} GiB, {len(cores)} cores, pool {npool}")
        return parent_pool(ctx, npool, cores)
    import fsps_mem

    t = time.time()
    sp = fsps.StellarPopulation(zcontinuous=1)
    for k in ("add_igm_absorption", "add_dust_emission", "add_neb_emission", "nebemlineinspec"):
        sp.params[k] = False  # as NebStepBasis sets them
    sp.params["sfh"] = 0
    room = commit_room()
    nbuild = max(1, min(10, int((room - GIB_MARGIN) // GIB_HELPER)))
    sp.build_ssps(processes=nbuild)
    freed = fsps_mem.free_speclib() if free else False
    room = commit_room()
    cores = sorted(os.sched_getaffinity(0))
    npool = npool or max(1, min(len(cores) - TORCH_THREADS, int((room - GIB_MARGIN) // gib_worker)))
    print(
        f"SSPs {time.time() - t:.0f} s with {nbuild} helpers, speclib freed {freed}, commit room "
        f"{room:.1f} GiB, {len(cores)} cores, pool {npool}",
        flush=True,
    )
    assert not jax_backend_started(), "the JAX backend started before the fork"
    return parent_pool(ctx, npool, cores)


def main():
    """Build the SSPs, fork the pool, then check or run (one MPI rank per node with --mpi)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "run", "predict"])
    parser.add_argument("--step", type=int, choices=sorted(STEMS), default=5)
    parser.add_argument("--pool", type=int, default=0, help="0 sizes it from the commit room")
    parser.add_argument("--gib-worker", type=float, default=GIB_WORKER, help="commit per worker")
    parser.add_argument("--no-free", action="store_true", help="keep speclib (MILES build)")
    parser.add_argument("--mpi", action="store_true", help="one MPI rank per node, see NodePool")
    args = parser.parse_args()
    action = partial({"check": check, "run": run, "predict": predict}[args.action], step=args.step)
    with node_pool(args.pool, args.gib_worker, not args.no_free) as pool:
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
