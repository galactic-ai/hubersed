"""Ladder steps 9-10 of 39627770174637084: fit step 7's model with masked lines and sampled broad EWs.

Changes from 2026-11-07 step 7: [OI], [SII], [ArIII]7135, [SIII]6312, 9069, [OIII]4363, [OII]7323,
7332 and [NII]5755 are masked within MASK_KMS of their wavelengths; [NeIII] stays with Cue; there are
no narrow free columns; the broad H, [OII], [OIII] and [NII] columns take their amplitudes from four
rest-frame EW parameters, TopHat(0, W) with W from ion_limits.json, sampled by pocoMC. No term is
added to the likelihood. Step 9 fits the DR1 coadd, step 10 the recoadd.

COMMAND: from the repository root, as 2026-11-07 fit.py,
``PYTHONPATH=<C3K_ER fsps site>:experiments/2026-10-15_broadline_free_amplitude:experiments/2026-10-31_poco_c3k_er_dark_39633433328091196
SPS_HOME=<fsps> uv run --no-sync python experiments/2026-11-25_emline_masked_broad/fit.py check --step 9``.
Use ``run`` for the fit and ``predict`` after it; on the cluster check.slurm and prod.slurm.

OUTPUT: results/2026-11-25_emline_masked_broad/<STEMS[step]>/ (pmc_<i>.state, .h5, predict.npz).
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
from prospect.models.sedmodel import HyperSpecModel
from prospect.sources.constants import ckms, lightspeed
from scipy.sparse import csc_matrix

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
OUT = PATHS["RESULTS"] / "2026-11-25_emline_masked_broad"
POCO = dict(n_effective=1024, n_active=512, n_max_steps=300, n_total=4096, n_evidence=4096)
STEMS = {
    9: f"step9_dr1_masked_broad_neff{POCO['n_effective']}",
    10: f"step10_recoadd_masked_broad_neff{POCO['n_effective']}",
}
RECOADD_STEPS = (10,)  # steps fitted to the recoadd instead of the DR1 coadd
MASK_KMS = 500.0  # half-width of each line mask
MASK_LINES = ("[O I] 6300", "[O I] 6363", "[S II] 6716", "[S II] 6731", "[Ar III] 7135",
              "[S III] 6312", "[S III] 9069", "[O III] 4363", "[O II] 7323", "[O II] 7332",
              "[N II] 5755")  # fmt: skip
BROAD_EW = {"H broad": "ew_broad_h", "[OII] broad": "ew_broad_oii", "[OIII] broad": "ew_broad_oiii",
            "[NII] broad": "ew_broad_nii"}  # fmt: skip
if os.environ.get("SPEC_DATA"):  # staged Parquet dataset; data/ is not mounted on compute nodes
    desi_io.load_spectrum.__defaults__ = (
        Path(os.environ["SPEC_DATA"]) / "datasets" / "desi_sv3_bright_v1",
    )


class SampledColumnModel(HyperSpecModel):
    """HyperSpecModel whose line columns take their amplitudes from rest-frame EW parameters.

    The fork marginalises the column amplitudes in ``_marginalise_columns``. Here each column's flux
    is its EW parameter (``BROAD_EW``) times (1 + z) times the continuum after R' at the column's
    strongest line, the conversion the fork uses for its upper bound. Nothing is added to
    ``_ln_eline_penalty``, so the likelihood is Prospector's Gaussian one.
    """

    def _marginalise_columns(self, obs, rprime, columns, spec, response):
        """Return the line spectrum of the columns at their sampled amplitudes, before the response."""
        idx = np.array([c[0] for c in columns])
        centre = np.array([c[3] for c in columns])
        names = [BROAD_EW[self.line_columns[j].name] for j in idx]
        ew = np.array([float(np.atleast_1d(self.params[n])[0]) for n in names])
        a = 1 + float(np.atleast_1d(self._zred)[0])
        cont = np.interp(centre, obs.wavelength, self._resolved_continuum)
        flux = ew * a * cont * lightspeed / centre**2
        rows = np.concatenate([c[1] for c in columns])
        cols = np.repeat(np.arange(len(columns)), [c[1].size for c in columns])
        fine = csc_matrix((np.concatenate([c[2] for c in columns]), (rows, cols)),
                          shape=(rprime.shape[1], len(columns)))  # fmt: skip
        self._line_marginal = None
        self._line_column_index = idx
        return (rprime @ fine).toarray() @ flux


def line_mask(wave_obs, z, rest, kms=MASK_KMS):
    """Return True at observed wavelengths within ``kms`` of any rest wavelength in ``rest``."""
    centre = np.asarray(rest, float)[None, :] * (1 + z)
    return np.any(np.abs(np.asarray(wave_obs)[:, None] / centre - 1) < kms / ckms, axis=1)


def load_data(recoadd=False):
    """Return the redshift, flux and uncertainty in maggies, fitted pixels and the R matrix.

    Parameters
    ----------
    recoadd : bool
        False takes load_spectrum and the DR1 R, True (step 6) the recoadd npz flux, ivar and R.
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
    """Return the parameter specification with the Cue model, fixed tau_in and 10-13 hyper priors.

    Parameters
    ----------
    lines : bool
        Add the line widths and broad offset (add_line_params).
    floor : float, optional
        The sigma_smooth prior floor, default the library resolution over rest 3320-9000 A.
    afe_zero : bool
        Add afe fixed at 0.
    widths : str
        "split" for steps 5-6, "ratio" for steps 7-8 and the mocks.
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
    """Return ([observation], model, cue) of step 1 (MILES) or 2 (C3K HR), with 10-08's data code."""
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


def build(step=9):
    """Return ([observation], model, cue) of ``step``, the likelihood builder of fit_model."""
    from prospect.models.priors import TopHat
    from prospect.observation import ResolutionMatrixSpectrum

    cue = build_cue_sps()
    assert cue.ssp.libraries[1] in ("c3k_er", b"c3k_er"), cue.ssp.libraries
    z, flux, unc, good, r = load_data(recoadd=step in RECOADD_STEPS)
    params = make_params(z, cue, lines=True, widths="ratio")
    limits = json.loads(LIMITS.read_text())
    for col, name in BROAD_EW.items():
        params[name] = dict(N=1, isfree=True, init=0.1 * limits[col], units="rest-frame EW, A",
                            prior=TopHat(mini=0.0, maxi=limits[col]))  # fmt: skip
    info = SampledColumnModel(params).emline_info
    names = np.char.strip(info["name"].astype(str))
    assert np.isin(MASK_LINES, names).all(), "a masked line is not in the model's line list"
    good = good & ~line_mask(WAVE_OBS, z, info["wave"][np.isin(names, MASK_LINES)])
    columns = [c for c in build_columns(names, limits) if c.name in BROAD_EW]
    model = SampledColumnModel(params, line_columns=columns)
    obs = ResolutionMatrixSpectrum(
        wavelength=r["wave"], resolution_matrix=r["res"], flux=np.where(good, flux, 0.0),
        uncertainty=np.where(good, unc, np.inf), mask=good,
    )  # fmt: skip
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
    print(f"masked lines {len(MASK_LINES)}; likelihood penalty {model._ln_eline_penalty}")
    assert model._ln_eline_penalty == 0, "the likelihood has an extra term"
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
    """Return the model spectrum at ``theta`` and, for step 5, the free-column EWs in Angstrom."""
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
    """Write predict.npz with data, max-likelihood model, percentiles of 100 posterior draws and chi."""
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


GIB_HELPER = 15.0  # commit of one build_ssps helper (10-31)
GIB_WORKER = 8.0  # commit of one worker after the free: 5.44 GiB without Cue + Cue
GIB_MARGIN = 8.0  # commit left free on each node
TORCH_THREADS = 16  # pocoMC's flow runs in the parent; it keeps this many cores, workers the rest
# Saved C3K ER SSP slots (2026-11-13_ssp_cache_c3k_er), next to the staged data on compute nodes.
SSP_CACHE = Path(
    os.environ.get(
        "SSP_CACHE",
        Path(os.environ.get("SPEC_DATA", PATHS["DATA"])) / "ssp_cache" / "c3k_er_afe_v3.npz",
    )
)


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
    """Return this node's forked pool, after the SSPs and the speclib free on TACC builds.

    The SSPs are loaded from SSP_CACHE when it matches this build and its SSP parameters, and built
    otherwise.
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

    from hubersed.sps.ssp_cache import load_ssp_cache

    t = time.time()
    sp = fsps.StellarPopulation(zcontinuous=1)
    for k in ("add_igm_absorption", "add_dust_emission", "add_neb_emission", "nebemlineinspec"):
        sp.params[k] = False  # as NebStepBasis sets them
    sp.params["sfh"] = 0
    try:
        how = f"{load_ssp_cache(sp, SSP_CACHE)} slots loaded from {SSP_CACHE}"
    except (OSError, ValueError, RuntimeError) as err:  # no cache, or one of another build or setup
        print(f"SSP cache not used: {err}", flush=True)
        room = commit_room()
        nbuild = max(1, min(10, int((room - GIB_MARGIN) // GIB_HELPER)))
        sp.build_ssps(processes=nbuild)
        how = f"built with {nbuild} helpers"
    freed = fsps_mem.free_speclib() if free else False
    room = commit_room()
    cores = sorted(os.sched_getaffinity(0))
    npool = npool or max(1, min(len(cores) - TORCH_THREADS, int((room - GIB_MARGIN) // gib_worker)))
    print(
        f"SSPs {time.time() - t:.0f} s, {how}, speclib freed {freed}, commit room "
        f"{room:.1f} GiB, {len(cores)} cores, pool {npool}",
        flush=True,
    )
    assert not jax_backend_started(), "the JAX backend started before the fork"
    return parent_pool(ctx, npool, cores)


def main():
    """Build the SSPs, fork the pool, then check or run (one MPI rank per node with --mpi)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "run", "predict"])
    parser.add_argument("--step", type=int, choices=sorted(STEMS), default=9)
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
