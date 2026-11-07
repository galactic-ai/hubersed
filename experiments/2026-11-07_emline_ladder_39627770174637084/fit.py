"""Ladder step 5 of 39627770174637084: Cue narrow lines plus free weak and broad line columns.

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

OUTPUT: results/2026-11-07_emline_ladder_39627770174637084/<STEM>/ (pmc_<i>.state, <STEM>.h5).
"""

import argparse
import json
import multiprocessing as mp
import resource
import sys
import time
import traceback

import astropy.units as u
import fsps
import numpy as np
from fit_fork import jax_backend_started, pin_worker, prebuild_ssps
from poco import NodePool, latest_state, serve

from hubersed.conversion import DESI_FLAM, ivar_to_maggies, to_maggies
from hubersed.io.desi import load_resolution_npz, load_spectrum
from hubersed.paths import PATHS
from hubersed.sps.config import build_continuum_model, build_full_cue_model, library_sigma_floor
from hubersed.sps.line_columns import add_line_params, build_columns
from hubersed.sps.parameter_file import WAVE_OBS, build_cue_sps
from hubersed.sps.utils import universe_age_gyr

TID = 39627770174637084
REST_MIN, REST_MAX = 3320.0, 9000.0
DR1_NPZ = (
    PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline" / str(TID) / "resolution_matrix.npz"
)
LIMITS = PATHS["RESULTS"] / "2026-11-06_line_ew_prior" / "ion_limits.json"
OUT = PATHS["RESULTS"] / "2026-11-07_emline_ladder_39627770174637084"
POCO = dict(n_effective=1024, n_active=512, n_max_steps=300, n_total=4096, n_evidence=4096)
STEM = f"step5_dr1_rmatrix_cols_neff{POCO['n_effective']}"


def load_data():
    """Return the redshift, flux and uncertainty in maggies, fitted pixels and the DR1 R.

    Flux and ivar come from load_spectrum, checked equal to the npz flux on the fitted pixels.
    Pixels outside ``good`` have zero flux and infinite uncertainty.
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
    unc = np.where(good, 1 / np.sqrt(np.where(good, ivar, 1.0)), np.inf)
    return z, np.where(good, flux, 0.0), unc, good, r


def make_params(z, cue):
    """Return the parameter specification: Cue model, fixed tau_in, 10-13 hyper priors, lines."""
    from prospect.models.priors import LogUniform

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
    return add_line_params(tmpl)


def build():
    """Return ([observation], model, cue) of step 5. The likelihood builder of fit_model."""
    from prospect.models.sedmodel import HyperSpecModel
    from prospect.observation import ResolutionMatrixSpectrum

    cue = build_cue_sps()
    assert cue.ssp.libraries[1] in ("c3k_er", b"c3k_er"), cue.ssp.libraries
    z, flux, unc, good, r = load_data()
    params = make_params(z, cue)
    names = HyperSpecModel(params).emline_info["name"]
    columns = build_columns(names, json.loads(LIMITS.read_text()))
    model = HyperSpecModel(params, line_columns=columns)
    obs = ResolutionMatrixSpectrum(
        wavelength=r["wave"], resolution_matrix=r["res"], flux=flux, uncertainty=unc, mask=good
    )
    obs.rectify()
    return [obs], model, cue


def check(pool):
    """Print the setup, compare main and pool likelihoods, time a call, send one pocoMC batch."""
    from prospect.fitting import ProcessLikelihood, fit_model, lnprobfn

    t = time.time()
    observations, model, cue = build()
    obs = observations[0]
    print(f"build {time.time() - t:.0f} s; {model.ndim} free: {model.theta_labels()}")
    print(
        f"n_pix {int(obs.mask.sum())}; sigma_smooth prior {model.config_dict['sigma_smooth']['prior']}"
    )
    print("columns: " + ", ".join(f"{c.name} (W {c.ew_max:.1f})" for c in model.line_columns))
    if hasattr(fsps.fsps, "_ssp_cache_key"):
        same = fsps.fsps._ssp_cache_inputs(cue.ssp.params) == fsps.fsps._ssp_cache_key
        print(f"Cue SSP inputs match the prebuilt SSPs: {same}")
    theta = model.theta.copy()
    main = lnprobfn(theta, model=model, observations=observations, sps=cue, nested=True)
    lik = ProcessLikelihood(build, lnprobfn, labels=model.theta_labels(), nested=True)
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
            out = pool.map(f, list(tasks))
            print(f"first pocoMC batch: {len(out)} calls, {np.isfinite(out).mean():.3f} finite")
            raise OneBatch

    try:
        fit_model(
            observations,
            model,
            None,
            likelihood_builder=build,
            nested_sampler="pocomc",
            nested_target_n_effective=POCO["n_total"],
            pool=FirstBatchPool(),
            random_state=0,
            **{k: POCO[k] for k in ("n_effective", "n_active", "n_max_steps")},
        )
    except OneBatch:
        print("check done")


def run(pool):
    """Run or resume pocoMC through fit_model and write the result."""
    from prospect.fitting import fit_model
    from prospect.io.write_results import write_hdf5

    observations, model, _ = build()
    folder = OUT / STEM
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
    )
    out = fit_model(
        observations,
        model,
        None,
        likelihood_builder=build,
        pool=pool,
        resume_state_path=None if resume is None else str(resume),
        **run_params,
    )
    sampling = out["sampling"]
    print(f"done in {sampling['duration']:.0f} s", flush=True)
    write_hdf5(str(folder / f"{STEM}.h5"), run_params, model, observations, sampling, None)


def main():
    """Build the SSPs, fork the pool, then check or run (one MPI rank per node with --mpi)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "run"])
    parser.add_argument("--pool", type=int, default=2)
    parser.add_argument("--mpi", action="store_true", help="one MPI rank per node, see NodePool")
    args = parser.parse_args()
    if hasattr(fsps.StellarPopulation, "build_ssps"):  # the TACC fork; older builds skip the share
        t = time.time()
        prebuild_ssps(fsps.StellarPopulation(zcontinuous=1))
        print(f"FSPS setup and SSPs {time.time() - t:.0f} s", flush=True)
    assert not jax_backend_started(), "the JAX backend started before the fork"
    action = check if args.action == "check" else run
    with mp.get_context("fork").Pool(args.pool, initializer=pin_worker) as pool:
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
