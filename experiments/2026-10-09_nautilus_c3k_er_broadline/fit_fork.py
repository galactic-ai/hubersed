"""Run fit.py with forked pool workers that share one FSPS setup and one set of SSPs.

With spawn, every pool worker runs the FSPS setup itself. With C3K_ER and AFE_FLAG a process
holds about 20 GiB at the setup peak and 15 GiB after it, which limits the pool to 8 on a
250 GiB ls6 node. Here the main process runs the FSPS setup once and then forks the pool, so the
workers inherit the setup arrays (speclib among them) copy on write. FSPS only reads them after
setup, so one physical copy serves every worker.

The main process also builds, before the fork, every SSP the fit can reach: all 13 metallicities
at afe 0 and +0.2, the two afe slots FSPS interpolates between at afe 0. It builds them in
parallel helper processes and stores them with get_ssp_slot and set_ssp_slot, which our
python-fsps fork adds at 2c5168f, so the fsps build must include that commit. Without this each worker
builds the SSPs of each new metallicity it meets, about 40 s per metallicity and afe slot on ls6.
A batch ends only when its slowest worker does, so with 48 workers nearly every batch waited on
one of these builds, and the pool ran at 96 calls per minute (test job 3500970). A new
StellarPopulation marks its SSPs out of date on its first spectrum, so each worker sets its
dirtiness to 1 after building its sources. That is valid because the fit sets no SSP parameter
to a value other than the FSPS default the main process built with (only imf_type = 2, which is
the default).

The fork must come before anything imports prospect. Importing prospect starts the JAX backend
(cuejax.utils makes jax arrays as default arguments), and JAX is not fork safe. So this script
imports only fsps before the fork, and each worker imports prospect in its initializer.

Each worker is pinned to one core before it imports JAX. XLA sizes its thread pools to the cores
a process may use, about 300 threads per process on a 128-core ls6 node. With 48 unpinned workers
that passed the 16384 threads a user may run there, and every worker died in pthread_create
(test job 3500937). The workers alternate between the node's two sockets, so their memory spreads
over both NUMA nodes.

A worker that dies is replaced by a fork of the main process, which by then has JAX loaded.

Takes the same options as fit.py. Run from the repository root with the same PYTHONPATH and
SPS_HOME as fit.py, for example ``.venv/bin/python
experiments/2026-10-09_nautilus_c3k_er_broadline/fit_fork.py --forbidden-broad shared --pool 48
--n-batch 480``.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import fsps


def build_slot(args):
    """Build one SSP slot in a helper process and return it with its indices."""
    ns, nt, zi, ai = args
    return (zi, ai, *fsps.fsps.driver.get_ssp_slot(ns, nt, zi, ai))


def prebuild_ssps(sp):
    """Build every SSP slot a fit at afe 0 can reach, in parallel helper processes.

    FSPS builds the two grid metallicities around ``logzsol`` and the two afe slots around
    ``afe`` (``compute_zdep`` in fsps.f90). At afe 0 these are afe slots 2 and 3 (afe 0 and +0.2)
    with AFE_FLAG, and slot 1 without it. So every metallicity in those afe slots covers the fit.
    This process first builds the slots around solar itself. That sets the SSP parameters and the
    IMF variables ssp_gen keeps in sps_vars, which the helpers then inherit. Each forked helper
    builds one slot with get_ssp_slot, and this process stores it with set_ssp_slot. The copy is
    exact, so the stored slots equal the ones this process would build.

    Parameters
    ----------
    sp : fsps.StellarPopulation
        A population with ``zcontinuous=1`` and default SSP parameters.
    """
    drv = fsps.fsps.driver
    assert sp.params["afe"] == 0.0
    sp.params["logzsol"] = 0.0
    sp.get_spectrum(tage=1.0)
    afe_slots = {1: [1], 5: [2, 3]}[drv.get_nafe()]
    ns, nt = drv.get_nspec(), drv.get_ntfull()
    jobs = [(ns, nt, zi, ai) for zi in range(1, len(sp.zlegend) + 1) for ai in afe_slots]
    n = min(len(jobs), len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 4)
    with mp.get_context("fork").Pool(n) as helpers:
        for zi, ai, spec, mass, lbol in helpers.imap_unordered(build_slot, jobs):
            drv.set_ssp_slot(zi, ai, spec, mass, lbol)


def pin_worker():
    """Pin this pool worker to one core, alternating between the two halves of the core list."""
    if not hasattr(os, "sched_setaffinity"):  # macOS
        return
    cores = sorted(os.sched_getaffinity(0))
    half = len(cores) // 2
    order = [c for pair in zip(cores[:half], cores[half:], strict=True) for c in pair]
    order += cores[2 * half :]
    # _identity is the worker's 1-based number in its pool; replacement workers count on
    i = mp.current_process()._identity[0] - 1
    os.sched_setaffinity(0, {order[i % len(order)]})


def init_worker():
    """Pin the worker, build its sources, and keep the SSPs it inherited from the main process."""
    pin_worker()
    from hubersed.fitting.map_fits import get_sps

    # the same call as fit.loglike, so loglike finds these sources in the get_sps cache
    sources = get_sps(zero_library_resolution=False)
    for src in (sources["sps"], sources["cue"]):
        src.ssp.params.dirtiness = 1


def main():
    """Run the FSPS setup and build the SSPs, fork the pool, then run fit.py's main with it."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--pool", type=int, default=4)
    n = pre.parse_known_args()[0].pool
    t = time.time()
    # prospect makes its StellarPopulation with the default setup flags, and fsps asserts that
    # later ones match the flags the setup ran with
    sp = fsps.StellarPopulation(zcontinuous=1)
    assert fsps.fsps.driver.is_setup
    print(f"FSPS setup {time.time() - t:.0f} s", flush=True)
    t = time.time()
    prebuild_ssps(sp)
    print(f"SSPs for {len(sp.zlegend)} metallicities built in {time.time() - t:.0f} s", flush=True)
    assert "jax" not in sys.modules, "jax was imported before the fork"
    with mp.get_context("fork").Pool(n, initializer=init_worker) as pool:
        import fit

        fit.main(fit.parse_args(), pool=pool)


if __name__ == "__main__":
    main()
