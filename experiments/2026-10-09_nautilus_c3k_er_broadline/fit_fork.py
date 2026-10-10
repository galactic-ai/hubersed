"""Run fit.py with forked pool workers that share one FSPS setup and one set of SSPs.

The main process runs the FSPS setup once and builds every SSP slot the fit can reach
with StellarPopulation.build_ssps (python-fsps dc3e41e), then forks the pool. Workers inherit both
copy on write, so each needs about 1 GiB instead of 15 GiB and none waits on SSP builds (job
3500970). python-fsps keeps the inherited SSP cache because the fit's SSP inputs equal the defaults
the main process built with. A worker whose inputs differ warns and rebuilds.

The fork must come before the JAX backend starts (JAX is not fork safe), so workers import
prospect in their initializer. Each worker is pinned to one core first, so XLA makes ~15 threads
instead of ~300 and the pool stays under the 16384-thread user limit on ls6 (job 3500937).

Takes the same options as fit.py, with the same PYTHONPATH and SPS_HOME, for example
``.venv/bin/python experiments/2026-10-09_nautilus_c3k_er_broadline/fit_fork.py
--forbidden-broad shared --pool 120 --n-batch 1200``.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings

import fsps


def prebuild_ssps(sp):
    """Build every SSP slot a fit at afe 0 can reach, in parallel helper processes.

    FSPS builds the two grid metallicities around ``logzsol`` and the two afe slots around
    ``afe`` (``compute_zdep`` in fsps.f90). At afe 0 these are afe slots 2 and 3 (afe 0 and +0.2)
    with AFE_FLAG, and slot 1 without it, so every metallicity in those afe slots covers the fit.

    Parameters
    ----------
    sp : fsps.StellarPopulation
        A population with ``zcontinuous=1`` and default SSP parameters.
    """
    assert sp.params["afe"] == 0.0
    afe_slots = {1: [1], 5: [2, 3]}[fsps.fsps.driver.get_nafe()]
    sp.build_ssps(afeindx=afe_slots)


def jax_backend_started():
    """Return True if this process has started a JAX backend, which makes forking unsafe."""
    if "jax" not in sys.modules:
        return False
    from jax._src import xla_bridge

    return xla_bridge.backends_are_initialized()


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
    """Pin the worker, build its sources, and warn if they would not keep the inherited SSPs."""
    pin_worker()
    from hubersed.fitting.map_fits import get_sps

    # the same call as fit.loglike, so loglike finds these sources in the get_sps cache
    sources = get_sps(zero_library_resolution=False)
    for name in ("sps", "cue"):
        params = sources[name].ssp.params
        if fsps.fsps._ssp_cache_inputs(params) != fsps.fsps._ssp_cache_key:
            msg = f"worker {name} SSP inputs differ from the inherited SSPs; rebuilding"
            warnings.warn(msg, stacklevel=2)


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
    assert not jax_backend_started(), "the JAX backend started before the fork"
    with mp.get_context("fork").Pool(n, initializer=init_worker) as pool:
        import fit

        fit.main(fit.parse_args(), pool=pool)


if __name__ == "__main__":
    main()
