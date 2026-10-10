"""Run fit.py with forked pool workers that share one FSPS setup.

With spawn, every pool worker runs the FSPS setup itself. With C3K_ER and AFE_FLAG a process
holds about 20 GiB at the setup peak and 15 GiB after it, which limits the pool to 8 on a
250 GiB ls6 node. Here the main process runs the FSPS setup once and then forks the pool, so the
workers inherit the setup arrays (speclib among them) copy on write. FSPS only reads them after
setup, so one physical copy serves every worker.

The fork must come before anything imports prospect. Importing prospect starts the JAX backend
(cuejax.utils makes jax arrays as default arguments), and JAX is not fork safe. So this script
imports only fsps before the fork and imports fit.py after it. Each worker imports fit.py and
builds its own Cue and model on its first likelihood call, as with spawn. Its StellarPopulation
skips the setup, but it still builds the SSPs it needs.

Each worker is pinned to one core before it imports JAX. XLA sizes its thread pools to the cores
a process may use, about 300 threads per process on a 128-core ls6 node. With 48 unpinned workers
that passed the 16384 threads a user may run there, and every worker died in pthread_create
(test job 3500937). The workers alternate between the node's two sockets, so their memory spreads
over both NUMA nodes.

A worker that dies is replaced by a fork of the main process, which by then has JAX loaded.

Takes the same options as fit.py. Run from the repository root with the same PYTHONPATH and
SPS_HOME as fit.py, for example ``uv run --no-sync python
experiments/2026-10-09_nautilus_c3k_er_broadline/fit_fork.py --forbidden-broad shared --pool 48
--n-batch 480``.
"""

import argparse
import multiprocessing as mp
import os
import sys

import fsps


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


def main():
    """Run the FSPS setup, fork the pool, then run fit.py's main with that pool."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--pool", type=int, default=4)
    n = pre.parse_known_args()[0].pool
    # prospect makes its StellarPopulation with the default setup flags, and fsps asserts that
    # later ones match the flags the setup ran with
    fsps.StellarPopulation(zcontinuous=1)
    assert fsps.fsps.driver.is_setup
    assert "jax" not in sys.modules, "jax was imported before the fork"
    with mp.get_context("fork").Pool(n, initializer=pin_worker) as pool:
        import fit

        fit.main(fit.parse_args(), pool=pool)


if __name__ == "__main__":
    main()
