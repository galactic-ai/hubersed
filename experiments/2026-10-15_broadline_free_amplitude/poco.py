"""Run pocoMC on the free-line fit of 39627770174637084.

QUESTION: nautilus had not left exploration after 2.7M calls (freeline2, 2026-10-16). Does pocoMC
(arXiv:2207.05652, pocomc 1.2.6) reach the posterior faster, and does it resume from a saved state?

Same model, data, priors and likelihood as fit.py. pocoMC samples the unit cube and the likelihood
applies model.prior_transform. The pool forks after the SSP build, as in fit_fork.py. With ``--mpi``
each node runs one MPI rank with its own pool, and rank 0 splits each batch over the ranks.

RESULT: ls6 free_dust, job 3517041, 2 nodes x 128 workers, 1024/512, cap 300, 2026-10-17. Beta
reached 1 at iteration 62 after 4 segments (~7 h, 5.78M calls), with ESS steady at ~1020. The cap
bound in 5 of iterations 27-33 and in all of 43-60 as eff fell to ~0.25. The first resume reproduced
iteration 33 exactly. Final log Z and posterior pending. Mac test runs: poco_run2.log,
poco_run3_resume.log.

COMMAND: on ls6 through poco.slurm. On the Mac, from the repository root,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix:experiments/2026-10-15_broadline_free_amplitude
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-15_broadline_free_amplitude/poco.py --pool 8``.

OUTPUT: poco_<TID>_<variant>_neff<n>/pmc_<iteration>.state in
results/2026-10-15_broadline_free_amplitude/.
"""

import argparse
import multiprocessing as mp
import re
import sys
import time
import traceback
from functools import partial

import fsps
import numpy as np
from fit_fork import init_worker, jax_backend_started, prebuild_ssps


def latest_state(folder):
    """Return the newest state file in ``folder``, or None if there is none."""
    states = [p for p in folder.glob("pmc_*.state") if re.fullmatch(r"pmc_\d+", p.stem)]
    return max(states, key=lambda p: int(p.stem.split("_")[1])) if states else None


class NodePool:
    """Map over MPI ranks, one per node, each with its own forked pool. Rank 0 runs the sampler."""

    def __init__(self, comm, local):
        self.comm, self.local, self.size = comm, local, comm.Get_size()
        self._sent = None

    def map(self, f, tasks):
        """Return ``[f(t) for t in tasks]``, split over the ranks in contiguous blocks."""
        tasks = list(tasks)
        blocks = [list(b) for b in np.array_split(np.arange(len(tasks)), self.size)]
        new = f is not self._sent
        for r in range(1, self.size):
            self.comm.send((f if new else None, [tasks[i] for i in blocks[r]]), dest=r)
        self._sent = f
        out = self.local.map(f, [tasks[i] for i in blocks[0]])
        for r in range(1, self.size):
            out += self.comm.recv(source=r)
        return out

    def close(self):
        """Stop the other ranks."""
        for r in range(1, self.size):
            self.comm.send("stop", dest=r)


def serve(comm, local):
    """Map the blocks rank 0 sends with the local pool until it says stop."""
    f = None
    while True:
        while not comm.Iprobe(source=0):
            time.sleep(0.005)  # no busy wait on a core the workers use
        msg = comm.recv(source=0)
        if msg == "stop":
            return
        g, block = msg
        f = g if g is not None else f
        comm.send(local.map(f, block), dest=0)


def run(args, pool):
    """Build the sampler and run or resume it."""
    import fit
    import pocomc as pc
    from prospect.fitting.nested import lnlike_of_unit_cube
    from scipy.stats import uniform

    model = fit.process_state(args.variant)[0]
    folder = fit.OUT / f"poco_{fit.TID}_{args.variant}_neff{args.n_effective}"
    lnl = partial(
        lnlike_of_unit_cube,
        prior_transform=model.prior_transform,
        likelihood_function=partial(fit.loglike, variant=args.variant),
    )
    sampler = pc.Sampler(
        prior=pc.Prior(model.ndim * [uniform(0.0, 1.0)]),
        likelihood=lnl,
        n_dim=model.ndim,
        n_effective=args.n_effective,
        n_active=args.n_active,
        n_max_steps=args.n_max_steps,
        pool=pool,
        output_dir=str(folder),
        output_label="pmc",
        random_state=0,
    )
    resume = latest_state(folder)
    print(f"{model.ndim} parameters, resume from {resume}", flush=True)
    t = time.time()
    sampler.run(
        n_total=args.n_total,
        n_evidence=args.n_evidence,
        save_every=args.save_every,
        resume_state_path=resume,
    )
    print(f"done in {time.time() - t:.0f} s, log Z {sampler.evidence()}", flush=True)


def main():
    """Run the FSPS setup and build the SSPs, fork the pool, then run pocoMC with it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="free")
    parser.add_argument("--pool", type=int, default=8)
    parser.add_argument("--n-effective", type=int, default=512)
    parser.add_argument("--n-active", type=int, default=256)
    parser.add_argument("--n-total", type=int, default=4096)
    parser.add_argument("--n-evidence", type=int, default=4096)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--n-max-steps", type=int, default=None, help="None is 10 * n_dim // 2")
    parser.add_argument("--mpi", action="store_true", help="one MPI rank per node, see NodePool")
    args = parser.parse_args()
    init = None
    if hasattr(fsps.StellarPopulation, "build_ssps"):  # the ls6 fork; older builds skip the share
        t = time.time()
        prebuild_ssps(fsps.StellarPopulation(zcontinuous=1))
        print(f"FSPS setup and SSPs {time.time() - t:.0f} s", flush=True)
        init = init_worker
    assert not jax_backend_started(), "the JAX backend started before the fork"
    with mp.get_context("fork").Pool(args.pool, initializer=init) as pool:
        if not args.mpi:
            run(args, pool)
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
            run(args, nodes)
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            comm.Abort(1)
        nodes.close()


if __name__ == "__main__":
    main()
