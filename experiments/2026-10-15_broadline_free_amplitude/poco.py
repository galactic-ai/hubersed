"""Test pocoMC (preconditioned Monte Carlo) on the free-line fit of 39627770174637084.

QUESTION: the nautilus run of the free variant spends most of its time building bounds on one core
and had not left exploration after 2.7M calls and 102 bounds (ls6 freeline2 chain, 2026-10-16).
Does pocoMC (Karamanis+22, arXiv:2207.05652; pocomc 1.2.6) climb from prior to posterior faster per
call and per hour, and does a new process resume from its saved state?

Same model, data, priors and likelihood as fit.py. pocoMC samples the unit cube with a Uniform(0, 1)
prior and the likelihood applies model.prior_transform, as fit.py does for nautilus, so the
posterior and log Z are those of the model priors. With the ls6 python-fsps (build_ssps) the pool
is forked after the SSP build, as in fit_fork.py; with older builds (the Mac) each worker builds its
own SSPs, ~6 GB each. State files go to ``poco_<TID>_<variant>_neff<n>/`` every ``--save-every``
iterations, and a rerun resumes from the newest one.

With ``--mpi`` (one MPI rank per node, e.g. through ibrun) each rank forks its own pool before MPI
starts, and rank 0 splits each batch of likelihood calls over the ranks (NodePool). A toy run with
the same seed gave the same log Z with a local pool and with two ranks (2026-10-17, Mac mpirun).

RESULT: 2026-10-16, Mac, pool 2, n_effective 256, n_active 128 (poco_run2.log). Stopped at
iteration 25 after 60 min and 87k calls, beta 0.00269. beta grew ~1.6x per iteration to iteration
14 (22k calls), then ~1.25x; calls per iteration rose from ~2k to ~15k as the flow efficiency fell
to ~0.35 (88-126 MCMC steps from iteration 24). A new process resumed from pmc_24.state and
reproduced iteration 25 (beta 0.00269; poco_run3_resume.log). Not reached: beta = 1, the peak and
posterior checks against nautilus.

COMMAND: from the repository root on the Mac with the C3K_ER build,
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
    """Pool over MPI ranks, one per node, each mapping its share with its own forked pool.

    Rank 0 runs the sampler and keeps the first share. The function is sent again only when it
    changes.
    """

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
