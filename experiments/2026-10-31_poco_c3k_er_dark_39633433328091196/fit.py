"""pocoMC posterior of the C3K ER continuum fit of the SV3 dark spectrum of 39633433328091196 (#17).

Changes from 2026-10-30_continuum_39633433328091196_dark_map/fit.py (dark MAP, spr job 3662995):
pocoMC 1.2.6 instead of the MAP, and sigma_reg, tau_eq, sigma_dyn and tau_dyn are free with the
#14 priors (2026-10-13_nautilus_resolution_matrix/model_setup.py, the 2026-09-25 priors that the
10-08 runs also used) instead of fixed at the C3K_HR max-L of the SV3 bright spectrum. Data,
resolution matrix, rest window 3320 to 9000 A, tau_in and the other priors are those of the dark
MAP. Sampler settings as #14 (2026-10-15_broadline_free_amplitude/poco.py): n_effective 1024,
n_active 512, unit-cube prior with the likelihood from lnprobfn(nested=True).

Stampede3 forbids memory overcommit, so the pool is not forked. Rank 0 runs the sampler and holds no
FSPS. Every other rank builds its own model, observation and C3K_ER source and evaluates the points
rank 0 sends, one point at a time to whichever rank is free. On each node at most ``--init-width``
ranks build FSPS at once, since one build peaks at about 25 GiB.

Steps: ``check`` (one process) checks the model against the dark MAP model at the MAP point and
times warm calls at prior draws. ``run`` (under ibrun or mpiexec) runs pocoMC or resumes the newest
state, after checking that every rank gives the same likelihood and logging each rank's memory.

FSPS must be the C3K_ER build with AFE_FLAG first on PYTHONPATH, with SPS_HOME holding the ER
spectra. Run on stampede3 through test.slurm, for example
``ibrun <venv>/bin/python experiments/2026-10-31_poco_c3k_er_dark_39633433328091196/fit.py run``.
"""

import argparse
import importlib.util
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from prospect.fitting import lnprobfn
from prospect.models.priors import LogUniform, TopHat

from hubersed.paths import PATHS
from hubersed.sps.config import build_continuum_model
from hubersed.sps.resolution_model import ResolutionMatrixModel
from hubersed.sps.utils import universe_age_gyr

EXPS = Path(__file__).resolve().parents[1]
OUT = PATHS["RESULTS"] / "2026-10-31_poco_c3k_er_dark_39633433328091196"
DARK = PATHS["RESULTS"] / "2026-10-30_continuum_39633433328091196_dark_map"
HYPERS = ("sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn")
# the #14 priors and starting values, as 2026-10-13_nautilus_resolution_matrix/model_setup.py:128-134
HYPER_PRIORS = {
    "sigma_dyn": (LogUniform(mini=0.01, maxi=1.0), 0.05),
    "tau_dyn": (LogUniform(mini=0.001, maxi=0.02), 0.005),
    "sigma_reg": (LogUniform(mini=0.1, maxi=3.0), None),
    "tau_eq": (LogUniform(mini=0.01, maxi=1.0), 0.1),
}

# the dark MAP fit.py, loaded by path since this file has the same name; it patches the 10-28 one
_spec = importlib.util.spec_from_file_location(
    "fit_dark", EXPS / "2026-10-30_continuum_39633433328091196_dark_map" / "fit.py"
)
dark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dark)
base = dark.base

_STATE = {}


def make_model(z):
    """Return the dark MAP model with the four SFH hyperparameters free under the #14 priors."""
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
    return ResolutionMatrixModel(t)


def state(with_sps=True):
    """Return this process's model, observation and source, built on first use."""
    if "model" not in _STATE:
        z, flux, unc, good = base.load_data()
        _STATE.update(model=make_model(z), obs=base.make_obs(flux, unc, good), z=z, good=good)
    if with_sps and "sps" not in _STATE:
        _STATE["sps"] = base.get_sps()
    return _STATE["model"], _STATE["obs"], _STATE.get("sps")


def loglike_unit(u):
    """Return the log likelihood of the unit-cube point ``u``."""
    model, obs, sps = state()
    theta = model.prior_transform(np.asarray(u, float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(lnprobfn(theta, model=model, observations=obs, sps=sps, nested=True))


def memory():
    """Return VmHWM and VmRSS of this process and the node's Committed_AS, in GiB."""
    out = {}
    for path, keys in (
        ("/proc/self/status", ("VmHWM", "VmRSS")),
        ("/proc/meminfo", ("Committed_AS",)),
    ):
        try:
            with open(path) as f:
                for line in f:
                    k = line.split(":")[0]
                    if k in keys:
                        out[k] = int(line.split()[1]) / 2**20
        except FileNotFoundError:  # macOS
            pass
    return {k: out.get(k, float("nan")) for k in ("VmHWM", "VmRSS", "Committed_AS")}


def check(n_calls):
    """Check the model against the dark MAP model and time warm calls at prior draws."""
    t = time.time()
    model, obs, sps = state()
    print(f"build {time.time() - t:.0f} s, memory {memory()}", flush=True)
    labels = model.theta_labels()
    print(f"z {_STATE['z']:.5f}, fitted px {_STATE['good'].sum()}, {model.ndim} free: {labels}")
    for k in model.free_params:
        print(f"  {k:16s} {model.config_dict[k]['prior']}")
    # the dark MAP point, with the hyperparameters it had fixed, under both models
    ref = base.make_model(_STATE["z"])
    m = np.load(DARK / f"{base.TID}_map.npz")
    point = dict(zip([str(x) for x in m["labels"]], m["best"], strict=True))
    point.update({k: float(ref.params[k][0]) for k in HYPERS})
    assert set(ref.theta_labels()) | set(HYPERS) == set(labels), "free parameters differ"
    for k in ref.free_params:
        assert repr(ref.config_dict[k]["prior"]) == repr(model.config_dict[k]["prior"]), k
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = lnprobfn(
            np.array([point[k] for k in labels]),
            model=model,
            observations=obs,
            sps=sps,
            nested=True,
        )
        b = lnprobfn(
            np.array([point[k] for k in ref.theta_labels()]),
            model=ref,
            observations=obs,
            sps=sps,
            nested=True,
        )
    print(f"lnlike at the dark MAP: this model {a:.6f}, dark MAP model {b:.6f}", flush=True)
    assert np.isclose(a, b, rtol=0, atol=1e-6), "the two models give different likelihoods"
    rng = np.random.default_rng(0)
    times, vals = [], []
    for _ in range(n_calls):
        u = rng.uniform(size=model.ndim)
        t = time.time()
        vals.append(loglike_unit(u))
        times.append(time.time() - t)
    times = np.array(times)
    print(
        f"{n_calls} calls at prior draws: first {times[0]:.1f} s, median of the rest "
        f"{np.median(times[1:]):.3f} s, max {times[1:].max():.1f} s, finite "
        f"{np.isfinite(vals).sum()} of {n_calls}; memory {memory()}",
        flush=True,
    )


class RankPool:
    """Map over the worker ranks, one point at a time to whichever rank is free."""

    def __init__(self, comm):
        self.comm, self.size = comm, comm.Get_size()
        self._sent = set()

    def map(self, f, tasks):
        """Return ``[f(t) for t in tasks]``, computed on ranks 1 to size - 1."""
        from mpi4py import MPI

        tasks = list(tasks)
        out = [None] * len(tasks)
        status = MPI.Status()
        nxt, busy = 0, 0
        for r in range(1, self.size):
            if nxt == len(tasks):
                break
            self._send(f, r, nxt, tasks[nxt])
            nxt, busy = nxt + 1, busy + 1
        while busy:
            i, val = self.comm.recv(source=MPI.ANY_SOURCE, tag=1, status=status)
            out[i] = val
            busy -= 1
            if nxt < len(tasks):
                self._send(f, status.Get_source(), nxt, tasks[nxt])
                nxt, busy = nxt + 1, busy + 1
        return out

    def _send(self, f, rank, i, task):
        """Send task ``i`` to ``rank``, with ``f`` only the first time that rank sees it."""
        new = (rank, id(f)) not in self._sent
        self.comm.send((f if new else None, i, task), dest=rank, tag=0)
        self._sent.add((rank, id(f)))

    def close(self):
        """Stop the worker ranks."""
        for r in range(1, self.size):
            self.comm.send("stop", dest=r, tag=0)


def serve(comm):
    """Evaluate the points rank 0 sends until it says stop."""
    f = None
    while True:
        msg = comm.recv(source=0, tag=0)
        if isinstance(msg, str) and msg == "stop":
            return
        g, i, task = msg
        f = g if g is not None else f
        comm.send((i, f(task)), dest=0, tag=1)


def build_in_turn(comm, width):
    """Build each worker rank's state, at most ``width`` ranks per node at a time."""
    from mpi4py import MPI

    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    nr, ns = node.Get_rank(), node.Get_size()
    if nr >= width:
        node.recv(source=nr - width, tag=7)
    t = time.time()
    if comm.Get_rank() > 0:
        state()
        loglike_unit(np.full(state()[0].ndim, 0.5))  # first call, which builds the first slots
    if nr + width < ns:
        node.send(None, dest=nr + width, tag=7)
    return time.time() - t


GIB_HELPER = 15.0  # commit of one build_ssps helper, a fork of a 14.7 GiB process (job 3664748)
GIB_WORKER = 6.0  # commit of one pool worker after the free, 5.44 GiB measured (job 3664817)
GIB_MARGIN = 8.0  # commit left free on each node


def commit_room():
    """Return CommitLimit minus Committed_AS of this node in GiB, NaN where unknown."""
    try:
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) for line in f}
        return (info["CommitLimit"] - info["Committed_AS"]) / 2**20
    except FileNotFoundError:  # macOS
        return float("nan")


def jax_backend_started():
    """Return True if this process has started a JAX backend, which makes forking unsafe."""
    if "jax" not in sys.modules:
        return False
    from jax._src import xla_bridge

    return xla_bridge.backends_are_initialized()


def pin_worker(counter, cores):
    """Pin this pool worker to its own core, skipping the first core, which the parent keeps."""
    import os

    with counter.get_lock():
        i = counter.value
        counter.value += 1
    os.sched_setaffinity(0, {cores[1 + i % (len(cores) - 1)]})


def node_setup(args):
    """Build every SSP slot, free speclib and fork this node's pool. Runs before MPI starts.

    Each build_ssps helper and each pool worker is a fork, charged under strict overcommit for all of
    its parent's private memory. So the helpers fork while speclib is held (~15 GiB each) and the
    pool forks after the free (~6 GiB each), with their numbers set from this node's commit room.
    """
    import multiprocessing as mp
    import os

    spec = importlib.util.spec_from_file_location(
        "fsps_mem", Path(__file__).with_name("fsps_mem.py")
    )
    fsps_mem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fsps_mem)
    t = time.time()
    model, obs, sps = state()
    room = commit_room()
    nbuild = args.build or max(1, min(10, int((room - GIB_MARGIN) // GIB_HELPER)))
    sps.ssp.build_ssps(processes=nbuild)
    info = dict(build_s=time.time() - t, build_processes=nbuild, room_before=room)
    info["freed"] = False if args.no_free else fsps_mem.free_speclib()
    u0 = np.random.default_rng(1).uniform(size=model.ndim)
    info["lnl0"] = loglike_unit(u0)  # in the parent, after the free: no slot may be missing
    info["room_after_free"] = room = commit_room()
    cores = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    npool = args.pool or max(1, min(len(cores) - 1, int((room - GIB_MARGIN) // GIB_WORKER)))
    info.update(cores=len(cores), pool=npool)
    assert not jax_backend_started(), "the JAX backend started before the fork"
    ctx = mp.get_context("fork")
    counter = ctx.Value("i", 0)
    init = (pin_worker, (counter, cores)) if len(cores) > 1 else (None, ())
    pool = ctx.Pool(npool, initializer=init[0], initargs=init[1])
    vals = pool.map(loglike_unit, [u0] * npool)
    info["workers_equal"] = all(v == info["lnl0"] for v in vals)
    info["room_with_pool"] = commit_room()
    info["parent_memory"] = memory()
    return pool, info


class NodePool:
    """Map over one MPI rank per node, each with its own forked pool, in blocks sized by pool."""

    def __init__(self, comm, local, sizes):
        self.comm, self.local, self.sizes = comm, local, np.asarray(sizes, float)
        self.size = int(self.sizes.sum()) + 1  # sampler plus workers, for the log line
        self._sent = None

    def map(self, f, tasks):
        """Return ``[f(t) for t in tasks]``, split over the nodes in contiguous blocks."""
        tasks = list(tasks)
        edges = np.round(np.cumsum(self.sizes) / self.sizes.sum() * len(tasks)).astype(int)
        starts = np.concatenate([[0], edges[:-1]])
        new = f is not self._sent
        for r in range(1, len(self.sizes)):
            self.comm.send((f if new else None, tasks[starts[r] : edges[r]]), dest=r, tag=0)
        self._sent = f
        out = self.local.map(f, tasks[starts[0] : edges[0]])
        for r in range(1, len(self.sizes)):
            out += self.comm.recv(source=r, tag=1)
        return out

    def close(self):
        """Stop the other nodes."""
        for r in range(1, len(self.sizes)):
            self.comm.send("stop", dest=r, tag=0)


def serve_node(comm, local):
    """Map the blocks rank 0 sends with the local pool until it says stop."""
    f = None
    while True:
        while not comm.Iprobe(source=0, tag=0):
            time.sleep(0.005)  # no busy wait on a core the workers use
        msg = comm.recv(source=0, tag=0)
        if isinstance(msg, str) and msg == "stop":
            return
        g, block = msg
        f = g if g is not None else f
        comm.send(local.map(f, block), dest=0, tag=1)


def run_fork(args):
    """Option 2: one MPI rank per node with a forked pool after the free; rank 0 also samples."""
    pool, info = node_setup(args)
    from mpi4py import MPI  # initialises MPI, so only after the fork

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    infos = comm.gather((rank, MPI.Get_processor_name(), info), root=0)
    sizes = comm.bcast([i[2]["pool"] for i in infos] if rank == 0 else None, root=0)
    if rank > 0:
        serve_node(comm, pool)
        pool.close()
        return
    for r, host, i in infos:
        print(f"rank {r} on {host}: {i}", flush=True)
    lnl = {i[2]["lnl0"] for i in infos}
    assert len(lnl) == 1 and all(i[2]["workers_equal"] for i in infos), f"nodes differ: {infos}"
    print(
        f"all {int(sum(sizes))} workers on {len(sizes)} nodes give lnlike {lnl.pop():.6f}",
        flush=True,
    )
    nodes = NodePool(comm, pool, sizes)
    try:
        sample(args, nodes)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        comm.Abort(1)
    nodes.close()
    pool.close()


def latest_state(folder):
    """Return the newest state file in ``folder``, or None if there is none."""
    states = [p for p in folder.glob("pmc_*.state") if re.fullmatch(r"pmc_\d+", p.stem)]
    return max(states, key=lambda p: int(p.stem.split("_")[1])) if states else None


def run(args):
    """Run or resume pocoMC with rank 0 as sampler and the other ranks as workers."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    assert size > 1, "run needs at least one worker rank"
    build_s = build_in_turn(comm, args.init_width)
    info = comm.gather((rank, MPI.Get_processor_name(), build_s, memory()), root=0)
    # every worker must give the same likelihood at one fixed point
    u0 = np.random.default_rng(1).uniform(size=state(with_sps=False)[0].ndim)
    vals = comm.gather(loglike_unit(u0) if rank > 0 else None, root=0)
    if rank > 0:
        serve(comm)
        print(f"rank {rank} done, memory {memory()}", flush=True)
        return
    for r, host, s, mem in info:
        print(f"rank {r} on {host}: build and first call {s:.0f} s, memory {mem}", flush=True)
    assert len(set(vals[1:])) == 1, f"ranks differ at the test point: {vals[1:]}"
    print(f"all {size - 1} workers give lnlike {vals[1]:.6f} at the test point", flush=True)
    pool = RankPool(comm)
    try:
        sample(args, pool)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        comm.Abort(1)
    pool.close()


def sample(args, pool):
    """Build the sampler on rank 0 and run or resume it."""
    import pocomc as pc
    from scipy.stats import uniform

    class Sampler(pc.Sampler):
        """pocoMC's Sampler, keeping this process's likelihood when a state is loaded.

        pocoMC saves its state with dill, which stores functions of __main__ by value. A loaded
        state would put that copy in place of loglike_unit, and pickling the copy to the pool
        fails (it's not the same object as __main__.loglike_unit, job 3664872 run 2).
        """

        def load_state(self, path):
            keep = self.log_likelihood
            super().load_state(path)
            self.log_likelihood = keep

    ndim = state(with_sps=False)[0].ndim
    folder = OUT / f"poco_neff{args.n_effective}{args.tag}"
    folder.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(
        prior=pc.Prior(ndim * [uniform(0.0, 1.0)]),
        likelihood=loglike_unit,
        n_dim=ndim,
        n_effective=args.n_effective,
        n_active=args.n_active,
        n_max_steps=args.n_max_steps,
        pool=pool,
        output_dir=str(folder),
        output_label="pmc",
        random_state=0,
    )
    resume = latest_state(folder)
    print(f"{ndim} parameters, {pool.size - 1} workers, resume from {resume}", flush=True)
    t = time.time()
    sampler.run(
        n_total=args.n_total,
        n_evidence=args.n_evidence,
        save_every=args.save_every,
        resume_state_path=resume,
    )
    print(f"done in {time.time() - t:.0f} s, log Z {sampler.evidence()}", flush=True)


def main():
    """Parse the step and run it."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("check", "run", "runfork"))
    ap.add_argument("--n-calls", type=int, default=20, help="check: calls at prior draws")
    ap.add_argument("--n-effective", type=int, default=1024)
    ap.add_argument("--n-active", type=int, default=512)
    ap.add_argument("--n-total", type=int, default=4096)
    ap.add_argument("--n-evidence", type=int, default=4096)
    ap.add_argument("--n-max-steps", type=int, default=None, help="None is 10 * n_dim // 2")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--init-width", type=int, default=2, help="FSPS builds at once per node")
    ap.add_argument("--tag", default="", help="suffix of the output folder, for test runs")
    ap.add_argument("--build", type=int, default=None, help="runfork: build_ssps processes")
    ap.add_argument("--pool", type=int, default=None, help="runfork: workers per node")
    ap.add_argument("--no-free", action="store_true", help="runfork: keep speclib (Mac tests)")
    args = ap.parse_args()
    if args.step == "check":
        check(args.n_calls)
    elif args.step == "run":
        run(args)
    else:
        run_fork(args)


if __name__ == "__main__":
    main()
