"""Time ways to speed up pocoMC's nsf6 flow inverse, the serial part of each MCMC step.

QUESTION: rank 0 of the pocoMC free_dust run (job 3517041) spends about half of each 1.8 s MCMC
step in ``flow.inverse`` on 512 x 30 points (py-spy, 2026-10-17) while the workers wait. How much
faster is the same nsf6 flow with more torch threads, or split by rows over forked processes?

The flow is pocomc.flow.Flow(30, "nsf6") with random weights. Its cost does not depend on the weights.
Rows are inverted independently, so a split result must match the serial one to float32 rounding.

COMMAND: on an idle compute node, ``OMP_WAIT_POLICY=PASSIVE uv run python
experiments/2026-10-17_flow_inverse_speed/analysis.py``. ``--threads`` and ``--chunks`` set the trials.
"""

import argparse
import multiprocessing as mp
import tempfile
import time
from functools import partial
from pathlib import Path

import numpy as np

N_DIM = 30
_CACHE = {}


def make_flow():
    """Return pocoMC's nsf6 flow for 30 parameters."""
    from pocomc.flow import Flow

    return Flow(N_DIM, "nsf6")


def worker_inverse(args):
    """Invert one block of rows with the weights in file ``path``, loading them only when new."""
    import torch
    from pocomc.tools import flow_numpy_wrapper

    path, rows = args
    if _CACHE.get("path") != path:
        torch.set_num_threads(1)
        flow = _CACHE.get("flow") or make_flow()
        flow.flow.load_state_dict(torch.load(path))
        _CACHE.update(path=path, flow=flow, wrap=flow_numpy_wrapper(flow))
    return _CACHE["wrap"].inverse(rows)


def chunked(pool, path, blocks):
    """Return the inverse of the stacked ``blocks``, one pool task per block."""
    out = pool.map(worker_inverse, [(path, b) for b in blocks], chunksize=1)
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


def timed(fn, repeat):
    """Return the median wall time of ``fn`` over ``repeat`` calls, after one warm-up call."""
    fn()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return float(np.median(times))


def main():
    """Time the serial inverse, then threads, then row blocks over a forked pool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--chunks", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    args = parser.parse_args()

    # Fork before torch starts any threads, as poco.py forks before pocomc is imported.
    pool = mp.get_context("fork").Pool(max(args.chunks))

    import torch
    from pocomc.tools import flow_numpy_wrapper

    torch.manual_seed(0)
    flow = make_flow()
    wrap = flow_numpy_wrapper(flow)
    x = np.random.default_rng(0).standard_normal((args.n, N_DIM))
    torch.set_num_threads(1)
    ref = wrap.inverse(x)
    print(f"torch {torch.__version__}, {args.n} x {N_DIM}, median of {args.repeat}")

    for k in args.threads:
        torch.set_num_threads(k)
        t = timed(lambda: wrap.inverse(x), args.repeat)
        diff = np.max(np.abs(wrap.inverse(x)[0] - ref[0]))
        print(f"threads {k:3d}: {t * 1e3:7.1f} ms, max |diff| {diff:.1e}")
    torch.set_num_threads(1)

    tmp = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())
    with tempfile.TemporaryDirectory(dir=tmp) as folder:
        for version, k in enumerate(args.chunks):
            # A new file per trial stands in for one retrained flow per pocoMC iteration.
            path = f"{folder}/flow_{version}.pt"
            t = time.perf_counter()
            torch.save(flow.flow.state_dict(), path)
            save = time.perf_counter() - t
            blocks = np.array_split(x, k)
            t = time.perf_counter()
            chunked(pool, path, blocks)
            first = time.perf_counter() - t
            t = timed(partial(chunked, pool, path, blocks), args.repeat)
            diff = np.max(np.abs(chunked(pool, path, blocks)[0] - ref[0]))
            print(
                f"chunks {k:3d}: {t * 1e3:7.1f} ms (save {save * 1e3:.0f} ms, first call "
                f"{first * 1e3:.0f} ms), max |diff| {diff:.1e}"
            )
    pool.close()


if __name__ == "__main__":
    main()
