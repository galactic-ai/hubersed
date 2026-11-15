"""Rebuild a random subset of the v2 mocks with the continuum and the emission lines saved apart.

The stored mocks keep only continuum + lines. This rebuilds mocks with the code that made them
(2026-11-09_mocks_v2_outliers/mocks.py, loaded by path; draw_params, setup_worker and the mock
model class unchanged, mock i drawn from default_rng([SEED, i]) and the same R bank row) and saves,
after model.predict, the returned spectrum, the prospect fork's _resolved_continuum (stars and
nebular continuum, after R) and _resolved_lines (all lines, free columns included). Subset: 20,000
indices from default_rng(0).choice(500000, 20000, replace=False), sorted.
COMMAND: ``python rebuild.py <mocks.py> <out.h5> [--n 20000] [--pool 0] [--gib-worker 8] [--cache
<ssp cache>] [--part k --nparts K]`` (stampede3, run.slurm; pool sized from the node's commit room as
mocks.py). With --cache the SSPs are loaded, not built; with --nparts the sorted subset is split
into K slices (np.array_split) and slice k is rebuilt.
OUTPUT: <out.h5> with index, redshift, flux (returned spectrum), continuum, lines, ok.
"""

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

N_TOTAL, SUBSET_SEED = 500_000, 0
_M = {}


def mocks_module(path):
    """Return mocks.py loaded once per process."""
    if "m" not in _M:
        spec = importlib.util.spec_from_file_location("mocks_v2", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _M["m"] = mod
    return _M["m"]


def one(m, i, r):
    """Return (flux, continuum, lines, z, ok) of mock i with R bank row r, as mocks.one_mock."""
    from prospect.observation import ResolutionMatrixSpectrum

    z, params, rng = m.draw_params(i)
    model = m._W["model_cls"](params, line_columns=m._W["columns"])
    model.mock_rng = rng
    wave = m._W["wave"]
    lad = m.ladder()
    rest = wave / (1 + z)
    obs = ResolutionMatrixSpectrum(
        wavelength=wave,
        resolution_matrix=np.asarray(m._W["res"][r], float),
        flux=np.zeros(wave.size),
        uncertainty=np.ones(wave.size),
        mask=(rest >= lad.REST_MIN) & (rest <= lad.REST_MAX),
    )
    obs.rectify()
    nan = np.full(wave.size, np.nan, np.float32)
    try:
        spec = model.predict(model.theta, [obs], sps=m._W["cue"])[0][0]
        return (np.asarray(spec, np.float32), np.asarray(model._resolved_continuum, np.float32),
                np.asarray(model._resolved_lines, np.float32), z, True)  # fmt: skip
    except Exception as err:  # a failed mock is recorded, as in mocks.py
        print(f"mock {i} failed: {err!r}", flush=True)
        return nan, nan, nan, z, False


def task(args):
    """Return the rebuilt mocks of one block."""
    path, idx, rows = args
    m = mocks_module(path)
    m.setup_worker()
    return idx, [one(m, i, r) for i, r in zip(idx, rows, strict=True)]


def cached_pool(lad, cache, npool, gib_worker):
    """Return the ladder's forked pool with the SSPs loaded from ``cache`` instead of built.

    As the ladder's node_pool (same population flags, speclib free, pool sizing and pinning), but
    the SSP slots come from hubersed.sps.ssp_cache (2026-11-13_ssp_cache_c3k_er), checked there.
    """
    import multiprocessing as mp
    import os

    import fsps
    import fsps_mem

    from hubersed.sps.ssp_cache import load_ssp_cache

    t = time.time()
    sp = fsps.StellarPopulation(zcontinuous=1)
    for k in ("add_igm_absorption", "add_dust_emission", "add_neb_emission", "nebemlineinspec"):
        sp.params[k] = False  # as NebStepBasis sets them, as node_pool
    sp.params["sfh"] = 0
    nslot = load_ssp_cache(sp, cache)
    freed = fsps_mem.free_speclib()
    room = lad.commit_room()
    cores = sorted(os.sched_getaffinity(0))
    npool = npool or max(
        1, min(len(cores) - lad.TORCH_THREADS, int((room - lad.GIB_MARGIN) // gib_worker))
    )
    print(f"{nslot} SSP slots loaded in {time.time() - t:.0f} s, speclib freed {freed}, commit "
          f"room {room:.1f} GiB, {len(cores)} cores, pool {npool}", flush=True)  # fmt: skip
    assert not lad.jax_backend_started(), "the JAX backend started before the fork"
    return lad.parent_pool(mp.get_context("fork"), npool, cores)


def main(path, out, n, pool_size, gib_worker, cache=None, part=0, nparts=1, block=20):
    """Rebuild part ``part`` of ``nparts`` of the subset and write it."""
    import h5py

    m = mocks_module(path)
    idx = np.sort(np.random.default_rng(SUBSET_SEED).choice(N_TOTAL, n, replace=False))
    idx = np.array_split(idx, nparts)[part]
    n = idx.size
    rows = np.random.default_rng(m.SEED).integers(0, 1000, size=N_TOTAL)[idx]
    tasks = [(path, idx[k : k + block], rows[k : k + block]) for k in range(0, n, block)]
    lad = m.ladder()
    if np.isnan(lad.commit_room()):  # macOS
        import multiprocessing as mp

        pool_ctx = mp.get_context("fork").Pool(pool_size or 2)
    elif cache is not None:
        pool_ctx = cached_pool(lad, cache, pool_size, gib_worker)
    else:
        pool_ctx = lad.node_pool(pool_size, gib_worker)
    t0 = time.time()
    pos = {int(i): k for k, i in enumerate(idx)}
    with pool_ctx as pool, h5py.File(out, "w") as hf:
        nw = 7781
        hf.create_dataset("index", data=idx)
        hf.create_dataset("rbank_index", data=rows)
        d = {k: hf.create_dataset(k, (n, nw), np.float32, compression="gzip")
             for k in ("flux", "continuum", "lines")}  # fmt: skip
        zs = hf.create_dataset("redshift", (n,), np.float64)
        ok = hf.create_dataset("ok", (n,), bool)
        done = 0
        for ids, res in pool.imap_unordered(task, tasks):
            for i, (f, c, ln, z, good) in zip(ids, res, strict=True):
                k = pos[int(i)]
                d["flux"][k], d["continuum"][k], d["lines"][k] = f, c, ln
                zs[k], ok[k] = z, good
            done += len(ids)
            print(f"{done}/{n} in {time.time() - t0:.0f} s", flush=True)
    print(f"wrote {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mocks_py", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--pool", type=int, default=0)
    p.add_argument("--gib-worker", type=float, default=8.0)
    p.add_argument("--cache", type=Path, default=None, help="SSP cache to load instead of building")
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--nparts", type=int, default=1)
    a = p.parse_args()
    main(a.mocks_py, a.out, a.n, a.pool, a.gib_worker, a.cache, a.part, a.nparts)
