"""Build the C3K ER SSP slots once, save them with hubersed.sps.ssp_cache, and check a reload.

The population is set up as the ladder's node_pool sets it before the fork (2026-11-07_emline_ladder
fit.py: zcontinuous=1; add_igm_absorption, add_dust_emission, add_neb_emission and nebemlineinspec
False, as NebStepBasis sets them; sfh=0), so the cache key matches the pools that will load it.
``build`` builds every slot with build_ssps and saves them with a reference spectrum (tage 1 Gyr,
logzsol 0). ``check`` runs in a new process: it loads the cache and compares that spectrum exactly.
COMMAND: ``python build.py build <cache.npz>`` then ``python build.py check <cache.npz>`` (build.slurm).
OUTPUT: <cache.npz> and <cache>_ref.npz.
"""

import argparse
import time
from pathlib import Path

import numpy as np

GIB_HELPER, GIB_MARGIN = 15.0, 10.0  # as the ladder's node_pool (10-31 job 3664748)


def population():
    """Return the population node_pool builds its SSPs with."""
    import fsps

    sp = fsps.StellarPopulation(zcontinuous=1)
    for k in ("add_igm_absorption", "add_dust_emission", "add_neb_emission", "nebemlineinspec"):
        sp.params[k] = False
    sp.params["sfh"] = 0
    return sp


def reference(sp):
    """Return the reference spectrum used by the check."""
    sp.params["logzsol"] = 0.0
    return sp.get_spectrum(tage=1.0)


def commit_room():
    """Return CommitLimit minus Committed_AS in GiB, NaN where unknown."""
    try:
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) for line in f}
        return (info["CommitLimit"] - info["Committed_AS"]) / 2**20
    except FileNotFoundError:
        return float("nan")


def build(path):
    """Build every slot, save the cache and the reference spectrum."""
    from hubersed.sps.ssp_cache import save_ssp_cache

    sp = population()
    t = time.time()
    room = commit_room()
    nbuild = 1 if np.isnan(room) else max(1, min(10, int((room - GIB_MARGIN) // GIB_HELPER)))
    sp.build_ssps(processes=nbuild)
    print(f"built in {time.time() - t:.0f} s with {nbuild} helpers", flush=True)
    t = time.time()
    save_ssp_cache(sp, path)
    print(
        f"saved {path} ({path.stat().st_size / 2**30:.2f} GiB) in {time.time() - t:.0f} s",
        flush=True,
    )
    wave, spec = reference(sp)
    np.savez(path.with_name(path.stem + "_ref.npz"), wave=wave, spec=spec)


def check(path):
    """Load the cache in this new process and compare the reference spectrum exactly."""
    from hubersed.sps.ssp_cache import load_ssp_cache

    sp = population()
    t = time.time()
    n = load_ssp_cache(sp, path)
    print(f"loaded {n} slots in {time.time() - t:.0f} s", flush=True)
    t = time.time()
    wave, spec = reference(sp)
    print(f"reference spectrum in {time.time() - t:.1f} s", flush=True)
    ref = np.load(path.with_name(path.stem + "_ref.npz"))
    same = np.array_equal(wave, ref["wave"]) and np.array_equal(spec, ref["spec"])
    print(f"reference spectrum identical: {same}", flush=True)
    if not same:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("action", choices=["build", "check"])
    p.add_argument("path", type=Path)
    a = p.parse_args()
    build(a.path) if a.action == "build" else check(a.path)
