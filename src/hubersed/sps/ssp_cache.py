"""Save built FSPS SSP slots to disk and load them in a later process.

FSPS builds each SSP slot (one metallicity index and one afe index) the first time a spectrum needs
it, which takes ~950 s for the 65 C3K ER slots. This module copies the slots out with the fork's
``driver.get_ssp_slot`` and back in with ``driver.set_ssp_slot`` (galactic-ai python-fsps, commit
2c5168f). It relies on private names of that fork (``_update_params``, ``_ssp_cache_inputs``), so it
fails if any is missing. A cache loads only into a process with the same build and the same SSP
inputs: the file holds the fork's cache key (the exact SSP parameters and ``smooth_velocity``) and a
build identity, and both must match.
"""

import os
from pathlib import Path

import numpy as np

from hubersed.paths import PATHS

CACHE_DIR = PATHS["DATA"] / "ssp_cache"


def _fsps_internals():
    """Return the fsps module and its driver, checking the fork names this module uses."""
    import fsps
    import fsps.fsps as ff
    from fsps._fsps import driver

    for name in ("get_ssp_slot", "set_ssp_slot", "get_nz", "get_nafe", "get_nspec", "get_ntfull"):
        if not hasattr(driver, name):
            raise RuntimeError(f"this FSPS build has no driver.{name}; ssp_cache needs the fork")
    if not hasattr(ff, "_ssp_cache_inputs"):
        raise RuntimeError("this python-fsps has no _ssp_cache_inputs; ssp_cache needs the fork")
    return fsps, ff, driver


def build_identity(sp):
    """Return a string that identifies the FSPS build and setup of a StellarPopulation.

    Parameters
    ----------
    sp : fsps.StellarPopulation
        The population, after its first spectrum or ``_update_params``.

    Returns
    -------
    str
        Version, module path, SPS_HOME, libraries, setup flags and array sizes.
    """
    fsps, _, driver = _fsps_internals()
    parts = [
        f"version {fsps.__version__}",
        f"module {Path(fsps.__file__).resolve().parent}",
        f"SPS_HOME {os.environ.get('SPS_HOME')}",
        f"libraries {tuple(x.decode() if isinstance(x, bytes) else str(x) for x in sp.libraries)}",
        f"setup {tuple(int(v) for v in driver.get_setup_vars())}",
        f"nspec {driver.get_nspec()} ntfull {driver.get_ntfull()}",
        f"nz {driver.get_nz()} nafe {driver.get_nafe()}",
    ]
    return "; ".join(parts)


def cache_key(sp):
    """Return the fork's SSP cache key of ``sp`` as a string, or raise if it cannot be compared.

    Parameters
    ----------
    sp : fsps.StellarPopulation
        The population whose SSP inputs define the key.

    Returns
    -------
    str
        ``repr`` of the key, which holds every SSP parameter with its exact type and value.
    """
    _, ff, _ = _fsps_internals()
    key = ff._ssp_cache_inputs(sp.params)
    if None in key:
        raise ValueError(
            "these SSP inputs cannot be cached (imf_type 5 or a non-flag nebular setting)"
        )
    return repr(key)


def save_ssp_cache(sp, path, zmet=None, afeindx=None):
    """Build (if needed) and save SSP slots of ``sp`` to ``path``.

    Parameters
    ----------
    sp : fsps.StellarPopulation
        The population whose SSP parameters the slots are built with.
    path : str or pathlib.Path
        Output ``.npz`` file.
    zmet, afeindx : sequence of int, optional
        Metallicity and afe indices (1-based). By default every slot.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    _, _, driver = _fsps_internals()
    sp._update_params()
    nspec, ntfull = driver.get_nspec(), driver.get_ntfull()
    zs = range(1, driver.get_nz() + 1) if zmet is None else zmet
    afes = range(1, driver.get_nafe() + 1) if afeindx is None else afeindx
    slots = [(int(z), int(a)) for z in zs for a in afes]
    spec = np.empty((len(slots), nspec, ntfull))
    mass = np.empty((len(slots), ntfull))
    lbol = np.empty((len(slots), ntfull))
    for k, (z, a) in enumerate(slots):
        spec[k], mass[k], lbol[k] = driver.get_ssp_slot(nspec, ntfull, z, a)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, slots=np.array(slots), spec=spec, mass=mass, lbol=lbol, key=cache_key(sp),
             identity=build_identity(sp))  # fmt: skip
    # _update_params cleared the dirty flag; the next spectrum must recompute the CSP
    sp.params.dirtiness = max(sp.params.dirtiness, 1)
    return path


def load_ssp_cache(sp, path):
    """Load saved SSP slots into this process for ``sp``.

    ``sp`` must have the SSP parameters the cache was built with. The first saved slot is built
    again (this sets FSPS IMF state) and must equal the saved one; the others are copied in. The next
    spectrum then uses the loaded slots as long as the SSP parameters stay unchanged.

    Parameters
    ----------
    sp : fsps.StellarPopulation
        The population to load the slots for.
    path : str or pathlib.Path
        A file written by ``save_ssp_cache``.

    Returns
    -------
    int
        The number of slots loaded.
    """
    _, _, driver = _fsps_internals()
    with np.load(path) as f:
        if str(f["identity"]) != build_identity(sp):
            raise ValueError(f"{path} was built with another FSPS build or setup")
        sp._update_params()  # sends the SSP parameters, clears the cache, records the key
        if str(f["key"]) != cache_key(sp):
            raise ValueError(f"{path} was built with other SSP parameters")
        slots, spec, mass, lbol = f["slots"], f["spec"], f["mass"], f["lbol"]
        nspec, ntfull = driver.get_nspec(), driver.get_ntfull()
        if spec.shape[1:] != (nspec, ntfull):
            raise ValueError(f"{path} has SSP arrays of another shape")
        # A real build of one slot sets the IMF state FSPS keeps outside the SSP arrays (the fork's
        # build_ssps does the same in its own process); it must also equal the saved slot.
        z0, a0 = (int(v) for v in slots[0])
        built = driver.get_ssp_slot(nspec, ntfull, z0, a0)
        if not all(
            np.array_equal(b, s) for b, s in zip(built, (spec[0], mass[0], lbol[0]), strict=True)
        ):
            raise ValueError(f"slot {z0, a0} built now differs from the one in {path}")
        for k, (z, a) in enumerate(slots[1:], start=1):
            driver.set_ssp_slot(int(z), int(a), spec[k], mass[k], lbol[k])
    # CSP-dirty only: the next spectrum recomputes the CSP and keeps the loaded SSPs
    sp.params.dirtiness = max(sp.params.dirtiness, 1)
    return len(slots)
