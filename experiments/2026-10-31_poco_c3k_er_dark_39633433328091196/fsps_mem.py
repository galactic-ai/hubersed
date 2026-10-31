"""Free FSPS's spectral library after the SSPs are built, so forked workers commit less memory.

Copied unchanged from experiments/2026-10-21_continuum_poco/fsps_mem.py on /work (never committed), so
that fit.py --fork can free speclib. The 2026-10-31 skx-dev job 3664817 (free_speclib_debug)
ran 24 forked workers after this free with bitwise equal spectra.

FSPS keeps the whole C3K_ER library in the allocatable module array sps_vars::speclib (10.1 GiB,
sps_vars.f90:544, allocated and filled once in sps_setup.f90:91-358). Only SSP building reads it
(getspec, via ssp_gen and stellar_spectrum). Once every SSP slot is built, spectra read only
spec_ssp_zz and the isochrones. On nodes with strict overcommit (vm.overcommit_memory = 2) each
forked child is charged for all of its parent's private memory, so freeing speclib in the parent
before the fork cuts that charge by 10 GiB per child.

free_speclib frees the array's buffer with libc free (gfortran allocates it with malloc) and sets
the descriptor's data pointer to NULL, so ALLOCATED(speclib) is false and sps_setup would allocate
it again. Any later SSP build reads a NULL pointer and crashes the process instead of returning a
wrong spectrum. The symbol __sps_vars_MOD_speclib must be in the dynamic symbol table of _fsps,
as in the c3k_er_afe_v3 build. It uses no FSPS code change.
"""

import ctypes
import ctypes.util

import fsps


def _speclib_base():
    """Return the ctypes view of speclib's data pointer, the first word of its descriptor."""
    lib = ctypes.CDLL(fsps._fsps.__file__)
    return ctypes.c_void_p.in_dll(lib, "__sps_vars_MOD_speclib")


def speclib_allocated():
    """Return True if speclib holds a buffer."""
    return bool(_speclib_base().value)


def free_speclib():
    """Free speclib's buffer and mark it unallocated. Return True if a buffer was freed."""
    base = _speclib_base()
    if not base.value:
        return False
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.free.argtypes = [ctypes.c_void_p]
    libc.free(base.value)
    base.value = None
    return True


def meminfo(key):
    """Return a /proc/meminfo field in GiB."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 2**20
    return float("nan")


def own_memory():
    """Return VmData and VmRSS of this process in GiB."""
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            k = line.split(":")[0]
            if k in ("VmData", "VmRSS"):
                out[k] = int(line.split()[1]) / 2**20
    return out["VmData"], out["VmRSS"]
