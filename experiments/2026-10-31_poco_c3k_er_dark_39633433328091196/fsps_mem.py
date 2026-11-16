"""Free FSPS's spectral library after the SSPs are built, so forked workers commit less memory.

Copied unchanged from experiments/2026-10-21_continuum_poco/fsps_mem.py (never committed).
free_speclib frees sps_vars::speclib (10.1 GiB) and nulls its pointer, so a later SSP build crashes.
Needs the symbol __sps_vars_MOD_speclib in the dynamic symbol table of _fsps (c3k_er_afe_v3 build).
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
