"""MAPs of the SV3 dark spectrum of 39633433328091196 with dust off and with MILES (#17).

Changes from 2026-10-29_nodust_map_39633433328091196/fit.py (SV3 bright): the data are the SV3 dark
DR1 coadd (1 exposure, EXPID 88050) of 2026-10-30_continuum_39633433328091196_dark_map, since the
SV3 bright coadd carries sky residual in the blue. All runs use 4 starts.

``--run er_nodust`` is the C3K ER fit of the dark map fit.py (dark spectrum and R matrix) with
dust off as in the 10-29 fit.py. Start 0 is the dark dust MAP best point (spr job 3662995).
``--run miles_dust`` and ``--run miles_nodust`` are the 10-29 MILES fits with logzsol capped at
+0.2, on the dark spectrum. Their SFH hyperparameters and start 0 are the 10-08 max-L of the SV3
bright spectrum, with logzsol moved to 0.195.

Steps check, fit and collect as in the 10-29 fit.py. check also asserts that the fitted flux is
the dark spectrum and not the bright one. Run on a stampede3 compute node through map.slurm, for
example ``<venv>/bin/python experiments/2026-10-30_nodust_miles_39633433328091196_dark/fit.py
fit --run er_nodust --start 0``.
"""

import argparse
import importlib.util
from pathlib import Path

import astropy.units as u
import numpy as np

from hubersed.conversion import to_maggies
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.sps.parameter_file import WAVE_OBS

EXPS = Path(__file__).resolve().parents[1]
DARK = EXPS / "2026-10-30_continuum_39633433328091196_dark_map" / "fit.py"
NODUST = EXPS / "2026-10-29_nodust_map_39633433328091196" / "fit.py"
OUT = PATHS["RESULTS"] / "2026-10-30_nodust_miles_39633433328091196_dark"
ZMAX = 0.2
RUNS = ("er_nodust", "miles_dust", "miles_nodust")


def load(name, path):
    """Return the module at ``path``, loaded by path since the experiment files share names."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dark = load("fit_dark", DARK)
nd = load("fit_nodust", NODUST)
DARK_MAP = dark.OUT / f"{nd.TID}_map.npz"


def setup_er():
    """Return the dark map step module with dust off and start 0 at the dark dust MAP."""
    base = dark.base
    make = base.ResolutionMatrixModel
    base.ResolutionMatrixModel = lambda t: make(nd.dust_off(t))
    best = nd.best_point(DARK_MAP)
    base.start_values = lambda nebular: dict(best)
    base.OUT = OUT / "er_nodust"
    return base, "C3K_ER + afe + R matrix, SV3 dark, dust off"


def setup_miles(dust):
    """Return the 10-29 MILES step module with logzsol capped and the dark spectrum."""
    read = nd.load

    def load_dark(name, path):
        module = read(name, path)
        if name == "fit_1008":
            module.load_spectrum = dark.dark_spectrum
        return module

    nd.load = load_dark
    nd.OUT = OUT
    base, label = nd.setup_miles(dust, ZMAX)
    return base, label + ", SV3 dark"


def check_dark(base):
    """Assert that the fitted flux is the dark spectrum and not the bright one."""
    data = base.build()[4]
    g = data["good"]
    dark_flux = load_spectrum_flux(dark.dark_spectrum(nd.TID))
    bright_flux = load_spectrum_flux(load_spectrum(nd.TID))
    assert np.allclose(data["flux"][g], dark_flux[g], rtol=1e-6, atol=0), "flux is not the dark one"
    assert not np.allclose(data["flux"][g], bright_flux[g], rtol=1e-2, atol=0), (
        "flux is the bright one"
    )
    ratio = np.median(dark_flux[g] / bright_flux[g])
    print(f"fitted px {g.sum()}, median dark over bright flux {ratio:.3f}")


def load_spectrum_flux(s):
    """Return the flux of a load_spectrum Spectrum in maggies on ``WAVE_OBS``."""
    return to_maggies(WAVE_OBS * u.AA, s.flux).value


def main():
    """Parse the run and step, and run it."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("check", "fit", "collect"))
    ap.add_argument("--run", choices=RUNS, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n-starts", type=int, default=4)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--maxfev", type=int, default=150000)
    ap.add_argument("--n-calls", type=int, default=7, help="warm calls timed by check")
    a = ap.parse_args()
    lib = a.run.split("_")[0]
    dust = a.run == "miles_dust"
    base, label = setup_er() if lib == "er" else setup_miles(dust)
    if a.step == "check":
        base.check(a.n_starts, a.jitter, a.n_calls)
        nd.extra_checks(base, lib, dust, None if lib == "er" else ZMAX)
        check_dark(base)
    elif a.step == "fit":
        base.fit(a.start, a.n_starts, a.jitter, a.maxfev, "")
    else:
        nd.collect(base, label, a.n_starts)


if __name__ == "__main__":
    main()
