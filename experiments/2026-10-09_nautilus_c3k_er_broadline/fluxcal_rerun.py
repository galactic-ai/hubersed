"""Rerun the iron flux calibration of 39627770174637084, as iron ran it and with the ivar fixes.

QUESTION: what is the calibration-vector ivar of this target's two exposures once the desispec
fixes to the calibration uncertainty are applied, with the calibration vector itself unchanged?

Iron (desispec 0.56.2) leaves an error floor of 20% on the standard-star ivars in the final
variance solve of compute_flux_calibration, which inflates the calibration variance (desispec
PR 2496). PR 2827 resets the normal matrix before that final solve. PR 2452 scales the calibration
ivar by the fiber aperture correction, as the calibration itself is scaled. This script runs
desi_compute_fluxcalibration of desispec 0.56.2 with iron's arguments (from the iron tile-night
logs), once unchanged and once with those three changes. They are applied to the text of the
installed desispec/fluxcalibration.py in memory and the result is loaded in its place, so the
installed package is not modified. Each anchor must match exactly once.

Checks: the unchanged run must reproduce the iron fluxcalib files, and the patched run must give the
same FLUXCALIB with a larger IVAR.

This runs with the separate desispec 0.56.2 environment, not the hubersed one. It is Python 3.11
with numpy 1.23.5 instead of iron's 3.10 and 1.22.4, and desimodel/io.py load_platescale catches
ValueError as well as IndexError for numpy 1.23 (desi_env_0562/PATCHES.txt). DESIMODEL points at
the desimodel 0.18.0 data folders focalplane and throughput.

INPUTS: the iron frame, sky, fiberflatexp, stdstars-1, calibstars and fluxcalib files of exposures
87113 and 88269, petal 1 (b1, r1, z1), under data/desi_iron_exposures/<night>/<expid>/. SEED: none.

COMMAND: on an ls6 compute node from the hubersed repository root,
``DESIMODEL=/work/11006/nikhilgaruda/ls6/research/desi_env_0562/desimodel
/work/11006/nikhilgaruda/ls6/research/desi_env_0562/venv/bin/python
experiments/2026-10-09_nautilus_c3k_er_broadline/fluxcal_rerun.py``.

OUTPUT: fluxcal_rerun.txt and fluxcalib_rerun/{unpatched,patched}/fluxcalib-*.fits.gz with logs in
results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/.
"""

import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
from astropy.io import fits

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "desi_iron_exposures"
OUT = ROOT / "results" / "2026-10-09_nautilus_c3k_er_broadline" / "39627770174637084"
EXPOSURES = {87113: "20210503", 88269: "20210513"}
CAMERAS = ("b1", "r1", "z1")
TARGET_ROW = {87113: 224, 88269: 204}

OLD_FINAL = """    # solve once again to get deconvolved variance
    #calibration,calibcovar=cholesky_solve_and_invert(A.todense(),B)
    calibcovar=np.linalg.inv(A)
"""
NEW_FINAL = """    # solve once again to get deconvolved variance
    # PR 2496: restart from the original ivar, keep the masked pixels; PR 2827: reset A
    current_ivar=stdstars.ivar*(current_ivar>0)
    sqrtw=np.sqrt(current_ivar)
    A *= 0.
    for star in range(nstds) :
        if badfiber[star] : continue
        R = stdstars.R[star]
        D1.setdiag(sqrtw[star]*scale[star])
        D2.setdiag(model_flux[star])
        sqrtwmodelR = D1.dot(R.dot(D2))
        A = A+(sqrtwmodelR.T*sqrtwmodelR).tocsr()
    minivar = np.min(current_ivar[current_ivar>0])
    epsilon = minivar/10000
    A = epsilon*np.eye(nwave) + A
    calibcovar=np.linalg.inv(A)
"""
OLD_PSF = """    ccalibration /= point_source_correction[:,None]
"""
NEW_PSF = """    ccalibration /= point_source_correction[:,None]
    ccalibivar *= (point_source_correction[:,None])**2  # PR 2452
"""


def patched_source():
    """Return the text of desispec/fluxcalibration.py with the three changes applied."""
    import desispec

    path = pathlib.Path(desispec.__file__).parent / "fluxcalibration.py"
    src = path.read_text()
    for old, new in ((OLD_FINAL, NEW_FINAL), (OLD_PSF, NEW_PSF)):
        assert src.count(old) == 1, f"anchor not found exactly once: {old.splitlines()[0]}"
        src = src.replace(old, new)
    return path, src


def run_one(mode):
    """Run the calibration for every exposure and camera, in a child process per mode."""
    args = []
    for expid, night in EXPOSURES.items():
        d = DATA / night / f"{expid:08d}"
        for cam in CAMERAS:
            out = OUT / "fluxcalib_rerun" / mode / f"fluxcalib-{cam}-{expid:08d}.fits.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            args.append(
                [
                    "--infile", str(d / f"frame-{cam}-{expid:08d}.fits.gz"),
                    "--sky", str(d / f"sky-{cam}-{expid:08d}.fits.gz"),
                    "--fiberflat", str(d / f"fiberflatexp-{cam}-{expid:08d}.fits.gz"),
                    "--models", str(d / f"stdstars-1-{expid:08d}.fits.gz"),
                    "--selected-calibration-stars", str(d / f"calibstars-{expid:08d}.csv"),
                    "--outfile", str(out),
                ]
            )  # fmt: skip
    return args


def child(mode, argv):
    """Load the patched module if asked, then run desi_compute_fluxcalibration once."""
    if mode == "patched":
        path, src = patched_source()
        spec = importlib.util.spec_from_loader("desispec.fluxcalibration", loader=None)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = str(path)
        exec(compile(src, str(path), "exec"), mod.__dict__)
        sys.modules["desispec.fluxcalibration"] = mod
        import desispec

        desispec.fluxcalibration = mod
    from desispec.scripts.fluxcalibration import main

    main(argv)


def compare(a, b, row):
    """Return max relative differences of FLUXCALIB and IVAR over all fibers and at one row."""
    out = {}
    with fits.open(a) as x, fits.open(b) as y:
        for h in ("FLUXCALIB", "IVAR"):
            u, v = x[h].data.astype(float), y[h].data.astype(float)
            ok = (u != 0) | (v != 0)
            r = np.abs(u - v)[ok] / np.maximum(np.abs(u[ok]), 1e-30)
            out[h] = (r.max(), np.median(v[row][u[row] > 0] / u[row][u[row] > 0]))
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3:])
        return
    lines = ["Iron flux calibration rerun of 39627770174637084 (desispec 0.56.2, petal 1)"]
    for mode in ("unpatched", "patched"):
        for argv in run_one(mode):
            log = pathlib.Path(argv[-1].replace(".fits.gz", ".log"))
            with open(log, "w") as fh:
                r = subprocess.run(
                    [sys.executable, __file__, "--child", mode, *argv], stdout=fh, stderr=fh
                )
            assert r.returncode == 0, f"{mode} failed, see {log}"
    lines.append(
        "unpatched vs iron, and patched vs unpatched: max rel diff over all fibers; for IVAR also"
        " the median ratio at the target row"
    )
    for expid, night in EXPOSURES.items():
        for cam in CAMERAS:
            name = f"fluxcalib-{cam}-{expid:08d}.fits.gz"
            iron = DATA / night / f"{expid:08d}" / name
            un = OUT / "fluxcalib_rerun" / "unpatched" / name
            pa = OUT / "fluxcalib_rerun" / "patched" / name
            c1 = compare(iron, un, TARGET_ROW[expid])
            c2 = compare(un, pa, TARGET_ROW[expid])
            lines.append(
                f"{expid} {cam}: unpatched/iron C {c1['FLUXCALIB'][0]:.1e} IVAR {c1['IVAR'][0]:.1e}; "
                f"patched/unpatched C {c2['FLUXCALIB'][0]:.1e}, IVAR ratio at target "
                f"{c2['IVAR'][1]:.3f} (sigma_C ratio {c2['IVAR'][1] ** -0.5:.3f})"
            )
    text = "\n".join(lines) + "\n"
    print(text)
    (OUT / "fluxcal_rerun.txt").write_text(text)


if __name__ == "__main__":
    main()
