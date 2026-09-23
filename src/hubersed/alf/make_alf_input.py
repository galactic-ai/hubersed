"""Write one DESI spectrum as an alf input file.

Run ``uv run python -m hubersed.alf.make_alf_input --tid TARGETID -o $ALF_HOME/indata/NAME.dat``
and then ``mpirun -np 8 $ALF_HOME/bin/alf.exe NAME``. Line numbers refer to alf commit 4ef7bb8.

Notes
-----
The file starts with one ``# l1 l2`` line per fitted interval, in microns, at most 10
(read_data.f90:54-84). Each data row is ``lam flx err wgt ires``. The wavelength is vacuum
Angstrom, flux and error are DESI f_lambda in 1e-17 erg/s/cm^2/A, the weight is between 0
and 1, and ires is the instrumental sigma in km/s (read_data.f90:93-107).

alf fits the continuum shape away with a polynomial per interval, so it only sees the
absorption features (alf manual section 1.1).

Pixels with wgt=0 still add a log jitter term to the fit_type=0 likelihood
(func.f90:118-124), so heavy masking pulls the fitted jitter down.

alf needs the VCJ SSP files from vcj_ssp.tar.gz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from hubersed.fitting.chi2 import WAVE_OBS, load_by_index, tids_to_indices
from hubersed.prospector.lsf import C_KMS, desi_resolution

# alf's model wavelength range (alf_vars.f90:127-128)
ALF_LAM_MIN, ALF_LAM_MAX = 3600.0, 11000.0
# Fitted intervals in microns. 0.64-0.80 is left out because the continuum polynomial
# over-fits TiO there (Beverage et al. 2025, arXiv:2407.02556).
DEFAULT_INTERVALS = "0.40,0.47,0.47,0.55,0.55,0.64,0.80,0.88"


def main(argv=None):
    """Write the alf input file for one TARGETID.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. By default they come from sys.argv.

    Returns
    -------
    int
        Exit status.
    """
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--tid", type=int, default=39633140817331167)
    p.add_argument("-o", "--out", required=True)
    p.add_argument(
        "--intervals",
        default=DEFAULT_INTERVALS,
        help="interval edges in microns, as l1,l2,l1,l2. The default covers "
        "4000-6400 and 8000-8800 A. It leaves out 6400-8000 A, where the continuum "
        "polynomial over-fits TiO, as Beverage et al. 2025 do. At most 10 intervals.",
    )
    p.add_argument(
        "--mask",
        default="5876,5913",
        help="rest-frame Angstrom edges to set to wgt=0, as l1,l2,l1,l2. "
        "The default is Na D, alf's NaD index band 5876.875-5909.375 A "
        "(air) padded about 2 A for vacuum. Na D picks up interstellar absorption "
        "(Conroy, Graves and van Dokkum 2014) and Beverage et al. 2025 mask it. "
        "Pass '' to disable.",
    )
    a = p.parse_args(argv)

    idx = int(tids_to_indices(np.array([a.tid], np.int64))[0])
    spec, ivar, z, tid_chk = load_by_index(idx)
    assert int(tid_chk) == a.tid, f"TARGETID mismatch: asked {a.tid}, got {tid_chk}"
    z = float(z)

    # alf's spectra are L_lambda (alf manual section 1.4.1), so DESI f_lambda goes in as is
    flux = np.asarray(spec, float)
    iv = np.asarray(ivar, float)
    good = (iv > 0) & np.isfinite(flux)
    err = np.where(good, 1.0 / np.sqrt(np.where(iv > 0, iv, np.inf)), 1.0)

    # DESI LSF as a velocity sigma in km/s, as in run_map_fits_outliers
    ires = (C_KMS / (2.355 * desi_resolution(WAVE_OBS))).astype(float)

    # alf reads the interval edges as rest frame and shifts them by its fitted velz
    # (func.f90:96-97), so the spectrum is written de-redshifted.
    lam = WAVE_OBS / (1.0 + z)
    frame = f"rest (de-redshifted by z={z:.7f})"

    ed = [float(v) for v in a.intervals.split(",")]
    assert len(ed) % 2 == 0, "--intervals needs an even number of edges"
    iv_pairs = list(zip(ed[::2], ed[1::2]))
    assert len(iv_pairs) <= 10, "alf's nlint_max is 10 intervals"

    # keep only what alf can model and what falls in a requested interval
    inrange = (lam > ALF_LAM_MIN) & (lam < ALF_LAM_MAX)
    wanted = np.zeros_like(lam, bool)
    for l1, l2 in iv_pairs:
        wanted |= (lam >= l1 * 1e4) & (lam <= l2 * 1e4)
    keep = inrange & wanted
    if not keep.any():
        raise SystemExit("no pixels survive the interval + model-range cut")

    # Bad pixels get wgt=0 instead of being dropped, so the wavelength grid stays contiguous.
    wgt = good[keep].astype(float)

    rest = lam[keep]
    med = [float(v) for v in a.mask.split(",") if v.strip()]
    assert len(med) % 2 == 0, "--mask needs an even number of edges"
    masked = []
    for m1, m2 in zip(med[::2], med[1::2]):
        sel = (rest >= m1) & (rest <= m2)
        # A count of 0 means the window missed the fitted intervals.
        masked.append((m1, m2, int(sel.sum())))
        wgt[sel] = 0.0

    # Bad pixels get interpolated flux, not 0. alf's printed rms has no weight term
    # (func.f90:152-153), so zeros would make it report the masked fraction.
    fill = np.copy(flux)
    if (~good).any() and good.any():
        fill[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), flux[good])
    flx = fill[keep]
    er = np.where(good, err, np.nanmax(err[good]) * 1e3)[keep]

    # alf's repo has no indata/ or results/, but alf.exe reads indata/ and writes results/
    # (read_data.f90:42, alf.f90:600). models/ is used by alf's helper programs.
    out_path = Path(a.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.parent.name == "indata":
        home = out_path.parent.parent
        for d in ("results", "models"):
            (home / d).mkdir(exist_ok=True)
        if not (home / "src" / "alf.f90").exists():
            print(f"  WARNING {home} has no src/alf.f90 -- is alf cloned there?")
            print(f"          git clone https://github.com/cconroy20/alf {home}")

    with open(out_path, "w") as f:
        for l1, l2 in iv_pairs:
            f.write(f"# {l1:.4f} {l2:.4f}\n")
        for L, F, E, W, R in zip(lam[keep], flx, er, wgt, ires[keep]):
            f.write(f"{L:10.4f} {F:14.6e} {E:14.6e} {W:5.2f} {R:9.3f}\n")

    snr = flux[keep][wgt > 0] / err[keep][wgt > 0]
    dl = float(np.median(np.diff(lam[keep])))
    print(f"wrote {out_path}")
    print(f"  TARGETID {a.tid}   z = {z:.7f}   frame: {frame}")
    print(f"  {int(keep.sum())} pixels, {int((wgt == 0).sum())} zero-weighted")
    for m1, m2, n in masked:
        print(
            f"  masked rest {m1:.1f}-{m2:.1f} A: {n} px"
            + ("   <-- ZERO, check frame/interval coverage" if n == 0 else "")
        )
    print(f"  lambda {lam[keep].min():.1f} - {lam[keep].max():.1f} A, median spacing {dl:.3f} A")
    print(f"  intervals (um): " + ", ".join(f"{l1}-{l2}" for l1, l2 in iv_pairs))
    print(f"  ires {ires[keep].min():.1f} - {ires[keep].max():.1f} km/s")
    print(f"  median S/N: {np.median(snr):.1f} /pixel, {np.median(snr) / np.sqrt(dl):.1f} /A")
    print("\n  alf's published mock tests span S/N = 20, 30, 50, 100 per A")
    print("  (Conroy et al. 2018, section 3.2.2).")
    print("\n  next:")
    print(f"    cd $ALF_HOME/src && make")
    print(f"    mpirun -np 8 $ALF_HOME/bin/alf.exe {out_path.stem} <tag>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
