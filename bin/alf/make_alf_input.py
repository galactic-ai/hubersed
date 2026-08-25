"""Write one DESI spectrum in the input format alf expects.

    uv run python bin/prospector/make_alf_input.py --tid 39633140817331167 \
        -o ~/alf/indata/desi_42580.dat

Then, on a machine with a Fortran compiler:

    brew install gcc open-mpi                 # gfortran + mpifort
    git clone https://github.com/cconroy20/alf ~/alf
    export ALF_HOME=~/alf/                    # trailing slash matters
    cd $ALF_HOME/src && make                  # Makefile defaults to F90=mpifort
    mpirun -np 8 $ALF_HOME/bin/alf.exe desi_42580

The 225 MB of SSPs and response functions are already in the repo's ``infiles/``;
there is no separate model download.

Format, verified from src/read_data.f90
---------------------------------------
Header: any number of lines ``# l1 l2`` giving the wavelength intervals to fit,
**in microns** (``read_data.f90:83-84`` multiplies by 1e4). At most ``nlint_max=10``.
Body: ``lam flx err wgt ires`` (``read_data.f90:92-93``), with

  lam   Angstroms, must satisfy 1e3 < lam < 5e4      (read_data.f90:96)
  flx   arbitrary units -- alf divides out a polynomial, so absolute
        flux calibration is irrelevant to it
  err   same units as flx
  wgt   0..1, hard-checked (read_data.f90:103). 0 = ignore this pixel.
  ires  instrumental resolution, 0..1e4 (read_data.f90:109). km/s sigma;
        setup.f90:489-497 interpolates it onto the SSP grid as a smoothing kernel.

Frame
-----
alf fits ``velz`` itself (``getvelz.f90``), and the prior is wide
(``set_pinit_priors.f90:143,207`` give -1e3 to 1e5 km/s), so either frame works.
This script de-redshifts by default, which is the usual convention and leaves velz
to absorb only the small residual. The interval header is written in the same frame
as the data.

What alf will and will not tell you
-----------------------------------
alf divides the data by a polynomial of order n = (lam_max - lam_min)/100 A before
computing chi2 (Conroy+2018, ms.tex:854, verified) precisely so that dust and flux
calibration do not enter -- they state fluxing is "rarely better than 5-10%"
(ms.tex:1568-1570). So alf reports the ABUNDANCE PATTERN and is deliberately blind to
the continuum. It cannot, by construction, speak to the 6500-9824 A continuum excess
that carries ~51% of this galaxy's chi2 in the Prospector fits.

The useful experiment is therefore two-step: get the abundance pattern from alf, then
put that pattern into a model WITH the continuum restored and ask whether the residual
shrinks. That is Choi+2019's "fit one thing, predict another" logic (they fit
continuum-normalised stacks and predicted ugriz colours, galaxy_sed.tex:129-131)
pointed at the continuum instead of the photometry.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from hubersed.conversion import flambda_to_maggies, ivar_flambda_to_ivar_maggies
from hubersed.fitting.chi2 import WAVE_OBS, load_by_index, tids_to_indices
from hubersed.prospector.lsf import C_KMS, desi_resolution

# alf's model grid runs nstart=100..nend=5830 on its own lambda array, which
# alf_vars.f90:127-128 annotates as 0.36 um .. 1.10 um.
ALF_LAM_MIN, ALF_LAM_MAX = 3600.0, 11000.0


def main(argv=None):
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--tid", type=int, default=39633140817331167)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--observed", action="store_true",
                   help="write observed-frame lambda instead of de-redshifting")
    p.add_argument("--intervals", default="0.40,0.47,0.47,0.55,0.55,0.70,0.70,0.88",
                   help="flat list of interval edges in MICRONS, l1,l2,l1,l2,... "
                        "Default is four intervals over 4000-8800 A, in the spirit of "
                        "the 3700-8850 A range Choi+2019 used (galaxy_sed.tex:131). "
                        "alf caps this at nlint_max=10 intervals.")
    p.add_argument("--mask", default="5876,5913",
                   help="flat list of REST-FRAME Angstrom edges l1,l2,l1,l2,... set to "
                        "wgt=0. Applied in the rest frame even under --observed. Default "
                        "is Na D: the allindices.dat NaD feature band 5876.875-5909.375 "
                        "(air), padded ~2 A to cover the air-to-vacuum offset. Masked "
                        "because Na I 5895 is 'well-known to be affected by' the ISM "
                        "(CvD14 ms.tex:879-882) and Beverage+2025 masks it "
                        "(suspense_abundances.tex:214) -- alf would read it as a stellar "
                        "Na abundance. Pass '' to disable.")
    a = p.parse_args(argv)

    idx = int(tids_to_indices(np.array([a.tid], np.int64))[0])
    spec, ivar, z, tid_chk = load_by_index(idx)
    assert int(tid_chk) == a.tid, f"TARGETID mismatch: asked {a.tid}, got {tid_chk}"
    z = float(z)

    flux = flambda_to_maggies(WAVE_OBS, spec)
    iv = ivar_flambda_to_ivar_maggies(WAVE_OBS, ivar)
    good = (iv > 0) & np.isfinite(flux)
    err = np.where(good, 1.0 / np.sqrt(np.where(iv > 0, iv, np.inf)), 1.0)

    # DESI LSF as a velocity sigma, the same quantity run_map_fits_outliers feeds
    # prospect as Spectrum(resolution=...): C_KMS / (2.355 * R).
    ires = (C_KMS / (2.355 * desi_resolution(WAVE_OBS))).astype(float)

    lam = WAVE_OBS if a.observed else WAVE_OBS / (1.0 + z)
    frame = "observed" if a.observed else f"rest (de-redshifted by z={z:.7f})"

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

    # wgt is a hard 0..1 mask (read_data.f90:103). Bad pixels go to 0 rather than
    # being dropped, so the wavelength grid stays contiguous.
    wgt = good[keep].astype(float)

    # Do NOT write flx=0 at bad pixels. alf's own rms diagnostic (func.f90:150) is
    # SQRT(SUM((flx/mflx-1)**2)/(i2-i1+1)) with NO weight term, so every zero-flux pixel
    # contributes exactly 1 and the printed rms becomes sqrt(masked fraction). Measured
    # on the first run: reported 24.9 / 32.0 / 62.2 % against sqrt(frac) of
    # 24.8 / 31.7 / 62.2 %, i.e. the diagnostic was reporting the mask, not the fit.
    # The chi2 is unaffected -- wgt=0 sends err to huge_number (alf.f90:315-317) -- so
    # this only repairs the diagnostic. Fill with a local median instead.

    # line masks, always in the rest frame: --observed changes the output column, not
    # where a stellar feature physically sits.
    rest = WAVE_OBS[keep] / (1.0 + z)
    med = [float(v) for v in a.mask.split(",") if v.strip()]
    assert len(med) % 2 == 0, "--mask needs an even number of edges"
    masked = []
    for m1, m2 in zip(med[::2], med[1::2]):
        sel = (rest >= m1) & (rest <= m2)
        # 0 here means the window fell outside the fitted intervals or the frame is
        # wrong -- alf would then happily fit a contaminated line as an abundance.
        masked.append((m1, m2, int(sel.sum())))
        wgt[sel] = 0.0

    fill = np.copy(flux)
    if (~good).any() and good.any():
        fill[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), flux[good])
    flx = fill[keep]
    er = np.where(good, err, np.nanmax(err[good]) * 1e3)[keep]

    # alf ships bin/ doc/ infiles/ scripts/ src/ subjobs/ and NOT indata/, results/ or
    # models/, but read_data.f90:41 reads indata/ and alf.f90:600 writes OUTDIR='results/'
    # (alf_vars.f90:12). Create them, or the run dies after the MCMC rather than before it.
    out_path = Path(a.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.parent.name == "indata":
        home = out_path.parent.parent
        for d in ("results", "models"):
            (home / d).mkdir(exist_ok=True)
        if not (home / "src" / "alf.f90").exists():
            print(f"  WARNING {home} has no src/alf.f90 -- is alf cloned there?")
            print( "          git clone https://github.com/cconroy20/alf "
                  f"{home}")

    with open(out_path, "w") as f:
        for l1, l2 in iv_pairs:
            f.write(f"# {l1:.4f} {l2:.4f}\n")
        for L, F, E, W, R in zip(lam[keep], flx, er, wgt, ires[keep]):
            f.write(f"{L:10.4f} {F:14.6e} {E:14.6e} {W:5.2f} {R:9.3f}\n")

    snr = (flux[keep][wgt > 0] / err[keep][wgt > 0])
    dl = float(np.median(np.diff(lam[keep])))
    print(f"wrote {out_path}")
    print(f"  TARGETID {a.tid}   z = {z:.7f}   frame: {frame}")
    print(f"  {int(keep.sum())} pixels, {int((wgt == 0).sum())} zero-weighted")
    for m1, m2, n in masked:
        print(f"  masked rest {m1:.1f}-{m2:.1f} A: {n} px"
              + ("   <-- ZERO, check frame/interval coverage" if n == 0 else ""))
    print(f"  lambda {lam[keep].min():.1f} - {lam[keep].max():.1f} A, "
          f"median spacing {dl:.3f} A")
    print(f"  intervals (um): " + ", ".join(f"{l1}-{l2}" for l1, l2 in iv_pairs))
    print(f"  ires {ires[keep].min():.1f} - {ires[keep].max():.1f} km/s")
    print(f"  median S/N: {np.median(snr):.1f} /pixel, "
          f"{np.median(snr) / np.sqrt(dl):.1f} /A")
    print("\n  alf's published mock tests span S/N = 20, 30, 50, 100 per A")
    print("  (Conroy+2018 sec 3.2, ms.tex:995-1053).")
    print("\n  next:")
    print(f"    cd $ALF_HOME/src && make")
    print(f"    mpirun -np 8 $ALF_HOME/bin/alf.exe {out_path.stem} <tag>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
