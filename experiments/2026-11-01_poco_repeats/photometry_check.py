"""Colours of each ladder step's best model against the Legacy Surveys photometry of one target.

QUESTION: do the best models reproduce the broadband colours, which they were not fitted to?
- Synthetic AB maggies use the photon-counting mean of F_nu over the band. They are made from each
  step's max-likelihood model and from its data spectrum (<step>_maxl.npz, observed frame,
  maggies).
- Curves are the speclite v1.0.0 files in data/filters/speclite_v1.0.0/: BASS g, r and MzLS z for
  PHOTSYS N, DECam otherwise.
- A band is used if 99 per cent of its weight lies where the spectrum is defined: the fitted
  pixels for the data, and the whole DESI grid for the model (the MILES fit window for MILES).
- Our spectra and models are not corrected for Milky Way extinction, so they are compared with the
  observed fluxes, not divided by MW_TRANSMISSION.
- Photometry: LS DR9 FIBERFLUX (1.5 arcsec) and FLUX (total) from the FastSpecFit v3.0 METADATA.
  Their errors are 1/sqrt(FLUX_IVAR) (total only).
- FastSpecFit v3.0 FLUX_SYNTH_G/R (from the spectrum) and FLUX_SYNTH_PHOTMODEL_* (its model,
  MW-free, so multiplied by MW_TRANSMISSION here) are listed for comparison.
- The r-band flux of each spectrum and model is compared with FIBERFLUX_R and FLUX_R, to place
  the aperture of our fitted mass. MILES is left out (its window does not cover r fully at all z).
- r-z, W1 and W2 lie beyond the spectrum and need the full FSPS SED; they are not done here.
COMMAND: ``uv run --no-sync python experiments/2026-11-01_poco_repeats/photometry_check.py --tid <TID>``.
OUTPUT: results/2026-11-01_poco_repeats/<TID>/photometry_check.txt.
"""

# %%
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from hubersed.paths import PATHS

HERE = Path(__file__).resolve().parent
FILT = PATHS["DATA"] / "filters" / "speclite_v1.0.0"
FSF = PATHS["DATA"] / "fastspecfit_v3.0" / "fastspec-iron-sv3-bright.fits"
NAMES = {"N": ("BASS-g", "BASS-r", "MzLS-z"), "S": ("decam2014-g", "decam2014-r", "decam2014-z")}
COVER = 0.99
MILES_REST = (3601.8, 7400.8)  # the MILES fit window of 2026-10-08 fit.py


def load(name, path):
    """Load the file at ``path`` as module ``name``."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def synth(wave, fnu, curve, lo, hi):
    """Return the AB maggies of F_nu (maggies) through a curve, or NaN if [lo, hi] covers too little."""
    w, r = np.asarray(curve["wavelength"], float), np.asarray(curve["response"], float)
    weight = r / w
    inside = (w >= lo) & (w <= hi)
    if np.trapezoid(weight * inside, w) / np.trapezoid(weight, w) < COVER:
        return np.nan
    f = np.interp(w, wave, fnu)
    return float(np.trapezoid(f * weight * inside, w) / np.trapezoid(weight * inside, w))


def colour(a, b):
    """Return -2.5 log10(a / b), NaN if either is not positive."""
    return float(-2.5 * np.log10(a / b)) if a > 0 and b > 0 and np.isfinite(a / b) else np.nan


def main(tid):
    """Write the colour table of ``tid``."""
    lp = load("ladder_plot", HERE / "ladder_plot.py")
    m = fits.getdata(FSF, "METADATA")
    r = m[m["TARGETID"] == tid][0]
    sys_ = "N" if str(r["PHOTSYS"]).strip() == "N" else "S"
    curves = {
        b[-1]: Table.read(FILT / f"{n}.ecsv") for b, n in zip("grz", NAMES[sys_], strict=True)
    }
    phot = {}
    for kind, pre in (("fibre", "FIBERFLUX_"), ("total", "FLUX_")):
        f = {b: float(r[pre + b.upper()]) for b in "grz"}
        phot[kind] = (colour(f["g"], f["r"]), colour(f["r"], f["z"]))
    err = {b: 1.0857 / (float(r["FLUX_" + b.upper()]) * np.sqrt(float(r["FLUX_IVAR_" + b.upper()])))
           for b in "grz"}  # fmt: skip
    s = fits.getdata(FSF, "SPECPHOT")
    s = s[s["TARGETID"] == tid][0]
    mw = {b: float(r["MW_TRANSMISSION_" + b.upper()]) for b in "grz"}
    pm = {b: float(s["FLUX_SYNTH_PHOTMODEL_" + b.upper()]) * mw[b] for b in "grz"}
    fsf = {"spec": colour(float(s["FLUX_SYNTH_G"]), float(s["FLUX_SYNTH_R"])),
           "model": colour(pm["g"], pm["r"]), "model_rz": colour(pm["r"], pm["z"])}  # fmt: skip
    lines = [f"TARGETID {tid}, PHOTSYS {sys_}, filters {NAMES[sys_]}; AB colours, observed (no MW correction)",
             f"  LS DR9 fibre: g-r {phot['fibre'][0]:.3f}, r-z {phot['fibre'][1]:.3f}",
             f"  LS DR9 total: g-r {phot['total'][0]:.3f} +- {np.hypot(err['g'], err['r']):.3f}, "
             f"r-z {phot['total'][1]:.3f} +- {np.hypot(err['r'], err['z']):.3f}",
             f"  FastSpecFit v3.0 synthesised from the spectrum: g-r {fsf['spec']:.3f}",
             f"  FastSpecFit v3.0 best model, total, reddened by MW_TRANSMISSION: g-r {fsf['model']:.3f}, "
             f"r-z {fsf['model_rz']:.3f}",
             "",
             f"{'step':36s} {'g-r data':>9s} {'g-r model':>10s} {'r-z data':>9s} {'r-z model':>10s}"]  # fmt: skip
    for label, path, _ in lp.steps(tid):
        if not path.exists():
            continue
        d = np.load(path)
        good = np.asarray(d["good"], bool)
        z1 = 1 + float(d["z"])
        span = {"flux": (d["wave"][good].min(), d["wave"][good].max())}
        # the model is a prediction wherever it is defined; MILES only inside its fit window
        span["sp_best"] = (MILES_REST[0] * z1, MILES_REST[1] * z1) if label == "MILES" else (
            d["wave"].min(), d["wave"].max())  # fmt: skip
        cells = []
        for b1, b2 in (("g", "r"), ("r", "z")):
            for x in ("flux", "sp_best"):
                fx = np.where(good, d[x], np.nan) if x == "flux" else np.asarray(d[x], float)
                keep = np.isfinite(fx)
                lo, hi = span[x]
                a = synth(d["wave"][keep], fx[keep], curves[b1], lo, hi)
                c = synth(d["wave"][keep], fx[keep], curves[b2], lo, hi)
                cells.append(colour(a, c))
        lines.append(
            f"{label:36s} "
            + " ".join(f"{c:9.3f}" if np.isfinite(c) else f"{'-':>9s}" for c in cells)
        )
    lines += [
        "",
        "'-': less than 99 per cent of a band's weight lies where the spectrum is defined.",
        "",
        f"r-band flux in nanomaggies, observed; LS DR9 FIBERFLUX_R {float(r['FIBERFLUX_R']):.3f}, "
        f"FLUX_R {float(r['FLUX_R']):.3f}",
    ]
    for label, path, _ in lp.steps(tid):
        if not path.exists() or label == "MILES":
            continue
        d = np.load(path)
        good = np.asarray(d["good"], bool)
        fx = np.where(good, d["flux"], np.nan)
        keep = np.isfinite(fx)
        fr = 1e9 * synth(d["wave"][keep], fx[keep], curves["r"], d["wave"][good].min(),
                         d["wave"][good].max())  # fmt: skip
        mr = 1e9 * synth(d["wave"], np.asarray(d["sp_best"], float), curves["r"], d["wave"].min(),
                         d["wave"].max())  # fmt: skip
        lines.append(f"  {label:36s} data {fr:7.3f} (x{fr / float(r['FIBERFLUX_R']):.2f} fibre, "
                     f"x{fr / float(r['FLUX_R']):.3f} total), model {mr:7.3f}")  # fmt: skip
    out = PATHS["RESULTS"] / HERE.name / str(tid) / "photometry_check.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tid", type=int, required=True)
    main(ap.parse_args().tid)
