"""Photometric colours of 39633433328091196 against the C3K ER SV3 dark MAPs with and without dust (#17).

QUESTION: do the Legacy Survey colours r-z, z-W1 and W1-W2 favour the C3K ER dark MAP with dust or
the one without? Does the SV3 dark spectrum give the fiber photometry g-r and r-z?
HYPOTHESIS: none registered.
INPUTS: <TID>_map.npz of 2026-10-30_continuum_39633433328091196_dark_map (with dust) and of
er_nodust/ in 2026-10-30_nodust_miles_39633433328091196_dark (without); SFH hyperparameters from
hr_maxl.npz and the model of 2026-10-29_continuum_39633433328091196_map; theta_of and ab_mag of
2026-10-30_uvj_models_39633433328091196/colours.py. Photometry from the METADATA of
data/fastspec-iron-sv3-bright.fits (PHOTSYS N). BASS g, r and MzLS z from speclite v1.0.0 (filters/),
WISE W1 and W2 from $SPS_HOME/data/allfilters.dat. dark_spectrum.npz of the dark map folder, and
the SV3 bright DR1 coadd of load_spectrum as a control.
SEED: none.
COMMAND: from the repository root on the Mac, ``SPS_HOME=~/Astronomy_Research/fsps OMP_NUM_THREADS=1
uv run --no-sync python experiments/2026-10-30_photcolour_dark_39633433328091196/analysis.py``.
RESULT: 2026-10-30, photcolour_dark.txt. The SV3 dark spectrum gives the fiber g-r within 0.02 and
r-z within 0.04 mag; the bright control is 0.72 bluer in g-r. The model rows do not decide: on the
dark spectrum's range the C3K_HR stand-in misses the dark colours by 0.03-0.06 mag, as large as the
dust against no-dust gap (r-z 0.09, z-W1 0.14). Against total photometry r-z is -4.5 sigma with
dust and -1.5 without; z-W1 lies between the models. Assumptions: C3K_HR with afe 0 for C3K ER,
fiber spectrum against total photometry (observed fiber r-z 0.029 redder), MzLS z 90% inside DESI.
FIGURES: none; table in results/2026-10-30_photcolour_dark_39633433328091196/photcolour_dark.txt.
"""

# %%
import importlib.util
import os
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from hubersed.io.desi import desi_spectrum, load_spectrum
from hubersed.paths import PATHS

TID = 39633433328091196
HERE = Path(__file__).resolve().parent
EXPS = HERE.parent
RES = PATHS["RESULTS"]
OUT = RES / HERE.name
NPZ = f"{TID}_map.npz"
DARK = RES / "2026-10-30_continuum_39633433328091196_dark_map"
MAPS = {
    "dust": DARK / NPZ,
    "no dust": RES / "2026-10-30_nodust_miles_39633433328091196_dark" / "er_nodust" / NPZ,
}
HR_MAXL = RES / "2026-10-29_continuum_39633433328091196_map" / "hr_maxl.npz"
SPECLITE = {"g": "BASS-g", "r": "BASS-r", "z": "MzLS-z"}
WISE = {"W1": 83, "W2": 84}  # FILTER_LIST index of allfilters.dat
COLUMN = {"g": "G", "r": "R", "z": "Z", "W1": "W1", "W2": "W2"}
FLOOR = {"g": 0.02, "r": 0.02, "z": 0.02, "W1": 0.05, "W2": 0.05}  # mag, as FastSpecFit 2.5.1
MODEL_COLOURS = [("r", "z"), ("z", "W1"), ("W1", "W2")]
SPEC_COLOURS = [("g", "r"), ("r", "z")]


def load(name, path):
    """Return the module at ``path``, loaded by path since experiment files share names."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_curves():
    """Return {band: (wave, response)} for BASS g, r, MzLS z and WISE W1, W2."""
    out = {}
    for b, name in SPECLITE.items():
        t = Table.read(HERE / "filters" / f"{name}.ecsv", format="ascii.ecsv")
        out[b] = np.array([np.asarray(t["wavelength"], float), np.asarray(t["response"], float)])
    blocks = []
    with open(Path(os.environ["SPS_HOME"]) / "data" / "allfilters.dat") as f:
        for line in f:
            if line.startswith("#"):
                blocks.append([])
            elif line.strip():
                blocks[-1].append([float(x) for x in line.split()[:2]])
    for b, index in WISE.items():
        out[b] = np.array(blocks[index - 1]).T
    return out


def coverage(wave, curve):
    """Return the photon-weighted fraction of a filter inside [wave.min(), wave.max()]."""
    w, r = curve
    inside = (w >= wave.min()) & (w <= wave.max())
    return np.trapezoid((r / w) * inside, w) / np.trapezoid(r / w, w)


colours = load("colours", EXPS / "2026-10-30_uvj_models_39633433328091196" / "colours.py")
curves = read_curves()
lines = []

# %% photometry, total corrected for Milky Way extinction, fiber as observed
meta = fits.getdata(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", "METADATA")
j = int(np.flatnonzero(meta["TARGETID"] == TID)[0])
assert meta["PHOTSYS"][j] == "N", meta["PHOTSYS"][j]
phot = {}
for b, c in COLUMN.items():
    flux, ivar = float(meta[f"FLUX_{c}"][j]), float(meta[f"FLUX_IVAR_{c}"][j])
    mag = -2.5 * np.log10(flux / float(meta[f"MW_TRANSMISSION_{c}"][j]))
    phot[b] = (mag, np.hypot(2.5 / np.log(10) / (flux * np.sqrt(ivar)), FLOOR[b]))
fiber = {b: -2.5 * np.log10(float(meta[f"FIBERFLUX_{COLUMN[b]}"][j])) for b in "grz"}

# %% model colours at the two dark MAP points, C3K_HR with afe 0 standing in for C3K ER
import fsps  # noqa: E402

from hubersed.sps.parameter_file import build_sps  # noqa: E402

target = load("fit_1029", EXPS / "2026-10-29_continuum_39633433328091196_map" / "fit.py")
dark = np.load(DARK / "dark_spectrum.npz")
assert int(dark["targetid"]) == TID
z = float(dark["z"])
model = target.base.make_model(z)
sps = build_sps(zcontinuous=1, zero_library_resolution=False)
assert sps.ssp.libraries[1] in ("c3k_hr", b"c3k_hr"), sps.ssp.libraries
assert fsps.fsps.driver.get_nafe() == 1, "this build has afe slots; revisit afe = 0"
hr = colours.as_dict(HR_MAXL)
mags, seds = {}, {}
for k, path in MAPS.items():
    point = colours.as_dict(path)
    point.update({h: hr[h] for h in colours.HYPERS})
    theta, zeroed = colours.theta_of(model, point, "dust" if k == "dust" else "nodust")
    model.set_parameters(theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wave, spec, _ = sps.get_galaxy_spectrum(**model.params)
    wobs = np.asarray(wave, float) * (1 + z)
    seds[k] = (wobs, np.asarray(spec, float))
    mags[k] = {b: colours.ab_mag(wobs, np.asarray(spec, float), c) for b, c in curves.items()}
    lines.append(
        f"model {k}: logzsol {point['logzsol']:.3f}, dust2 {float(np.ravel(sps.ssp.params['dust2'])[0]):.3f}, "
        f"dust1 {float(np.ravel(sps.ssp.params['dust1'])[0]):.3f}, set to 0 {zeroed}"
    )
npix = {}
for b, (cw, cr) in curves.items():
    inband = cw[cr > 0.01 * cr.max()]
    npix[b] = int(np.sum((wobs >= inband.min()) & (wobs <= inband.max())))
lines.append(f"FSPS wavelength points inside each band (response > 1% of peak): {npix}")

lines += [
    "",
    f"Total photometry (PHOTSYS N, MW corrected) against the models at z {z:.5f}. Sigma from "
    f"FLUX_IVAR with floors {FLOOR} mag added in quadrature.",
    f"{'colour':>8} {'data':>14} {'model dust':>11} {'d - m':>7} {'sigma':>6} "
    f"{'model no dust':>14} {'d - m':>7} {'sigma':>6}",
]
for a, b in MODEL_COLOURS:
    col = phot[a][0] - phot[b][0]
    sig = np.hypot(phot[a][1], phot[b][1])
    row = f"{a + '-' + b:>8} {col:7.3f} +- {sig:.3f}"
    for k in MAPS:
        mc = mags[k][a] - mags[k][b]
        row += f" {mc:{11 if k == 'dust' else 14}.3f} {col - mc:+7.3f} {(col - mc) / sig:+6.2f}"
    lines.append(row)

# %% spectrum colours against fiber photometry, both as observed (no MW correction)
lines += [
    "",
    "Spectra against fiber photometry, both without MW correction; masked pixels interpolated. "
    "Fraction of each filter (photon weighted) inside the spectrum in brackets.",
]
spectra = {
    "SV3 dark (EXPID 88050)": desi_spectrum(dark["flux"], dark["ivar"], z, TID),
    "SV3 bright DR1 coadd (control)": load_spectrum(TID),
}
row = "  fiber photometry: " + ", ".join(
    f"{a}-{b} {fiber[a] - fiber[b]:.3f}" for a, b in SPEC_COLOURS
)
lines.append(row)
spec_mags, spec_range = {}, {}
for name, s in spectra.items():
    w = np.asarray(s.spectral_axis.value, float)
    f = np.asarray(s.flux.value, float)
    good = ~np.asarray(s.mask, bool)
    fnu = np.interp(w, w[good], f[good]) * w**2
    sm = {b: colours.ab_mag(w, fnu, curves[b]) for b in "grz"}
    spec_mags[name], spec_range[name] = sm, (w[good].min(), w[good].max())
    cov = {b: coverage(w[good], curves[b]) for b in "grz"}
    parts = [
        f"{a}-{b} {sm[a] - sm[b]:.3f} (spec - fiber {sm[a] - sm[b] - (fiber[a] - fiber[b]):+.3f})"
        for a, b in SPEC_COLOURS
    ]
    lines.append(
        f"  {name}: "
        + ", ".join(parts)
        + f" [g {cov['g']:.2f}, r {cov['r']:.2f}, z {cov['z']:.2f}]"
    )

# %% stand-in check: model colours on the dark spectrum's range against the dark spectrum
# The spectrum is corrected for MW extinction with the photometric MW_TRANSMISSION of each band.
name = "SV3 dark (EXPID 88050)"
lo, hi = spec_range[name]
trans = {b: float(meta[f"MW_TRANSMISSION_{COLUMN[b]}"][j]) for b in "grz"}
lines += ["", f"Stand-in check: models cut to the dark spectrum's range ({lo:.0f}-{hi:.0f} A)."]
for a, b in SPEC_COLOURS:
    sc = spec_mags[name][a] - spec_mags[name][b] + 2.5 * np.log10(trans[a] / trans[b])
    row = f"  {a}-{b}: dark spectrum, MW corrected {sc:.3f}"
    for k, (wobs, spec) in seds.items():
        sel = (wobs >= lo) & (wobs <= hi)
        mc = colours.ab_mag(wobs[sel], spec[sel], curves[a]) - colours.ab_mag(
            wobs[sel], spec[sel], curves[b]
        )
        row += f"; model {k} {mc:.3f} (spec - model {sc - mc:+.3f})"
    lines.append(row)

text = "\n".join(lines) + "\n"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "photcolour_dark.txt").write_text(text)
print(text)
