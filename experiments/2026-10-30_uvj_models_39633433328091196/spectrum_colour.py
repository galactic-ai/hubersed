"""Rest-frame U-V of 39633433328091196 measured on the DESI spectrum, against the model spectra.

QUESTION: is the model U-V (1.6-2.0) below the CIGALE and FastSpecFit 2.15 because of the fits,
or does the DESI spectrum itself give a low U-V?
The SV3 DR1 coadd (load_spectrum) covers rest 2586-7058 A at z 0.392, so the FSPS U and V curves
of colours.py are inside it. Masked pixels are filled by linear interpolation. The model spectra
are the own-point full rest spectra of colours.py (colours_<lib>.npz, with dust). Each colour is
also given with every spectrum cut to the rest range the DESI data cover, so the data and models
see the same wavelengths. No Milky Way extinction correction (E(B-V) 0.029 in the fastspec
METADATA) is applied to the data.

COMMAND: from the repository root on the Mac, with SPS_HOME=~/Astronomy_Research/fsps,
``uv run --no-sync python experiments/2026-10-30_uvj_models_39633433328091196/spectrum_colour.py``.
OUTPUT: results/2026-10-30_uvj_models_39633433328091196/spectrum_colour.txt.
"""

import importlib.util
from pathlib import Path

import numpy as np

from hubersed.io.desi import load_spectrum

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("colours", HERE / "colours.py")
colours = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colours)

s = load_spectrum(colours.TID)
z = float(s.meta["redshift"]) if "redshift" in s.meta else float(s.redshift)
wave_obs = np.asarray(s.spectral_axis.value, float)
flam = np.asarray(s.flux.value, float)
good = ~np.asarray(s.mask, bool) if s.mask is not None else np.isfinite(flam)
rest = wave_obs / (1 + z)
flam_filled = np.interp(rest, rest[good], flam[good])
fnu_data = flam_filled * rest**2  # f_nu up to a constant, rest-frame wavelength axis
cover = (rest[0], rest[-1])

curves = colours.read_filters()
lines = [
    f"z {z:.5f}, DESI rest coverage {cover[0]:.0f}-{cover[1]:.0f} A, "
    f"{(~good).sum()} of {good.size} pixels masked and interpolated",
]
for b in ("U", "V"):
    w, r = curves[b]
    w_in = w[r > 0.01 * r.max()]
    lines.append(f"{b} curve, response above 1% of peak: {w_in.min():.0f}-{w_in.max():.0f} A")


def uv(wave, fnu, lo=None, hi=None):
    """Return U-V for f_nu on rest wavelengths, optionally cut to [lo, hi]."""
    sel = np.ones(wave.size, bool) if lo is None else (wave >= lo) & (wave <= hi)
    return colours.ab_mag(wave[sel], fnu[sel], curves["U"]) - colours.ab_mag(
        wave[sel], fnu[sel], curves["V"]
    )


lines.append(f"DESI SV3 bright spectrum: U-V {uv(rest, fnu_data):.3f}")

# the SV3 dark coadd through spender, as in 2026-10-30_sv3_bright_dark_39633433328091196
from spender.data.desi import DESI  # noqa: E402

dark_file = colours.ROOT / "data" / "sv3_dark_coadds" / "coadd-sv3-dark-10231.fits"
spec_d, w_d, z_d, tid_d, norm_d, _ = DESI.prepare_spectra(str(dark_file))
k = int(np.flatnonzero(tid_d.numpy() == colours.TID)[0])
flam_d = spec_d[k].numpy() * float(norm_d[k])
good_d = w_d[k].numpy() > 0
rest_d = wave_obs / (1 + float(z_d[k]))
fnu_d = np.interp(rest_d, rest_d[good_d], flam_d[good_d]) * rest_d**2
lines.append(
    f"DESI SV3 dark spectrum (z {float(z_d[k]):.5f}, {(~good_d).sum()} px masked): "
    f"U-V {uv(rest_d, fnu_d):.3f}"
)

# the PR 2377 recoadd of the SV3 bright exposures (2026-10-30_recoadd_39633433328091196)
rec = np.load(
    colours.RESULTS
    / "2026-10-30_recoadd_39633433328091196"
    / "sv3"
    / "resolution_matrix_rows_ivar.npz"
)
good_r = rec["ivar"] > 0
fnu_r = np.interp(rest, rest[good_r], rec["flux"][good_r]) * rest**2
lines.append(f"SV3 bright recoadd: U-V {uv(rest, fnu_r):.3f}")
for lib, point in (("c3k_hr", "c3k"), ("miles", "miles")):
    d = np.load(colours.OUT / f"colours_{lib}.npz")
    w, f = d["full_wave"], d[f"full_spec_{point}_dust"]
    lines.append(
        f"{lib} at its own point: U-V full {uv(w, f):.3f}, cut to DESI coverage {uv(w, f, *cover):.3f}"
    )
text = "\n".join(lines)
(colours.OUT / "spectrum_colour.txt").write_text(text + "\n")
print(text)
