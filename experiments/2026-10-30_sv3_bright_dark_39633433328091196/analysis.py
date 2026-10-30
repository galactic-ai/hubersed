"""DR1 SV3 bright against SV3 dark coadd spectrum of the continuum outlier 39633433328091196 (#17).

QUESTION: is the blue excess of the SV3 bright spectrum (synthetic g 3.1 times the fibre g, against
1.6 in r and z) also in the independent SV3 dark spectrum?
HYPOTHESIS: none registered.
INPUTS: SV3 bright DR1 coadd from hubersed.io.desi.load_spectrum (spender chunk files). SV3 dark DR1
healpix coadd data/sv3_dark_coadds/coadd-sv3-dark-10231.fits and redrock-sv3-dark-10231.fits
(iron, sha256 checked; 1 exposure, EXPID 88050, 921 s), put through the same spender step
(spender DESI.prepare_spectra: ivar-weighted b, r, z stitching onto the 7781 px grid, DESI mask and
sky-line mask), with the spender norm undone as hubersed.io.desi.load_by_index does. Spender reads
the redrock file at the coadd path with "coadd" replaced, so a copy sits in data/sv3_dark_redrocks/.
SEED: none.
COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-30_sv3_bright_dark_39633433328091196/analysis.py``.
RESULT: 2026-10-30, bright_dark.txt. Dark/bright median flux is 0.10 at observed 3600-4000 A, 0.26
at 4000-4620, 0.55 at 4620-5500 and 0.79-0.87 redward: the bright spectrum's blue excess is not in
the dark one. Assumptions: DR1 coadds as released, no recoadd, 51 px median filter.
FIGURES: results/2026-10-30_sv3_bright_dark_39633433328091196/bright_dark.png and bright_dark.txt.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from spender.data.desi import DESI

from hubersed.io.desi import desi_spectrum, load_spectrum
from hubersed.paths import get_paths
from hubersed.plotting.style import use_apj_style

TID = 39633433328091196
PATHS = get_paths()
DARK = PATHS["DATA"] / "sv3_dark_coadds"
OUT = PATHS["RESULTS"] / "2026-10-30_sv3_bright_dark_39633433328091196"
OUT.mkdir(parents=True, exist_ok=True)
KERNEL = 51  # pixels of the median filter, about 41 A at 0.8 A per pixel
BINS = [(3600, 4000), (4000, 4620), (4620, 5500), (5500, 6500), (6500, 7600), (7600, 9824)]


def smooth(flux, good):
    """Return the median-filtered flux on good pixels, NaN elsewhere."""
    out = np.full(flux.size, np.nan)
    out[good] = medfilt(flux[good], KERNEL)
    return out


def as_dict(spec):
    """Return wave, flux and good pixels of a specutils spectrum from hubersed.io.desi."""
    return {
        "wave": np.asarray(spec.spectral_axis.value, float),
        "flux": np.asarray(spec.flux.value, float),
        "good": ~np.asarray(spec.mask, bool),
        "z": float(spec.redshift),
    }


# %% both spectra through spender, on the same grid
bright = as_dict(load_spectrum(TID))
spec, w, z, tid, norm, _ = DESI.prepare_spectra(str(DARK / "coadd-sv3-dark-10231.fits"))
i = int(np.flatnonzero(tid.numpy() == TID)[0])
n = float(norm[i])
dark = as_dict(desi_spectrum((spec[i] * n).numpy(), (w[i] / n**2).numpy(), float(z[i]), TID))
assert np.allclose(bright["wave"], dark["wave"]), "different wavelength grids"

# %% table: median flux in observed bins, and ratio dark / bright
lines = [f"bright z {bright['z']:.5f}, dark z {dark['z']:.5f}"]
lines.append(
    f"{'obs A':>11} {'bright':>8} {'dark':>8} {'dark/bright':>11}  (median 1e-17 erg/s/cm2/A)"
)
for lo, hi in BINS:
    sel = (bright["wave"] >= lo) & (bright["wave"] < hi)
    mb = np.median(bright["flux"][sel & bright["good"]])
    md = np.median(dark["flux"][sel & dark["good"]])
    lines.append(f"{lo:5d}-{hi:5d} {mb:8.3f} {md:8.3f} {md / mb:11.3f}")
text = "\n".join(lines)
(OUT / "bright_dark.txt").write_text(text + "\n")
print(text)

# %% figure: both spectra and their median filters in one panel
use_apj_style()
fig, ax = plt.subplots(figsize=(16, 6))
colors = {"bright": ("#f4a582", "#b2182b"), "dark": ("#92c5de", "#2166ac")}
for name, d in (("bright", bright), ("dark", dark)):
    raw, med = colors[name]
    ax.plot(d["wave"], np.where(d["good"], d["flux"], np.nan), color=raw, lw=0.4,
            label=f"SV3 {name} DR1 coadd")  # fmt: skip
for name, d in (("bright", bright), ("dark", dark)):
    ax.plot(d["wave"], smooth(d["flux"], d["good"]), color=colors[name][1], lw=1.2,
            label=f"SV3 {name}, median filter {KERNEL} px")  # fmt: skip
ax.set_xlabel(r"observed wavelength [$\AA$]", fontsize=14)
ax.set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
ax.legend(frameon=False, fontsize=12, loc="upper left")
fig.savefig(OUT / "bright_dark.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'bright_dark.png'}")
