"""SV3 spectra and imaging of six quiescent continuum outliers (#17).

QUESTION: what do the SV3 spectra and Legacy Survey images of the quiescent continuum outliers
look like, before one is picked as the next target?
HYPOTHESIS: none registered.
INPUTS: SV3 DR1 coadds from hubersed.io.desi.load_spectrum. RA and Dec from the METADATA of
data/fastspec-iron-sv3-bright.fits. Legacy Survey DR10 griz jpeg cutouts (256 px, 0.262"/px)
from spender DESI.get_image, cached in results/2026-10-29_quiescent_candidates_spectra/cutouts/.
The six TARGETIDs are the Halpha EW 0 rows (ranks 1, 2, 3, 4, 5, 7) of
results/2026-10-21_continuum_outlier_snr/table.txt, whose code is not committed.
SEED: none.
COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-29_quiescent_candidates_spectra/analysis.py``.
RESULT: pending.
FIGURES: results/2026-10-29_quiescent_candidates_spectra/spectra_cutouts.png.
"""

# %%
import io

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from PIL import Image
from scipy.signal import medfilt
from spender.data.desi import DESI

from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TIDS = [
    39627775706927918,
    39633162581573659,
    39633315497510808,
    39633433328091196,
    39627799887088692,
    39627757533007793,
]
KERNEL = 9
OUT = PATHS["RESULTS"] / "2026-10-29_quiescent_candidates_spectra"
CUTOUTS = OUT / "cutouts"
VAC = PATHS["DATA"] / "fastspec-iron-sv3-bright.fits"


# %%
def radec(tids):
    """Return a dict of TARGETID to (RA, Dec) in degrees from the VAC METADATA."""
    meta = Table.read(VAC, hdu="METADATA")
    rows = {int(t): i for i, t in enumerate(meta["TARGETID"])}
    return {t: (float(meta["RA"][rows[t]]), float(meta["DEC"][rows[t]])) for t in tids}


def cutout(tid, ra, dec):
    """Return the cutout image of one target, fetched once and then read from the cache."""
    path = CUTOUTS / f"{tid}.jpg"
    if not path.exists():
        CUTOUTS.mkdir(parents=True, exist_ok=True)
        path.write_bytes(DESI.get_image(ra=ra, dec=dec, size=256, pixscale=0.262, bands="griz"))
    return Image.open(io.BytesIO(path.read_bytes()))


# %%
use_apj_style()
coords = radec(TIDS)
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
for ax, tid in zip(axes.ravel(), TIDS, strict=True):
    spec = load_spectrum(tid)
    z = float(spec.redshift)
    good = ~spec.mask
    wave = spec.spectral_axis.value / (1 + z)
    flux = spec.flux.value.copy()
    flux[~good] = np.nan
    smooth = np.full_like(flux, np.nan)
    smooth[good] = medfilt(flux[good], KERNEL)
    ax.plot(wave, flux, color="0.75", lw=0.5)
    ax.plot(wave, smooth, color="k", lw=0.8)
    ax.set_title(f"{tid}  z = {z:.3f}")
    ax.set_xlabel(r"rest wavelength [$\AA$]")
    ax.set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    inset = ax.inset_axes([0.76, 0.52, 0.23, 0.46])
    inset.imshow(cutout(tid, *coords[tid]))
    inset.set_xticks([])
    inset.set_yticks([])
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "spectra_cutouts.png", dpi=150)
print("saved", OUT / "spectra_cutouts.png")
