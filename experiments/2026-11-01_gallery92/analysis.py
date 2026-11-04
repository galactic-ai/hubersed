"""DR1 coadd spectrum and Legacy DR9 cutout of each of the 92 canonical continuum outliers (#17).

QUESTION: what do the 92 canonical common continuum outliers look like, spectrum and image?
HYPOTHESIS: none registered; a visual look.
INPUTS: cont of results/prescreened_outliers_postsimbad.npz (the canonical 92, outlier log Addendum
5); DR1 coadds from the DESIchunk1024 pkl through load_by_index; RA and DEC from the METADATA of
data/fastspec-iron-sv3-bright.fits; 96 px ls-dr9 cutouts at 0.262''/px from the Legacy Survey
viewer, reused from results/2026-10-31_logp_vs_colour/cutouts/ when there, else downloaded to
results/2026-11-01_gallery92/cutouts/.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-01_gallery92/analysis.py``.
RESULT: figures only.
FIGURES: results/2026-11-01_gallery92/<NN>_<TARGETID>.png, rest-frame pixels with ivar > 0 (grey)
and their 9 px median filter (black), in the npz order.
"""

import io
import time
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import medfilt
from spender.data.desi import DESI

from hubersed.io.desi import load_by_index, tids_to_indices
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

RES = PATHS["RESULTS"]
OUT = RES / "2026-11-01_gallery92"
CACHE = RES / "2026-10-31_logp_vs_colour" / "cutouts"
CUT = "https://www.legacysurvey.org/viewer/cutout.jpg?ra={}&dec={}&layer=ls-dr9&pixscale=0.262&size=96"
N_MEDIAN = 9


def cutout(t, ra, dec):
    """Return the jpg bytes of the cutout of target ``t``, downloading it if not cached."""
    for p in (CACHE / f"{t}.jpg", OUT / "cutouts" / f"{t}.jpg"):
        if p.exists():
            return p.read_bytes()
    data = urllib.request.urlopen(CUT.format(ra, dec), timeout=60).read()
    (OUT / "cutouts" / f"{t}.jpg").write_bytes(data)
    time.sleep(0.5)
    return data


def main():
    """Write one spectrum-and-cutout figure per target."""
    tids = [int(t) for t in np.load(RES / "prescreened_outliers_postsimbad.npz")["cont"]]
    assert len(tids) == 92, len(tids)
    md = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits")["METADATA"].data
    pos = {int(t): (float(r), float(d)) for t, r, d in zip(md["TARGETID"], md["RA"], md["DEC"],
                                                           strict=True)}  # fmt: skip
    (OUT / "cutouts").mkdir(parents=True, exist_ok=True)
    wave = DESI._wave_obs.numpy()
    use_apj_style()
    for k, t in enumerate(tids, 1):
        flux, ivar, z, _ = load_by_index(int(tids_to_indices(np.array([t]))[0]))
        flux, ivar, z = np.asarray(flux, float), np.asarray(ivar, float), float(z)
        good = ivar > 0
        rest = wave[good] / (1 + z)
        fig, ax = plt.subplots(figsize=(18, 4.6))
        ax.plot(rest, flux[good], color="0.6", lw=0.5)
        ax.plot(rest, medfilt(flux[good], N_MEDIAN), color="k", lw=0.6)
        ax.set_title(f"{k}/92   {t}   z {z:.3f}", fontsize=12)
        ax.set_xlabel(r"rest wavelength [\AA]", fontsize=11)
        ax.set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]", fontsize=11)
        ins = ax.inset_axes([0.83, 0.45, 0.16, 0.53])
        ins.imshow(plt.imread(io.BytesIO(cutout(t, *pos[t])), format="jpg"), origin="upper")
        ins.set_xticks([])
        ins.set_yticks([])
        fig.savefig(OUT / f"{k:02d}_{t}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {len(tids)} figures to {OUT}")


if __name__ == "__main__":
    main()
