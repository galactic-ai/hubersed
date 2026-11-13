"""DR1 coadd spectrum and Legacy DR9 cutout of each of the 105 new v2-mock outliers.

QUESTION: what do the 105 v2-mock outliers after SIMBAD that are not in the 310 look like?
HYPOTHESIS: none registered; a visual look.
INPUTS: full and cont of results/2026-11-09_mocks_v2_outliers/postsimbad_v2.npz minus target_ids
of results/outliers310_sample.npz, sorted by TARGETID; DR1 coadds through load_by_index; RA and
DEC from the METADATA of data/fastspec-iron-sv3-bright.fits; 96 px ls-dr9 cutouts at 0.262''/px,
reused from earlier galleries when there, else downloaded to results/2026-11-13_gallery105/cutouts/.
The cutout sits in its own panel right of the spectrum, not as an inset, so it hides no pixels.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-13_gallery105/analysis.py``.
RESULT: figures only.
FIGURES: results/2026-11-13_gallery105/<NN>_<TARGETID>.png, rest-frame pixels with ivar > 0 (grey)
and their 9 px median filter (black).
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
OUT = RES / "2026-11-13_gallery105"
CACHES = [RES / "2026-10-31_logp_vs_colour" / "cutouts", RES / "2026-11-01_gallery92" / "cutouts"]
CUT = "https://www.legacysurvey.org/viewer/cutout.jpg?ra={}&dec={}&layer=ls-dr9&pixscale=0.262&size=96"
N_MEDIAN = 9


def targets():
    """Return the sorted TARGETIDs of the v2 outliers after SIMBAD that are not in the 310."""
    new = np.load(RES / "2026-11-09_mocks_v2_outliers" / "postsimbad_v2.npz")
    old = set(np.load(RES / "outliers310_sample.npz")["target_ids"].tolist())
    tids = sorted((set(new["full"].tolist()) | set(new["cont"].tolist())) - old)
    assert len(tids) == 105, len(tids)
    return [int(t) for t in tids]


def cutout(t, ra, dec):
    """Return the jpg bytes of the cutout of target ``t``, downloading it if not cached."""
    for p in [c / f"{t}.jpg" for c in CACHES] + [OUT / "cutouts" / f"{t}.jpg"]:
        if p.exists():
            return p.read_bytes()
    data = urllib.request.urlopen(CUT.format(ra, dec), timeout=60).read()
    (OUT / "cutouts" / f"{t}.jpg").write_bytes(data)
    time.sleep(0.5)
    return data


def main():
    """Write one spectrum-and-cutout figure per target."""
    tids = targets()
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
        fig, (ax, ins) = plt.subplots(
            1, 2, figsize=(18, 4.6), gridspec_kw={"width_ratios": [5, 1], "wspace": 0.03}
        )
        ax.plot(rest, flux[good], color="0.6", lw=0.5)
        ax.plot(rest, medfilt(flux[good], N_MEDIAN), color="k", lw=0.6)
        ax.set_title(f"{k}/{len(tids)}   {t}   z {z:.3f}", fontsize=12)
        ax.set_xlabel(r"rest wavelength [\AA]", fontsize=11)
        ax.set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]", fontsize=11)
        ins.imshow(plt.imread(io.BytesIO(cutout(t, *pos[t])), format="jpg"), origin="upper")
        ins.set_xticks([])
        ins.set_yticks([])
        fig.savefig(OUT / f"{k:02d}_{t}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {len(tids)} figures to {OUT}")


if __name__ == "__main__":
    main()
