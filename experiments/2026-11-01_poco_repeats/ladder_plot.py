"""Model ladder of one poco_repeats target: its SV3 recoadd with the recoadd fit, and chi of six models.

As 2026-10-08_continuum_outlier/spectrum_ladder_recoadd.py for 39627757533007793, with pocoMC
max-likelihood models in place of the nautilus and MAP ones. Top: the recoadd flux on the pixels its
fit used, a 9 pixel median filter, and the recoadd model. Below, one chi panel per step, each on the
data and uncertainty that step was fitted to or predicted for: MILES, C3K HR (degraded), C3K HR +
afe (degraded), C3K ER + afe (median LSF, predicted at the R-matrix fit's max-likelihood point), C3K
ER + afe + R matrix on the DR1 coadd, and on the recoadd. As the 10-08 figure, the top panel starts
at zero and the x range spans every step's fitted pixels. A step whose file does not exist yet gets
an empty panel marked "not run yet", and until the recoadd fit exists the top panel shows the DR1
coadd with its R-matrix model. The index bands of 2026-10-29_continuum_recoadd/analysis.py
($SPS_HOME/data/allindices.dat) are shaded in every panel, the legend sits centred above the top
panel, and the target's Legacy Survey DR10 griz cutout (spender DESI.get_image, 96 px at
0.262''/px, RA and Dec from the METADATA of data/fastspec-iron-sv3-bright.fits as in
2026-10-29_quiescent_candidates_spectra) sits in the top right of the top panel, with the 1.5''
DESI fibre aperture drawn at its centre as in tmp/fetch_cutouts.py.
COMMAND: ``uv run --no-sync python experiments/2026-11-01_poco_repeats/ladder_plot.py --tid <TARGETID>``.
OUTPUT: results/2026-11-01_poco_repeats/<TARGETID>/ladder.png.
"""

# %%
import argparse
import io
import os
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.patches import Circle
from PIL import Image
from scipy.signal import medfilt
from spender.data.desi import DESI

from hubersed.conversion import to_flambda
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

RES = PATHS["RESULTS"] / "2026-11-01_poco_repeats"
CUTOUTS = RES / "cutouts"
VAC = PATHS["DATA"] / "fastspec-iron-sv3-bright.fits"
CUT_PX, PIXSCALE = 96, 0.262  # cutout size in px and arcsec per px
FIBRE_DIAM = 1.5  # arcsec, DESI fibre on the sky
INDEX_NAMES = "Dn4000 HdeltaA CN1 Ca4227 G4300 C4668 Hbeta Mgb Fe5270 Fe5335 NaD TiO1 TiO2".split()
N_MEDIAN = 9


def steps(tid):
    """Return (label, file, colour) of each step, in ladder order."""
    d = RES / str(tid)
    return (
        ("MILES", d / "miles" / f"{tid}_miles_maxl.npz", "#b2182b"),
        ("C3K HR (degraded)", d / "c3k_hr" / f"{tid}_c3k_hr_maxl.npz", "#2166ac"),
        ("C3K HR + afe (degraded)", d / "c3k_hr_afe" / f"{tid}_c3k_hr_afe_maxl.npz", "#e66101"),
        ("C3K ER + afe", d / "median_lsf" / f"{tid}_median_lsf_pred.npz", "#1b7837"),
        ("C3K ER + afe + R matrix, DR1 coadd", d / f"{tid}_poco_maxl.npz", "#762a83"),
        ("C3K ER + afe + R matrix, recoadd", d / "recoadd" / f"{tid}_recoadd_maxl.npz", "#1b9e77"),
    )


def index_bands():
    """Return each index's shaded rest-frame spans, as in 2026-10-29_continuum_recoadd."""
    table = np.genfromtxt(
        Path(os.environ["SPS_HOME"]) / "data" / "allindices.dat",
        dtype=None,
        encoding=None,
        comments="#",
    )
    out = {}
    for r in table:
        name = str(r[-1])
        if name in INDEX_NAMES:
            f0, f1, b0, b1, r0, r1, kind = (r[0], r[1], r[2], r[3], r[4], r[5], int(r[6]))
            out[name] = [(b0, b1), (r0, r1)] if kind == 3 else [(f0, f1)]
    return out


def radec(tid):
    """Return the RA and Dec of ``tid`` in degrees from the VAC METADATA."""
    meta = Table.read(VAC, hdu="METADATA")
    (i,) = np.flatnonzero(np.asarray(meta["TARGETID"], np.int64) == tid)
    return float(meta["RA"][i]), float(meta["DEC"][i])


def cutout(tid):
    """Return the DR10 cutout of ``tid``, fetched once with spender and then read from the cache."""
    path = CUTOUTS / f"{tid}.jpg"
    if not path.exists():
        ra, dec = radec(tid)
        CUTOUTS.mkdir(parents=True, exist_ok=True)
        path.write_bytes(DESI.get_image(ra=ra, dec=dec, size=CUT_PX, pixscale=PIXSCALE,
                                        bands="griz"))  # fmt: skip
    return Image.open(io.BytesIO(path.read_bytes()))


def flam(d, x):
    """Return maggies on the file's wavelength grid in 1e-17 erg/s/cm^2/A, NaN off its fit."""
    f = to_flambda(d["wave"] * u.AA, np.asarray(x, float) * u.mgy).value
    return np.where(d["good"], f, np.nan)


def main(tid):
    """Draw the ladder of ``tid``."""
    st = steps(tid)
    top = st[-1] if st[-1][1].exists() else st[-2]
    rc = np.load(top[1])
    z = float(rc["z"])
    rest = rc["wave"] / (1 + z)
    data = flam(rc, rc["flux"])
    med = np.full(data.size, np.nan)
    med[rc["good"]] = medfilt(data[rc["good"]], N_MEDIAN)
    chis, lo, hi = [], np.inf, -np.inf
    for label, path, _ in st:
        if not path.exists():
            chis.append(None)
            print(f"{label:36s} not run yet")
            continue
        d = np.load(path)
        assert np.allclose(d["wave"], rc["wave"]), f"{path.name} is on another grid"
        assert abs(float(d["z"]) - z) < 1e-6, f"{path.name} has another redshift"
        with np.errstate(divide="ignore", invalid="ignore"):
            chi = np.where(d["good"], (d["flux"] - d["sp_best"]) / d["unc"], np.nan)
        chis.append(chi)
        r = d["wave"][d["good"]] / (1 + z)
        lo, hi = min(lo, r.min()), max(hi, r.max())
        g = np.isfinite(chi)
        print(f"{label:36s} chi2 {np.sum(chi[g] ** 2):8.1f} on {g.sum()} px, "
              f"chi2/N {np.mean(chi[g] ** 2):.3f}")  # fmt: skip

    use_apj_style()
    fig, ax = plt.subplots(1 + len(st), 1, figsize=(14, 4 + 1.6 * len(st)), sharex=True,
                           gridspec_kw={"height_ratios": [3] + [1] * len(st), "hspace": 0.06})  # fmt: skip
    ax[0].plot(rest, data, color="0.6", lw=0.6, label="DESI Spectrum")
    ax[0].plot(rest, med, color="k", lw=0.5, label="Median Filter")
    ax[0].plot(rest, flam(rc, rc["sp_best"]), color=top[2], lw=0.9, label="Model Fit")
    ax[0].set_ylim(bottom=0)
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
    ax[0].legend(frameon=False, fontsize=16, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3)
    for a, (label, _, colour), chi in zip(ax[1:], st, chis, strict=True):
        if chi is None:
            label = f"{label} (not run yet)"
        else:
            a.plot(rest, chi, color=colour, lw=0.5)
        for h in (-2, 2):
            a.axhline(h, color="0.3", ls="--", lw=1)
        a.axhline(0, color="0.3", lw=0.6)
        a.set_ylim(-5, 5)
        a.set_ylabel(r"$\chi$", fontsize=14)
        a.text(0.01, 0.93, label, transform=a.transAxes, ha="left", va="top", fontsize=11,
               bbox=dict(facecolor="white", edgecolor=colour, boxstyle="round,pad=0.25"))  # fmt: skip
    ax[-1].set_xlim(lo, hi)
    for name, spans in index_bands().items():
        for s0, s1 in spans:
            for a in ax:
                a.axvspan(s0, s1, color="0.5", alpha=0.12, lw=0)
        mid = 0.5 * (spans[0][0] + spans[-1][1])
        if lo < mid < hi:
            ax[0].text(mid, 1.01, name, transform=ax[0].get_xaxis_transform(), rotation=90,
                       fontsize=9, ha="center", va="bottom")  # fmt: skip
    ins = ax[0].inset_axes([0.885, 0.64, 0.11, 0.34])
    img = cutout(tid)
    ins.imshow(img, origin="upper")
    c = ((img.width - 1) / 2, (img.height - 1) / 2)
    ins.add_patch(Circle(c, 0.5 * FIBRE_DIAM / PIXSCALE, fill=False, ec="lime", lw=1.0))
    ins.set_axis_off()
    ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
    path = RES / str(tid) / "ladder.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tid", type=int, required=True)
    main(ap.parse_args().tid)
