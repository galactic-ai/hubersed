"""Extract sky-line windows of all fibers in the b and r frames of the target exposures.

Divides each iron frame by its fiberflatexp as desispec 0.56.5 apply_fiberflat does, finds the peaks of
the camera mean sky model by the desispec rule, and keeps windows around them for every fiber. Run
through extract.slurm, with the 2026-10-09 folder on PYTHONPATH.
OUTPUT: results/2026-11-16_exposure_wavelength_checks/sky_<EXPID>_<CAMERA>.npz.
"""

import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from ivar_recompute import DATA
from targets import RESULTS, TARGETS

# the 2026-11-16_exposure_line_centroids targets with repeat exposures (34 has one)
TIDS = (39627818010675704, 39633318999755849, 39627806493117457, 39633127609469239,
        39627770174637084)  # fmt: skip
ROOT = Path(os.environ.get("RESULTS_ROOT", RESULTS))
OUT = ROOT / "2026-11-16_exposure_wavelength_checks"
KEYS = ("NIGHT", "EXPID", "TILEID", "MJD-OBS", "EXPTIME", "HELIOCOR", "AIRMASS", "SEEING")
HALF = 10  # pixels kept on each side of a peak; the fits use 4
BADFIBERFLAT = 2**4  # desispec maskbits.py


def flat_fielded(fr, ff):
    """Return flux, ivar and mask of a frame divided by its fiberflat, as desispec apply_fiberflat."""
    wave = fr["WAVELENGTH"].data.astype(float)
    flux, ivar = fr["FLUX"].data.astype(float), fr["IVAR"].data.astype(float)
    mask = fr["MASK"].data.astype(np.int64)
    flat, fivar = ff[0].data.astype(float), ff["IVAR"].data.astype(float)
    fwave = ff["WAVELENGTH"].data.astype(float)
    assert np.allclose(wave, fwave), "frame and fiberflat grids differ"
    if "HELIOCOR" in fr[0].header:
        # the flat is moved to the frame grid, which is labelled in the barycentric frame
        for i in range(flat.shape[0]):
            good = fivar[i] > 0
            if good.any():
                flat[i] = np.interp(wave, fwave[good] * fr[0].header["HELIOCOR"], flat[i][good])
    num = (ivar > 0) * (fivar > 0) * (flat > 0)
    var = 1 / ((ivar + (ivar == 0)) * (flat**2 + (flat == 0)))
    var = var + flux**2 / (fivar * flat**4 + (fivar * flat == 0))
    ivar = num / var
    ok = flat > 0
    flux[ok] = flux[ok] / flat[ok]
    mask[(flat == 0) | (fivar == 0) | (ff["MASK"].data != 0)] |= BADFIBERFLAT
    return wave, flux, ivar, mask


def peaks_of(meansky):
    """Return the sky peaks: local maxima above 10% of the largest, 10 pixels from the edges."""
    m = meansky
    tmp = (m[1:-1] > m[2:]) * (m[1:-1] > m[:-2]) * (m[1:-1] > 0.1 * np.max(m))
    p = np.where(tmp)[0] + 1
    return p[(p > 10) & (p < m.size - 10)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for tid in TIDS:
        for expid, (night, petal, _) in TARGETS[tid]["exposures"].items():
            jobs.setdefault((night, expid, petal), []).append(tid)
    for (night, expid, petal), tids in sorted(jobs.items(), key=lambda j: j[0][1]):
        d = DATA / night / f"{expid:08d}"
        for arm in "br":
            cam = f"{arm}{petal}"
            name = f"{cam}-{expid:08d}.fits.gz"
            with (
                fits.open(d / f"frame-{name}") as fr,
                fits.open(d / f"fiberflatexp-{name}") as ff,
                fits.open(d / f"sky-{name}") as sk,
            ):
                wave, flux, ivar, mask = flat_fielded(fr, ff)
                sky = sk[0].data.astype(float)
                assert np.allclose(sk["WAVELENGTH"].data, wave), "sky grid differs"
                fm = fr["FIBERMAP"].data
                head = fr[0].header
                meta = [float(head.get(k, np.nan)) for k in KEYS]
                hdus = [h.name for h in sk]
                fiber, objtype = np.array(fm["FIBER"]), np.array(fm["OBJTYPE"]).astype(str)
                status, targetid = np.array(fm["FIBERSTATUS"]), np.array(fm["TARGETID"])
            meansky = sky.mean(0)
            peaks = peaks_of(meansky)
            cols = peaks[:, None] + np.arange(-HALF, HALF + 1)
            rows = [int(np.where(targetid == t)[0][0]) for t in tids]
            np.savez(
                OUT / f"sky_{expid}_{cam}.npz",
                tids=np.array(tids), target_rows=np.array(rows), wave=wave, meansky=meansky,
                peaks=peaks, win_wave=wave[cols], win_flux=flux[:, cols].astype("f4"),
                win_ivar=ivar[:, cols].astype("f4"), win_mask=mask[:, cols].astype(np.int32),
                win_sky=sky[:, cols].astype("f4"), fiber=fiber, objtype=objtype,
                fiberstatus=status, targetid=targetid, meta=np.array(meta),
                meta_keys=np.array(KEYS), sky_hdus=np.array(hdus),
            )  # fmt: skip
            print(f"{expid} {cam}: {peaks.size} peaks at {np.round(wave[peaks], 2)}", flush=True)


if __name__ == "__main__":
    main()
