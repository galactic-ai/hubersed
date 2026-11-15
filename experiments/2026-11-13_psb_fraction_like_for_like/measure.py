"""Measure the same spectral indices on the DESI parent, the noised v2 mocks and the rebuilt mocks.

For analysis.py, which compares post-starburst fractions in matched redshift bins.
INPUTS: results/prescreened_sample_v1.npz (keep, the parent) through load_by_index (DR1 SV3 bright
coadds, f_lambda); data/fastspec-iron-sv3-bright.fits (LOGMSTAR); data/prospector_model/noised_cuev2/
DESIcuev2_*.pkl (the noised v2 mocks the flows trained on: f_lambda, ivar, z, mock index; written by
scripts/make_prospector_noisy_sed.py); data/prospector_model/mocks_v2_500000.h5 (logmass);
optionally the rebuilt noiseless subset of 2026-11-13_mock_psb_coverage (rebuilt_part*.h5, maggies).
METHOD: every index on the raw spectrum (lines included), good pixels only (ivar > 0), NaN when a
band has under 25% good pixels:
- HdA: Worthey & Ottaviani (1997) bands taken as air and moved to vacuum, (l2 - l1)(1 - <F/F_c>).
- Dn4000: mean f_nu in 4000-4100 over 3850-3950 A (Balogh et al. 1999, PSG.tex l.326).
- EW(Halpha): DAP bands (Westfall et al. 2019, emissionlines.tex l.61), main 6557.6-6571.6, sidebands
  6483.0-6513.0 and 6623.0-6653.0 A; EW([OII]): Yan et al. (2006, ms.tex l.1392) 3716.3-3738.3,
  3696.3-3716.3 and 3738.3-3758.3 A. Pseudo-continuum: line between the sideband medians; EW = band
  width times <(F - C) / C>, rest frame, emission positive. Band media unstated; taken as vacuum.
COMMAND: ``uv run --no-sync python experiments/2026-11-13_psb_fraction_like_for_like/measure.py
[--rebuilt <rebuilt.h5 or quoted glob of rebuilt_part*.h5>]``.
OUTPUT: results/2026-11-13_psb_fraction_like_for_like/indices_{desi,noised,rebuilt}.npz (z, logmass,
hda, dn4000, ew_ha, ew_oii) and snr_noised.npz (median per-pixel S/N of the noised mocks).
"""

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
from spender.data.desi import DESI

from hubersed.io.desi import load_by_index, tids_to_indices
from hubersed.paths import PATHS
from hubersed.sps.utils import _airtovac

OUT = PATHS["RESULTS"] / "2026-11-13_psb_fraction_like_for_like"
NOISED = PATHS["DATA"] / "prospector_model" / "noised_cuev2"
MOCKS = PATHS["DATA"] / "prospector_model" / "mocks_v2_500000.h5"
HDA = tuple((_airtovac(lo), _airtovac(hi)) for lo, hi in
            ((4083.50, 4122.25), (4041.60, 4079.75), (4128.50, 4161.00)))  # fmt: skip
DN = ((3850.0, 3950.0), (4000.0, 4100.0))
HA = ((6557.6, 6571.6), (6483.0, 6513.0), (6623.0, 6653.0))
OII = ((3716.3, 3738.3), (3696.3, 3716.3), (3738.3, 3758.3))
MIN_GOOD = 0.25


def _sel(rest, good, band):
    """Return the good pixels of a band, or None when under MIN_GOOD of them are good."""
    s = (rest >= band[0]) & (rest <= band[1])
    if not s.any() or good[s].mean() < MIN_GOOD:
        return None
    return s & good


def hda(rest, flam, good):
    """Return HdA (A, absorption positive)."""
    sel = [_sel(rest, good, b) for b in HDA]
    if any(s is None for s in sel):
        return np.nan
    (f0, f1), (b0, b1), (r0, r1) = HDA
    cb, cr = flam[sel[1]].mean(), flam[sel[2]].mean()
    cont = cb + (cr - cb) * (rest[sel[0]] - (b0 + b1) / 2) / ((r0 + r1) / 2 - (b0 + b1) / 2)
    return float((f1 - f0) * (1 - np.mean(flam[sel[0]] / cont)))


def dn4000(rest, flam, good):
    """Return Dn4000 from mean f_nu."""
    fnu = flam * rest**2
    b, r = (_sel(rest, good, x) for x in DN)
    return np.nan if b is None or r is None else float(fnu[r].mean() / fnu[b].mean())


def ew(rest, flam, good, bands):
    """Return the rest-frame EW (A, emission positive) over a sideband pseudo-continuum."""
    sel = [_sel(rest, good, b) for b in bands]
    if any(s is None for s in sel):
        return np.nan
    (m0, m1), (b0, b1), (r0, r1) = bands
    cb, cr = np.median(flam[sel[1]]), np.median(flam[sel[2]])
    xb, xr = (b0 + b1) / 2, (r0 + r1) / 2
    cont = cb + (cr - cb) * (rest[sel[0]] - xb) / (xr - xb)
    return float((m1 - m0) * np.mean((flam[sel[0]] - cont) / cont))


def measure(rest, flam, good):
    """Return (hda, dn4000, ew_ha, ew_oii) of one spectrum."""
    return (
        hda(rest, flam, good),
        dn4000(rest, flam, good),
        ew(rest, flam, good, HA),
        ew(rest, flam, good, OII),
    )


def desi(wave):
    """Measure the DESI parent."""
    keep = np.load(PATHS["RESULTS"] / "prescreened_sample_v1.npz")["keep"]
    with fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits") as h:
        tid, lm = (
            np.asarray(h["METADATA"].data["TARGETID"]),
            np.asarray(h["FASTSPEC"].data["LOGMSTAR"]),
        )
    mass = dict(zip(tid.tolist(), lm.tolist(), strict=True))
    rows = []
    for t, g in zip(keep, tids_to_indices(keep), strict=True):
        f, iv, z, _ = load_by_index(int(g))
        f, iv, z = np.asarray(f, float), np.asarray(iv, float), float(z)
        rows.append((z, mass.get(int(t), np.nan), *measure(wave / (1 + z), f, iv > 0)))
    return np.array(rows), np.asarray(keep)


def noised(wave):
    """Measure every noised v2 mock."""
    with h5py.File(MOCKS, "r") as h:
        labels = [s.decode() if isinstance(s, bytes) else str(s) for s in h.attrs["theta_labels"]]
        logmass = h["priors/theta"][:, labels.index("logmass")]
    rows, ids = [], []
    w = wave[:-1]  # the noised mocks drop the last pixel (make_prospector_noisy_sed.py)
    for p in sorted(NOISED.glob("DESIcuev2_*.pkl"), key=lambda q: int(q.stem.split("_")[1])):
        with open(p, "rb") as fh:
            f, iv, z, idx, _, _ = pickle.load(fh)
        f, iv, z, idx = (np.asarray(x) for x in (f, iv, z, idx))
        for k in range(z.size):
            rows.append(
                (
                    z[k],
                    logmass[int(idx[k])],
                    *measure(w / (1 + z[k]), f[k].astype(float), iv[k] > 0),
                )
            )
            ids.append(int(idx[k]))
    return np.array(rows), np.array(ids)


def rebuilt(pattern, wave):
    """Measure the noiseless rebuilt mocks (maggies converted to f_lambda) of one file or a glob."""
    with h5py.File(MOCKS, "r") as h:
        labels = [s.decode() if isinstance(s, bytes) else str(s) for s in h.attrs["theta_labels"]]
        logmass = h["priors/theta"][:, labels.index("logmass")]
    paths = sorted(Path(pattern).parent.glob(Path(pattern).name))
    rows, ids = [], []
    for path in paths:
        with h5py.File(path, "r") as h:
            idx, z, ok, flux = h["index"][:], h["redshift"][:], h["ok"][:], h["flux"][:]
        for k in np.flatnonzero(ok):
            flam = flux[k].astype(float) / wave**2
            rows.append(
                (z[k], logmass[idx[k]], *measure(wave / (1 + z[k]), flam, np.isfinite(flam)))
            )
        ids.append(idx[ok])
    print(f"rebuilt: {len(paths)} files", flush=True)
    return np.array(rows), np.concatenate(ids)


def snr_noised():
    """Return mock index and median per-pixel S/N (flux sqrt(ivar), ivar > 0) of every noised mock.

    The definition of the latent S/N cut (scripts/get_latent_space.py); the mock normalisation
    cancels in flux sqrt(ivar).
    """
    ids, snr = [], []
    for p in sorted(NOISED.glob("DESIcuev2_*.pkl"), key=lambda q: int(q.stem.split("_")[1])):
        with open(p, "rb") as fh:
            f, iv, _, idx, _, _ = pickle.load(fh)
        f, iv = np.asarray(f, float), np.asarray(iv, float)
        s = np.where(iv > 0, f * np.sqrt(np.clip(iv, 0, None)), np.nan)
        snr.append(np.nanmedian(s, axis=1))
        ids.append(np.asarray(idx, int))
    return np.concatenate(ids), np.concatenate(snr)


def save(name, rows, ids):
    """Write one sample's table."""
    z, lm, h, d, eha, eoii = rows.T
    np.savez(
        OUT / f"indices_{name}.npz",
        id=ids,
        z=z,
        logmass=lm,
        hda=h,
        dn4000=d,
        ew_ha=eha,
        ew_oii=eoii,
    )
    print(f"{name}: {rows.shape[0]} spectra, HdA finite {np.isfinite(h).mean():.3f}", flush=True)


def main():
    """Measure each sample."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rebuilt", default=None, help="rebuilt h5 file, or a quoted glob of parts")
    p.add_argument("--only", choices=["desi", "noised", "rebuilt", "snr"], default=None)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    wave = DESI._wave_obs.numpy()
    if a.only in (None, "desi"):
        save("desi", *desi(wave))
    if a.only in (None, "noised"):
        save("noised", *noised(wave))
    if a.only in (None, "snr"):
        ids, snr = snr_noised()
        np.savez(OUT / "snr_noised.npz", id=ids, snr=snr)
        print(f"snr: {ids.size} noised mocks, median {np.nanmedian(snr):.1f}", flush=True)
    if a.rebuilt is not None and a.only in (None, "rebuilt"):
        save("rebuilt", *rebuilt(a.rebuilt, wave))


if __name__ == "__main__":
    main()
