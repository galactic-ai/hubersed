"""Measure the same emission-line EWs on the 58 emission_line outliers and on every v2 mock.

Raw-spectrum rest-frame EWs in Westfall et al. (2019) bands, for analysis.py.
COMMAND: ``uv run --no-sync python experiments/2026-11-14_emline_outlier_coverage/lines.py``.
OUTPUT: results/2026-11-14_emline_outlier_coverage/{ew_desi.npz, ew_mocks.npz}.
"""

import h5py
import numpy as np
import pandas as pd
from spender.data.desi import DESI

from hubersed.io.desi import load_by_index, tids_to_indices
from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-14_emline_outlier_coverage"
MOCKS = PATHS["DATA"] / "prospector_model" / "mocks_v2_500000.h5"
BANDS = {
    "oii": ((3716.3, 3738.3), (3696.3, 3716.3), (3738.3, 3758.3)),
    "hb": ((4852.7, 4872.7), (4798.9, 4838.9), (4885.6, 4925.6)),
    "oiii": ((4998.2, 5018.2), (4978.2, 4998.2), (5018.2, 5038.2)),
    "ha": ((6557.6, 6571.6), (6483.0, 6513.0), (6623.0, 6653.0)),
    "nii": ((6575.3, 6595.3), (6483.0, 6513.0), (6623.0, 6653.0)),
    "sii": ((6711.3, 6739.7), (6673.0, 6703.0), (6748.0, 6778.0)),
}
MIN_GOOD = 0.25
CHUNK = 5000


def ew_batch(rest, flam, good, bands, err=None):
    """Return the EWs (and errors if ``err`` is given) for one line of (n, npix) spectra."""
    (m0, m1), (b0, b1), (r0, r1) = bands
    inb = [(rest >= lo) & (rest <= hi) for lo, hi in bands]
    ok = np.ones(rest.shape[0], bool)
    for s in inb:
        n_all = s.sum(1)
        ok &= (n_all > 0) & ((s & good).sum(1) >= MIN_GOOD * np.maximum(n_all, 1))
    side = []
    for s in inb[1:]:
        x = np.where(s & good, flam, np.nan)
        side.append(np.nanmedian(x, axis=1) if x.size else np.full(rest.shape[0], np.nan))
    cb, cr = side
    xb, xr = (b0 + b1) / 2, (r0 + r1) / 2
    cont = cb[:, None] + (cr - cb)[:, None] * (rest - xb) / (xr - xb)
    m = inb[0] & good
    nm = m.sum(1)
    with np.errstate(all="ignore"):
        ew = (m1 - m0) * np.where(m, (flam - cont) / cont, 0).sum(1) / nm
        ew = np.where(ok & (nm > 0), ew, np.nan)
        if err is None:
            return ew
        cmain = np.where(m, cont, np.nan)
        cbar = np.nanmean(cmain, axis=1)
        var_line = np.where(m, err**2, 0).sum(1) / nm**2 / cbar**2
        var_side = []
        for s in inb[1:]:
            ns = (s & good).sum(1)
            sig = np.sqrt(np.nanmean(np.where(s & good, err**2, np.nan), axis=1))
            var_side.append((1.253 * sig) ** 2 / np.maximum(ns, 1))
        flux_ratio = np.nanmean(np.where(m, flam, np.nan), axis=1) / cbar
        var_cont = flux_ratio**2 * 0.25 * (var_side[0] + var_side[1]) / cbar**2
        sig_ew = (m1 - m0) * np.sqrt(var_line + var_cont)
        return ew, np.where(np.isfinite(ew), sig_ew, np.nan)


def desi_targets():
    """Return the target ids and redshifts of the 58 emission_line outliers."""
    rev = pd.read_csv(
        PATHS["ROOT"] / "experiments" / "2026-11-13_gallery105" / "review.tsv", sep="\t"
    )
    tab = pd.read_csv(PATHS["RESULTS"] / "2026-11-13_gallery105" / "gallery105.tsv", sep="\t")
    g = rev.merge(tab[["nn", "target_id", "z", "branch"]], on="nn")
    g = g[g.review_class == "emission_line"]
    return g.target_id.to_numpy(np.int64), g.branch.to_numpy(str), g.nn.to_numpy(int)


def main():
    """Measure the DESI outliers and the mocks."""
    OUT.mkdir(parents=True, exist_ok=True)
    wave = DESI._wave_obs.numpy().astype(float)
    tids, branch, nn = desi_targets()
    fl, iv, zz = [], [], []
    for g in tids_to_indices(tids):
        f, i, z, _ = load_by_index(int(g))
        fl.append(np.asarray(f, float))
        iv.append(np.asarray(i, float))
        zz.append(float(z))
    fl, iv, zz = np.array(fl), np.array(iv), np.array(zz)
    rest = wave[None, :] / (1 + zz[:, None])
    err = np.where(iv > 0, 1 / np.sqrt(np.clip(iv, 1e-30, None)), np.nan)
    out = {"id": tids, "nn": nn, "branch": branch, "z": zz}
    for k, b in BANDS.items():
        out[k], out[k + "_err"] = ew_batch(rest, fl, iv > 0, b, err)
    np.savez(OUT / "ew_desi.npz", **out)
    print(f"DESI: {tids.size} targets measured", flush=True)

    with h5py.File(MOCKS, "r") as h:
        n = h["fluxes"].shape[0]
        zs, failed = h["priors/redshifts"][:], h["failed"][:]
        res = {k: np.full(n, np.nan) for k in BANDS}
        for a in range(0, n, CHUNK):
            b = min(a + CHUNK, n)
            flam = h["fluxes"][a:b].astype(float) / wave[None, :] ** 2
            restm = wave[None, :] / (1 + zs[a:b, None])
            good = np.isfinite(flam) & (flam > 0)
            for k, bd in BANDS.items():
                res[k][a:b] = ew_batch(restm, flam, good, bd)
            if (a // CHUNK) % 10 == 0:
                print(f"mocks: {b} of {n}", flush=True)
    np.savez(OUT / "ew_mocks.npz", z=zs, failed=failed, **res)
    print("mocks done", flush=True)


if __name__ == "__main__":
    main()
