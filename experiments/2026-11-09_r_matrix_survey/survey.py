"""Effective LSF width of the DR1 coadd resolution matrix for random parent targets.

Changed from the previous experiment: resolution_check.py (2026-10-09) merged one target's matrix
over the arms. This does the same merge for 1,000 random targets of the 104,990 parent and keeps
the effective sigma of every row.

``select`` (on the Mac) draws the targets from results/prescreened_sample_v1.npz with seed 0 and
writes targets.txt (TARGETID, HEALPIX from data/fastspec-iron-sv3-bright.fits). ``run`` (on a
stampede3 compute node) reads each target's B/R/Z_RESOLUTION from the DR1 coadd on /work, merges
the arms with each column weighted by the unmasked ivar of its pixel (desispec coadd_cameras, as
resolution_check.save_matrix), and stores per row the normalised second moment in km/s.

COMMAND: on the Mac, ``uv run python experiments/2026-11-09_r_matrix_survey/survey.py select``.
On stampede3, ``sbatch experiments/2026-11-09_r_matrix_survey/survey.slurm``.

OUTPUT: targets.txt next to this file; results/2026-11-09_r_matrix_survey/survey.npz.
"""

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TARGETS = HERE / "targets.txt"
DATA = Path("/work/11006/nikhilgaruda/ls6/research/spender/DATA")
N_TARGETS = 1000
SEED = 0
PIX_A = 0.8  # DESI merged grid step in Angstrom
C_KMS = 299792.458


def select():
    """Draw the targets and write targets.txt."""
    from astropy.io import fits

    from hubersed.paths import PATHS

    keep = np.load(PATHS["RESULTS"] / "prescreened_sample_v1.npz")["keep"]
    with fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True) as h:
        meta = h["METADATA"].data
        hp = dict(zip(np.asarray(meta["TARGETID"]), np.asarray(meta["HEALPIX"]), strict=True))
    rng = np.random.default_rng(SEED)
    tids = np.sort(rng.choice(keep, size=N_TARGETS, replace=False))
    rows = np.array([(t, hp[t]) for t in tids], dtype=np.int64)
    np.savetxt(TARGETS, rows, fmt="%d", header="TARGETID HEALPIX")
    print(f"wrote {TARGETS}: {len(rows)} targets in {len(np.unique(rows[:, 1]))} healpix")


def merged_matrix(h, idx, wave):
    """Return the banded matrix of one target merged over the arms.

    Each column is weighted by the unmasked ivar of its pixel, as in
    resolution_check.save_matrix of 2026-10-09_nautilus_c3k_er_broadline.
    """
    res = {b: np.asarray(h[f"{b.upper()}_RESOLUTION"].data[idx], float) for b in "brz"}
    ndiag = max(r.shape[0] for r in res.values())
    rdata = np.zeros((ndiag, wave.size))
    ivar_un = np.zeros(wave.size)
    for b in "brz":
        bw = np.asarray(h[f"{b.upper()}_WAVELENGTH"].data, float)
        iv = np.asarray(h[f"{b.upper()}_IVAR"].data[idx], float)
        imin = np.argmin(abs(bw[0] - wave))
        w = np.arange(imin, imin + bw.size)
        assert np.allclose(bw, wave[w], atol=1e-4)
        nd = res[b].shape[0]
        ivar_un[w] += iv
        rdata[(ndiag - nd) // 2 : (ndiag + nd) // 2, w] += iv * res[b]
    ok = ivar_un > 0
    rdata[:, ok] /= ivar_un[ok]
    return rdata


def row_sigma(rdata):
    """Return each row's sum and normalised second moment in pixels.

    Row i holds R[i, i + o] = data[k, i + o] for offset o = ndiag//2 - k (desispec storage).
    """
    ndiag, n = rdata.shape
    offsets = ndiag // 2 - np.arange(ndiag)
    rows = np.zeros((n, ndiag))
    for k, o in enumerate(offsets):
        j = np.arange(n) + o
        ok = (j >= 0) & (j < n)
        rows[ok, k] = rdata[k, j[ok]]
    s = rows.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = (rows * offsets).sum(1) / s
        var = (rows * (offsets - mu[:, None]) ** 2).sum(1) / s
    return s, np.sqrt(np.clip(var, 0, None))


def run():
    """Read the matrices and write survey.npz."""
    from astropy.io import fits

    from hubersed.paths import PATHS
    from hubersed.sps.lsf import DESI_WAV

    out_dir = PATHS["RESULTS"] / "2026-11-09_r_matrix_survey"
    out_dir.mkdir(parents=True, exist_ok=True)
    tids, hps = np.loadtxt(TARGETS, dtype=np.int64, unpack=True)
    wave = np.asarray(DESI_WAV, float)
    sig = np.full((tids.size, wave.size), np.nan, np.float32)
    rsum = np.full((tids.size, wave.size), np.nan, np.float32)
    fiber = np.full(tids.size, -1)
    found = np.zeros(tids.size, bool)
    for hp in np.unique(hps):
        path = DATA / str(hp) / f"coadd-sv3-bright-{hp}.fits"
        with fits.open(path, memmap=True) as h:
            ftid = np.asarray(h["FIBERMAP"].data["TARGETID"])
            # the healpix coadd FIBERMAP has no FIBER; take the first exposure's if stored
            ef = h["EXP_FIBERMAP"].data if "EXP_FIBERMAP" in h else None
            has_fiber = ef is not None and "FIBER" in ef.columns.names
            for i in np.where(hps == hp)[0]:
                idx = np.where(ftid == tids[i])[0]
                if idx.size == 0:
                    print(f"missing {tids[i]} in {path}")
                    continue
                s, sp = row_sigma(merged_matrix(h, idx[0], wave))
                sig[i] = sp * PIX_A / wave * C_KMS
                rsum[i] = s
                found[i] = True
                if has_fiber:
                    e = np.where(np.asarray(ef["TARGETID"]) == tids[i])[0]
                    fiber[i] = int(ef["FIBER"][e[0]]) if e.size else -1
    print(f"found {found.sum()} of {tids.size}")
    np.savez(
        out_dir / "survey.npz",
        targetid=tids,
        healpix=hps,
        fiber=fiber,
        found=found,
        wave=wave,
        sigma_kms=sig,
        row_sum=rsum,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("action", choices=["select", "run"])
    {"select": select, "run": run}[p.parse_args().action]()
