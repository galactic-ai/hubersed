"""Rerun the iron flux calibration on a sample of exposure-petals and keep a per-fiber summary.

QUESTION: how is the corrected calibration error f = sigma_C / C distributed over the exposures of
our SV3 bright sample, and does it depend on the fiber aperture correction FLAT_TO_PSF_FLUX?

For one target (experiments/2026-10-09_nautilus_c3k_er_broadline/fluxcal_rerun.py) the corrected
calibrated variance is the iron variance plus (f F')^2, less the small iron calibration term,
with f = sigma_C / C from the flux calibration rerun with desispec PRs 2496, 2827 and 2452. With PR 2452 the calibration of each fiber
and its ivar are both scaled by FLAT_TO_PSF_FLUX, so f should not depend on it. In iron it did,
by a factor FLAT_TO_PSF_FLUX. This script measures f for every fiber of each listed exposure-petal.

For each unit (night, exposure, petal) it downloads the iron frame, sky, fiberflatexp and fluxcalib
of the three cameras, the stdstars file and calibstars csv, and checks each against the iron
sha256sum file. Downloads go to the public AWS S3 copy of DR1 (bucket desidata), and optionally
also to data.desi.lbl.gov (--nersc-streams, it returns 503 above about 6), with retries. A camera
whose files are not in the iron exposure is skipped. Units whose petal is flagged BADPETALSTDSTAR in
the exposure QA (column N_BADPETALSTDSTAR of units_all.csv from make_units.py) are recorded and
skipped: desispec then uses a default calibration with 100% uncertainty, which needs
DESI_SPECTRO_CALIB. Each unit writes its own tables under units/, so a rerun skips finished units. It then runs desi_compute_fluxcalibration of desispec 0.56.2
with iron's arguments, unchanged and patched, through fluxcal_rerun.py --child of the 10-09
folder. It checks that the unchanged run reproduces the iron fluxcalib and that the patched one
leaves the calibration vector unchanged, and writes one row per camera and fiber.

Downloaded inputs are kept unless --discard is given, which removes the unit's own working folder
under the scratch directory after its summary is written.

This runs with the separate desispec 0.56.2 environment (desi_env_0562, see fluxcal_rerun.py), not
the hubersed one.

INPUTS: a unit list csv with NIGHT, EXPID, TILEID, PETAL (units_all.csv from make_units.py, or
pilot_units.csv), and data/all_target_ids.npy. SEED: none.

COMMAND: on an ls6 compute node, see fluxcal_sample.slurm.

OUTPUT: fibers.fits, cameras.fits, units/<expid>-p<petal>/ and run.log in
results/2026-10-13_fluxcal_sample/<name>/.
"""

import argparse
import concurrent.futures as cf
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

REPO = Path(__file__).resolve().parents[2]
RERUN = REPO / "experiments" / "2026-10-09_nautilus_c3k_er_broadline" / "fluxcal_rerun.py"
NERSC = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/exposures"
S3 = "https://desidata.s3.amazonaws.com/dr1/spectro/redux/iron/exposures"
CAMERAS = ("b", "r", "z")
NBIN = 10  # wavelength bins per camera for the binned f
LOG_LOCK = threading.Lock()


def log(path, msg):
    """Append a time-stamped line to the run log and print it."""
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with LOG_LOCK:
        print(line, flush=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")


def unit_files(expid, petal):
    """Return the iron file names one unit needs."""
    e = f"{expid:08d}"
    names = [f"stdstars-{petal}-{e}.fits.gz", f"calibstars-{e}.csv"]
    for c in CAMERAS:
        for kind in ("frame", "sky", "fiberflatexp", "fluxcalib"):
            names.append(f"{kind}-{c}{petal}-{e}.fits.gz")
    return names


class Fetcher:
    """Download files from NERSC and S3 with per-host stream limits, retries and sha256 checks."""

    def __init__(self, nersc_streams, s3_streams):
        self.sem = {
            "nersc": threading.Semaphore(max(nersc_streams, 1)),
            "s3": threading.Semaphore(max(s3_streams, 1)),
        }
        self.use = [h for h, n in (("s3", s3_streams), ("nersc", nersc_streams)) if n > 0]
        self.base = {"nersc": NERSC, "s3": S3}
        self.count = 0
        self.lock = threading.Lock()

    def _get(self, host, rel, dest):
        with self.sem[host]:
            t0 = time.time()
            with (
                urllib.request.urlopen(f"{self.base[host]}/{rel}", timeout=120) as r,
                open(dest, "wb") as fh,
            ):
                shutil.copyfileobj(r, fh, length=1 << 20)
            return time.time() - t0

    def get(self, rel, dest, sha=None, tries=6):
        """Download one file, trying hosts in turn; return host, seconds and bytes."""
        with self.lock:
            self.count += 1
            k0 = self.count % 5 == 0 and len(self.use) > 1
        hosts = self.use[::-1] if k0 else self.use
        err = None
        for k in range(tries):
            host = hosts[k % len(hosts)]
            try:
                dt = self._get(host, rel, dest)
                if sha is not None:
                    h = hashlib.sha256()
                    with open(dest, "rb") as fh:
                        for chunk in iter(lambda: fh.read(1 << 22), b""):
                            h.update(chunk)
                    if h.hexdigest() != sha:
                        raise ValueError(f"sha256 mismatch for {rel}")
                return host, dt, dest.stat().st_size
            except (urllib.error.URLError, OSError, ValueError) as e:
                err = e
                time.sleep(min(2**k, 30))
        raise RuntimeError(f"failed {rel}: {err}")


def sha_table(fetcher, night, expid, workdir):
    """Return the iron sha256 of every file of an exposure."""
    rel = f"{night}/{expid:08d}/redux_iron_exposures_{night}_{expid:08d}.sha256sum"
    dest = workdir / Path(rel).name
    fetcher.get(rel, dest)
    out = {}
    for line in dest.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[Path(parts[1]).name] = parts[0]
    return out


def run_calibration(workdir, expid, petal, logpath, cameras=CAMERAS):
    """Run the unchanged and patched calibration of the three cameras; return time and peak memory."""
    e = f"{expid:08d}"
    procs = []
    t0 = time.time()
    for mode in ("unpatched", "patched"):
        (workdir / mode).mkdir(exist_ok=True)
        for c in cameras:
            cam = f"{c}{petal}"
            args = [
                "--infile", str(workdir / f"frame-{cam}-{e}.fits.gz"),
                "--sky", str(workdir / f"sky-{cam}-{e}.fits.gz"),
                "--fiberflat", str(workdir / f"fiberflatexp-{cam}-{e}.fits.gz"),
                "--models", str(workdir / f"stdstars-{petal}-{e}.fits.gz"),
                "--selected-calibration-stars", str(workdir / f"calibstars-{e}.csv"),
                "--outfile", str(workdir / mode / f"fluxcalib-{cam}-{e}.fits.gz"),
            ]  # fmt: skip
            timer = (
                ["/usr/bin/time", "-f", "MAXRSS_KB %M"] if Path("/usr/bin/time").exists() else []
            )
            cmd = [*timer, sys.executable, str(RERUN), "--child", mode, *args]
            fh = open(workdir / mode / f"log-{cam}.txt", "w")
            procs.append(
                (mode, cam, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh)
            )
    rss, failed = [], []
    for mode, cam, p, fh in procs:
        p.wait()
        fh.close()
        text = (workdir / mode / f"log-{cam}.txt").read_text()
        for line in text.splitlines():
            if line.startswith("MAXRSS_KB"):
                rss.append(int(line.split()[1]))
        if p.returncode != 0:
            failed.append(f"{mode} {cam}")
    if failed:
        log(logpath, f"{expid} p{petal}: calibration failed for {failed}")
    return time.time() - t0, max(rss) / 1024 if rss else np.nan, failed


def summarise(workdir, night, expid, tile, petal, sample):
    """Return the per-fiber and per-camera rows of one unit."""
    e = f"{expid:08d}"
    fib_rows, cam_rows = [], []
    for c in CAMERAS:
        cam = f"{c}{petal}"
        name = f"fluxcalib-{cam}-{e}.fits.gz"
        iron, un, pa = workdir / name, workdir / "unpatched" / name, workdir / "patched" / name
        if not (un.exists() and pa.exists()):
            continue
        with (
            fits.open(iron) as hi,
            fits.open(un) as hu,
            fits.open(pa) as hp,
            fits.open(workdir / f"frame-{cam}-{e}.fits.gz") as hf,
        ):
            C, ivi = hi["FLUXCALIB"].data.astype(float), hi["IVAR"].data.astype(float)
            ivp = hp["IVAR"].data.astype(float)
            d_un = max(
                float(np.max(np.abs(hu[k].data.astype(float) - hi[k].data.astype(float))))
                for k in ("FLUXCALIB", "IVAR")
            )
            d_c = float(np.max(np.abs(hp["FLUXCALIB"].data.astype(float) - C)))
            fc = hi["FIBERCORR"].data
            fm = hf["FIBERMAP"].data
            wave = hi["WAVELENGTH"].data.astype(float)
            nstd = len(hi["STDSTAR_FIBERMAP"].data) if "STDSTAR_FIBERMAP" in hi else -1
        cam_rows.append(
            dict(NIGHT=night, EXPID=expid, TILEID=tile, PETAL=petal, CAMERA=c, NSTD_USED=nstd,
                 UNPATCHED_VS_IRON_MAXABS=d_un, PATCHED_C_VS_IRON_MAXABS=d_c)
        )  # fmt: skip
        edges = np.linspace(wave[0], wave[-1], NBIN + 1)
        for i in range(C.shape[0]):
            ok_i = (C[i] > 0) & (ivi[i] > 0)
            ok_p = (C[i] > 0) & (ivp[i] > 0)
            fi = 1 / (C[i][ok_i] * np.sqrt(ivi[i][ok_i])) if ok_i.any() else np.array([np.nan])
            fp = np.full(C.shape[1], np.nan)
            fp[ok_p] = 1 / (C[i][ok_p] * np.sqrt(ivp[i][ok_p]))
            fbin = [
                np.nanmedian(fp[(wave >= a) & (wave < b)])
                if np.isfinite(fp[(wave >= a) & (wave < b)]).any()
                else np.nan
                for a, b in zip(edges[:-1], edges[1:], strict=True)
            ]
            fpv = fp[np.isfinite(fp)] if np.isfinite(fp).any() else np.array([np.nan])
            tid = int(fm["TARGETID"][i])
            fib_rows.append(
                dict(
                    NIGHT=night, EXPID=expid, TILEID=tile, PETAL=petal, CAMERA=c,
                    FIBER=int(fm["FIBER"][i]), TARGETID=tid,
                    FIBERSTATUS=int(fm["FIBERSTATUS"][i]) if "FIBERSTATUS" in fm.names else -1,
                    OBJTYPE=str(fm["OBJTYPE"][i]) if "OBJTYPE" in fm.names else "",
                    IN_SAMPLE=bool(tid in sample),
                    FIBER_X=float(fm["FIBER_X"][i]) if "FIBER_X" in fm.names else np.nan,
                    FIBER_Y=float(fm["FIBER_Y"][i]) if "FIBER_Y" in fm.names else np.nan,
                    DELTA_X=float(fm["DELTA_X"][i]) if "DELTA_X" in fm.names else np.nan,
                    DELTA_Y=float(fm["DELTA_Y"][i]) if "DELTA_Y" in fm.names else np.nan,
                    FLAT_TO_PSF_FLUX=float(fc["FLAT_TO_PSF_FLUX"][i]),
                    C_MED=float(np.median(C[i][C[i] > 0])) if (C[i] > 0).any() else np.nan,
                    F_IRON_P05=float(np.nanpercentile(fi, 5)), F_IRON_P50=float(np.nanmedian(fi)),
                    F_IRON_P95=float(np.nanpercentile(fi, 95)),
                    F_PATCH_P05=float(np.nanpercentile(fpv, 5)), F_PATCH_P50=float(np.nanmedian(fpv)),
                    F_PATCH_P95=float(np.nanpercentile(fpv, 95)),
                    F_PATCH_BINS=np.array(fbin, dtype=float),
                    NPIX_IVARC0=int((ivp[i] <= 0).sum()),
                )
            )  # fmt: skip
    return fib_rows, cam_rows


def unit_rows_empty(night, expid, tile, petal, reason):
    """Return camera rows for a unit that is not recalibrated, with the reason."""
    return [
        dict(NIGHT=night, EXPID=expid, TILEID=tile, PETAL=petal, CAMERA=c, NSTD_USED=-1,
             UNPATCHED_VS_IRON_MAXABS=np.nan, PATCHED_C_VS_IRON_MAXABS=np.nan, SKIPPED=reason,
             DOWNLOAD_S=np.nan, DOWNLOAD_MB=np.nan, CALIB_S=np.nan, MAXRSS_MB=np.nan, FAILED="",
             N_NERSC=0, N_S3=0)
        for c in CAMERAS
    ]  # fmt: skip


def process(unit, fetcher, scratch, out, sample, logpath, discard):
    """Download, calibrate and summarise one unit, or load it if it is already done."""
    night, expid, tile, petal = (int(unit[k]) for k in ("NIGHT", "EXPID", "TILEID", "PETAL"))
    udir = out / "units" / f"{expid:08d}-p{petal}"
    if (udir / "cameras.fits").exists():
        cam = [dict(zip(r.colnames, r, strict=True)) for r in Table.read(udir / "cameras.fits")]
        fp = udir / "fibers.fits"
        fib = [dict(zip(r.colnames, r, strict=True)) for r in Table.read(fp)] if fp.exists() else []
        return fib, cam, True
    udir.mkdir(parents=True, exist_ok=True)
    if int(unit.get("N_BADPETALSTDSTAR", 0) or 0) > 0:
        cam = unit_rows_empty(night, expid, tile, petal, "BADPETALSTDSTAR")
        Table(rows=cam).write(udir / "cameras.fits", overwrite=True)
        log(logpath, f"{expid} p{petal}: skipped, BADPETALSTDSTAR")
        return [], cam, False
    workdir = scratch / f"{expid:08d}-p{petal}"
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sha = sha_table(fetcher, night, expid, workdir)
    names = [n for n in unit_files(expid, petal) if n in sha]
    stats = []
    with cf.ThreadPoolExecutor(max_workers=max(len(names), 1)) as ex:
        futs = [
            ex.submit(fetcher.get, f"{night}/{expid:08d}/{n}", workdir / n, sha[n]) for n in names
        ]
        for f in cf.as_completed(futs):
            stats.append(f.result())
    nbytes = sum(s[2] for s in stats)
    t_dl = time.time() - t0
    hosts = {h: sum(1 for s in stats if s[0] == h) for h in ("nersc", "s3")}
    e = f"{expid:08d}"
    cams_ok = [
        c
        for c in CAMERAS
        if all(
            (workdir / f"{k}-{c}{petal}-{e}.fits.gz").exists()
            for k in ("frame", "sky", "fiberflatexp", "fluxcalib")
        )
    ]
    if not (workdir / f"stdstars-{petal}-{e}.fits.gz").exists() or not cams_ok:
        cam = unit_rows_empty(night, expid, tile, petal, "MISSING_INPUTS")
        Table(rows=cam).write(udir / "cameras.fits", overwrite=True)
        log(logpath, f"{expid} p{petal}: skipped, missing inputs (cameras with files {cams_ok})")
        if discard:
            shutil.rmtree(workdir)
        return [], cam, False
    t_run, rss, failed = run_calibration(workdir, expid, petal, logpath, cams_ok)
    fib, cam = summarise(workdir, night, expid, tile, petal, sample)
    for r in cam:
        r.update(SKIPPED="", DOWNLOAD_S=t_dl, DOWNLOAD_MB=nbytes / 1e6, CALIB_S=t_run, MAXRSS_MB=rss,
                 FAILED=";".join(failed), N_NERSC=hosts["nersc"], N_S3=hosts["s3"])  # fmt: skip
    missing = [c for c in CAMERAS if c not in {r["CAMERA"] for r in cam}]
    for r in unit_rows_empty(night, expid, tile, petal, "CAMERA_MISSING_OR_FAILED"):
        if r["CAMERA"] in missing:
            cam.append(r)
    if fib:
        Table(rows=fib).write(udir / "fibers.fits", overwrite=True)
    Table(rows=cam).write(udir / "cameras.fits", overwrite=True)
    log(
        logpath,
        f"{expid} p{petal}: {nbytes / 1e6:.0f} MB in {t_dl:.0f} s ({nbytes / 1e6 / max(t_dl, 1e-3):.1f} MB/s, "
        f"{hosts}), calibration {t_run:.0f} s, max RSS {rss:.0f} MB, failed {failed or 'none'}, "
        f"cameras {cams_ok}",
    )
    if discard:
        shutil.rmtree(workdir)
    return fib, cam, False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("units", help="unit list csv with NIGHT, EXPID, TILEID, PETAL")
    ap.add_argument("name", help="output folder name under results/2026-10-13_fluxcal_sample")
    ap.add_argument("--scratch", default=os.environ.get("SCRATCH", "/tmp"))
    ap.add_argument("--nersc-streams", type=int, default=0)
    ap.add_argument("--s3-streams", type=int, default=40)
    ap.add_argument("--units-parallel", type=int, default=8)
    ap.add_argument("--discard", action="store_true", help="remove each unit's downloaded inputs")
    ap.add_argument(
        "--tag", default="", help="suffix of the merged tables, for runs that split a list"
    )
    a = ap.parse_args()
    out = REPO / "results" / "2026-10-13_fluxcal_sample" / a.name
    out.mkdir(parents=True, exist_ok=True)
    logpath = out / "run.log"
    scratch = Path(a.scratch) / "fluxcal_sample" / a.name
    sample = set(np.load(REPO / "data" / "all_target_ids.npy").astype(np.int64).tolist())
    units = list(csv.DictReader(open(a.units)))
    log(
        logpath,
        f"{len(units)} units, scratch {scratch}, streams nersc {a.nersc_streams} s3 {a.s3_streams}",
    )
    fetcher = Fetcher(a.nersc_streams, a.s3_streams)
    fibs, cams = [], []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.units_parallel) as ex:
        futs = [
            ex.submit(process, u, fetcher, scratch, out, sample, logpath, a.discard) for u in units
        ]
        for f in cf.as_completed(futs):
            try:
                fib, cam, _ = f.result()
                fibs += fib
                cams += cam
            except Exception as e:  # keep going; the log says which unit failed
                log(logpath, f"unit failed: {e!r}")
    Table(rows=fibs).write(out / f"fibers{a.tag}.fits", overwrite=True)
    Table(rows=cams).write(out / f"cameras{a.tag}.fits", overwrite=True)
    log(
        logpath,
        f"done in {time.time() - t0:.0f} s, {len(cams)} camera rows, {len(fibs)} fiber rows",
    )


if __name__ == "__main__":
    main()
