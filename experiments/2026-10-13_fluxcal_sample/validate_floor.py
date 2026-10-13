"""Compare the exact corrected per-exposure variance with fractional floor models on VALIDATE units.

QUESTION: does the floor Var = 1/ivar_iron + (f F')^2, with f from the 10 wavelength bins of
F_PATCH_BINS, reproduce the exact corrected variance of every sample fiber, per exposure and
camera, at all S/N and at the camera edges?

HYPOTHESIS: sigma_floor / sigma_exact stays within a few percent everywhere. The largest errors
are in b, at high S/N and at the blue end, where f changes fastest with wavelength and the dropped
iron calibration term is largest.

Iron (desispec 0.56.2 apply_flux_calibration) gives Var_iron = 1/(ivar_F C^2) + F'^2 /
(ivar_C,old C^4), with F' the calibrated flux. The corrected variance (PR 2422 formula, ivar_C,new
from the rerun with PRs 2496, 2827 and 2452) is Var_new = 1/(ivar_F C^2) + F'^2 / (ivar_C,new C^2).
So Var_exact = 1/ivar_iron - F'^2 / (ivar_C,old C^4) + F'^2 / (ivar_C,new C^2). The models are
the proposed floor with f interpolated linearly between the bin centres of F_PATCH_BINS and flat
beyond them, a scalar f per exposure-petal-camera (column F of f_table), the binned f with the
iron term subtracted (binning error only), and the exact f with the iron term kept (double count
only). The bare iron variance is listed for scale.

For each unit and camera the script reads the iron, unpatched and patched fluxcalib from the
unit folder that fluxcal_sample.py leaves without --discard, and the iron cframe, which it
downloads from the public S3 copy of DR1 and checks against the iron sha256sum file. It checks
that the unpatched rerun equals iron, that the patched C equals iron C, that the cframe was
calibrated with this fluxcalib on the same wavelength grid and fibers, that ivar_iron is zero
wherever iron zeroes it, and that 1/ivar_iron less the iron calibration term stays positive. It
also recomputes the 10 bins from the patched file and compares them with the stored F_PATCH_BINS.
It does not rebuild the cframe from the frame, since the cframe step also rejects cosmics and
corrects fiber crosstalk. Half of the units are anchors from five exposures, so the summary is
repeated for the stratified units alone.

With --make-units it writes validate_units.csv instead. These are the VALIDATE rows of
fluxcal_candidates_500.csv with the units_all.csv columns, less units that are BADPETALSTDSTAR or
not ok in f_table.

INPUTS: validate_units.csv, the unit folders of fluxcal_sample.py run with name validate,
fibers_part0.fits, fibers_part1.fits and f_table.fits of the run all, and the iron cframes.
SEED: none.

COMMAND: on an ls6 compute node after fluxcal_sample.py, see validate.slurm. To remake the unit
list, from the repository root, ``uv run python
experiments/2026-10-13_fluxcal_sample/validate_floor.py --make-units``.

OUTPUT: validate_floor.txt and validate_floor_fibers.fits (fibers and cameras HDUs) in
results/2026-10-13_fluxcal_sample/validate/.
"""

import argparse
import concurrent.futures as cf
import csv
import hashlib
import os
import shutil
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from hubersed.paths import PATHS

HERE = PATHS["ROOT"] / "experiments" / "2026-10-13_fluxcal_sample"
RUN_ALL = PATHS["RESULTS"] / "2026-10-13_fluxcal_sample" / "all"
OUT = PATHS["RESULTS"] / "2026-10-13_fluxcal_sample" / "validate"
HOSTS = (
    "https://desidata.s3.amazonaws.com/dr1/spectro/redux/iron/exposures",
    "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/exposures",
)
NBIN = 10  # as fluxcal_sample.py
SN_BINS = ((5, 20), (20, 40), (40, 80), (80, np.inf))
MODELS = ("iron", "bins", "scalar", "bins_noiron", "double")
LABELS = {
    "iron": "iron, no floor",
    "bins": "floor, 10-bin f (proposal)",
    "scalar": "floor, scalar F",
    "bins_noiron": "10-bin f, iron term removed",
    "double": "exact f, iron term kept",
}
UNIT_COLS = (
    "NIGHT", "EXPID", "TILEID", "PETAL", "NSAMPLE", "NSTDSTAR", "STARRMS", "NCFRAME",
    "N_BADPETALPOS", "N_BADPETALSTDSTAR", "N_BADPETALFLUXCAL", "ROLE",
)  # fmt: skip


def make_units(candidates, units_all, f_table, out):
    """Write the VALIDATE units in the unit list format of fluxcal_sample.py.

    Returns
    -------
    list of str
        One line per dropped unit and a closing count.
    """
    cand = list(csv.DictReader(open(candidates)))
    allu = {(int(r["EXPID"]), int(r["PETAL"])): r for r in csv.DictReader(open(units_all))}
    t = Table.read(f_table)
    status = {}
    for r in t:
        k = (int(r["EXPID"]), int(r["PETAL"]))
        status.setdefault(k, []).append(str(r["STATUS"]).strip())
    rows, notes = [], []
    for c in cand:
        if c["VALIDATE"].strip() != "True":
            continue
        k = (int(c["EXPID"]), int(c["PETAL"]))
        u = dict(allu[k], ROLE=c["ROLE"])
        bad = [s for s in status.get(k, ["MISSING"]) if s != "ok"]
        if int(u["N_BADPETALSTDSTAR"]) > 0 or bad:
            notes.append(
                f"dropped {k[0]} p{k[1]}: N_BADPETALSTDSTAR {u['N_BADPETALSTDSTAR']}, {bad}"
            )
            continue
        rows.append({col: u[col] for col in UNIT_COLS})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=UNIT_COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    pairs = sum(int(r["NSAMPLE"]) for r in rows)
    notes.append(f"{len(rows)} units written to {out}, {pairs} sample target-exposure pairs")
    return notes


def fetch(rel, dest, sha=None, tries=6):
    """Download one iron file, alternating hosts on failure, and check its sha256."""
    err = None
    for k in range(tries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(f"{HOSTS[k % 2]}/{rel}", timeout=120) as r:
                with open(tmp, "wb") as fh:
                    shutil.copyfileobj(r, fh, length=1 << 20)
            if sha is not None and sha256(tmp) != sha:
                raise ValueError(f"sha256 mismatch for {rel}")
            tmp.replace(dest)
            return dest.stat().st_size
        except (urllib.error.URLError, OSError, ValueError) as e:
            err = e
            time.sleep(min(2**k, 30))
    raise RuntimeError(f"failed {rel}: {err}")


def sha256(path):
    """Return the sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_table(unit_dir, night, expid):
    """Return the iron sha256 of every file of an exposure, downloading the list if needed."""
    name = f"redux_iron_exposures_{night}_{expid:08d}.sha256sum"
    path = unit_dir / name
    if not path.exists():
        unit_dir.mkdir(parents=True, exist_ok=True)
        fetch(f"{night}/{expid:08d}/{name}", path)
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[Path(parts[1]).name] = parts[0]
    return out


def get_cframes(units, unit_root, cframe_root, cameras, streams, log):
    """Download the missing cframes of all units, sha256 checked; return the bytes fetched."""
    jobs = []
    for u in units:
        night, expid, petal = int(u["NIGHT"]), int(u["EXPID"]), int(u["PETAL"])
        udir = unit_root / f"{expid:08d}-p{petal}"
        cdir = (cframe_root or unit_root) / f"{expid:08d}-p{petal}"
        cdir.mkdir(parents=True, exist_ok=True)
        sha = sha_table(udir, night, expid)
        for c in cameras:
            name = f"cframe-{c}{petal}-{expid:08d}.fits.gz"
            dest = cdir / name
            if name in sha and not (dest.exists() and sha256(dest) == sha[name]):
                jobs.append((f"{night}/{expid:08d}/{name}", dest, sha[name]))
    log(f"{len(jobs)} cframes to download")
    nbytes = 0
    with cf.ThreadPoolExecutor(max_workers=max(streams, 1)) as ex:
        futs = {ex.submit(fetch, *j): j[0] for j in jobs}
        for f in cf.as_completed(futs):
            try:
                nbytes += f.result()
            except RuntimeError as e:
                log(str(e))
    return nbytes


def bin_f(wave, C, ivc, nbin=NBIN):
    """Return the binned f = sigma_C / C of every fiber, as fluxcal_sample.py computes it.

    Edges are equal in wavelength over the fluxcalib grid. The last pixel is in no bin.
    """
    ok = (C > 0) & (ivc > 0)
    fp = np.full(C.shape, np.nan)
    fp[ok] = 1 / (C[ok] * np.sqrt(ivc[ok]))
    edges = np.linspace(wave[0], wave[-1], nbin + 1)
    out = np.full((C.shape[0], nbin), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            out[:, k] = np.nanmedian(fp[:, (wave >= a) & (wave < b)], axis=1)
    return out


def interp_f(wave, fbins):
    """Interpolate binned f linearly between bin centres, flat beyond; NaN bins are skipped."""
    edges = np.linspace(wave[0], wave[-1], fbins.shape[-1] + 1)
    cen = 0.5 * (edges[1:] + edges[:-1])
    fb = np.atleast_2d(fbins)
    out = np.full((fb.shape[0], wave.size), np.nan)
    for i, row in enumerate(fb):
        good = np.isfinite(row)
        if good.any():
            out[i] = np.interp(wave, cen[good], row[good])
    return out.reshape(np.shape(fbins)[:-1] + (wave.size,))


def variances(flux, ivar, C, ivc_old, ivc_new, f_interp, f_scalar):
    """Return the exact variance, the model variances, both calibration terms and the good mask.

    All arrays share one shape (pixels, or fibers by pixels). Bad pixels get inf variance.
    """
    ok = (ivar > 0) & (C > 0) & (ivc_old > 0) & (ivc_new > 0)
    vi = np.where(ok, 1 / np.where(ok, ivar, 1), np.inf)
    iron_term = np.where(ok, flux**2 / np.where(ok, ivc_old * C**4, 1), 0.0)
    new_term = np.where(ok, flux**2 / np.where(ok, ivc_new * C**2, 1), 0.0)
    bins_term = np.where(ok, (f_interp * flux) ** 2, 0.0)  # NaN where no finite bin
    var = {
        "exact": vi - iron_term + new_term,
        "iron": vi,
        "bins": vi + bins_term,
        "scalar": vi + np.where(ok, (f_scalar * flux) ** 2, 0.0),
        "bins_noiron": vi - iron_term + bins_term,
        "double": vi + new_term,
    }
    return var, iron_term, new_term, ok


def ratios(var, ok):
    """Return sigma_model / sigma_exact for each model and the S/N |F'| / sigma_exact."""
    good = ok & (var["exact"] > 0)
    se = np.sqrt(np.where(good, var["exact"], np.nan))
    r = {m: np.sqrt(np.where(good, var[m], np.nan)) / se for m in MODELS}
    return r, good, se


def read_fluxcalib(path):
    """Return wavelength, calibration vector and its ivar of a fluxcalib file."""
    with fits.open(path) as h:
        return (
            h["WAVELENGTH"].data.astype(float),
            h["FLUXCALIB"].data.astype(float),
            h["IVAR"].data.astype(float),
        )


def maxrel(a, b):
    """Return the largest relative difference of b from a over entries where either is nonzero."""
    m = (a != 0) | (b != 0)
    if not m.any():
        return 0.0
    return float(np.max(np.abs(b[m] - a[m]) / np.maximum(np.abs(a[m]), 1e-300)))


def process_camera(task):
    """Check one unit-camera and compare the models for its sample fibers.

    Returns
    -------
    tuple
        Fiber rows, the camera row and per-pixel arrays (S/N >= 5) for the summary.
    """
    u, c, unit_dir, cframe_dir, stored, f_scalar = task
    night, expid, petal = int(u["NIGHT"]), int(u["EXPID"]), int(u["PETAL"])
    e = f"{expid:08d}"
    cam = f"{c}{petal}"
    name = f"fluxcalib-{cam}-{e}.fits.gz"
    paths = {
        "iron": unit_dir / name,
        "unpatched": unit_dir / "unpatched" / name,
        "patched": unit_dir / "patched" / name,
        "cframe": cframe_dir / f"cframe-{cam}-{e}.fits.gz",
    }
    role = u.get("ROLE", "")
    row = dict(NIGHT=night, EXPID=expid, PETAL=petal, CAMERA=c, ROLE=role, STATUS="ok")
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        row["STATUS"] = "missing " + ",".join(missing)
        return [], row, None
    wave, C, ivo = read_fluxcalib(paths["iron"])
    _, Cu, ivu = read_fluxcalib(paths["unpatched"])
    _, Cp, ivn = read_fluxcalib(paths["patched"])
    with fits.open(paths["cframe"]) as h:
        cwave = h["WAVELENGTH"].data.astype(float)
        flux = h["FLUX"].data.astype(float)
        ivar = h["IVAR"].data.astype(float)
        fm = h["FIBERMAP"].data
        fibers, tids = np.asarray(fm["FIBER"]), np.asarray(fm["TARGETID"]).astype(np.int64)
        fstatus = np.asarray(fm["FIBERSTATUS"])
        in_calib = str(h[0].header.get("IN_CALIB", ""))
        hdr = h[0].header
        depver = next(
            (
                str(hdr[f"DEPVER{k[6:]}"])
                for k in hdr
                if k.startswith("DEPNAM") and hdr[k] == "desispec"
            ),
            "",
        )
    row.update(
        UNPATCHED_C_MAXREL=maxrel(C, Cu),
        UNPATCHED_IVAR_MAXREL=maxrel(ivo, ivu),
        PATCHED_C_MAXABS=float(np.max(np.abs(Cp - C))),
        WAVE_MAXABS=float(np.max(np.abs(cwave - wave))) if cwave.shape == wave.shape else np.inf,
        IN_CALIB_OK=Path(in_calib).name.split(".fits")[0] == name.split(".fits")[0],
        IN_CALIB=in_calib,
        CFRAME_DESISPEC=depver,
        MASK_VIOL=int(np.sum((ivar > 0) & ((C <= 0) | (ivo <= 0)))),
    )
    # stored per-fiber values of the run all, matched on FIBER
    idx = np.searchsorted(stored["FIBER"], fibers)
    idx = np.clip(idx, 0, len(stored["FIBER"]) - 1)
    have = stored["FIBER"][idx] == fibers
    row["TID_MISMATCH"] = int(np.sum(have & (stored["TARGETID"][idx] != tids)))
    row["NO_STORED_ROW"] = int(np.sum(~have))
    fstored = np.where(have[:, None], stored["F_PATCH_BINS"][idx], np.nan)
    frecomp = bin_f(wave, C, ivn)
    both = np.isfinite(fstored) & np.isfinite(frecomp)
    row["BINS_REPRO_MAXREL"] = (
        float(np.max(np.abs(frecomp[both] / fstored[both] - 1))) if both.any() else np.nan
    )
    row["BINS_NAN_MISMATCH"] = int(np.sum(np.isfinite(fstored) != np.isfinite(frecomp)))
    row["F_SCALAR"] = f_scalar
    var, it, nt, ok = variances(flux, ivar, C, ivo, ivn, interp_f(wave, fstored), f_scalar)
    with np.errstate(divide="ignore", invalid="ignore"):
        varf = np.where(ok, 1 / np.where(ok, ivar, 1) - it, np.inf)
    row["NNEG_VARF"] = int(np.sum(ok & (varf <= 0)))
    r, good, se = ratios(var, ok)
    sn = np.abs(flux) / se
    thirds = np.linspace(wave[0], wave[-1], 4)
    third = np.clip(np.searchsorted(thirds, wave, side="right") - 1, 0, 2).astype(np.int8)
    sample = have & stored["IN_SAMPLE"][idx]
    fib_rows, pix = [], {k: [] for k in ("sn", "third", "term", *MODELS)}
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(nt > 0, it / nt, np.nan)
    for i in np.flatnonzero(sample):
        g = good[i]
        if not g.any():
            continue
        hi = g & (sn[i] >= 5)
        fr = dict(
            NIGHT=night, EXPID=expid, PETAL=petal, CAMERA=c, ROLE=role, FIBER=int(fibers[i]),
            TARGETID=int(tids[i]), FIBERSTATUS=int(fstatus[i]),
            NPIX_OK=int(g.sum()), NPIX_SN5=int(hi.sum()),
            SN_P50=float(np.median(sn[i][g])), SN_MAX=float(np.max(sn[i][g])),
            F_SCALAR=f_scalar, F_PATCH_BINS=fstored[i].astype(float),
            BINS_REPRO_MAXREL=float(np.nanmax(np.abs(frecomp[i] / fstored[i] - 1)))
            if np.isfinite(frecomp[i] / fstored[i]).any() else np.nan,
            TERM_RATIO_P50=float(np.nanmedian(term[i][g])),
            TERM_RATIO_P95=float(np.nanpercentile(term[i][g], 95)),
            NNEG_VARF=int(np.sum(g & (varf[i] <= 0))),
        )  # fmt: skip
        for m in MODELS:
            x = r[m][i][hi]
            x = x[np.isfinite(x)]
            fr[f"R_{m.upper()}_P50"] = float(np.median(x)) if x.size else np.nan
            fr[f"R_{m.upper()}_MIN"] = float(np.min(x)) if x.size else np.nan
            fr[f"R_{m.upper()}_MAX"] = float(np.max(x)) if x.size else np.nan
            fr[f"FRAC5_{m.upper()}"] = float(np.mean(np.abs(x - 1) > 0.05)) if x.size else np.nan
            pix[m].append(r[m][i][hi].astype(np.float32))
        pix["sn"].append(sn[i][hi].astype(np.float32))
        pix["third"].append(third[hi])
        pix["term"].append(term[i][hi].astype(np.float32))
        fib_rows.append(fr)
    row["NFIB_SAMPLE"] = len(fib_rows)
    pix = {k: np.concatenate(v) if v else np.zeros(0) for k, v in pix.items()}
    return fib_rows, row, dict(pix, camera=c, role=role, edges=thirds)


def load_stored(run_all, keys):
    """Return the stored fiber values of the run all for the given (EXPID, PETAL) keys."""
    cols = ("EXPID", "PETAL", "CAMERA", "FIBER", "TARGETID", "IN_SAMPLE", "F_PATCH_BINS")
    want = {k[0] * 10 + k[1] for k in keys}
    parts = []
    for k in (0, 1):
        with fits.open(run_all / f"fibers_part{k}.fits", memmap=True) as h:
            d = h[1].data
            key = np.asarray(d["EXPID"]).astype(np.int64) * 10 + np.asarray(d["PETAL"])
            m = np.isin(key, list(want))
            parts.append({c: np.array(d[c][m]) for c in cols})
    d = {c: np.concatenate([p[c] for p in parts]) for c in cols}
    d["CAMERA"] = np.char.strip(d["CAMERA"].astype(str))
    out = {}
    for ex, pe, ca in {
        (int(a), int(b), str(c)) for a, b, c in zip(*(d[x] for x in cols[:3]), strict=True)
    }:
        m = (d["EXPID"] == ex) & (d["PETAL"] == pe) & (d["CAMERA"] == ca)
        o = np.argsort(d["FIBER"][m])
        out[(ex, pe, ca)] = {c: d[c][m][o] for c in cols[3:]}
    return out


def qline(x, label):
    """Return one summary line of a ratio sample."""
    nbad = int(np.sum(~np.isfinite(x)))
    x = x[np.isfinite(x)]
    if x.size < 3:
        return f"    {label:<30} n {x.size}"
    q = np.percentile(x, [1, 5, 50, 95, 99])
    return (
        f"    {label:<30} p50 {q[2]:.4f} [p1 {q[0]:.4f} p5 {q[1]:.4f} p95 {q[3]:.4f} "
        f"p99 {q[4]:.4f}] >5% {np.mean(np.abs(x - 1) > 0.05):.4f} >10% "
        f"{np.mean(np.abs(x - 1) > 0.10):.4f} n {x.size}" + (f" non-finite {nbad}" if nbad else "")
    )


def summary_lines(pixels, cameras, thirds=True):
    """Return the text summary of sigma_model / sigma_exact by camera, S/N bin and third."""
    lines = []
    for c in cameras:
        ps = [p for p in pixels if p is not None and p["camera"] == c]
        if not ps:
            continue
        p = {k: np.concatenate([x[k] for x in ps]) for k in ("sn", "third", "term", *MODELS)}
        e = ps[0]["edges"]
        lines.append("")
        lines.append(f"camera {c}, thirds edges {e[0]:.1f} {e[1]:.1f} {e[2]:.1f} {e[3]:.1f} A")
        t = p["term"][np.isfinite(p["term"])]
        if t.size:
            q = np.percentile(t, [50, 95, 99])
            lines.append(
                f"  iron term / new term at S/N >= 5: p50 {q[0]:.4f} p95 {q[1]:.4f} p99 {q[2]:.4f}"
            )
        for lo, hi in SN_BINS:
            m = (p["sn"] >= lo) & (p["sn"] < hi)
            lines.append(f"  S/N {lo:g}-{hi:g}, all wavelengths")
            for mod in MODELS:
                lines.append(qline(p[mod][m], LABELS[mod]))
        for k in range(3 if thirds else 0):
            for lo, hi in SN_BINS:
                m = (p["third"] == k) & (p["sn"] >= lo) & (p["sn"] < hi)
                lines.append(f"  third {k} ({e[k]:.0f}-{e[k + 1]:.0f} A), S/N {lo:g}-{hi:g}")
                for mod in ("bins", "scalar", "bins_noiron", "double"):
                    lines.append(qline(p[mod][m], LABELS[mod]))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--make-units", action="store_true", help="write validate_units.csv and stop")
    ap.add_argument("--units", default=str(HERE / "validate_units.csv"))
    ap.add_argument(
        "--unit-root",
        default=str(Path(os.environ.get("SCRATCH", "/tmp")) / "fluxcal_sample" / "validate"),
        help="folder with the <expid>-p<petal> unit folders of fluxcal_sample.py",
    )
    ap.add_argument("--cframe-root", default=None, help="cframe folder, default the unit folders")
    ap.add_argument("--run-all", default=str(RUN_ALL), help="folder of the run all tables")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--cameras", default="brz")
    ap.add_argument("--streams", type=int, default=24)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    run_all = Path(a.run_all)
    if a.make_units:
        notes = make_units(
            HERE / "fluxcal_candidates_500.csv",
            HERE / "units_all.csv",
            run_all / "f_table.fits",
            Path(a.units),
        )
        print("\n".join(notes))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    units = list(csv.DictReader(open(a.units)))
    unit_root = Path(a.unit_root)
    cframe_root = Path(a.cframe_root) if a.cframe_root else None
    cameras = tuple(a.cameras)
    t0 = time.time()
    nbytes = get_cframes(units, unit_root, cframe_root, cameras, a.streams, log)
    log(f"cframes {nbytes / 1e6:.0f} MB in {time.time() - t0:.0f} s")
    keys = [(int(u["EXPID"]), int(u["PETAL"])) for u in units]
    stored = load_stored(run_all, keys)
    ft = Table.read(run_all / "f_table.fits")
    fval = np.ma.filled(np.ma.asarray(ft["F"], dtype=float), np.nan)
    fscal = {
        (int(r["EXPID"]), int(r["PETAL"]), str(r["CAMERA"]).strip()): float(v)
        for r, v in zip(ft, fval, strict=True)
    }
    tasks = []
    for u in units:
        ex, pe = int(u["EXPID"]), int(u["PETAL"])
        udir = unit_root / f"{ex:08d}-p{pe}"
        cdir = (cframe_root or unit_root) / f"{ex:08d}-p{pe}"
        for c in cameras:
            if (ex, pe, c) not in stored:
                log(f"{ex} p{pe} {c}: no stored fibers in the run all, skipped")
                continue
            tasks.append((u, c, udir, cdir, stored[(ex, pe, c)], fscal.get((ex, pe, c), np.nan)))
    fibs, cams, pixels = [], [], []
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        for fr, cr, px in ex.map(process_camera, tasks):
            fibs += fr
            cams.append(cr)
            pixels.append(px)
    log(f"processed {len(cams)} unit-cameras in {time.time() - t0:.0f} s")
    cam_t = Table(rows=cams)
    fib_t = Table(rows=fibs)
    hdus = [fits.PrimaryHDU()]
    for t, nm in ((fib_t, "FIBERS"), (cam_t, "CAMERAS")):
        h = fits.table_to_hdu(t)
        h.name = nm
        hdus.append(h)
    fits.HDUList(hdus).writeto(out / "validate_floor_fibers.fits", overwrite=True)

    ok = np.asarray(cam_t["STATUS"]) == "ok"
    lines.append(f"Exact corrected variance vs floor models, {len(units)} units from {a.units}")
    lines.append(f"unit folders {unit_root}, stored bins and F from {run_all}")
    lines.append(f"unit-cameras {len(cam_t)}, ok {int(ok.sum())}")
    for r in cam_t[~ok]:
        lines.append(f"  not ok: {r['EXPID']} p{r['PETAL']} {r['CAMERA']} {r['STATUS']}")
    if ok.any():
        c = cam_t[ok]

        def mx(col):
            return float(np.nanmax(np.asarray(c[col], float)))

        lines.append("checks, max over ok unit-cameras")
        lines.append(f"  unpatched vs iron C, max rel {mx('UNPATCHED_C_MAXREL'):.2e}")
        lines.append(f"  unpatched vs iron IVAR, max rel {mx('UNPATCHED_IVAR_MAXREL'):.2e}")
        lines.append(f"  patched C vs iron C, max abs {mx('PATCHED_C_MAXABS'):.2e}")
        lines.append(f"  cframe vs fluxcalib wavelength, max abs {mx('WAVE_MAXABS'):.2e} A")
        lines.append(f"  cframe IN_CALIB not this fluxcalib {int(np.sum(~c['IN_CALIB_OK']))}")
        lines.append(f"  cframe desispec versions {sorted(set(c['CFRAME_DESISPEC']))}")
        lines.append(
            f"  pixels with ivar_iron > 0 where C or ivar_C,old <= 0 {int(c['MASK_VIOL'].sum())}"
        )
        lines.append(f"  good pixels with 1/ivar_iron - iron term <= 0 {int(c['NNEG_VARF'].sum())}")
        lines.append(f"  fibers with TARGETID unlike the run all {int(c['TID_MISMATCH'].sum())}")
        lines.append(f"  fibers without a stored row {int(c['NO_STORED_ROW'].sum())}")
        lines.append(
            f"  recomputed 10-bin f vs stored F_PATCH_BINS, max rel {mx('BINS_REPRO_MAXREL'):.2e}, "
            f"NaN mismatches {int(c['BINS_NAN_MISMATCH'].sum())}"
        )
        pairs = {(r["TARGETID"], r["EXPID"]) for r in fibs}
        lines.append(f"  sample fibers compared {len(fib_t)}, target-exposure pairs {len(pairs)}")
    lines.append("")
    lines.append(
        "sigma_model / sigma_exact per exposure-camera pixel of sample fibers, per-exposure S/N "
        "= |F'| / sigma_exact"
    )
    lines += summary_lines(pixels, cameras)
    strat = [p for p in pixels if p is not None and p["role"] == "stratified"]
    lines.append("")
    lines.append(f"stratified units only (anchors left out), {len(strat)} unit-cameras")
    lines += summary_lines(strat, cameras, thirds=False)
    text = "\n".join(lines) + "\n"
    print(text)
    (out / "validate_floor.txt").write_text(text)


if __name__ == "__main__":
    main()
