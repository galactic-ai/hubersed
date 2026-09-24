"""Fit DESI spectra by MAP, one TARGETID at a time, with prospector and Cue nebular emission.

Run as ``uv run python scripts/run_map_fits_outliers.py``. Each galaxy gets a pickle with
the best fit and its chi2 split into line and continuum pixels, plus spectrum and SFH figures.
The fitting is in ``hubersed.fitting.map_fits``.
"""

import argparse
import multiprocessing as mp
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# One BLAS thread per process. This has to run before numpy loads BLAS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from hubersed.fitting.map_fits import (
    AIR_LINES,
    BALMER,
    FLAT_SFH_RANGE,
    FROZEN_HYPERS,
    _quiet_process,
    _worker,
    fit_one,
    get_sps,
)
from hubersed.paths import PATHS


def main(argv=None):
    """Fit every TARGETID in the sample file and write ``summary.pkl`` next to the fits."""
    _quiet_process()
    p = argparse.ArgumentParser(
        description="MAP fits (Cue, free PSD) for emission-line OOD outliers."
    )
    p.add_argument(
        "-s", "--sample", type=str, default=str(PATHS["RESULTS"] / "emline_outlier_sample20.npz")
    )
    p.add_argument("-o", "--outdir", type=str, default=str(PATHS["RESULTS"] / "emline_map_fits"))
    p.add_argument("-n", "--n-seeds", type=int, default=6)
    p.add_argument("-m", "--maxfev", type=int, default=120_000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("-w", "--workers", type=int, default=1)
    p.add_argument(
        "--freeze-hypers",
        action="store_true",
        help="fix the 5 SFH hyperparameters at FROZEN_HYPERS",
    )
    p.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="fix a parameter at one value for every galaxy, e.g. --fix logzsol=-2.5",
    )
    p.add_argument(
        "--flat-sfh-prior",
        action="store_true",
        help=f"use SpecModel with TopHat(+/-{FLAT_SFH_RANGE} dex) on logsfr_ratios "
        f"instead of the stochastic SFH prior",
    )
    p.add_argument(
        "--zcontinuous",
        type=int,
        default=1,
        choices=[1, 2],
        help="FSPS metallicity interpolation mode, passed to build_sps",
    )
    p.add_argument(
        "--optimizer",
        default="Powell",
        choices=["Powell", "Nelder-Mead"],
        help="scipy minimize method",
    )
    p.add_argument(
        "--spectra-npz",
        default=None,
        metavar="PATH",
        help="npz with target_ids, spec, ivar and z on the WAVE_OBS grid in "
        "1e-17 erg/s/cm^2/A, used instead of the chunk files",
    )
    p.add_argument(
        "--continuum-only",
        action="store_true",
        help="fit build_continuum_model on line-masked pixels with FSPS and no "
        "nebular emission; its SFH hyperparameters are fixed",
    )
    p.add_argument(
        "--free-dust1",
        action="store_true",
        help="fit dust1 with TopHat(0, 3) instead of dust2 * dust_ratio; "
        "ignored with --continuum-only",
    )
    args = p.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.sample)
    tids = d["target_ids"].astype(np.int64)
    if args.limit:
        tids = tids[: args.limit]

    # Start logzsol at the sample's value when it has that column, else at -1.0.
    _z0 = d["logzsol"] if "logzsol" in d.files else np.full(len(d["target_ids"]), -1.0)
    seeds = {
        int(t): {"logmass": float(m), "eline_sigma": float(s), "logzsol": float(zz)}
        for t, m, s, zz in zip(d["target_ids"], d["logmstar"], d["narrow_sigma"], _z0, strict=True)
    }

    fixed = dict(kv.split("=") for kv in args.fix)
    fixed = {k: float(v) for k, v in fixed.items()}

    if args.freeze_hypers or fixed:
        print(f"frozen hypers: {args.freeze_hypers}   fixed: {fixed}", flush=True)

    todo = [int(t) for t in tids if not (args.skip_existing and (out / f"{t}.pkl").exists())]
    print(f"{len(todo)}/{len(tids)} to fit, {args.workers} worker(s)", flush=True)

    if args.workers > 1:
        results = {}
        # spawn, not fork. A forked child can deadlock on locks held by the parent's threads.
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context("spawn")
        ) as ex:
            futs = {
                ex.submit(
                    _worker,
                    t,
                    args.n_seeds,
                    args.maxfev,
                    str(out),
                    seeds.get(t),
                    args.freeze_hypers,
                    fixed,
                    flat_sfh=args.flat_sfh_prior,
                    cont_only=args.continuum_only,
                    method=args.optimizer,
                    zcontinuous=args.zcontinuous,
                    spectra_npz=args.spectra_npz,
                    free_dust1=args.free_dust1,
                ): t
                for t in todo
            }
            for n, fu in enumerate(as_completed(futs), 1):
                t = futs[fu]
                try:
                    results[t] = fu.result()
                except Exception as e:
                    results[t] = {"target_id": t, "status": f"error:{type(e).__name__}: {e}"}
                print(f"[{n}/{len(todo)}] {t} done: {results[t].get('status')}", flush=True)
        recs = [results[t] for t in todo if t in results]
    else:
        S = get_sps(zcontinuous=args.zcontinuous)
        recs = []
        for i, tid in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {tid}", flush=True)
            try:
                recs.append(
                    fit_one(
                        tid,
                        S["sps"],
                        S["cue"],
                        S["lines"],
                        S["line_waves"],
                        args.n_seeds,
                        args.maxfev,
                        out,
                        seeds=seeds.get(tid),
                        frozen=args.freeze_hypers,
                        fixed=fixed,
                        flat_sfh=args.flat_sfh_prior,
                        cont_only=args.continuum_only,
                        method=args.optimizer,
                        zcontinuous=args.zcontinuous,
                        spectra_npz=args.spectra_npz,
                        free_dust1=args.free_dust1,
                    )
                )
            except Exception as e:
                print(f"    FAILED {type(e).__name__}: {e}", flush=True)
                recs.append({"target_id": int(tid), "status": f"error:{type(e).__name__}"})

    summary = []

    def par(rec, name, default=np.nan):
        """Return a parameter's value whether it was free, fixed or a frozen hyperparameter.

        Looks in ``theta_dict``, then ``fixed``, then ``hyper_values``, else ``default``.
        """
        td = rec.get("theta_dict") or {}
        if name in td:
            return float(np.atleast_1d(td[name])[0])
        for k in ("fixed", "hyper_values"):
            m = rec.get(k) or {}
            if name in m:
                return float(m[name])
        return float(default)

    lines = next((r["lines"] for r in recs if r.get("status") == "ok"), AIR_LINES)
    for rec in recs:
        tid = rec["target_id"]
        if rec["status"] != "ok":
            summary.append({"target_id": int(tid), "status": rec["status"]})
            continue
        v = np.array([rec["line_ratios"][k] for k in lines])
        b = np.array([rec["line_ratios"][k] for k in BALMER])
        summary.append(
            {
                "target_id": int(tid),
                "z": rec["z"],
                "status": "ok",
                "chi2_red": rec["stats"]["chi2_red"],
                "line_per_pix": rec["stats_tight"]["chi2_per_pix_line"],
                "cont_per_pix": rec["stats_tight"]["chi2_per_pix_cont"],
                "rms_all": float(np.sqrt(np.nanmean((v - 1) ** 2))),
                "rms_balmer": float(np.sqrt(np.nanmean((b - 1) ** 2))),
                # Use par for every parameter. Which are free depends on the flags, and a
                # missing key here fails after all the fits are written.
                "eline_sigma": par(rec, "eline_sigma"),
                "logzsol": par(rec, "logzsol"),
                "logmass": par(rec, "logmass"),
                "sigma_reg": par(rec, "sigma_reg", FROZEN_HYPERS["sigma_reg"]),
                "cont_only": bool(rec.get("cont_only", False)),
                "cmf_dev": float(np.max(np.abs(rec["sfh"]["cmf"] - rec["sfh"]["cmf_flat_null"]))),
                "n_converged": rec["optim"]["n_converged"],
                "gap_best_second": rec["optim"]["gap_best_second"],
                "seconds": rec["optim"].get("seconds", np.nan),
            }
        )

    with open(out / "summary.pkl", "wb") as f:
        pickle.dump(summary, f)

    okr = [s for s in summary if s["status"] == "ok"]
    print(
        f"\n{'TARGETID':>19}{'z':>8}{'chi2':>8}{'line':>8}{'cont':>7}{'rmsAll':>8}"
        f"{'rmsBal':>8}{'e_sig':>7}{'logZ':>7}{'cmfdev':>8}{'conv':>6}"
    )
    for s in okr:
        print(
            f"{s['target_id']:>19d}{s['z']:>8.4f}{s['chi2_red']:>8.3f}{s['line_per_pix']:>8.2f}"
            f"{s['cont_per_pix']:>7.3f}{s['rms_all']:>8.4f}{s['rms_balmer']:>8.4f}"
            f"{s['eline_sigma']:>7.1f}{s['logzsol']:>7.2f}{s['cmf_dev']:>8.3f}"
            f"{s['n_converged']:>6d}"
        )
    print(f"\n{len(okr)}/{len(tids)} ok -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
