"""Fit DESI spectra with prospector at the MAP and report the reduced chi-squared.

Each galaxy gets a continuum-only fit first, which seeds a full fit with nebular emission.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache

import astropy.units as u
import numpy as np
from prospect.fitting import lnprobfn

from hubersed.conversion import DESI_FLAM, ivar_to_maggies, to_maggies
from hubersed.fitting.result import MapFitResult
from hubersed.io.desi import load_by_index, tids_to_indices
from hubersed.paths import PATHS
from hubersed.sps import parameter_file as P
from hubersed.sps.config import build_continuum_model, build_full_cue_model, build_full_model
from hubersed.sps.rebin import common_obs_edges

DATA_PATH = PATHS["DATA"]
RESULTS_PATH = PATHS["RESULTS"]
WAVE_OBS = P.WAVE_OBS
N_TOTAL = 254976
Z_FLOOR = 0.01

EDGES = common_obs_edges()
WAVE_C = (0.5 * (EDGES[1:] + EDGES[:-1])).astype(np.float32)  # coarse centers, for the checkpoint


from scipy.optimize import minimize


def _map_optimize(neg, theta_init, n_seeds=3, jitter=0.03, maxfev=20_000, max_tries=100):
    """Minimize an objective with Powell from several starting points and keep the best.

    Parameters
    ----------
    neg : callable
        Negative log probability of a parameter vector. It returns 1e18 for invalid points,
        for example when Cue is asked for parameters outside its training range.
    theta_init : np.ndarray
        Starting parameter vector. It is always the first start.
    n_seeds : int
        Number of extra starts, each ``theta_init`` plus Gaussian jitter.
    jitter : float
        Standard deviation of the jitter, in the units of each parameter.
    maxfev : int
        Maximum number of objective calls for each Powell run.
    max_tries : int
        Most jitter draws for one extra start before that start is dropped.

    Returns
    -------
    scipy.optimize.OptimizeResult or None
        The run with the lowest objective, or None if no run finished below 1e10.

    Notes
    -----
    The jitter for start ``s`` is drawn with ``np.random.default_rng(s)``, so every galaxy
    gets the same offsets. The value 1e18 is finite, so a plain ``np.isfinite`` check would
    not catch it, which is why starts at or above 1e17 are treated as invalid.

    An invalid jittered start is drawn again from the same generator, up to ``max_tries``
    times, so every run starts from a different valid point. Each candidate is evaluated
    once. A start with no valid draw is dropped, and an invalid ``theta_init`` is not run.
    ``tests/test_equal_budget.py`` checks this.
    """

    def valid(theta):
        v = neg(theta)
        return np.isfinite(v) and v < 1e17

    starts = [theta_init] if valid(theta_init) else []
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        for _ in range(max_tries):
            st = theta_init + rng.normal(0, jitter, theta_init.shape)
            if valid(st):
                starts.append(st)
                break
    best = None
    for st in starts:
        r = minimize(
            neg,
            st,
            method="Powell",
            options={"maxiter": maxfev // 10, "maxfev": maxfev, "ftol": 1e-6},
        )
        if np.isfinite(r.fun) and r.fun < 1e10:
            best = r if (best is None or r.fun < best.fun) else best
    return best


@cache
def _fsps():
    """Build the FSPS stellar population source once per process."""
    return P.build_sps()


@cache
def _cue():
    """Build the Cue nebular emission source once per process."""
    return P.build_cue_sps()


@cache
def _lsf_sigma_kms():
    """Return the DESI instrumental resolution as a Gaussian sigma for prospect.

    Returns
    -------
    np.ndarray
        Sigma in km/s on ``WAVE_OBS``, computed as ``C_KMS / (2.355 * R)``.

    Notes
    -----
    Passing this to prospect is safe because ``build_sps`` sets the library resolution to
    zero, so prospect does not refuse data that is sharper than the templates.
    """
    from hubersed.sps.lsf import C_KMS, desi_resolution

    R = desi_resolution(WAVE_OBS)
    return (C_KMS / (2.355 * R)).astype(np.float64)


def map_chi2_one(gidx, use_cue=False, cont_nseeds=1, full_nseeds=1, maxfev=3_000):
    """Fit one galaxy at the MAP and return its reduced chi-squared.

    The continuum fit uses a mask that hides emission lines. Its best values seed the full
    fit, which uses every good pixel.

    Parameters
    ----------
    gidx : int
        Global index of the galaxy, as used by ``load_by_index``.
    use_cue : bool
        Model nebular emission with Cue instead of FSPS.
    cont_nseeds, full_nseeds : int
        Extra jittered starts for the continuum and full fits.
    maxfev : int
        Maximum objective calls for each Powell run.

    Returns
    -------
    dict
        Always has ``gidx`` and ``status``. When ``status`` is ``"ok"`` it also has
        ``id`` (TARGETID), ``z``, ``chi2``, ``ndof``, ``chi2_red``, ``npix``, ``theta``,
        ``theta_labels``, ``theta_dict``, and the spectra ``model``, ``flux``, ``unc`` and
        ``mask`` in maggies. Other statuses are ``load_fail:<error>``, ``below_zfloor``,
        ``too_masked``, ``cont_fail`` and ``full_fail``.

    Notes
    -----
    ``theta_labels`` is ``model.theta_labels()``, one label per entry of ``theta``, with
    vector parameters named ``logsfr_ratios_1`` and so on. ``MapFitResult`` checks it
    against ``theta_dict``.
    """
    try:
        spec, ivar, redshift, tid = load_by_index(gidx)
    except Exception as e:
        return dict(gidx=gidx, status=f"load_fail:{type(e).__name__}")
    if redshift < Z_FLOOR:
        return dict(gidx=gidx, id=tid, z=redshift, status="below_zfloor")

    spec_maggies = to_maggies(WAVE_OBS * u.AA, spec * DESI_FLAM).value
    ivar_maggies = ivar_to_maggies(WAVE_OBS * u.AA, ivar * DESI_FLAM**-2).value
    sigma = 1 / np.sqrt(np.where(ivar_maggies > 0, ivar_maggies, np.inf))
    mask = (sigma > 0) & np.isfinite(sigma) & np.isfinite(spec_maggies)
    if mask.sum() < 100:
        return dict(gidx=gidx, id=tid, z=redshift, status="too_masked")

    sps = _fsps()
    fw = sps.ssp.emline_wavelengths
    fopt = fw[(fw > 3600) & (fw < 9824)]
    mask_em = P.mask_spectral_lines(WAVE_OBS, mask, redshift, halfwidth_kms=1500.0, line_waves=fopt)
    res = _lsf_sigma_kms()

    obs_em = P.build_obs(
        spec=spec_maggies, unc=sigma, mask=mask_em, resolution=res, wavelength=WAVE_OBS
    )
    obs_full = P.build_obs(
        spec=spec_maggies, unc=sigma, mask=mask, resolution=res, wavelength=WAVE_OBS
    )

    # continuum MAP (seeds logmass/logzsol/sigma_smooth for the full model)
    cmodel, ctemplate = build_continuum_model(redshift)

    def neg_cont(th):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lp = lnprobfn(th, model=cmodel, observations=obs_em, sps=sps, nested=False)
                return -lp if np.isfinite(lp) else 1e18
            except Exception:
                return 1e18

    bc = _map_optimize(neg_cont, cmodel.theta.copy(), n_seeds=cont_nseeds, maxfev=maxfev)
    if bc is None:
        return dict(gidx=gidx, id=tid, z=redshift, status="cont_fail")
    theta_cont = bc.x

    # full MAP (continuum + nebular); Cue or FSPS
    if use_cue:
        sps = _cue()
        fmodel, ftemplate = build_full_cue_model(ctemplate, theta_cont, cmodel)
    else:
        fmodel, ftemplate = build_full_model(ctemplate, theta_cont, cmodel)

    def neg_full(th):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lp = lnprobfn(th, model=fmodel, observations=obs_full, sps=sps, nested=False)
                return -lp if np.isfinite(lp) else 1e18
            except Exception:
                return 1e18

    bf = _map_optimize(neg_full, fmodel.theta.copy(), n_seeds=full_nseeds, maxfev=maxfev)
    if bf is None:
        return dict(gidx=gidx, id=tid, z=redshift, status="full_fail")
    theta_map = bf.x

    preds, _ = fmodel.predict(theta_map, observations=obs_full, sps=sps)
    sp = preds[0]
    m = obs_full[0].mask
    resid = (obs_full[0].flux[m] - sp[m]) / obs_full[0].uncertainty[m]
    chi2 = float(np.nansum(resid**2))
    ndof = int(m.sum()) - len(theta_map)
    res = MapFitResult(
        int(tid),
        float(redshift),
        {k: np.asarray(theta_map[v], dtype=np.float32) for k, v in fmodel.theta_index.items()},
        tuple(fmodel.theta_labels()),
    )
    return dict(
        gidx=gidx,
        id=tid,
        z=redshift,
        status="ok",
        chi2=chi2,
        ndof=ndof,
        chi2_red=chi2 / ndof,
        npix=int(m.sum()),
        theta=res.vector().astype(np.float32),
        theta_labels=list(res.labels),
        theta_dict=res.theta,
        model=np.asarray(sp, dtype=np.float32),  # MAP model spectrum (full grid)
        flux=np.asarray(obs_full[0].flux, dtype=np.float32),
        unc=np.asarray(obs_full[0].uncertainty, dtype=np.float32),
        mask=np.asarray(m, dtype=bool),
    )


def _work(args):
    """Unpack one task tuple for the process pool and run ``map_chi2_one``."""
    gi, use_cue, cns, fns, mf = args
    return map_chi2_one(int(gi), use_cue=use_cue, cont_nseeds=cns, full_nseeds=fns, maxfev=mf)


def main():
    """Fit a random sample or the flagged outliers and save the results.

    The first argument is the number of random galaxies (default 100). Options are
    ``--cue``, ``--seed``, ``--workers``, ``--limit``, ``--outfile``, ``--tag``, ``--rich``
    for a larger optimizer budget, ``--outliers`` to fit the TARGETIDs in ``--outfile``,
    and ``--worst`` to take the outliers in order of isolation forest score. Results go to
    ``results/map_chi2_*.npy`` and ``results/map_chi2_*_full.pkl``.
    """
    N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100
    use_cue = "--cue" in sys.argv
    seed = 0
    workers = 1
    limit = None
    outfile = "desi_outliers_cue_snr3.pt"
    argv = sys.argv

    def _opt(name, cast):
        for i, a in enumerate(argv):
            if a == name and i + 1 < len(argv):  # "--name value"
                return cast(argv[i + 1])
            if a.startswith(name + "="):  # "--name=value"
                return cast(a.split("=", 1)[1])
        return None

    v = _opt("--seed", int)
    seed = v if v is not None else seed
    v = _opt("--workers", int)
    workers = v if v is not None else workers
    v = _opt("--limit", int)
    limit = v if v is not None else limit
    v = _opt("--outfile", str)
    outfile = v if v is not None else outfile
    runtag = _opt("--tag", str)  # appended to output filenames (e.g. --tag=KC13dust)
    rich = "--rich" in sys.argv
    cns, fns, mf = (3, 5, 30_000) if rich else (1, 1, 3_000)  # optimizer budget
    if rich:
        print("RICH budget: cont 3 seeds, full 5 seeds, maxfev 30k")
    if "--outliers" in sys.argv:
        import torch

        worst = "--worst" in sys.argv
        blob = torch.load(RESULTS_PATH / outfile, weights_only=False)
        tids = np.asarray(blob["outlier_target_ids"]).astype(np.int64)  # TARGETIDs (canonical)
        # order the outliers by IsoForest score (lower = more anomalous) if requested
        if worst and "scores_desi" in blob and "desi_target_ids" in blob:
            dtid = np.asarray(blob["desi_target_ids"]).astype(np.int64)
            dscore = np.asarray(blob["scores_desi"])
            score_of = dict(zip(dtid.tolist(), dscore.tolist()))
            tids = tids[np.argsort([score_of[int(t)] for t in tids])]  # most anomalous first
        idxs = tids_to_indices(tids)  # -> numeric global indices
        # self-check: load_by_index must return the SAME TARGETID we asked for
        for g, t in list(zip(idxs, tids))[:3]:
            assert load_by_index(int(g))[3] == int(t), "TID->index map mismatch!"
        if limit is not None and limit < idxs.size:
            idxs = (
                idxs[:limit]
                if worst
                else np.random.default_rng(seed).choice(idxs, limit, replace=False)
            )
        N = idxs.size
        modetag = (
            "outliers" if limit is None else (f"outliers_worst{N}" if worst else f"outliers_n{N}")
        )
        print(
            f"fitting {N} outliers by TARGETID (from {outfile}; {'WORST by IsoForest score' if worst else 'random subset' if limit else 'all'}); self-check passed"
        )
    else:
        rng = np.random.default_rng(seed)
        idxs = np.sort(rng.choice(N_TOTAL, N, replace=False))
        modetag = f"randN{N}_seed{seed}"

    out = []
    # output paths up front so the serial loop can checkpoint the FULL pkl (theta+spectra)
    tag = "cue" if use_cue else "fsps"
    suffix = f"{modetag}_lsf" + ("_rich" if rich else "") + (f"_{runtag}" if runtag else "")
    full_pkl = RESULTS_PATH / f"map_chi2_{tag}_{suffix}_full.pkl"

    def _checkpoint():
        with open(full_pkl, "wb") as f:
            pickle.dump(
                {
                    "wave": np.asarray(WAVE_C, dtype=np.float32),
                    "use_cue": use_cue,
                    "rich": rich,
                    "results": out,
                },
                f,
            )

    if workers <= 1:
        for k, gi in enumerate(idxs):
            r = map_chi2_one(int(gi), use_cue=use_cue, cont_nseeds=cns, full_nseeds=fns, maxfev=mf)
            out.append(r)
            s = f"chi2_red {r['chi2_red']:.2f}" if r.get("status") == "ok" else r.get("status")
            print(f"[{k + 1}/{N}] idx {gi}  {s}")
            if (k + 1) % 50 == 0:  # incremental full-pkl checkpoint -> survive crash/interrupt
                _checkpoint()
                print(f"  [checkpoint {k + 1}/{N} -> {full_pkl.name}]", flush=True)
    else:
        tag0 = "cue" if use_cue else "fsps"
        ckpt = RESULTS_PATH / f"map_chi2_{tag0}_{modetag}.npy"
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_work, (int(gi), use_cue, cns, fns, mf)): int(gi) for gi in idxs}
            for k, fut in enumerate(as_completed(futs)):
                r = fut.result()
                out.append(r)
                s = f"chi2_red {r['chi2_red']:.2f}" if r.get("status") == "ok" else r.get("status")
                print(f"[{k + 1}/{N}] idx {r.get('gidx')}  {s}", flush=True)
                if (k + 1) % 25 == 0:  # incremental checkpoint -> survive crash/interrupt
                    np.save(
                        ckpt,
                        np.array(
                            [
                                (
                                    rr.get("id", -1),
                                    rr.get("z", np.nan),
                                    rr.get("chi2_red", np.nan),
                                    rr.get("gidx", -1),
                                )
                                for rr in out
                            ]
                        ),
                    )
                    _checkpoint()  # also dump full pkl (theta + spectra) so a crash isn't total loss

    ok = [r for r in out if r.get("status") == "ok"]
    if ok:
        c = np.array([r["chi2_red"] for r in ok])
        print(f"\n=== {len(ok)}/{N} fit  (cue={use_cue}) ===")
        print(f"chi2_red p16,50,84,95,99 = {np.percentile(c, [16, 50, 84, 95, 99]).round(2)}")
        print(
            f"frac chi2_red > 2: {100 * np.mean(c > 2):.1f}%   > 3: {100 * np.mean(c > 3):.1f}%   > 5: {100 * np.mean(c > 5):.1f}%"
        )
    fname = f"map_chi2_{tag}_{suffix}.npy"  # tag/suffix/full_pkl computed before the loop
    np.save(
        RESULTS_PATH / fname,
        np.array(
            [
                (
                    r.get("id", -1),
                    r.get("z", np.nan),
                    r.get("chi2_red", np.nan),
                    r.get("gidx", -1),
                )
                for r in out
            ]
        ),
    )
    _checkpoint()  # final full pkl (theta + MAP model spectrum + data)
    print(f"saved -> {fname}  and  {full_pkl.name} (theta + model spectra)")


if __name__ == "__main__":
    main()
