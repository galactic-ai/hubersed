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

import numpy as np
from prospect.fitting import lnprobfn

from hubersed.conversion import flambda_to_maggies, ivar_flambda_to_ivar_maggies
from hubersed.fitting.config import build_continuum_model, build_full_cue_model, build_full_model
from hubersed.paths import PATHS
from hubersed.prospector import parameter_file as P
from hubersed.prospector.rebin import common_obs_edges

DATA_PATH = PATHS["DATA"]
RESULTS_PATH = PATHS["RESULTS"]
WAVE_OBS = P.WAVE_OBS
CHUNK = 1024
N_TOTAL = 254976
Z_FLOOR = 0.01

EDGES = common_obs_edges()
WAVE_C = (0.5 * (EDGES[1:] + EDGES[:-1])).astype(np.float32)  # coarse centers, for the checkpoint


from scipy.optimize import minimize


def _map_optimize(neg, theta_init, n_seeds=3, jitter=0.03, maxfev=20_000):
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

    Returns
    -------
    scipy.optimize.OptimizeResult or None
        The run with the lowest objective, or None if no run finished below 1e10.

    Notes
    -----
    The jitter for start ``s`` is drawn with ``np.random.default_rng(s)``, so every galaxy
    gets the same offsets. The value 1e18 is finite, so a plain ``np.isfinite`` check would
    not catch it, which is why starts at or above 1e17 are treated as invalid.

    If a jittered start is invalid, Powell runs again from ``theta_init`` instead. The
    number of runs stays at ``n_seeds + 1``, but the number of distinct starting points
    then varies from galaxy to galaxy. We plan to change this to redraw until the start is
    valid. ``tests/test_equal_budget.py`` pins the current behaviour.
    """
    starts = [theta_init] + [
        theta_init + np.random.default_rng(s).normal(0, jitter, theta_init.shape)
        for s in range(n_seeds)
    ]
    best = None
    for st in starts:
        if not np.isfinite(neg(st)) or neg(st) >= 1e17:
            st = theta_init  # fall back to the known-valid init
            if neg(st) >= 1e17:
                continue  # init itself bad -> skip seed
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
    from hubersed.prospector.lsf import C_KMS, desi_resolution

    R = desi_resolution(WAVE_OBS)
    return (C_KMS / (2.355 * R)).astype(np.float64)


def load_by_index(gidx):
    """Load one DESI spectrum by its global index and undo the spender normalization.

    Parameters
    ----------
    gidx : int
        Global index. The spectrum is row ``gidx % 1024`` of the file
        ``DESIchunk1024_{gidx // 1024}.pkl``, counting chunks in numeric order.

    Returns
    -------
    spec : np.ndarray
        Flux density f_lambda in units of 1e-17 erg/s/cm^2/A, on ``WAVE_OBS``.
    ivar : np.ndarray
        Inverse variance of ``spec``, in (1e-17 erg/s/cm^2/A)^-2.
    z : float
        Redshift.
    tid : int
        DESI TARGETID. Callers should check it matches the galaxy they asked for.

    Notes
    -----
    spender reads the same files in string order (0, 1, 10, 100, ...), so a row in an
    encoder output file is not a global index. Go from TARGETID to index with
    ``tids_to_indices``. ``tests/test_targetid.py`` pins both orders.
    """
    chunk, row = gidx // CHUNK, gidx % CHUNK
    with open(DATA_PATH / "desi_spectra" / f"DESIchunk1024_{chunk}.pkl", "rb") as f:
        s, w, z, tid, norm, *_ = pickle.load(f)
    s = s * norm[:, None]
    w = w / norm[:, None] ** 2  # un-normalize
    to = lambda x: x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    s, w, z, tid = to(s)[row], to(w)[row], float(to(z)[row]), int(to(tid)[row])
    return s, w, z, tid


def tids_to_indices(tids):
    """Find the global index of each TARGETID.

    Parameters
    ----------
    tids : np.ndarray
        TARGETIDs as int64.

    Returns
    -------
    np.ndarray
        Global indices for ``load_by_index``, in the same order as ``tids``.

    Raises
    ------
    ValueError
        If any TARGETID is missing from ``all_target_ids.npy``. Nothing is dropped.

    Notes
    -----
    ``all_target_ids.npy`` lists TARGETIDs in numeric chunk order, the order
    ``load_by_index`` uses. This was checked against all 249 chunk files on 2026-09-22.
    """
    all_tids = np.load(DATA_PATH / "all_target_ids.npy").astype(np.int64)
    order = np.argsort(all_tids)
    sa = all_tids[order]
    pos = np.clip(np.searchsorted(sa, tids), 0, len(sa) - 1)
    ok = sa[pos] == tids
    if not ok.all():
        raise ValueError(
            f"{int((~ok).sum())}/{len(tids)} TARGETIDs not found in all_target_ids.npy"
        )
    return order[pos].astype(int)


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
    ``theta_labels`` holds ``free_params``, one name per parameter, while ``theta`` has
    one entry per value. ``logsfr_ratios`` has 9 values, so the two lists do not line up
    by position. Read values from ``theta_dict``. ``tests/test_theta_dict.py`` shows the
    shift.
    """
    try:
        spec, ivar, redshift, tid = load_by_index(gidx)
    except Exception as e:
        return dict(gidx=gidx, status=f"load_fail:{type(e).__name__}")
    if redshift < Z_FLOOR:
        return dict(gidx=gidx, id=tid, z=redshift, status="below_zfloor")

    spec_maggies = flambda_to_maggies(WAVE_OBS, spec)
    ivar_maggies = ivar_flambda_to_ivar_maggies(WAVE_OBS, ivar)
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
    theta_dict = {
        k: np.asarray(theta_map[v], dtype=np.float32) for k, v in fmodel.theta_index.items()
    }  # keyed by name (multi-elem safe)
    return dict(
        gidx=gidx,
        id=tid,
        z=redshift,
        status="ok",
        chi2=chi2,
        ndof=ndof,
        chi2_red=chi2 / ndof,
        npix=int(m.sum()),
        theta=np.asarray(theta_map, dtype=np.float32),  # flat vector
        theta_labels=list(fmodel.free_params),
        theta_dict=theta_dict,  # {param_name: value(s)}
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
