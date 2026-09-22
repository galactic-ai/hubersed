import sys, pickle, warnings, csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from prospect.fitting import lnprobfn
from prospect.sources import SSPBasis

from hubersed.paths import PATHS
from hubersed.fitting.config import build_continuum_model
from hubersed.fitting.chi2 import load_by_index, tids_to_indices, _map_optimize, WAVE_OBS
from hubersed.prospector.parameter_file import build_obs, build_sps, mask_spectral_lines
from hubersed.prospector.rebin import prep_spectrum, common_obs_edges, MILES_LAM_MIN, MILES_LAM_MAX
from hubersed.prospector.lsf import desi_resolution, C_KMS
from hubersed.conversion import flambda_to_maggies, ivar_flambda_to_ivar_maggies

SAMPLE = PATHS["RESULTS"] / "oldnew_sample.npz"
OUT    = "./oldnew_results.pkl"
EDGES  = common_obs_edges()
LSF    = (C_KMS / (2.355 * desi_resolution(WAVE_OBS))).astype(np.float64)   # DESI LSF sigma [km/s]
KEYS   = ("logzsol", "logmass", "sigma_smooth")

# per-worker SPS, built once by the pool initializer (see main()).
_SPS = None


def _emlines(sps):
    fw = sps.ssp.emline_wavelengths
    return fw[(fw > 3600) & (fw < 9824)]


def _fit(model, obs, sps):
    def neg(th):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lp = lnprobfn(th, model=model, observations=obs, sps=sps, nested=False)
                return -lp if np.isfinite(lp) else 1e18
            except Exception:
                return 1e18
    best = _map_optimize(neg, model.theta.copy(), n_seeds=2, maxfev=8000)
    if best is None:
        return {k: np.nan for k in KEYS}
    return {k: float(best.x[model.theta_index[k]][0]) for k in KEYS}


def _load(idx):
    spec, ivar, z, tid = load_by_index(idx)
    fm = flambda_to_maggies(WAVE_OBS, spec)
    iv = ivar_flambda_to_ivar_maggies(WAVE_OBS, ivar)
    ok = (iv > 0) & np.isfinite(fm)
    return fm, np.where(ok, iv, 0.0), ok, z, tid


def fit_new(idx, sps):
    fm, iv, ok, z, tid = _load(idx)
    wave_c, flux_c, ivar_c, good_c = prep_spectrum(WAVE_OBS, fm, iv, z, EDGES)
    sig_c = 1.0 / np.sqrt(np.where(ivar_c > 0, ivar_c, np.inf))
    m = mask_spectral_lines(wave_c, good_c, z, halfwidth_kms=1500.0, line_waves=_emlines(sps))
    obs = build_obs(spec=flux_c, unc=sig_c, mask=m, resolution=None, wavelength=wave_c)
    model, _ = build_continuum_model(z)
    return _fit(model, obs, sps), z, tid


def fit_old(idx, sps):
    fm, iv, ok, z, tid = _load(idx)
    sig = 1.0 / np.sqrt(np.where(iv > 0, iv, np.inf))
    # restrict OLD to the SAME MILES window NEW uses (rest 3750-7200 A) so the ONLY
    # difference between the two fits is the resolution handling, not which pixels
    # got fit (NEW drops the BaSeL red/blue where the model is only R~200; see issue #15).
    lam_rest = WAVE_OBS / (1.0 + z)
    inwin = (lam_rest >= MILES_LAM_MIN) & (lam_rest <= MILES_LAM_MAX)
    m = mask_spectral_lines(WAVE_OBS, ok & inwin, z, halfwidth_kms=1500.0, line_waves=_emlines(sps))
    obs = build_obs(spec=fm, unc=sig, mask=m, resolution=LSF, wavelength=WAVE_OBS)   # LSF onto the model
    model, _ = build_continuum_model(z)
    return _fit(model, obs, sps)


# ----------------------------------------------------------------------------
# worker-pool plumbing
# ----------------------------------------------------------------------------
def _init_new():
    """NEW pool: build sps with the library resolution INTACT (the fix)."""
    global _SPS
    _SPS = build_sps()


def _init_old():
    """OLD pool: zero the library resolution in THIS worker, then build sps (the bug)."""
    global _SPS
    SSPBasis.spectral_resolution = property(lambda self: np.zeros_like(self.ssp.wavelengths))
    _SPS = build_sps()


def _do_new(task):
    label, idx, tid_e = task
    try:
        res, z, tid = fit_new(int(idx), _SPS)
        return label, int(idx), int(tid), float(z), res
    except Exception:
        return label, int(idx), int(tid_e), float("nan"), {k: np.nan for k in KEYS}


def _do_old(idx):
    try:
        return fit_old(int(idx), _SPS)
    except Exception:
        return {k: np.nan for k in KEYS}


def main():
    n = None
    workers = 1
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])

    s = np.load(SAMPLE)
    bins = {"SF_lowM": s["sf_tids"], "QU_highM": s["qu_tids"]}

    tasks = []
    for label, tids in bins.items():
        tids = tids[:n] if n else tids
        tids = np.asarray(tids, np.int64)
        idxs = tids_to_indices(tids)
        for tid_e, idx in zip(tids, idxs):
            tasks.append((label, int(idx), int(tid_e)))
    print(f"[run] n={n} workers={workers} ntasks={len(tasks)}", flush=True)

    ctx = mp.get_context("spawn")

    # --- NEW fits: one pool, library resolution intact ---
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_new) as ex:
        new_out = list(ex.map(_do_new, tasks))
    rows = []
    for k, (label, idx, tid, z, res) in enumerate(new_out):
        rows.append(dict(bin=label, tid=tid, z=z, idx=idx, new=res))
        print(f"[NEW {label} {k+1}] logzsol={res['logzsol']:+.2f} sigma={res['sigma_smooth']:.0f}", flush=True)

    # --- OLD fits: fresh pool, library resolution zeroed in each worker ---
    idx_list = [r["idx"] for r in rows]
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_old) as ex:
        old_out = list(ex.map(_do_old, idx_list))
    for r, old in zip(rows, old_out):
        r["old"] = old
        print(f"[OLD {r['bin']}] logzsol={old['logzsol']:+.2f} sigma={old['sigma_smooth']:.0f}", flush=True)

    # binary pkl (local convenience only -- corrupts if dragged over the device bridge)
    with open(OUT, "wb") as f:
        pickle.dump(rows, f)
    # text CSV -- survives the bridge; this is the artifact to keep / stage for plotting.
    csv_path = PATHS["RESULTS"] / "oldnew_results.csv"
    with open(csv_path, "w", newline="") as fcsv:
        wc = csv.writer(fcsv)
        wc.writerow(["bin", "tid", "z", "idx",
                     "lz_new", "lz_old", "lm_new", "lm_old", "sig_new", "sig_old"])
        for r in rows:
            wc.writerow([r["bin"], r["tid"], r["z"], r["idx"],
                         r["new"]["logzsol"], r["old"]["logzsol"],
                         r["new"]["logmass"], r["old"]["logmass"],
                         r["new"]["sigma_smooth"], r["old"]["sigma_smooth"]])
    print(f"also wrote {csv_path}")

    # --- summary ---
    print(f"\n{'bin':>9} | {'N':>3} | {'sig_new':>7} {'sig_old':>7} | {'d_logzsol(new-old)':>18} | {'d_sigma':>8}")
    for label in bins:
        sub = [r for r in rows if r["bin"] == label
               and np.isfinite(r["new"]["logzsol"]) and np.isfinite(r["old"]["logzsol"])]
        if not sub:
            print(f"{label:>9} | (no fits)")
            continue
        sn = np.median([r["new"]["sigma_smooth"] for r in sub])
        so = np.median([r["old"]["sigma_smooth"] for r in sub])
        dlz = np.median([r["new"]["logzsol"] - r["old"]["logzsol"] for r in sub])
        dsg = np.median([r["new"]["sigma_smooth"] - r["old"]["sigma_smooth"] for r in sub])
        print(f"{label:>9} | {len(sub):>3} | {sn:7.0f} {so:7.0f} | {dlz:+18.3f} | {dsg:+8.0f}")
    print(f"\nExpect: SF_lowM shows OLD sigma railed low + a nonzero d_logzsol; "
          f"QU_highM ~ 0 (control).\nsaved -> {OUT}  and  {csv_path}")


if __name__ == "__main__":
    main()
