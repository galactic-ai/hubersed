import argparse
import os
import pickle
import subprocess
import sys
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np

np.seterr(divide="ignore", invalid="ignore", over="ignore", under="ignore")

import dynesty
from dynesty.utils import resample_equal

from prospect.fitting import lnprobfn
from prospect.models.sedmodel import HyperSpecModel
from prospect.sources import SSPBasis

from hubersed.conversion import flambda_to_maggies, ivar_flambda_to_ivar_maggies
from hubersed.fitting.chi2 import WAVE_OBS, load_by_index, tids_to_indices
from hubersed.fitting.config import build_continuum_model, build_full_cue_model
from hubersed.paths import PATHS
from hubersed.prospector.lsf import C_KMS, desi_resolution
from hubersed.prospector.parameter_file import build_cue_sps, build_obs, build_sps, mask_spectral_lines
from hubersed.prospector.utils import universe_age_gyr

LSF = (C_KMS / (2.355 * desi_resolution(WAVE_OBS))).astype(np.float64)
FROZEN_HYPERS = {"sigma_reg": 1.5, "sigma_dyn": 0.1, "tau_eq": 2.5, "tau_dyn": 0.025}


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_one(tid, args, out):
    idx = int(tids_to_indices(np.array([tid], np.int64))[0])
    spec, ivar, z, tid_chk = load_by_index(idx)
    assert int(tid_chk) == tid, f"TARGETID mismatch: asked {tid}, got {tid_chk}"

    flux = flambda_to_maggies(WAVE_OBS, spec)
    iv = ivar_flambda_to_ivar_maggies(WAVE_OBS, ivar)
    mask = (iv > 0) & np.isfinite(flux)
    iv = np.where(mask, iv, 0.0)
    unc = 1.0 / np.sqrt(np.where(iv > 0, iv, np.inf))
    obs = build_obs(spec=flux, unc=unc, mask=mask, resolution=LSF, wavelength=WAVE_OBS)

    SSPBasis.spectral_resolution = property(lambda self: np.zeros_like(self.ssp.wavelengths))
    sps, cue_sps = build_sps(), build_cue_sps()
    line_waves = sps.ssp.emline_wavelengths
    line_waves = line_waves[(line_waves > 3600) & (line_waves < 9824)]
    line_pix = mask & ~mask_spectral_lines(WAVE_OBS, mask, z, halfwidth_kms=1500.0,
                                           line_waves=line_waves)

    cont_model, cont_tmpl = build_continuum_model(z)
    model, tmpl = build_full_cue_model(cont_tmpl, cont_model.theta, cont_model, z)
    if args.freeze_hypers:
        for k, v in dict(FROZEN_HYPERS, tau_in=universe_age_gyr(z)).items():
            tmpl[k]["isfree"] = False
            tmpl[k]["init"] = float(v)
        model = HyperSpecModel(tmpl)
    assert model._need_lines, "analytic eline path off; set nebemlineinspec=False"

    labels = list(model.theta_labels())
    ndim = len(labels)
    ordered = args.order_taus and ("tau_eq" in labels) and ("tau_in" in labels)
    i_eq, i_in = (labels.index("tau_eq"), labels.index("tau_in")) if ordered else (None, None)

    def ptform(u):
        t = model.prior_transform(u)
        if ordered:
            a, b = t[i_eq], t[i_in]
            t[i_eq], t[i_in] = min(a, b), max(a, b)
        return t

    def loglike(x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ll = lnprobfn(x, model=model, observations=obs, sps=cue_sps, nested=True)
                return float(ll) if np.isfinite(ll) else -1e300
            except Exception:
                return -1e300

    nlive = args.nlive or 10 * ndim * (1 if ordered else 2)
    ckpt = str(out / f"{tid}_dynesty.save")
    print(f"  {tid}: ndim={ndim} nlive={nlive} ordered={ordered} "
          f"sample={args.sample} bound={args.bound}", flush=True)

    # dynesty checkpoints by pickling the whole sampler, which closes over loglike ->
    # model, obs and the FSPS/Cue SPS objects. Stdlib pickle cannot do local functions
    # at all, and the Fortran-backed SPS may defeat dill too. checkpoint_every defaults
    # to 60 s, so a failure here would kill the run a minute in. Decide up front.
    use_ckpt = False
    if not args.no_checkpoint:
        try:
            import dill

            dynesty.utils.pickle_module = dill
            dill.dumps((loglike, ptform))
            use_ckpt = True
        except Exception as e:
            print(f"  {tid}: checkpointing DISABLED ({type(e).__name__}: {e}); "
                  f"a killed run will restart from scratch", flush=True)

    t0 = time.time()
    run_kw = dict(dlogz=args.dlogz, maxcall=args.maxcall, print_progress=True)
    if use_ckpt:
        run_kw.update(checkpoint_file=ckpt, checkpoint_every=args.checkpoint_every)

    if use_ckpt and Path(ckpt).exists():
        print(f"  {tid}: resuming from {ckpt}", flush=True)
        ds = dynesty.NestedSampler.restore(ckpt)
        ds.run_nested(resume=True, **run_kw)
    else:
        ds = dynesty.NestedSampler(loglike, ptform, ndim, nlive=nlive, bound=args.bound,
                                   sample=args.sample,
                                   rstate=np.random.default_rng(args.seed))
        ds.run_nested(**run_kw)
    res = ds.results
    secs = time.time() - t0

    wt = np.exp(res.logwt - res.logz[-1])
    eq = resample_equal(res.samples, wt / wt.sum())
    best = res.samples[np.argmax(res.logl)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds, _ = model.predict(best, observations=obs, sps=cue_sps)
    sp = np.asarray(preds[0], float)
    m = mask & np.isfinite(sp) & (unc > 0)
    chi2 = float(np.sum(((flux[m] - sp[m]) / unc[m]) ** 2))

    rec = {
        "target_id": int(tid), "z": float(z), "status": "ok",
        "labels": labels, "ndim": ndim,
        "eq_samples": eq, "logz": float(res.logz[-1]), "logzerr": float(res.logzerr[-1]),
        "logl_max": float(res.logl.max()), "theta_max_logl": best,
        "niter": int(res.niter), "ncall": int(np.sum(res.ncall)), "seconds": secs,
        "model": sp, "wave": WAVE_OBS, "flux": flux, "unc": unc, "mask": mask,
        "line_pix": line_pix,
        "stats": {"chi2": chi2, "chi2_red": chi2 / max(int(m.sum()) - ndim, 1),
                  "npix": int(m.sum()), "ntheta": ndim},
        "config": {"nlive": nlive, "bound": args.bound, "sample": args.sample,
                   "dlogz": args.dlogz, "maxcall": args.maxcall, "seed": args.seed,
                   "ordered_taus": ordered, "hypers": "frozen" if args.freeze_hypers else "free",
                   "nebular": "cue_stellar_nebular", "lsf": "median desi_resolution",
                   "git_sha": git_sha()},
    }
    with open(out / f"{tid}.pkl", "wb") as f:
        pickle.dump(rec, f)
    print(f"  {tid}: done  logz={rec['logz']:.1f}+/-{rec['logzerr']:.1f}  "
          f"chi2_red={rec['stats']['chi2_red']:.3f}  ncall={rec['ncall']:.3g}  "
          f"{secs / 3600:.2f} h", flush=True)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description="dynesty posteriors for the emission-line outliers.")
    p.add_argument("-t", "--tid", type=int, action="append", required=True,
                   help="repeatable")
    p.add_argument("-o", "--outdir", default=str(PATHS["RESULTS"] / "emline_dynesty"))
    p.add_argument("--nlive", type=int, default=None, help="default 10*ndim per mode")
    p.add_argument("--sample", default="rslice")
    p.add_argument("--bound", default="multi")
    p.add_argument("--dlogz", type=float, default=1.0)
    p.add_argument("--maxcall", type=int, default=5_000_000,
                   help="~17 h at 12.5 ms/call; dlogz should stop the run well before this")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--checkpoint-every", type=float, default=900.0, help="seconds")
    p.add_argument("--order-taus", action="store_true",
                   help="impose tau_eq < tau_in, breaking the exact exchange symmetry")
    p.add_argument("--freeze-hypers", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for tid in args.tid:
        if args.skip_existing and (out / f"{tid}.pkl").exists():
            print(f"  {tid}: exists, skipping", flush=True)
            continue
        run_one(int(tid), args, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
