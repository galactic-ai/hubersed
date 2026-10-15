"""MAP fits of injected mocks of 39627770174637084, to measure the bias the line model causes.

What changed from 2026-10-13_nautilus_resolution_matrix/map.py: the data are replaced by noiseless
mocks (mocks.py) plus Gaussian noise with the corrected uncertainties, and each mock is fitted
with three line models from three starts. The pipeline, pixels, mask, priors, optimizer (map_fit,
one Powell start per call, maxfev 40000) and the SFH hyperparameters fixed at the 10-09 max-L are
those of map.py rmatrix mask.

Jobs. A job is (scenario, noise seed, fit model, start).

- scenario S0, S1, S2, S3 of injection.py.
- noise seed -1 is the noiseless mock. Seed s >= 0 adds noise drawn with
  numpy.random.default_rng(NOISE_BASE + s), the same in every scenario.
- fit model shared, free or none (injection.make_fit_params).
- start truth (the truth point), maxl (the 10-09 max-L point, map.py's start) or jitter (truth
  jittered with injection.jitter_start, seed JITTER_BASE + s). A free forbidden fraction starts at
  the start's eline_fbroad.

Workers claim jobs in order with an exclusive lock file, so any number can run at once and a
restart skips jobs already done. S0 and S1 come first in the job list.

COMMAND: from the repository root on the Mac with the C3K_ER build, after mocks.py, start N
workers (6 were used) with
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_broadline_injection/fit.py run --seeds -1 0 1 2 3 4``
once per worker, and ``... fit.py status`` to count done jobs. ``--scenarios``, ``--models`` and
``--starts`` limit the jobs. ``--unmask "[SII]6731"`` also fits the pixels model_setup.UNMASK_LINES
restores and writes to separate folders (the 2026-10-14 run used S0 S1 S3, shared, truth maxl).

OUTPUT: fits/<scenario>_<seed>_<model>_<start>.npz in results/2026-10-13_broadline_injection/, or
fits_unmask_<lines>/ with ``--unmask``.
"""

import argparse
import os
import socket
import time
import warnings

import numpy as np
from injection import (
    FIT_MODELS,
    OUT,
    SCENARIOS,
    jitter_start,
    make_fit_params,
    noise,
    prior_bounds,
    setup,
    theta_for,
)
from model_setup import get_cue, make_obs
from prospect.fitting import lnprobfn

from hubersed.fitting.map_fits import map_fit
from hubersed.sps.resolution_model import ResolutionMatrixModel

NOISE_BASE = 1000
JITTER_BASE = 7000
STARTS = ("truth", "maxl", "jitter")
MAXFEV = 40000
FITS = OUT / "fits"
CLAIMS = OUT / "claims"
MAGGIE_TO_1E17 = 3631e-23 / 1e-17
KEY_LINES = {
    "Halpha": ("Ba-alpha 6563",),
    "Hbeta": ("Ba-beta 4861",),
    "[OIII]5007": ("[O III] 5007",),
    "[NII]6584": ("[N II] 6584",),
    "[SII]6716+6731": ("[S II] 6716", "[S II] 6731"),
    "[OII]3726+3729": ("[O II] 3726", "[O II] 3729"),
    "[OI]6300": ("[O I] 6300",),
    "[NeIII]3869": ("[Ne III] 3869",),
}
EDGE_FRAC = 1e-3  # a fitted value within this fraction of the prior range of a bound is at the edge


def job_name(scenario, seed, fit_model, start):
    """Return the file stem of one job."""
    return f"{scenario}_{seed}_{fit_model}_{start}"


def jobs(scenarios, seeds, models=FIT_MODELS, starts=STARTS):
    """Return the job list, scenario by scenario, then seed, fit model and start."""
    return [(sc, s, fm, st) for sc in scenarios for s in seeds for fm in models for st in starts]


def claim(name):
    """Return True if this process took the job, by creating its lock file exclusively."""
    if (FITS / f"{name}.npz").exists():
        return False
    try:
        fd = os.open(CLAIMS / f"{name}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.ctime()}\n".encode())
    os.close(fd)
    return True


def line_fluxes(model):
    """Return the Cue fluxes of ``KEY_LINES`` at the last prediction, in 1e-17 erg/s/cm2."""
    names = np.char.strip(np.asarray(model.emline_info["name"]).astype(str))
    elum = model._eline_lum * model.flux_norm() / (1 + model._zred) * MAGGIE_TO_1E17
    return np.array([sum(elum[names == n][0] for n in v) for v in KEY_LINES.values()])


def chi2_lnp(model, theta, obs, cue):
    """Return chi2 on the fitted pixels and -ln posterior at ``theta``, and the line fluxes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = model.predict(theta, observations=obs, sps=cue)[0][0]
        fl = line_fluxes(model)
        lp = lnprobfn(theta, model=model, observations=obs, sps=cue, nested=False)
    o = obs[0]
    chi2 = float(np.sum(((o.flux - sp)[o.mask] / o.uncertainty[o.mask]) ** 2))
    return chi2, -float(lp), fl


def run_job(job, s, cue, mocks, models):
    """Fit one job and save its result."""
    sc, seed, fm, st = job
    name = job_name(*job)
    mock = mocks[list(SCENARIOS).index(sc)]
    flux = mock + (noise(s["unc"], s["good"], NOISE_BASE + seed) if seed >= 0 else 0.0)
    flux = np.where(s["good"], flux, mock)
    obs = make_obs(flux, s["unc"], s["good"], "rmatrix")
    model = models[fm]
    t_truth = theta_for(model, s["truth"], fill={"eline_fbroad_forb": s["truth"]["eline_fbroad"]})
    if st == "truth":
        theta0 = t_truth
    elif st == "maxl":
        theta0 = theta_for(model, s["maxl"], fill={"eline_fbroad_forb": s["maxl"]["eline_fbroad"]})
    else:
        theta0 = jitter_start(model, t_truth, seed=JITTER_BASE + seed)
    chi2_t, fun_t, fl_t = chi2_lnp(model, t_truth, obs, cue)
    chi2_0, fun_0, _ = chi2_lnp(model, theta0, obs, cue)
    t0 = time.time()
    best, info = map_fit(model, obs, cue, n_seeds=0, maxfev=MAXFEV, theta0=theta0, tag=name)
    secs = time.time() - t0
    chi2_f, fun_f, fl_f = chi2_lnp(model, best.x, obs, cue)
    assert np.isclose(fun_f, best.fun, rtol=1e-8), (fun_f, best.fun)
    lo, hi = prior_bounds(model)
    w = hi - lo
    edge = np.isfinite(w) & ((best.x - lo < EDGE_FRAC * w) | (hi - best.x < EDGE_FRAC * w))
    np.savez(
        FITS / f"{name}.npz",
        scenario=sc,
        seed=seed,
        noise_seed=NOISE_BASE + seed if seed >= 0 else -1,
        jitter_seed=JITTER_BASE + seed if st == "jitter" else -1,
        fit_model=fm,
        start=st,
        labels=np.array(model.theta_labels()),
        theta_start=theta0,
        theta_truth=t_truth,
        theta=best.x,
        fun=best.fun,
        fun_start=fun_0,
        fun_truth=fun_t,
        chi2=chi2_f,
        chi2_start=chi2_0,
        chi2_truth=chi2_t,
        npix=int(s["good"].sum()),
        nfev=best.nfev,
        success=best.success,
        seconds=secs,
        edge=edge,
        line_names=np.array(list(KEY_LINES)),
        lines_fit=fl_f,
        lines_truth=fl_t,
        maxfev=MAXFEV,
    )
    print(
        f"{name}: chi2 {chi2_f:.1f} (truth {chi2_t:.1f}), -lnp {best.fun:.2f} (truth {fun_t:.2f}),"
        f" nfev {best.nfev}, {secs:.0f} s",
        flush=True,
    )


def run(scenarios, seeds, fit_models=FIT_MODELS, starts=STARTS, unmasked_lines=()):
    """Claim and run jobs until none is left. ``unmasked_lines`` adds pixels to the fit."""
    FITS.mkdir(parents=True, exist_ok=True)
    CLAIMS.mkdir(parents=True, exist_ok=True)
    d = np.load(OUT / "mocks.npz")
    assert [str(x) for x in d["scenarios"]] == list(SCENARIOS)
    s = setup(unmasked_lines)
    # restored pixels may only add to the mocks' pixels, with the same uncertainty elsewhere
    g = d["good"]
    assert np.all(s["good"][g]) and np.array_equal(d["unc"][g], s["unc"][g])
    assert unmasked_lines or np.array_equal(g, s["good"])
    cue = get_cue()
    models = {
        fm: ResolutionMatrixModel(make_fit_params(s["z"], fm, s["hypers"])) for fm in FIT_MODELS
    }
    t_sh = theta_for(models["shared"], s["truth"])
    assert np.array_equal(t_sh, d["truth_theta"])
    for job in jobs(scenarios, seeds, fit_models, starts):
        if claim(job_name(*job)):
            run_job(job, s, cue, d["mocks"], models)


def status(scenarios, seeds, models=FIT_MODELS, starts=STARTS):
    """Print how many jobs are done, claimed but not done, and left."""
    done = claimed = left = 0
    for job in jobs(scenarios, seeds, models, starts):
        n = job_name(*job)
        if (FITS / f"{n}.npz").exists():
            done += 1
        elif (CLAIMS / f"{n}.claim").exists():
            claimed += 1
        else:
            left += 1
    print(f"done {done}, running or stale {claimed}, left {left}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("run", "status"))
    ap.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[-1, 0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", choices=FIT_MODELS, default=list(FIT_MODELS))
    ap.add_argument("--starts", nargs="+", choices=STARTS, default=list(STARTS))
    ap.add_argument("--unmask", nargs="+", default=[], help="UNMASK_LINES names, e.g. [SII]6731")
    a = ap.parse_args()
    if a.unmask:
        # separate output, so these fits are not mixed with the masked ones
        tag = "_".join(n.replace("[", "").replace("]", "").lower() for n in a.unmask)
        FITS, CLAIMS = OUT / f"fits_unmask_{tag}", OUT / f"claims_unmask_{tag}"
    if a.action == "run":
        warnings.filterwarnings("ignore")
        np.seterr(all="ignore")
        run(a.scenarios, a.seeds, a.models, a.starts, tuple(a.unmask))
    else:
        status(a.scenarios, a.seeds, a.models, a.starts)
