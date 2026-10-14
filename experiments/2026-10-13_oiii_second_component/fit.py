"""MAP fits of 39627770174637084 with a separate second-component fraction for [O III].

Changed from map_fbfree.py of 2026-10-13_nautilus_resolution_matrix: the high-ionization lines
([O III], [Ne III], [Ar IV], [Ne IV], He II) get their own narrow width ``eline_sigma_forb_hi``
and their own second-component fraction ``eline_fbroad_forb_hi`` (fixed at 0 in F0, free in F1).
The other non-Balmer lines keep ``eline_sigma_forb`` and a free ``eline_fbroad_forb``. The data,
pixels, missing-line mask, resolution matrix, Cue with C3K_ER and the SFH hyperparameters fixed at
the 10-09 max-L are those of the rmatrix mask run of map.py there (see oiii_setup.py).

Actions.

- ``run FIT START [START ...]`` runs ``map_fit`` (Powell, one start, no jitter) from each named
  start of ``oiii_setup.starts`` and saves fit_<FIT>_<START>.npz.
- ``rerun FIT`` reruns ``map_fit`` from the best saved point of FIT, to check that Powell did not
  stop early, and saves fit_<FIT>_rerun<k>.npz.
- ``profile {hi,lo} V [V ...]`` fixes ``eline_fbroad_forb_hi`` (hi) or ``eline_fbroad_forb``
  (lo) at each value V, re-optimizes the other line profile parameters (``oiii_setup.LINE_PROFILE``
  and ``eline_fbroad_forb_hi``) and the gas parameters (``GAS``) from the best F1 point with every
  other parameter held there, and saves profile_<hi|lo>_<V>.npz. The objective is -ln p minus the
  best F1 value, so Powell's relative tolerance acts on a number of order chi2 differences.

COMMAND: from the repository root on the Mac with the C3K_ER build, at most two at once,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix:experiments/2026-10-13_oiii_second_component
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_oiii_second_component/fit.py run F0 A B`` and ``... run F1 A0 A1 B``, then
``... rerun F0`` and ``... rerun F1`` until the rerun does not move, then
``... profile hi 0 0.05 0.1 ...`` and ``... profile lo 0 0.05 0.1 ...``. analysis.py reads the outputs.

OUTPUT: fit_*.npz and profile_*.npz in results/2026-10-13_oiii_second_component/.
"""

import argparse
import time
import warnings

import numpy as np
from oiii_setup import GAS, LINE_PROFILE, OUT, build, chi2_parts, starts
from prospect.fitting import lnprobfn
from scipy.optimize import minimize

from hubersed.fitting.map_fits import map_fit

MAXFEV = 40000
PROFILE_MAXFEV = 8000


def saved(fit):
    """Return the saved fit_<fit>_*.npz files as a dict of name to loaded arrays."""
    return {p.stem: dict(np.load(p)) for p in sorted(OUT.glob(f"fit_{fit}_*.npz"))}


def best_saved(fit):
    """Return the name and contents of the saved point of ``fit`` with the lowest -ln p."""
    runs = saved(fit)
    name = min(runs, key=lambda k: float(runs[k]["fun"]))
    return name, runs[name]


def save(path, model, obs, cue, theta0, best, maxfev, start, t):
    """Save one optimizer result with its chi2 and the time ``t`` it took in seconds."""
    _, parts, _ = chi2_parts(model, obs, cue, best.x)
    np.savez(
        path,
        labels=np.array(model.theta_labels()),
        theta0=theta0,
        theta=best.x,
        fun=best.fun,
        nfev=best.nfev,
        success=best.success,
        maxfev=maxfev,
        start=start,
        chi2=parts["all"],
        seconds=t,
    )
    print(
        f"{path.name}: -lnp {best.fun:.3f}, chi2 {parts['all']:.2f}, nfev {best.nfev}, "
        f"success {best.success}, {t:.0f} s",
        flush=True,
    )


def run(fit, names, maxfev):
    """Run map_fit from each named start of ``fit``."""
    model, obs, cue = build(fit)
    OUT.mkdir(parents=True, exist_ok=True)
    pts = starts(fit, model)
    for name in names:
        t0 = time.time()
        best, _ = map_fit(model, obs, cue, n_seeds=0, maxfev=maxfev, theta0=pts[name], tag=name)
        dt = time.time() - t0
        save(OUT / f"fit_{fit}_{name}.npz", model, obs, cue, pts[name], best, maxfev, name, dt)


def rerun(fit, maxfev):
    """Rerun map_fit from the best saved point of ``fit``."""
    model, obs, cue = build(fit)
    name, r = best_saved(fit)
    assert list(r["labels"]) == model.theta_labels()
    k = 1 + sum(n.startswith(f"fit_{fit}_rerun") for n in saved(fit))
    t0 = time.time()
    best, _ = map_fit(model, obs, cue, n_seeds=0, maxfev=maxfev, theta0=r["theta"], tag="rerun")
    dt = time.time() - t0
    save(OUT / f"fit_{fit}_rerun{k}.npz", model, obs, cue, r["theta"], best, maxfev, name, dt)
    print(f"from {name} at -lnp {float(r['fun']):.3f}: moved by {best.fun - float(r['fun']):+.3f}")


PROFILED = {"hi": "eline_fbroad_forb_hi", "lo": "eline_fbroad_forb"}


def profile(which, values, maxfev):
    """Fix one second-component fraction at each value and re-optimize the profile and gas.

    ``which`` is hi (``eline_fbroad_forb_hi``) or lo (``eline_fbroad_forb``).
    """
    model, obs, cue = build("F1")
    name, r = best_saved("F1")
    labels = model.theta_labels()
    ref = float(r["fun"])
    ihi = labels.index(PROFILED[which])
    free = (*LINE_PROFILE, "eline_fbroad_forb_hi", *GAS)
    sub = [labels.index(k) for k in free if k != PROFILED[which]]
    assert ihi not in sub and len(sub) == len(free) - 1

    def neg(x, base):
        th = base.copy()
        th[sub] = x
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lp = lnprobfn(th, model=model, observations=obs, sps=cue, nested=False)
            except Exception:
                return 1e18
        return -lp - ref if np.isfinite(lp) else 1e18

    for v in values:
        base = np.array(r["theta"], dtype=float)
        base[ihi] = v
        t0 = time.time()
        res = minimize(
            neg,
            base[sub],
            args=(base,),
            method="Powell",
            options={"maxiter": maxfev // 10, "maxfev": maxfev, "ftol": 1e-6},
        )
        th = base.copy()
        th[sub] = res.x
        res.x, res.fun = th, res.fun + ref
        save(
            OUT / f"profile_{which}_{v:.3f}.npz",
            model,
            obs,
            cue,
            base,
            res,
            maxfev,
            name,
            time.time() - t0,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("run", "rerun", "profile"))
    ap.add_argument(
        "args", nargs="+", help="FIT [START ...] for run and rerun, hi or lo and values for profile"
    )
    ap.add_argument("--maxfev", type=int, default=None)
    a = ap.parse_args()
    if a.action == "run":
        run(a.args[0], a.args[1:], a.maxfev or MAXFEV)
    elif a.action == "rerun":
        rerun(a.args[0], a.maxfev or MAXFEV)
    else:
        profile(a.args[0], [float(v) for v in a.args[1:]], a.maxfev or PROFILE_MAXFEV)
