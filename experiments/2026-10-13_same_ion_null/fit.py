"""MAP fit and profiles of the same-ion null test of the second-component fractions.

Changed from fit.py of 2026-10-13_oiii_second_component: [O III] 4959 and 5007 and [N II] 6548 and
6584 each get their own second-component fraction (same_ion.py). Data, pixels, mask, resolution
matrix, Cue with C3K_ER and the SFH hyperparameters fixed at the 10-09 max-L are unchanged.

Actions.

- ``check`` sets every per-line fraction to its group value at the best F1 point and asserts that
  the spectrum equals the F1 model's and the chi2 equals the saved F1 chi2.
- ``run`` runs ``map_fit`` (Powell, one start) from that point and saves fit_run.npz.
- ``rerun`` reruns ``map_fit`` from the best saved fit and saves fit_rerun<k>.npz.
- ``profile NAME V [V ...]`` fixes the per-line fraction NAME at each value V, re-optimizes the
  other per-line fractions, the line profile parameters and the gas parameters from the best fit
  with every other parameter held there, and saves profile_<NAME>_<V>.npz.

COMMAND: from the repository root on the Mac with the C3K_ER build, at most two at once, with
``P=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix:experiments/2026-10-13_oiii_second_component:experiments/2026-10-13_same_ion_null``,
``PYTHONPATH=$P SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_same_ion_null/fit.py check``, then ``... run``, ``... rerun`` until it
does not move, then ``... profile eline_fbroad_oiii5007 0.1 0.125 ...`` for each per-line fraction.

OUTPUT: fit_*.npz and profile_*.npz in results/2026-10-13_same_ion_null/.
"""

import argparse
import time
import warnings

import numpy as np
from oiii_setup import build as oiii_build
from prospect.fitting import lnprobfn
from same_ion import GAS, LINE_PROFILE, OUT, PER_LINE, build, chi2_parts, f1_best, start
from scipy.optimize import minimize

from hubersed.fitting.map_fits import map_fit

MAXFEV = 40000
PROFILE_MAXFEV = 8000


def save(path, model, obs, cue, theta0, res, t, start_name):
    """Save one optimizer result with its chi2 per window."""
    _, parts = chi2_parts(model, obs, cue, res.x)
    np.savez(
        path,
        labels=np.array(model.theta_labels()),
        theta0=theta0,
        theta=res.x,
        fun=res.fun,
        nfev=res.nfev,
        success=res.success,
        chi2=parts["all"],
        windows=np.array(list(parts)),
        window_chi2=np.array(list(parts.values())),
        start=start_name,
        seconds=t,
    )
    print(
        f"{path.name}: -lnp {res.fun:.3f}, chi2 {parts['all']:.2f}, nfev {res.nfev}, {t:.0f} s",
        flush=True,
    )


def check():
    """Assert that the per-line model at group values reproduces the best F1 fit."""
    model, obs, cue = build()
    theta = start(model)
    spec, parts = chi2_parts(model, obs, cue, theta)
    p, name, _ = f1_best()
    f1_model, f1_obs, _ = oiii_build("F1")
    f1_theta = np.array([p[k] for k in f1_model.theta_labels()], dtype=float)
    f1_spec = f1_model.predict(f1_theta, observations=f1_obs, sps=cue)[0][0]
    saved = float(np.load(OUT.parent / "2026-10-13_oiii_second_component" / f"{name}.npz")["chi2"])
    diff = np.max(np.abs(spec - f1_spec) / np.maximum(np.abs(f1_spec), 1e-30))
    print(
        f"start from {name}: chi2 {parts['all']:.4f}, saved F1 chi2 {saved:.4f}, max rel {diff:.2e}"
    )
    assert np.allclose(spec, f1_spec, rtol=1e-12, atol=0)
    assert np.isclose(parts["all"], saved, rtol=1e-9)
    print("OK: per-line fractions at their group values reproduce F1")


def best_saved():
    """Return the name and contents of the saved fit with the lowest -ln p."""
    runs = {p.stem: dict(np.load(p)) for p in sorted(OUT.glob("fit_*.npz"))}
    name = min(runs, key=lambda k: float(runs[k]["fun"]))
    return name, runs[name]


def run(maxfev):
    """Run map_fit from the F1 best point with per-line fractions at their group values."""
    model, obs, cue = build()
    OUT.mkdir(parents=True, exist_ok=True)
    theta0 = start(model)
    t0 = time.time()
    res, _ = map_fit(model, obs, cue, n_seeds=0, maxfev=maxfev, theta0=theta0, tag="run")
    save(OUT / "fit_run.npz", model, obs, cue, theta0, res, time.time() - t0, "F1 best")


def rerun(maxfev):
    """Rerun map_fit from the best saved fit."""
    model, obs, cue = build()
    name, r = best_saved()
    k = 1 + len(list(OUT.glob("fit_rerun*.npz")))
    t0 = time.time()
    res, _ = map_fit(model, obs, cue, n_seeds=0, maxfev=maxfev, theta0=r["theta"], tag="rerun")
    save(OUT / f"fit_rerun{k}.npz", model, obs, cue, r["theta"], res, time.time() - t0, name)
    print(f"from {name} at -lnp {float(r['fun']):.3f}: moved by {res.fun - float(r['fun']):+.3f}")


def profile(which, values, maxfev):
    """Fix the per-line fraction ``which`` at each value and re-optimize the rest of the lines."""
    assert which in PER_LINE
    model, obs, cue = build()
    name, r = best_saved()
    labels = model.theta_labels()
    ref = float(r["fun"])
    ifix = labels.index(which)
    free = (*LINE_PROFILE, *(k for k in PER_LINE if k != which), *GAS)
    sub = [labels.index(k) for k in free]
    assert ifix not in sub

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
        base[ifix] = v
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
            OUT / f"profile_{which}_{v:.3f}.npz", model, obs, cue, base, res, time.time() - t0, name
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("check", "run", "rerun", "profile"))
    ap.add_argument("args", nargs="*", help="for profile: the per-line fraction and its values")
    ap.add_argument("--maxfev", type=int, default=None)
    a = ap.parse_args()
    if a.action == "check":
        check()
    elif a.action == "run":
        run(a.maxfev or MAXFEV)
    elif a.action == "rerun":
        rerun(a.maxfev or MAXFEV)
    else:
        profile(a.args[0], [float(v) for v in a.args[1:]], a.maxfev or PROFILE_MAXFEV)
