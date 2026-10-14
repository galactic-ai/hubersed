"""Re-evaluate the parameter vectors where prospect's likelihood call raised during the fits.

prospect's lnprobfn prints "There was an error during the likelihood call at parameters [...]"
and re-raises any exception other than ValueError. map_fit then returns 1e18 for that point, so
the optimizer sees a wall. This script reads every such vector from the worker logs, finds the
job it belongs to (the next job that finishes in the same log), predicts the model there outside
the optimizer and records the exception. It then compares each point with the job's final
optimum, as a fraction of each prior range.

COMMAND: from the repository root on the Mac with the C3K_ER build, after the fits,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_broadline_injection/likelihood_errors.py``

OUTPUT: likelihood_errors.txt in results/2026-10-13_broadline_injection/.
"""

import glob
import re
import traceback
import warnings

import numpy as np
from injection import FIT_MODELS, OUT, make_fit_params, prior_bounds, setup, theta_for
from model_setup import get_cue

from hubersed.sps.resolution_model import ResolutionMatrixModel

ERR = "There was an error during the likelihood call at parameters ["
DONE = re.compile(r"^(S\d_-?\d+_(shared|free|none)_(truth|maxl|jitter)): chi2")


def parse_logs():
    """Return (log, job name or None, theta) for every error vector in the worker logs."""
    found = []
    for path in sorted(glob.glob(str(OUT / "logs" / "worker*.log"))):
        lines = open(path).read().splitlines()
        pending = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith(ERR):
                text = line[len(ERR) :]
                while "]" not in text:
                    i += 1
                    text += " " + lines[i]
                vals = np.array(text.split("]")[0].split(), float)
                pending.append(vals)
            m = DONE.match(line)
            if m:
                found += [(path, m.group(1), v) for v in pending]
                pending = []
            i += 1
        found += [(path, None, v) for v in pending]
    return found


def inst2_fails(value, nline=86):
    """Return True if ``_profile_widths`` would give a nonzero inst2 for this eline_sigma.

    It repeats the arithmetic of TwoCompLineModel._profile_widths when ResolutionMatrixModel
    sets ``_eline_sigma_kms`` to eline_sigma: an array squared by numpy minus a Python float
    squared by ``**``.
    """
    p = float(value)
    sig0 = np.full(nline, p) * np.ones(nline)
    return bool(np.any(np.clip(sig0**2 - p**2, 0, None) != 0))


def mechanism_check(model, s, cue, n=200000, seed=0):
    """Test the rounding mechanism of the AssertionError in ResolutionMatrixModel.fine_lines.

    The printed vectors carry only 9 significant digits, so they need not reproduce the exact
    failing value. This finds eline_sigma values where the arithmetic gives a nonzero inst2,
    predicts the model at the truth with one of them, and with the next float up.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(10.0, 45.0, n)
    bad = np.array([inst2_fails(v) for v in x])
    out = [
        "Mechanism. ResolutionMatrixModel.fine_lines sets _eline_sigma_kms = eline_sigma and then",
        "asserts inst2 == 0 (src/hubersed/sps/resolution_model.py, fine_lines). _profile_widths",
        "computes inst2 = clip(sig0**2 - eline_sigma**2, 0): numpy squares the array, Python's **",
        "squares the float, and the two can differ by one ulp.",
        f"   eline_sigma uniform in [10, 45]: {bad.sum()} of {n} values ({bad.mean():.3%}) give inst2 != 0",
    ]
    t = theta_for(model, s["truth"])
    i = model.theta_index["eline_sigma"].start
    v_bad = float(x[bad][0])
    for v in (v_bad, float(np.nextafter(v_bad, np.inf)), s["truth"]["eline_sigma"]):
        th = t.copy()
        th[i] = v
        try:
            model.predict(th, observations=s["obs"], sps=cue)
            res = "no exception"
        except AssertionError as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            res = f"AssertionError at {tb.filename.split('/')[-1]}:{tb.lineno} ({tb.line})"
        out.append(f"   truth with eline_sigma = {v!r}: {res}")
    return out


def main():
    """Re-evaluate every error vector and write likelihood_errors.txt."""
    s = setup()
    cue = get_cue()
    models = {
        fm: ResolutionMatrixModel(make_fit_params(s["z"], fm, s["hypers"])) for fm in FIT_MODELS
    }
    by_ndim = {m.ndim: fm for fm, m in models.items()}
    out = ["Likelihood-call errors during the fits, re-evaluated outside the optimizer", ""]
    found = parse_logs()
    out.append(f"{len(found)} error vectors in the logs")
    kinds = {}
    for path, job, theta in found:
        fm = by_ndim[theta.size]
        model = models[fm]
        labels = model.theta_labels()
        lo, hi = prior_bounds(model)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.predict(theta, observations=s["obs"], sps=cue)
            msg = "no exception on re-evaluation"
            where = ""
        except Exception as e:  # noqa: BLE001, the point is to record any exception
            msg = f"{type(e).__name__}: {e}"
            tb = traceback.extract_tb(e.__traceback__)
            where = " <- ".join(
                f"{f.filename.split('/')[-1]}:{f.lineno} {f.name}" for f in tb[::-1][:4]
            )
        kinds[msg] = kinds.get(msg, 0) + 1
        w = hi - lo
        near = [
            f"{lab}={v:.6g} ({'low' if v - a < 1e-3 * (b - a) else 'high'} edge)"
            for lab, v, a, b in zip(labels, theta, lo, hi, strict=True)
            if np.isfinite(b - a) and (v - a < 1e-3 * (b - a) or b - v < 1e-3 * (b - a))
        ]
        line = ["", f"{path.split('/')[-1]} job {job} ({fm})", f"   {msg}", f"   at {where}"]
        line.append(f"   params within 0.1% of a prior edge: {near or 'none'}")
        if job is not None:
            r = np.load(OUT / "fits" / f"{job}.npz")
            dx = np.abs(theta - r["theta"]) / np.where(np.isfinite(w), w, np.nan)
            k = np.nanargmax(dx)
            line.append(
                f"   final optimum: -lnp {float(r['fun']):.2f}; largest |error point - optimum| is"
                f" {labels[k]} at {dx[k]:.3f} of its prior range; median over bounded params"
                f" {np.nanmedian(dx):.4f}"
            )
            fin_near = [lab for lab, e in zip(labels, r["edge"], strict=True) if e]
            line.append(f"   final optimum at a prior edge in: {fin_near or 'none'}")
        out += line
    out += ["", "Exception counts:"] + [f"   {n:4d}  {k}" for k, n in kinds.items()]
    out += ["", *mechanism_check(models["shared"], s, cue)]
    text = "\n".join(out) + "\n"
    print(text)
    (OUT / "likelihood_errors.txt").write_text(text)


if __name__ == "__main__":
    main()
