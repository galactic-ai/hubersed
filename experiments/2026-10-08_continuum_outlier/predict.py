"""Save the posterior and model spectra of one converged fit for analysis.py.

FSPS fixes its library at compile time, so each run needs its own process, and MILES runs need
the MILES build first on PYTHONPATH, as in run.sh. The output is
results/2026-10-08_continuum_outlier/<run>_pred.npz with the data the fit used, the posterior
points that carry weight, the maximum likelihood spectrum and 100 weighted draws.

Run from the repository root on a compute node with
``uv run python experiments/2026-10-08_continuum_outlier/predict.py --tid 39633433328091196
--lib c3k_hr --window full --nebular off``.
"""

import argparse
import warnings

import numpy as np
from fit import _STATE, load_data, loglike, make_model
from nautilus import Sampler

from hubersed.paths import PATHS
from hubersed.sps.parameter_file import WAVE_OBS

OUT = PATHS["RESULTS"] / "2026-10-08_continuum_outlier"
N_DRAW = 100


def run_name(tid, lib, window, nebular, degrade):
    """Return the checkpoint stem that fit.py uses for this run."""
    neb = "on" if nebular else "off"
    return f"{tid}_{lib}_{window}_neb{neb}{'' if degrade else '_nodeg'}"


def main(tid, lib, window, nebular, degrade):
    name = run_name(tid, lib, window, nebular, degrade)
    z, flux, unc, good, _ = load_data(tid, lib, window, degrade)
    model = make_model(z, nebular)

    # Only reads the checkpoint. The likelihood is never called.
    s = Sampler(
        model.prior_transform,
        lambda x: 0.0,
        n_dim=model.ndim,
        n_live=1000,
        filepath=str(OUT / f"{name}_seed0.h5"),
        resume=True,
    )
    assert s.explored, name
    points, log_w, log_l = s.posterior()
    best = points[np.argmax(log_l)]
    # points more than e^-30 below the heaviest one add nothing to any statistic
    keep = log_w > log_w.max() - 30
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    print(f"{name}: {len(points)} points, {keep.sum()} kept, N_eff {s.n_eff:.0f}")

    # The likelihood of the fit, called once to check the setup matches. It caches the source.
    lnl = loglike(best, tid, lib, window, nebular, degrade)
    assert np.isclose(lnl, log_l.max(), atol=0.1), (lnl, log_l.max())
    _, obs, sps = _STATE[(tid, lib, window, nebular, degrade)]

    def predict(theta):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds, _ = model.predict(np.asarray(theta, float), observations=obs, sps=sps)
        return np.asarray(preds[0], float)

    draws = np.random.default_rng(0).choice(len(w), size=N_DRAW, p=w)
    np.savez(
        OUT / f"{name}_pred.npz",
        wave=WAVE_OBS,
        z=z,
        flux=flux,
        unc=unc,
        good=good,
        labels=np.array(model.theta_labels()),
        points=points[keep],
        log_w=log_w[keep],
        log_l=log_l[keep],
        log_z=s.log_z,
        n_eff=s.n_eff,
        n_like=s.n_like,
        best=best,
        sp_best=predict(best),
        draw_theta=points[draws],
        draw_spec=np.array([predict(points[i]) for i in draws]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tid", type=int, default=39627757533007793)
    parser.add_argument("--lib", choices=["miles", "c3k_hr"], required=True)
    parser.add_argument("--window", choices=["miles", "full"], required=True)
    parser.add_argument("--nebular", choices=["on", "off"], required=True)
    parser.add_argument("--degrade", choices=["on", "off"], default="on")
    args = parser.parse_args()
    main(args.tid, args.lib, args.window, args.nebular == "on", args.degrade == "on")
