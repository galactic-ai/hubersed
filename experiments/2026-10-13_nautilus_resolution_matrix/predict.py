"""Save the rmatrix posterior and model spectra to <stem>_pred.npz for analysis.py.

Same keys as the 10-09 predict.py. The checkpoint may be a copy taken between chain links.
Run with the C3K_ER build first on PYTHONPATH, as for analysis.py.
"""

import argparse
import warnings

import numpy as np
from fit import MASKS, N_LIVE, OUT, loglike, process_state, stem
from model_setup import load_data
from nautilus import Sampler

from hubersed.fitting.chi2 import WAVE_OBS

N_DRAW = 100


def main(lsf, mask):
    name = stem(lsf, mask)
    z, flux, unc, good = load_data(MASKS[mask])
    model, obs, cue = process_state(lsf, mask)

    # Only reads the checkpoint. The likelihood is never called.
    s = Sampler(
        lambda x: x,
        lambda x: 0.0,
        n_dim=model.ndim,
        n_live=N_LIVE,
        filepath=str(OUT / f"{name}.h5"),
        resume=True,
    )
    assert s.explored, name
    unit, log_w, log_l = s.posterior()
    best, max_l = model.prior_transform(unit[np.argmax(log_l)]), log_l.max()
    # points more than e^-30 below the heaviest one add nothing to any statistic
    keep = log_w > log_w.max() - 30
    print(f"{name}: {len(unit)} points, {keep.sum()} kept, N_eff {s.n_eff:.0f}")
    points = np.array([model.prior_transform(x) for x in unit[keep]])
    log_w, log_l = log_w[keep], log_l[keep]
    w = np.exp(log_w - log_w.max())
    w /= w.sum()

    # The likelihood of the fit, called once to check the setup matches.
    lnl = loglike(best, lsf, mask)
    assert np.isclose(lnl, max_l, atol=0.1), (lnl, max_l)

    def predict(theta):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds, _ = model.predict(np.asarray(theta, float), observations=obs, sps=cue)
        return np.asarray(preds[0], float)

    narrow = best.copy()
    narrow[list(model.theta_labels()).index("eline_fbroad")] = 0.0
    draws = np.random.default_rng(0).choice(len(w), size=N_DRAW, p=w)
    np.savez(
        OUT / f"{name}_pred.npz",
        wave=WAVE_OBS,
        z=z,
        flux=flux,
        unc=unc,
        good=good,
        labels=np.array(model.theta_labels()),
        points=points,
        log_w=log_w,
        log_l=log_l,
        log_z=s.log_z,
        n_eff=s.n_eff,
        n_like=s.n_like,
        best=best,
        max_lnl=max_l,
        sp_best=predict(best),
        sp_narrow=predict(narrow),
        draw_theta=points[draws],
        draw_spec=np.array([predict(points[i]) for i in draws]),
    )
    print(f"wrote {name}_pred.npz, max lnL {max_l:.2f} (recomputed {lnl:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lsf", choices=["rmatrix", "median"], default="rmatrix")
    parser.add_argument("--mask", choices=list(MASKS), default="mask")
    args = parser.parse_args()
    main(args.lsf, args.mask)
