"""Save the rmatrix posterior and model spectra to <stem>_pred.npz for analysis.py.

Same keys as the 10-09 predict.py. The checkpoint may be a copy taken between chain links.
Run with the C3K_ER build first on PYTHONPATH, as for analysis.py.
"""

import argparse
import multiprocessing as mp
import warnings

import numpy as np
from fit import MASKS, N_LIVE, OUT, loglike, process_state, stem
from model_setup import load_data, make_model, make_params
from nautilus import Sampler
from tqdm import tqdm

from hubersed.fitting.chi2 import WAVE_OBS

N_DRAW = 100
N_PROC = 8
_MODEL = {}


def _init(z, lsf):
    """Build one model per worker process."""
    _MODEL["m"] = make_model(make_params(z), lsf)


def _transform(rows):
    """Return the prior transform of each row with the worker's model."""
    return np.array([_MODEL["m"].prior_transform(x) for x in rows])


def prior_transform_all(unit, z, lsf):
    """Return model.prior_transform of every row of ``unit``, spread over N_PROC processes.

    The SFH prior covariance depends on each point's hyperparameters, so points are done one by one.
    """
    chunks = np.array_split(unit, max(1, len(unit) // 5000))
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_PROC, initializer=_init, initargs=(z, lsf)) as pool:
        parts = list(tqdm(pool.imap(_transform, chunks), total=len(chunks), desc="prior transform"))
    return np.vstack(parts)


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
    points = prior_transform_all(unit[keep], z, lsf)
    assert np.allclose(points[:100], [model.prior_transform(x) for x in unit[keep][:100]])
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
