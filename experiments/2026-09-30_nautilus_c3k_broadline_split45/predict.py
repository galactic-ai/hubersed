"""Save the posterior and model spectra of a converged HR split 45 fit, as the ER predict.py does.

The output is results/2026-09-30_nautilus_c3k_broadline_split45/<stem>_pred.npz with the data the
fit used, the posterior points that carry weight, the maximum likelihood spectrum, the same point
with the broad component switched off (eline_fbroad = 0), and 100 weighted draws. The keys match
the ER ``_pred.npz`` of experiments/2026-10-09_nautilus_c3k_er_broadline/predict.py.

The fit gave nautilus the identity prior, so the checkpoint holds unit cube points. They are read
with the identity here and only the kept ones go through model.prior_transform.

The data are those of the fit: degraded to C3K_HR with the original uncertainty kept, and with the
SPARCL sky pixels unmasked unless ``--keep-sky-mask`` is given.

Run from the repository root with the default C3K_HR build. Command for run A of 37084:
``SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-09-30_nautilus_c3k_broadline_split45/predict.py --forbidden-broad shared``.
"""

import argparse
import warnings

import numpy as np
from fit import _STATE, N_LIVE, load_data, loglike, make_model
from nautilus import Sampler

from hubersed.fitting.chi2 import WAVE_OBS
from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-09-30_nautilus_c3k_broadline_split45"
N_DRAW = 100


def main(tid, fb, split, unmask, rest_max):
    mask_tag = "unmask" if unmask else "skymask"
    stem = f"{tid}_broad_{fb}_split{split:g}_{mask_tag}_rest{rest_max:g}_seed0"
    z, flux, unc, good, _ = load_data(tid, rest_max, unmask)
    model = make_model(z, fb, split)

    # Only reads the checkpoint. The likelihood is never called.
    s = Sampler(
        lambda x: x,
        lambda x: 0.0,
        n_dim=model.ndim,
        n_live=N_LIVE,
        filepath=str(OUT / f"{stem}.h5"),
        resume=True,
    )
    assert s.explored, stem
    unit, log_w, log_l = s.posterior()
    best, max_l = model.prior_transform(unit[np.argmax(log_l)]), log_l.max()
    # points more than e^-30 below the heaviest one add nothing to any statistic
    keep = log_w > log_w.max() - 30
    print(f"{stem}: {len(unit)} points, {keep.sum()} kept, N_eff {s.n_eff:.0f}")
    points = np.array([model.prior_transform(x) for x in unit[keep]])
    log_w, log_l = log_w[keep], log_l[keep]
    w = np.exp(log_w - log_w.max())
    w /= w.sum()

    # The likelihood of the fit, called once to check the setup matches. It caches the source.
    lnl = loglike(best, tid, fb, split, unmask, rest_max)
    assert np.isclose(lnl, max_l, atol=0.1), (lnl, max_l)
    model, obs, cue = _STATE[(tid, fb, split, unmask, rest_max)]

    def predict(theta):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds, _ = model.predict(np.asarray(theta, float), observations=obs, sps=cue)
        return np.asarray(preds[0], float)

    narrow = best.copy()
    narrow[list(model.theta_labels()).index("eline_fbroad")] = 0.0
    draws = np.random.default_rng(0).choice(len(w), size=N_DRAW, p=w)
    np.savez(
        OUT / f"{stem}_pred.npz",
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
    print(f"wrote {stem}_pred.npz, max lnL {max_l:.2f} (recomputed {lnl:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tid", type=int, default=39627770174637084)
    parser.add_argument("--forbidden-broad", choices=["free", "shared"], required=True)
    parser.add_argument("--sigma-split", type=float, default=45.0)
    parser.add_argument("--keep-sky-mask", action="store_true")
    parser.add_argument("--rest-max", type=float, default=9000.0)
    args = parser.parse_args()
    main(args.tid, args.forbidden_broad, args.sigma_split, not args.keep_sky_mask, args.rest_max)
