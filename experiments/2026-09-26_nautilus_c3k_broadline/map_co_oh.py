"""MAP of C/O free or tied to O/H from best point of the broad-line posterior."""

import argparse
import copy
import pickle
import time
from pathlib import Path

import numpy as np
from fit import load_data, make_obs, make_params
from nautilus import Sampler
from prospect.fitting import lnprobfn
from prospect.models import priors

from hubersed.fitting.map_fits import get_sps, map_fit, sfh_from_theta
from hubersed.paths import PATHS
from hubersed.sps import broadline

HYPERS = ("sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn")

# From Li+24 for the solar abundance
CUE_LOGOH_SUN = -3.07
CUE_LOGCO_SUN = -0.37
CUE_LOGCO_RANGE = (-1.0, np.log10(5.4))
CO_SCATTER_DEX = 0.17  # from Berg+19
RUNS = {"free": (False, False), "tied": (True, False), "tied_fixed": (True, True)}


def co_from_oh(gas_logz=0.0, gas_dlogco=0.0, **extras):
    """Return Cue's ``gas_logco`` from O/H with the Nicholls et al. (2017) relation.

    log(C/O) = log10(10^-0.8 + 10^(log(O/H) + 2.72)) plus the offset ``gas_dlogco``, relative to
    Cue's solar C/O and clipped to the Cue grid.
    """
    log_oh = CUE_LOGOH_SUN + np.atleast_1d(gas_logz)
    log_co = np.log10(10**-0.8 + 10 ** (log_oh + 2.72))
    return np.clip(log_co - CUE_LOGCO_SUN + np.atleast_1d(gas_dlogco), *CUE_LOGCO_RANGE)


def best_point(z, checkpoint):
    """Return max-likelihood point of nautilus checkpoint as a dictionary.

    Only reads the checkpoint file.
    """
    run_model = broadline.TwoCompLineModel(make_params(z))
    s = Sampler(
        run_model.prior_transform,
        lambda x: 0.0,
        n_dim=run_model.ndim,
        n_live=1000,
        filepath=str(checkpoint),
        resume=True,
    )
    points, _, log_l = s.posterior()
    i = int(np.argmax(log_l))
    print(f"checkpoint: {len(points)} points, N_eff {s.n_eff:.0f}, max lnL {log_l[i]:.2f}")
    return dict(zip(run_model.theta_labels(), points[i], strict=True)), float(log_l[i])


def template(z, best, tie, fixed_scatter, forbidden_broad, sigma_split):
    t = make_params(z)
    broadline.add_broad_params(
        t, separate_forbidden_width=True, forbidden_broad=forbidden_broad, sigma_split=sigma_split
    )

    for k in HYPERS:
        t[k].update(isfree=False, init=float(best[k]))

    if tie:
        t["gas_dlogco"] = dict(
            N=1,
            isfree=not fixed_scatter,
            init=0.0,
            prior=priors.Normal(mean=0.0, sigma=CO_SCATTER_DEX),
        )
        t["gas_logco"] = dict(t["gas_logco"], isfree=False, depends_on=co_from_oh)
    return copy.deepcopy(t)


def main(tid, window, checkpoint, runs, maxfev, n_seeds, forbidden_broad, sigma_split, out):
    z, flux, unc, good = load_data(tid, window)
    obs = make_obs(flux, unc, good)
    cue = get_sps(zero_library_resolution=False)["cue"]
    assert cue.ssp.libraries[1] in ("c3k_hr", b"c3k_hr"), cue.ssp.libraries
    best, lnl_run = best_point(z, checkpoint)
    print("fixed hyperparameters:", {k: round(float(best[k]), 4) for k in HYPERS})

    res = {
        "best": best,
        "max_lnl_run": lnl_run,
        "checkpoint": str(checkpoint),
        "data": dict(z=z, flux=flux, unc=unc, good=good),
    }
    for name in runs:
        tie, fixed = RUNS[name]
        model = broadline.TwoCompLineModel(
            template(z, best, tie, fixed, forbidden_broad, sigma_split)
        )

        theta0 = np.array(
            [best.get(lab, v) for lab, v in zip(model.theta_labels(), model.theta, strict=True)]
        )
        lnl0 = lnprobfn(theta0, model=model, observations=obs, sps=cue, nested=True)
        t0 = time.time()
        mfit, info = map_fit(
            model, obs, cue, n_seeds=n_seeds, maxfev=maxfev, theta0=theta0, tag=name
        )
        lnl = lnprobfn(mfit.x, model=model, observations=obs, sps=cue, nested=True)
        spec = np.asarray(model.predict(mfit.x, observations=obs, sps=cue)[0][0], float)
        chi2 = float(np.sum(((flux - spec) / unc)[good] ** 2))
        chi2_red = chi2 / (good.sum() - model.ndim)
        co = float(np.atleast_1d(model.params["gas_logco"])[0])

        print(
            f"{name}: lnL {lnl0:.2f} -> {lnl:.2f}, chi2 {chi2:.1f}, chi2_red {chi2_red:.3f}, "
            f"gas_logco {co:+.3f}, gas_logz {float(model.params['gas_logz'][0]):+.3f}, "
            f"converged {mfit.success}, {time.time() - t0:.0f}s",
            flush=True,
        )

        res[name] = dict(
            labels=list(model.theta_labels()),
            theta=mfit.x,
            lnpost=-mfit.fun,
            lnl=lnl,
            chi2=chi2,
            chi2_red=chi2_red,
            gas_logco=co,
            info=info,
            spec=spec,
            sfh=sfh_from_theta(model, mfit.x),
        )
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{tid}_map_co_{window}.pkl", "wb") as f:
            pickle.dump(res, f)
    if "free" in res and "tied" in res:
        print(f"tie costs delta chi2 {2 * (res['free']['lnl'] - res['tied']['lnl']):+.1f}")


if __name__ == "__main__":
    out_dir = PATHS["RESULTS"] / "2026-09-26_nautilus_c3k_broadline"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tid", type=int, default=39627770174637084)
    parser.add_argument("--window", choices=["full", "miles"], default="miles")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--runs", default="free,tied", help="comma-separated: " + ",".join(RUNS))
    parser.add_argument("--maxfev", type=int, default=8000)
    parser.add_argument("--n-seeds", type=int, default=2, help="jittered starts added to the first")
    parser.add_argument("--forbidden-broad", choices=["shared", "free"], default=None)
    parser.add_argument("--sigma-split", type=float, default=100.0)
    parser.add_argument("--out", type=Path, default=out_dir / "map_co")
    a = parser.parse_args()
    ckpt = a.checkpoint or out_dir / "run2" / f"{a.tid}_broad_{a.window}_seed0.h5"
    main(
        a.tid,
        a.window,
        ckpt,
        a.runs.split(","),
        a.maxfev,
        a.n_seeds,
        a.forbidden_broad,
        a.sigma_split,
        a.out,
    )
