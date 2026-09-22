"""
SFH old-age resolution test for 42580, PARALLEL + many randomized starts.

Runs (binning x start) tasks across CPUs. Each task: build the model for one
binning, seed an OLD SFH + random perturbation, MAP-optimize (cheap params, no
SSP recompute). Keep the BEST chi2 per binning -> escapes local minima.

Binnings: n_sub subdivision of the old bins (lookback > t_obs-11, i.e. cosmic
time < 11): n_sub=1 -> standard(10); 3,5,8 -> more old bins.

Config at top: N_START (starts per binning), MAXFEV, N_WORKERS.
Uses spawn (safe with JAX/Cue). Each worker loads FSPS+Cue once (reused).

Run from hubersed root:
    python tmp/sfh_resolution_parallel_42580.py
"""

import os, sys, pickle, warnings
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "bin/prospector")

PKL, KEY = "results/mapfit_cont_line_examples.pkl", "continuum-only"
OUT = "results/sfh_resolution_parallel_42580.pkl"
NSUBS = [1, 3, 5, 8]  # 1=standard(10); 3->14, 5->18, 8->24 bins
N_START = 10  # randomized starts PER binning
MAXFEV = 60000
N_WORKERS = min(os.cpu_count() or 4, len(NSUBS) * N_START, 8)  # cap for memory


def _agebins(z, n_sub):
    from hubersed.prospector.utils import make_stochastic_agebins
    from astropy.cosmology import Planck18 as cosmo
    import astropy.units as u

    std = make_stochastic_agebins(z)
    if n_sub <= 1:
        return std
    t = cosmo.age(z).to_value(u.Gyr)
    split = t - 11.0
    edges = np.unique(np.round(10 ** std.ravel() / 1e9, 6))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > split:
            out.extend(np.linspace(lo, hi, n_sub + 1)[:-1])
        else:
            out.append(lo)
    out.append(edges[-1])
    ab = np.array([[out[i], out[i + 1]] for i in range(len(out) - 1)])
    return np.log10(ab * 1e9)


def run_one(task):
    n_sub, start_idx = task
    import numpy as np, pickle
    from hubersed.prospector import parameter_file as P; from hubersed.fitting import config as FC, chi2 as MC
    from prospect.models.sedmodel import HyperSpecModel
    from prospect.models.templates import adjust_stochastic_params
    from prospect.fitting import lnprobfn

    try:
        d = pickle.load(open(PKL, "rb"))
        wave = np.asarray(d["wave"], float)
        r = d["results"][KEY]
        z = r["z"]
        flux = np.asarray(r["flux"], float)
        unc = np.asarray(r["unc"], float)
        mask = np.asarray(r["mask"], bool)
        ab = _agebins(z, n_sub)
        nb = ab.shape[0]

        FC.make_stochastic_agebins = lambda zz, ab=ab: ab
        MC._fsps()
        obs = P.build_obs(spec=flux, unc=unc, mask=mask, resolution=MC._lsf_sigma_kms())
        cmodel, ctemplate = FC.build_continuum_model(z)
        _, ft = FC.build_full_cue_model(ctemplate, cmodel.theta.copy(), cmodel, z)
        FREE = [
            "logsfr_ratios",
            "logzsol",
            "dust2",
            "dust_ratio",
            "dust_index",
            "logmass",
            "sigma_smooth",
            "gas_logz",
            "gas_logu",
            "gas_lognH",
            "gas_logno",
            "gas_logco",
            "eline_sigma",
        ]
        for k in list(ft.keys()):
            if isinstance(ft[k], dict) and "isfree" in ft[k]:
                ft[k]["isfree"] = k in FREE
        for k, vv in {
            "sigma_reg": 1.5,
            "sigma_dyn": 0.1,
            "tau_eq": 1.0,
            "tau_in": 1.0,
            "tau_dyn": 0.025,
        }.items():
            ft[k]["isfree"] = False
            ft[k]["init"] = vv
        ft = adjust_stochastic_params(ft)
        fm = HyperSpecModel(ft)
        sps = MC._cue()
        sps.ssp.params["tpagb_norm_type"] = 2
        sps.ssp.params["add_agb_dust_model"] = True
        m = obs[0].mask

        def neg(th):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    lp = lnprobfn(th, model=fm, observations=obs, sps=sps, nested=False)
                    return -lp if np.isfinite(lp) else 1e18
                except Exception:
                    return 1e18

        # old seed (equivalent old-bias) + random perturbation on the SFH for this start
        rng = np.random.default_rng(1000 * n_sub + start_idx)
        th0 = fm.theta.copy()
        li = fm.theta_index["logsfr_ratios"]
        base = -10.8 / (nb - 1)
        th0[li] = base + rng.normal(0, 0.5, size=np.atleast_1d(th0[li]).shape)
        bf = MC._map_optimize(neg, th0, n_seeds=1, maxfev=MAXFEV)
        preds, _ = fm.predict(bf.x, observations=obs, sps=sps)
        sp = np.asarray(preds[0], float)
        chi2 = float(np.nansum(((obs[0].flux[m] - sp[m]) / obs[0].uncertainty[m]) ** 2))
        ndof = int(m.sum()) - len(bf.x)
        wr = wave / (1 + z)

        def cw(lo, hi):
            s = m & (wr >= lo) & (wr < hi)
            return (
                float(
                    np.nansum(((obs[0].flux[s] - sp[s]) / obs[0].uncertainty[s]) ** 2)
                )
                / s.sum()
            )

        lr = np.asarray(bf.x[fm.theta_index["logsfr_ratios"]], float)
        lm = float(np.atleast_1d(bf.x[fm.theta_index["logmass"]])[0])
        td = {k: np.asarray(bf.x[v], float) for k, v in fm.theta_index.items()}
        return dict(
            n_sub=n_sub,
            nb=nb,
            start=start_idx,
            ok=True,
            chi2_red=chi2 / ndof,
            bump=cw(6500, 7200),
            control=cw(4500, 5500),
            model=sp,
            theta=np.asarray(bf.x, float),
            theta_labels=list(fm.free_params),
            agebins=np.asarray(ab, float),
            logsfr_ratios=lr,
            logmass=lm,
            theta_dict=td,
        )
    except Exception as e:
        return dict(n_sub=n_sub, start=start_idx, ok=False, err=repr(e))


if __name__ == "__main__":
    tasks = [(ns, s) for ns in NSUBS for s in range(N_START)]
    print(
        f"{len(tasks)} tasks ({len(NSUBS)} binnings x {N_START} starts) on {N_WORKERS} workers, MAXFEV={MAXFEV}"
    )
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
        allres = list(ex.map(run_one, tasks))
    good = [r for r in allres if r.get("ok")]
    nbad = len(allres) - len(good)
    if nbad:
        print(f"  ({nbad} starts failed)")
    best = {}
    for ns in NSUBS:
        cand = [r for r in good if r["n_sub"] == ns]
        if cand:
            best[ns] = min(cand, key=lambda r: r["chi2_red"])
    print(
        f"\n{'binning':10s} {'Nbins':>5s} {'best chi2_red':>13s} {'bump 6500-7200':>15s} {'control 4500-5500':>18s}  (of {N_START} starts)"
    )
    for ns in NSUBS:
        if ns in best:
            b = best[ns]
            spread = np.ptp([r["chi2_red"] for r in good if r["n_sub"] == ns])
            print(
                f"n_sub={ns:<4d} {b['nb']:>5d} {b['chi2_red']:>13.3f} {b['bump']:>15.2f} {b['control']:>18.2f}   (chi2_red spread {spread:.2f})"
            )
    pickle.dump(dict(z=None, best=best, all=allres, wave=None), open(OUT, "wb"))
    # re-attach wave/data for plotting convenience
    d = pickle.load(open(PKL, "rb"))
    r = d["results"][KEY]
    save = pickle.load(open(OUT, "rb"))
    save["wave"] = np.asarray(d["wave"])
    save["z"] = r["z"]
    save["flux"] = np.asarray(r["flux"])
    save["unc"] = np.asarray(r["unc"])
    save["mask"] = np.asarray(r["mask"])
    pickle.dump(save, open(OUT, "wb"))
    print(f"\nsaved {OUT}")
    print(
        "Converged check: best chi2_red should be MONOTONIC non-increasing with Nbins."
    )
