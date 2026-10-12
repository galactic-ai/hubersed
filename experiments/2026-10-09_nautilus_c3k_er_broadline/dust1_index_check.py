"""Would a free dust1_index fix the Balmer decrement misfit of the ER fit of 39627770174637084?

QUESTION: at the ER max-L point the data Hbeta/Halpha is about 4.7% below the model. The young
CSP is attenuated by tau1 = dust1 (lambda/5500)^dust1_index, and dust1_index is not a model
parameter, so it stays at the FSPS default of -1. How much chi2 could a free dust1_index buy, and
does it move the Balmer windows the right way without hurting the continuum?

Two parts. A grid moves dust1_index with every other parameter at the max-L point ``best``. It
gives the total chi2, the chi2 in each line window of jacobian.py and in the rest (continuum), and
the model Balmer line ratios from ``model._eline_lum``. A profile re-optimizes dust2, dust_ratio,
logmass, logzsol, logsfr_ratios_1 and logsfr_ratios_2 at five dust1_index values with Powell
(at most 400 likelihood calls each), starting from ``best``. The objective is the likelihood
only. Points outside the prior give minus infinity, and Powell is bounded by the prior bounds.
The profile chi2 is quoted against the same re-optimization at dust1_index = -1.

dust1_index is set on ``ssp.params`` of the cached Cue source. prospect's ``update`` only copies
model parameters into ``ssp.params``, and dust1_index is not one, so it stays where it is set.
``get_spectrum`` of nebssp_basis.py passes it to ``add_dust``. FSPS gets dust1 = dust2 = 0, so the
FSPS spectra do not depend on it, which the script checks. It is put back to -1 at the end.

INPUTS: the ER ``_pred.npz`` of predict.py, jacobian.npz of jacobian.py (posterior widths that
scale the optimizer steps) and the same data, model and build as the fit. SEED: none
(deterministic).

Run from the repository root on the Mac with the C3K_ER build, as jacobian.py. COMMAND:
``PYTHONPATH=<site with the C3K_ER fsps>:experiments/2026-10-09_nautilus_c3k_er_broadline
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-09_nautilus_c3k_er_broadline/dust1_index_check.py``.

OUTPUT: results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/dust1_index_check.txt.
"""

import time
import warnings

import numpy as np
from fit import loglike, process_state
from jacobian import WINDOWS
from scipy.optimize import minimize

from hubersed.paths import PATHS

TID = 39627770174637084
FB, SPLIT, REST_MAX = "shared", 45.0, 9000.0
OUT = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline"
FIG = OUT / str(TID)
STEM = f"{TID}_er_broad_{FB}_split{SPLIT:g}_rest{REST_MAX:g}_seed0"
GRID = [-1.6, -1.4, -1.2, -1.1, -1.0, -0.9, -0.8, -0.6, -0.4]
PROFILE = [-1.6, -1.3, -1.0, -0.7, -0.4]
FREE = ["dust2", "dust_ratio", "logmass", "logzsol", "logsfr_ratios_1", "logsfr_ratios_2"]
MAXFEV = 400
# rest-frame vacuum wavelengths of the Balmer lines
BALMER = {"Halpha": 6564.6, "Hbeta": 4862.7, "Hgamma": 4341.7, "Hdelta": 4102.9}
# (data - model) / model line flux at best, from line_flux_check.txt
MISFIT = {"Halpha": 0.008, "Hbeta": -0.039, "Hgamma": -0.025, "Hdelta": -0.040}
SHOW = ["Hbeta", "Halpha", "Hgamma", "Hdelta", "continuum"]


def main():
    d = np.load(OUT / f"{STEM}_pred.npz")
    labels = [str(x) for x in d["labels"]]
    best, good, flux, unc = d["best"], d["good"], d["flux"], d["unc"]
    rest = d["wave"] / (1 + float(d["z"]))
    model, obs, sps = process_state(TID, FB, SPLIT, REST_MAX)
    assert list(model.theta_labels()) == labels
    ssp = sps.ssp
    out = []

    def say(s=""):
        print(s, flush=True)
        out.append(s)

    def lnl(theta):
        return loglike(theta, TID, FB, SPLIT, REST_MAX)

    def predict(theta):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = np.asarray(model.predict(theta, observations=obs, sps=sps)[0][0], float)
        return spec, np.array(model._eline_wave, float), np.array(model._eline_lum, float)

    # how dust1_index reaches the model
    say(
        f"source {type(sps).__name__}, dust1_index in model.params: {'dust1_index' in model.params}"
    )
    assert "dust1_index" not in model.params
    say(
        f"FSPS dust1 {ssp.params['dust1']}, dust2 {ssp.params['dust2']}, default dust1_index "
        f"{ssp.params['dust1_index']}, marginalize_elines "
        f"{model.params.get('marginalize_elines', False)}, use_stellar_ionizing "
        f"{model.params.get('use_stellar_ionizing', False)}"
    )
    assert ssp.params["dust1_index"] == -1.0
    assert ssp.params["dust1"] == 0 and ssp.params["dust2"] == 0

    try:
        lnl0 = lnl(best)
        t0 = time.time()
        lnl(best)
        dt = time.time() - t0
        say(f"lnL at best, dust1_index -1.0: {lnl0:.4f} (pred.npz {float(d['max_lnl']):.4f})")
        say(f"second likelihood call {dt:.2f} s (the first one compiles)")
        assert np.isclose(lnl0, float(d["max_lnl"]), atol=0.1)

        # with dust1 = 0 the spectrum must not depend on dust1_index
        th0 = best.copy()
        th0[labels.index("dust_ratio")] = 0.0
        s_a = predict(th0)[0]
        ssp.params["dust1_index"] = -1.6
        s_b = predict(th0)[0]
        ssp.params["dust1_index"] = -1.0
        say(
            f"dust_ratio 0: max |spec(-1.6) - spec(-1.0)| / unc {np.max(np.abs(s_a - s_b)[good] / unc[good]):.2e}"
        )
        # a set value survives a model call
        ssp.params["dust1_index"] = -1.3
        lnl(best)
        say(
            f"set -1.3, after a likelihood call ssp.params['dust1_index'] = {ssp.params['dust1_index']}"
        )
        assert ssp.params["dust1_index"] == -1.3
        ssp.params["dust1_index"] = -1.0

        regions = [*WINDOWS.items(), ("continuum", (None, None))]

        def window_chi2(spec):
            chi = (flux - spec) / unc
            res, used = {}, np.zeros_like(good)
            for name, (a, b) in regions:
                k = good & ~used if a is None else good & (rest >= a) & (rest < b)
                used |= k
                res[name] = float(np.sum(chi[k] ** 2))
            return float(np.sum(chi[good] ** 2)), res

        def balmer(ew, el):
            idx = {k: int(np.argmin(np.abs(ew - w))) for k, w in BALMER.items()}
            for k, i in idx.items():
                assert abs(ew[i] - BALMER[k]) < 3, (k, ew[i])
            return {k: el[i] for k, i in idx.items()}, {k: ew[i] for k, i in idx.items()}

        # grid with everything else at best
        grid = {}
        for n in GRID:
            ssp.params["dust1_index"] = n
            spec, ew, el = predict(best)
            c2, win = window_chi2(spec)
            lum, lw = balmer(ew, el)
            grid[n] = dict(lnl=lnl(best), chi2=c2, win=win, lum=lum)
        ssp.params["dust1_index"] = -1.0
        say(f"Cue line wavelengths used: {', '.join(f'{k} {v:.2f}' for k, v in lw.items())}")

        ref = grid[-1.0]
        want = {
            k: (1 + MISFIT[k]) / (1 + MISFIT["Halpha"]) - 1 for k in ("Hbeta", "Hgamma", "Hdelta")
        }
        say("\nGRID, all other parameters at best. dchi2 against dust1_index -1.0.")
        say("Ratio columns are the % change of the model L/L(Halpha) against -1.0.")
        say(
            "Data want "
            + ", ".join(f"H{k[1]}/Ha {100 * v:+.1f}%" for k, v in want.items())
            + " (Balmer misfits are lower bounds)."
        )
        head = f"{'d1_index':>8s} {'dchi2':>8s} {'-2dlnL':>8s}" + "".join(
            f" {s[:9]:>9s}" for s in SHOW
        )
        head += f" {'Hb/Ha%':>7s} {'Hg/Ha%':>7s} {'Hd/Ha%':>7s}"
        say(head)
        for n, g in grid.items():
            row = f"{n:8.2f} {g['chi2'] - ref['chi2']:+8.2f} {-2 * (g['lnl'] - ref['lnl']):+8.2f}"
            row += "".join(f" {g['win'][s] - ref['win'][s]:+9.2f}" for s in SHOW)
            for k in ("Hbeta", "Hgamma", "Hdelta"):
                r = g["lum"][k] / g["lum"]["Halpha"] / (ref["lum"][k] / ref["lum"]["Halpha"])
                row += f" {100 * (r - 1):+7.2f}"
            say(row)
        say(
            f"chi2 at -1.0: total {ref['chi2']:.1f}, "
            + ", ".join(f"{s} {ref['win'][s]:.1f}" for s in SHOW)
        )
        say("All windows, dchi2 at -1.6 and -0.4 against -1.0:")
        for name, _ in regions:
            say(
                f"  {name:12s} {grid[-1.6]['win'][name] - ref['win'][name]:+8.2f} "
                f"{grid[-0.4]['win'][name] - ref['win'][name]:+8.2f}"
            )

        # profile over the dust and young SFH parameters
        ix = [labels.index(k) for k in FREE]
        bounds = np.array(model.theta_bounds(), float)[ix]
        jz = np.load(FIG / "jacobian.npz")
        assert [str(x) for x in jz["labels"]] == labels
        sd = jz["post_sd"][ix]
        say(
            f"\nPROFILE over {', '.join(FREE)}, Powell, at most {MAXFEV} calls, in units of post_sd."
        )
        say(
            "prior bounds: "
            + ", ".join(f"{k} [{a:.3g}, {b:.3g}]" for k, (a, b) in zip(FREE, bounds, strict=True))
        )

        def theta_of(u):
            th = best.copy()
            th[ix] = best[ix] + u * sd
            return th

        prof = {}
        for n in PROFILE:
            ssp.params["dust1_index"] = n
            start = lnl(best)

            def f(u, start=start):
                v = lnl(theta_of(u))
                return 1e10 if not np.isfinite(v) else -(v - start)

            t0 = time.time()
            res = minimize(
                f,
                np.zeros(len(ix)),
                method="Powell",
                bounds=list(
                    zip((bounds[:, 0] - best[ix]) / sd, (bounds[:, 1] - best[ix]) / sd, strict=True)
                ),
                options=dict(maxfev=MAXFEV, xtol=1e-3, ftol=1e-6),
            )
            th = theta_of(res.x)
            spec, ew, el = predict(th)
            c2, win = window_chi2(spec)
            lum, _ = balmer(ew, el)
            inside = bool(np.all((th[ix] >= bounds[:, 0]) & (th[ix] <= bounds[:, 1])))
            prof[n] = dict(
                lnl=lnl(th),
                start=start,
                theta=th[ix],
                chi2=c2,
                win=win,
                lum=lum,
                nfev=res.nfev,
                inside=inside,
            )
            say(
                f"  dust1_index {n:+.2f}: lnL start {start:.3f}, end {prof[n]['lnl']:.3f}, "
                f"{res.nfev} calls, {time.time() - t0:.0f} s, inside prior {inside}"
            )
        ssp.params["dust1_index"] = -1.0

        pref = prof[-1.0]
        say(
            "\ndchi2 = -2 (lnL - lnL of the profile at -1.0). Window columns are chi2 changes against the profile at -1.0."
        )
        say(
            f"{'d1_index':>8s} {'dchi2':>8s} {'dchi2_0':>8s}"
            + "".join(f" {s[:9]:>9s}" for s in SHOW)
            + f" {'Hb/Ha%':>7s}"
            + "".join(f" {k[:10]:>10s}" for k in FREE)
        )
        for n, p in prof.items():
            r = (
                p["lum"]["Hbeta"]
                / p["lum"]["Halpha"]
                / (ref["lum"]["Hbeta"] / ref["lum"]["Halpha"])
            )
            say(
                f"{n:8.2f} {-2 * (p['lnl'] - pref['lnl']):+8.2f} {-2 * (p['lnl'] - lnl0):+8.2f}"
                + "".join(f" {p['win'][s] - pref['win'][s]:+9.2f}" for s in SHOW)
                + f" {100 * (r - 1):+7.2f}"
                + "".join(f" {v:10.4f}" for v in p["theta"])
            )
        say(
            f"{'best':>8s} {'':>8s} {'':>8s}"
            + "".join(f" {'':>9s}" for _ in SHOW)
            + f" {'':>7s}"
            + "".join(f" {v:10.4f}" for v in best[ix])
        )
        say("dchi2_0 is against best at -1.0 (not re-optimized). Hb/Ha% is against best at -1.0.")
    finally:
        ssp.params["dust1_index"] = -1.0

    lnl_end = lnl(best)
    say(f"\nrestored dust1_index {ssp.params['dust1_index']}, lnL at best {lnl_end:.4f}")
    assert np.isclose(lnl_end, lnl0, atol=1e-6)
    (FIG / "dust1_index_check.txt").write_text("\n".join(out) + "\n")
    print("wrote dust1_index_check.txt")


if __name__ == "__main__":
    main()
