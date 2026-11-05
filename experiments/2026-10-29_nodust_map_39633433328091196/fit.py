"""MAPs of the continuum outlier 39633433328091196 with dust attenuation off, for C3K_ER and MILES.

QUESTION (C. Hahn, 2026-10-29): is the best fit without dust about as good as the one with dust?

``--lib er`` changes 2026-10-29_continuum_39633433328091196_map/fit.py (dusty MAP, icx job 3657895)
in one way: dust2 is fixed at 0, so dust1 = dust2 * dust_ratio is 0 and dust_index scales nothing;
dust_ratio and dust_index are fixed as well. Start 0 is the dusty MAP best point. All else is that
fit (C3K_ER + afe, DR1 R matrix, rest 3320-9000 A, SFH hyperparameters fixed at the C3K_HR max-L,
7 Powell starts).

``--lib miles`` is a MAP with the model, data and likelihood of the 10-08 run
miles_miles_neboff_nodeg (2026-10-08_continuum_outlier/fit.py: MILES build, native data, MILES
window, nebular off), with dust as in that run (``--dust on``) or off. The four SFH
hyperparameters, free in the 10-08 run, are fixed at its max-L point, as the ER fit fixes them,
since with them free a MAP drifts to their prior floors (hubersed.sps.config, notes of
build_full_cue_model). Start 0 is the 10-08 max-L point.

``--mask features`` (follow-up 3, C3K ER only) drops rest 3850-4450 A and 4600-4750 A from the
fitted pixels, where C3K and MILES SSPs differ most at fixed age and metallicity
(2026-10-29_c3k_vs_miles_ssp: Ca H and K, G band, C2 4668), and fits with dust on and off. Outputs go
to er_dust_features and er_nodust_features.

``--zmax 0.2`` (follow-up 2) caps the logzsol prior at +0.2 (Uniform -2.5 to 0.2 instead of 0.5),
the top of FSPS's MILES spectral grid (Z 0.030, zsol_spec 0.019; MILES spectra are not extrapolated,
sps_setup.f90), for either library; starts above the cap are moved to 0.195. Outputs carry _zmax0.2.

Steps check, fit and collect as in 2026-10-28_continuum_c3k_er_rmatrix_map/fit.py, whose step
code this file reuses. check also asserts, with dust off, that dust_index and dust_ratio leave the
spectrum unchanged, and for MILES with dust on, that the log likelihood at start 0 is the 10-08 max
log L. Run on a compute node through er.slurm (stampede3) or miles.slurm (ls6).
"""

import argparse
import importlib.util
import warnings
from pathlib import Path

import numpy as np
from prospect.fitting import lnprobfn
from prospect.models.priors import Uniform

from hubersed.paths import PATHS

TID = 39633433328091196
EXPS = Path(__file__).resolve().parents[1]
OUT = PATHS["RESULTS"] / "2026-10-29_nodust_map_39633433328091196"
ER_MAP = PATHS["RESULTS"] / "2026-10-29_continuum_39633433328091196_map" / f"{TID}_map.npz"
MILES_PRED = (
    PATHS["RESULTS"] / "2026-10-08_continuum_outlier" / f"{TID}_miles_miles_neboff_nodeg_pred.npz"
)
MILES_HYPERS = ("sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn")
MASKS = {"features": ((3850.0, 4450.0), (4600.0, 4750.0))}  # rest A left out of the fit


def load(name, path):
    """Return the module at ``path``, loaded by path since the experiment files share names."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def best_point(path):
    """Return the saved best point of a fit as a dict of label to value."""
    d = np.load(path)
    return dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))


def dust_off(t):
    """Fix the dust parameters of template ``t`` so that no attenuation is applied."""
    t["dust2"].update(isfree=False, init=0.0)
    t["dust_ratio"].update(isfree=False, init=1.0)
    t["dust_index"].update(isfree=False, init=0.0)
    return t


def cap_z(t, zmax):
    """Cap the logzsol prior of template ``t`` at ``zmax`` (the lower bound stays at -2.5)."""
    t["logzsol"]["prior"] = Uniform(mini=-2.5, maxi=zmax)
    t["logzsol"]["init"] = min(float(np.atleast_1d(t["logzsol"]["init"])[0]), zmax - 0.005)
    return t


def clip_z(point, zmax):
    """Return ``point`` with logzsol moved below ``zmax`` when it lies above."""
    out = dict(point)
    if zmax is not None:
        out["logzsol"] = min(out["logzsol"], zmax - 0.005)
    return out


def setup_er(dust=False, mask=None, zmax=None):
    """Return the 10-28 step module set up for the C3K_ER fit, dust on or off, optionally masked."""
    target = load("fit_target", EXPS / "2026-10-29_continuum_39633433328091196_map" / "fit.py")
    base = target.base
    if not dust:
        make = base.ResolutionMatrixModel
        base.ResolutionMatrixModel = lambda t: make(dust_off(t))
    if mask:
        read = base.load_data

        def load_masked():
            z, flux, unc, good = read()
            rest = base.WAVE_OBS / (1 + z)
            for lo, hi in MASKS[mask]:
                good = good & ~((rest >= lo) & (rest <= hi))
            return z, np.where(good, flux, 0.0), np.where(good, unc, np.inf), good

        base.load_data = load_masked
    if zmax is not None:
        make_z = base.ResolutionMatrixModel
        base.ResolutionMatrixModel = lambda t: make_z(cap_z(t, zmax))
    best = clip_z(best_point(ER_MAP), zmax)
    base.start_values = lambda nebular: dict(best)
    name = f"er_{'dust' if dust else 'nodust'}" + (f"_{mask}" if mask else "")
    base.OUT = OUT / (name + (f"_zmax{zmax}" if zmax is not None else ""))
    label = f"C3K_ER + afe + R matrix, dust {'on' if dust else 'off'}"
    return base, label + (f", mask {mask}" if mask else "") + (
        f", logzsol <= {zmax}" if zmax is not None else ""
    )


def setup_miles(dust, zmax=None):
    """Return the 10-28 step module set up for the MILES fit, with dust on or off."""
    f08 = load("fit_1008", EXPS / "2026-10-08_continuum_outlier" / "fit.py")
    base = load("fit_1028", EXPS / "2026-10-28_continuum_c3k_er_rmatrix_map" / "fit.py")
    best = clip_z(best_point(MILES_PRED), zmax)
    make = f08.HyperSpecModel

    def model_of(t):
        for k in MILES_HYPERS:
            t[k].update(isfree=False, init=best[k])
        if zmax is not None:
            cap_z(t, zmax)
        return make(t if dust else dust_off(t))

    f08.HyperSpecModel = model_of

    def build(nebular=False):
        z, flux, unc, good, target_kms = f08.load_data(TID, "miles", "miles", False)
        sps = f08.make_sps(False, False)
        obs = f08.make_obs(z, flux, unc, good, target_kms, sps)
        model = f08.make_model(z, False, "none")
        theta0 = np.array([best[k] for k in model.theta_labels()], float)
        return model, obs, sps, theta0, dict(z=z, flux=flux, unc=unc, good=good)

    base.build = build
    base.TID = TID
    name = "miles_dust" if dust else "miles_nodust"
    base.OUT = OUT / (name + (f"_zmax{zmax}" if zmax is not None else ""))
    label = f"MILES, dust {'on' if dust else 'off'}"
    return base, label + (f", logzsol <= {zmax}" if zmax is not None else "")


def predict(model, obs, sps, theta):
    """Return the model spectrum at ``theta``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.predict(theta, observations=obs, sps=sps)[0][0], float)


def extra_checks(base, lib, dust, zmax=None):
    """Check that dust is off where it should be, and the MILES setup against the 10-08 fit."""
    model, obs, sps, theta0, _ = base.build()
    print(f"free {model.theta_labels()}")
    if not dust:
        s0 = predict(model, obs, sps, theta0)
        model.params["dust_index"] = np.array([-1.0])
        model.params["dust_ratio"] = np.array([0.5])
        s1 = predict(model, obs, sps, theta0)
        print(
            f"dust off: max rel change from dust_index and dust_ratio {np.max(np.abs(s1 / s0 - 1)):.2e}"
        )
        assert np.allclose(s1, s0, rtol=1e-10, atol=0), "dust is not off"
    if lib == "miles" and dust and zmax is None:  # the 10-08 max-L lies above any cap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lnl = lnprobfn(theta0, model=model, observations=obs, sps=sps, nested=True)
        max_l = float(np.load(MILES_PRED)["log_l"].max())
        print(f"MILES log L at start 0 {lnl:.2f}, 10-08 max log L {max_l:.2f}")
        assert np.isclose(lnl, max_l, atol=0.1), "MILES setup differs from the 10-08 run"


def collect(base, label, n_starts):
    """Keep the start with the lowest -lnp, save its spectrum and chi2, and write map.txt."""
    out = base.OUT
    paths = {k: out / f"map_s{k}.npz" for k in range(n_starts)}
    missing = [k for k, p in paths.items() if not p.exists()]
    runs = {k: np.load(p) for k, p in paths.items() if p.exists()}
    runs = {k: r for k, r in runs.items() if np.isfinite(float(r["fun"]))}
    k_best = min(runs, key=lambda k: float(runs[k]["fun"]))
    r = runs[k_best]
    model, obs, sps, theta0, data = base.build()
    spec = predict(model, obs, sps, r["theta"])
    g = data["good"]
    chi2 = float(np.sum(((data["flux"][g] - spec[g]) / data["unc"][g]) ** 2))
    np.savez(
        out / f"{TID}_map.npz",
        wave=base.WAVE_OBS,
        z=data["z"],
        flux=data["flux"],
        unc=data["unc"],
        good=g,
        labels=r["labels"],
        best=r["theta"],
        sp_best=spec,
        chi2=chi2,
        start=k_best,
        fun_all=np.array([float(runs[k]["fun"]) for k in sorted(runs)]),
    )
    lines = [f"MAP of {TID}, {label}, best start {k_best} of {n_starts}"]
    lines += [f"missing starts {missing}", f"chi2 {chi2:.2f} on {g.sum()} px", ""]
    lines += [
        f"start {k}: -lnp {float(runs[k]['fun']):.2f}, nfev {int(runs[k]['nfev'])} of "
        f"{int(runs[k]['maxfev'])}, success {bool(runs[k]['success'])}"
        for k in sorted(runs)
    ]
    lines += ["", f"{'parameter':>16} {'start 0':>10} {'MAP':>10}"]
    for name, a, b in zip(r["labels"], theta0, r["theta"], strict=True):
        lines.append(f"{name!s:>16} {a:10.4f} {b:10.4f}")
    text = "\n".join(lines) + "\n"
    (out / "map.txt").write_text(text)
    print(text)


def main():
    """Parse the library, dust switch and step, and run it."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("check", "fit", "collect"))
    ap.add_argument("--lib", choices=("er", "miles"), required=True)
    ap.add_argument("--dust", choices=("on", "off"), default="off")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n-starts", type=int, default=7)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--maxfev", type=int, default=150000)
    ap.add_argument("--n-calls", type=int, default=7, help="warm calls timed by check")
    ap.add_argument("--mask", choices=("none", *MASKS), default="none")
    ap.add_argument("--zmax", type=float, default=None, help="cap of the logzsol prior")
    a = ap.parse_args()
    dust = a.dust == "on"
    mask = None if a.mask == "none" else a.mask
    assert a.lib == "er" or mask is None, "masks are for the C3K_ER follow-up only"
    assert a.lib == "miles" or not dust or mask or a.zmax is not None, (
        "the unmasked dusty C3K_ER MAP is job 3657895"
    )
    base, label = setup_er(dust, mask, a.zmax) if a.lib == "er" else setup_miles(dust, a.zmax)
    if a.step == "check":
        base.check(a.n_starts, a.jitter, a.n_calls)
        extra_checks(base, a.lib, dust, a.zmax)
    elif a.step == "fit":
        base.fit(a.start, a.n_starts, a.jitter, a.maxfev, "")
    else:
        collect(base, label, a.n_starts)


if __name__ == "__main__":
    main()
