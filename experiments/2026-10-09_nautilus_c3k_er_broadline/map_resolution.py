"""MAP refit of the ER max-L point of 39627770174637084 with its resampled resolution matrix.

QUESTION: when the model goes through this target's own DESI resolution matrix (resampled onto a
fine grid as R', resolution_model.py) instead of the median Gaussian LSF, how do the line
luminosities, line widths, dust and recent star formation move, and does the chi2 improve?

HYPOTHESIS: R keeps 86-90% of the strong-line flux at the max-L widths (eval_resolution.txt), so
the Cue line fluxes rise by 10-15%, through the young SFH bins, dust or gas. eline_sigma_forb leaves
its 10 km/s floor toward 16-21 km/s. dust_ratio or dust2 shift, since R lowers Hbeta/Halpha by about
2.5%.

Two runs from the same start, the ER max-L point of the 10-09 nautilus run.

- control, the fit's own model with the median LSF.
- R, ResampledRModel with this target's merged matrix from resolution_matrix.npz.

In both the SFH hyperparameters sigma_reg, tau_eq, sigma_dyn and tau_dyn are fixed at max-L (MAP
does not recover them, so the 10-08 decision dropped MAP fits), and tau_in is fixed as in fit.py.
Every other parameter is free. One Powell start (hubersed.fitting.map_fits.map_fit, no jitter
starts). The control separates the optimizer's own drift from the change of resolution model.

INPUTS: the ``_pred.npz`` of predict.py, resolution_matrix.npz of resolution_check.py, and the DESI
spectrum as fit.load_data reads it. SEED: none (one start, no jitter).

COMMAND: from the repository root on the Mac with the C3K_ER build, one process per run,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-09_nautilus_c3k_er_broadline
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-09_nautilus_c3k_er_broadline/map_resolution.py {control,R} [--maxfev N]``,
then ``... map_resolution.py report`` once both are done.

OUTPUT: map_resolution_{control,R}.npz and map_resolution.txt in
results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/.
"""

import argparse
import time

import numpy as np
from fit import load_data, make_obs, make_params
from profile_check import FB, OUT, REST_MAX, SPLIT, STEM, TID, WINDOWS
from resolution_model import ResampledRModel, fine_grid, resampled_matrix

from hubersed.fitting.map_fits import get_sps, map_fit, sfh_from_theta
from hubersed.sps import broadline

HYPERS = ("sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn")
DEST = OUT / str(TID)
MAGGIE_HZ_TO_1E17 = 3631e-23 / 1e-17  # a line flux in maggies Hz, in 1e-17 erg/s/cm2
LINES = {
    "[OII]3726": "[O II] 3726",
    "[OII]3729": "[O II] 3729",
    "Hbeta": "Ba-beta 4861",
    "[OIII]5007": "[O III] 5007",
    "Halpha": "Ba-alpha 6563",
    "[NII]6584": "[N II] 6584",
    "[SII]6716": "[S II] 6716",
}


def build(which):
    """Return the model, observation, Cue source and start point of one run."""
    d = np.load(OUT / f"{STEM}_pred.npz", allow_pickle=True)
    labels = [str(x) for x in d["labels"]]
    best = dict(zip(labels, d["best"], strict=True))
    z, flux, unc, good = load_data(TID, REST_MAX)
    assert np.isclose(z, float(d["z"]))
    params = make_params(z, FB, SPLIT)
    for k in HYPERS:
        params[k]["isfree"] = False
        params[k]["init"] = float(best[k])
    if which == "R":
        rm = np.load(DEST / "resolution_matrix.npz")
        fine = fine_grid(rm["wave"], rm["res"].shape[0])
        model = ResampledRModel(params)
        model.set_resolution(resampled_matrix(rm["wave"], rm["res"], fine), fine)
    else:
        model = broadline.TwoCompLineModel(params)
    theta0 = np.array([best[k] for k in model.theta_labels()])
    obs = make_obs(flux, unc, good)
    cue = get_sps(zero_library_resolution=False)["cue"]
    assert cue.ssp.libraries[1] in ("c3k_er", b"c3k_er"), cue.ssp.libraries
    return model, obs, cue, theta0


def run(which, maxfev):
    """Run one MAP and save the start, the result and the optimizer info."""
    model, obs, cue, theta0 = build(which)
    t0 = time.time()
    pred0 = model.predict(theta0, observations=obs, sps=cue)[0][0]
    o = obs[0]
    chi2_0 = float(np.sum(((o.flux - pred0)[o.mask] / o.uncertainty[o.mask]) ** 2))
    print(
        f"{which}: one prediction {time.time() - t0:.2f} s, chi2 at start {chi2_0:.1f}", flush=True
    )
    assert np.all(np.isfinite(pred0))
    best, info = map_fit(model, obs, cue, n_seeds=0, maxfev=maxfev, theta0=theta0, tag=which)
    np.savez(
        DEST / f"map_resolution_{which}.npz",
        labels=np.array(model.theta_labels()),
        theta0=theta0,
        theta=best.x,
        fun=best.fun,
        nfev=best.nfev,
        success=best.success,
        maxfev=maxfev,
        fun_all=np.array(info["fun_all"]),
    )
    print(f"{which}: done, -lnp {best.fun:.2f}, nfev {best.nfev}, success {best.success}")


def describe(model, obs, cue, theta, linepix):
    """Return the spectrum, chi2 parts, line fluxes and recent sSFR at ``theta``."""
    spec = model.predict(theta, observations=obs, sps=cue)[0][0]
    o = obs[0]
    flux, unc, good = o.flux, o.uncertainty, o.mask
    z = float(np.atleast_1d(model.params["zred"])[0])
    rest = o.wavelength / (1 + z)

    def c2(m):
        return float(np.sum(((flux - spec)[m] / unc[m]) ** 2))

    parts = {"all": c2(good), "lines": c2(good & linepix), "continuum": c2(good & ~linepix)}
    for key, ((a, b), _) in WINDOWS.items():
        parts[key] = c2(good & (rest >= a) & (rest < b))
    names = np.char.strip(np.asarray(model.emline_info["name"]).astype(str))
    elum = model._eline_lum * model.flux_norm() / (1 + model._zred) * MAGGIE_HZ_TO_1E17
    fluxes = {k: float(elum[np.where(names == v)[0][0]]) for k, v in LINES.items()}
    sfh = sfh_from_theta(model, theta)
    return parts, fluxes, sfh


def report():
    """Compare the start and both MAP results and write map_resolution.txt."""
    runs = {w: np.load(DEST / f"map_resolution_{w}.npz") for w in ("control", "R")}
    out = [f"MAP refits of {TID} from the ER max-L point. SFH hyperparameters fixed at max-L."]
    for w, r in runs.items():
        out.append(
            f"{w}: -lnp start -> end, {float(r['fun_all'].max()):.2f} (one start) end {float(r['fun']):.2f}, "
            f"nfev {int(r['nfev'])} of maxfev {int(r['maxfev'])}, success {bool(r['success'])}"
        )
    states = {}
    for w in ("control", "R"):
        model, obs, cue, theta0 = build(w)
        if w == "control":
            model.predict(theta0, observations=obs, sps=cue)
            linepix = model._fix_eline_pixelmask.copy()
            states["max-L (control model)"] = describe(model, obs, cue, theta0, linepix)
            states["max-L through R"] = None
        states[f"MAP {w}"] = describe(model, obs, cue, runs[w]["theta"], linepix)
        if w == "R":
            states["max-L through R"] = describe(model, obs, cue, theta0, linepix)
    cols = list(states)
    out.append("")
    out.append(f"{'chi2 on':>12} " + " ".join(f"{c:>22}" for c in cols))
    for k in states[cols[0]][0]:
        out.append(f"{k:>12} " + " ".join(f"{states[c][0][k]:22.1f}" for c in cols))
    out.append("")
    out.append("Cue line fluxes in 1e-17 erg/s/cm2 (before any LSF)")
    for k in LINES:
        out.append(f"{k:>12} " + " ".join(f"{states[c][1][k]:22.1f}" for c in cols))
    hb_ha = " ".join(f"{states[c][1]['Hbeta'] / states[c][1]['Halpha']:22.4f}" for c in cols)
    out.append(f"{'Hbeta/Halpha':>12} {hb_ha}")
    out.append("")
    out.append("log sSFR [1/yr] in the first three age bins")
    for b in range(3):
        e = states[cols[0]][2]["edges_gyr"]
        lab = f"{e[b] * 1e3:.2g}-{e[b + 1] * 1e3:.2g} Myr"
        out.append(
            f"{lab:>12} " + " ".join(f"{np.log10(states[c][2]['ssfr'][b]):22.3f}" for c in cols)
        )
    out.append("")
    labels = [str(x) for x in runs["control"]["labels"]]
    assert labels == [str(x) for x in runs["R"]["labels"]]
    out.append(
        f"{'parameter':>17} {'max-L':>10} {'MAP control':>12} {'MAP R':>10} {'R - control':>12}"
    )
    for i, k in enumerate(labels):
        a, b, c = runs["control"]["theta0"][i], runs["control"]["theta"][i], runs["R"]["theta"][i]
        out.append(f"{k:>17} {a:10.4f} {b:12.4f} {c:10.4f} {c - b:+12.4f}")
    text = "\n".join(out) + "\n"
    print(text)
    (DEST / "map_resolution.txt").write_text(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("which", choices=("control", "R", "report"))
    ap.add_argument("--maxfev", type=int, default=6000)
    args = ap.parse_args()
    if args.which == "report":
        report()
    else:
        run(args.which, args.maxfev)
