"""Quick MAP of four line outliers with the free-line model, SFH hyperparameters fixed at #14's.

QUESTION: how well does the free-line model of 39627770174637084 (#14) fit four more line outliers
from the v2 MAP ranking, before any posterior run?

Each fit is map.py's ``free`` variant (experiments/2026-10-15_broadline_free_amplitude) on the
target's own data: the PR 2377 recoadd with the patched calibration ivar and the target's
resolution matrix (results/2026-10-18_candidate_resolution_matrix/<TID>/), with spender's mask and
the missing-line masks of model_setup.load_data. sigma_reg, tau_eq, sigma_dyn and tau_dyn are fixed
at the #14 values map.py used, the max-L point of the 10-13 rmatrix chain. The start is the #14
``free`` MAP point with logmass set to the target's v2 MAP value. One Powell start, 40000 calls.

model_setup reads TID, COADD and DR1_COADD as module globals, so they are set per target here.

INPUTS: map_free.npz of map.py, the 10-13 _pred.npz, the candidate recoadds and the v2 MAP pkls in
results/map_fits_outliers310_v2/. SEED: none.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=experiments/2026-10-15_broadline_free_amplitude:~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-18_candidate_map.py {fit,report} --tid <TID>``.

OUTPUT: <TID>_map_free.npz, <TID>_map.txt and <TID>_map.png in results/2026-10-18_candidate_map/.
"""

import argparse
import pickle
import time

import matplotlib.pyplot as plt
import model_setup
import numpy as np
from map import HYPERS, SPLIT, START, WINDOWS, amplitude_lines, chi2
from model_setup import MISSING_LINES, get_cue, load_data, make_obs, make_params

from hubersed.fitting.map_fits import map_fit
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style
from hubersed.sps.free_lines import FreeLineModel, add_free_line_params

OUT = PATHS["RESULTS"] / "2026-10-18_candidate_map"
REF_MAP = PATHS["RESULTS"] / "2026-10-15_broadline_free_amplitude" / "map_free.npz"
CANDIDATES = (39628357133926597, 39633278973513279, 39632961393395307, 39633001071509599)
SHOW = ("logmass", "logzsol", "dust2", "dust_ratio", "dust_index", "gas_logz", "gas_logu",
        "gas_lognH", "gas_logno", "eline_sigma", "eline_sigma_broad", "eline_sigma_broad_forb",
        "eline_vbroad")  # fmt: skip


def set_target(tid):
    """Point model_setup at the target's recoadd and DR1 matrix files."""
    folder = PATHS["RESULTS"] / "2026-10-18_candidate_resolution_matrix" / str(tid)
    model_setup.TID = tid
    model_setup.COADD = folder / "resolution_matrix_rows_ivar.npz"
    model_setup.DR1_COADD = folder / "resolution_matrix.npz"


def build(tid):
    """Return the model, observation, Cue source and start point of one target."""
    set_target(tid)
    d = np.load(START, allow_pickle=True)
    hyper = dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))
    z, flux, unc, good = load_data(tuple(MISSING_LINES))
    params = make_params(z)
    for k in HYPERS:
        params[k]["isfree"] = False
        params[k]["init"] = float(hyper[k])
    params = add_free_line_params(params, sigma_split=SPLIT, auroral_width=np.inf)
    model = FreeLineModel(params)
    ref = np.load(REF_MAP)
    start = dict(zip([str(x) for x in ref["labels"]], ref["theta"], strict=True))
    v2 = pickle.load(open(PATHS["RESULTS"] / "map_fits_outliers310_v2" / f"{tid}.pkl", "rb"))
    start["logmass"] = float(np.atleast_1d(v2["theta_dict"]["logmass"])[0])
    theta0 = np.array([start[k] for k in model.theta_labels()])
    return model, make_obs(flux, unc, good, "rmatrix"), get_cue(), theta0


def fit(tid, maxfev):
    """Run one MAP and save it."""
    model, obs, cue, theta0 = build(tid)
    t = time.time()
    model.predict(theta0, observations=obs, sps=cue)
    print(f"{tid}: one prediction {time.time() - t:.2f} s", flush=True)
    best, _ = map_fit(model, obs, cue, n_seeds=0, maxfev=maxfev, theta0=theta0, tag=str(tid))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / f"{tid}_map_free.npz",
        labels=np.array(model.theta_labels()),
        theta0=theta0,
        theta=best.x,
        fun=best.fun,
        nfev=best.nfev,
        success=best.success,
        maxfev=maxfev,
    )
    print(f"{tid}: -lnp {best.fun:.2f}, nfev {best.nfev}, success {best.success}", flush=True)


def report(tid):
    """Write the chi2 table and parameters, and plot the spectrum and line windows."""
    r = np.load(OUT / f"{tid}_map_free.npz")
    model, obs, cue, _ = build(tid)
    spec = model.predict(r["theta"], observations=obs, sps=cue)[0][0]
    o = obs[0]
    z = float(np.atleast_1d(model.params["zred"])[0])
    rest = o.wavelength / (1 + z)
    c2, n = chi2(obs, spec)
    nfree = len(r["theta"])
    out = [
        f"MAP of {tid} (z {z:.5f}), free-line model, SFH hyperparameters fixed at #14's",
        f"-lnp {float(r['fun']):.2f}, nfev {int(r['nfev'])} of {int(r['maxfev'])}, "
        f"success {bool(r['success'])}, marginal term {float(model._ln_eline_penalty):.2f}",
        f"chi2 {c2:.1f} on {n} px, chi2/(n - {nfree}) {c2 / (n - nfree):.3f}",
        "",
        "chi2 per window (pixels)",
    ]
    for key, (a, b) in WINDOWS.items():
        cw, nw = chi2(obs, spec, (rest >= a) & (rest < b))
        out.append(f"  {key:>13} {cw:8.1f} ({nw})")
    labels = list(r["labels"])
    out += ["", f"{'parameter':>24} {'start':>9} {'MAP':>9}"]
    for k in SHOW:
        i = labels.index(k)
        out.append(f"{k:>24} {r['theta0'][i]:9.3f} {r['theta'][i]:9.3f}")
    try:
        out += ["", *amplitude_lines(model)]
    except ZeroDivisionError:  # an ion with zero narrow or Cue flux; list the raw amplitudes
        out += ["", "amplitude_lines failed (zero narrow or Cue flux); raw free amplitudes:"]
        out += [
            f"  {c['name']:>12} {c['kind']:>6} amp {c['amp']:.4g} +- {c['sd']:.2g}, "
            f"cue {c['cue']:.4g}, seen {c['seen']}"
            for c in model.free_amplitudes
        ]
    text = "\n".join(out) + "\n"
    (OUT / f"{tid}_map.txt").write_text(text)
    print(text)

    use_apj_style()
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(3, len(WINDOWS), height_ratios=[2.2, 1.6, 0.8], hspace=0.35)
    top = fig.add_subplot(gs[0, :])
    m = o.mask
    top.plot(rest, np.where(m, o.flux, np.nan), color="0.2", lw=0.5, label="data")
    top.plot(rest, np.where(m, spec, np.nan), color="C3", lw=0.7, label="MAP")
    top.set_xlabel(r"rest wavelength [$\AA$]")
    top.set_ylabel("flux [maggies]")
    top.legend(loc="upper right")
    top.set_title(f"{tid}: chi2/(n - {nfree}) = {c2 / (n - nfree):.3f}, n = {n}")
    for j, (key, (a, b)) in enumerate(WINDOWS.items()):
        sel = m & (rest >= a) & (rest < b)
        ax = fig.add_subplot(gs[1, j])
        axc = fig.add_subplot(gs[2, j], sharex=ax)
        ax.plot(rest[sel], o.flux[sel], color="0.2", lw=0.9)
        ax.plot(rest[sel], spec[sel], color="C3", lw=0.9)
        axc.plot(rest[sel], ((o.flux - spec) / o.uncertainty)[sel], color="C3", lw=0.8)
        axc.axhline(0, color="0.5", lw=0.6)
        cw, nw = chi2(obs, spec, (rest >= a) & (rest < b))
        ax.set_title(f"{key}\nchi2/N {cw / max(nw, 1):.1f}", fontsize=9)
        ax.tick_params(labelsize=7)
        axc.tick_params(labelsize=7)
    fig.savefig(OUT / f"{tid}_map.png", dpi=110, bbox_inches="tight")


def main():
    """Parse the step and target."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("fit", "report"))
    ap.add_argument("--tid", type=int, required=True, choices=CANDIDATES)
    ap.add_argument("--maxfev", type=int, default=40000)
    a = ap.parse_args()
    fit(a.tid, a.maxfev) if a.step == "fit" else report(a.tid)


if __name__ == "__main__":
    main()
