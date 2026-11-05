"""MAP fits of 39627770174637084 with the free-amplitude broadline model against model A.

QUESTION: does the free-line model (hubersed.sps.free_lines, spec in UberSED
knowledge/broadline_free_amplitude_spec_2026-10-15.md) fit as well as model A, and how do the gas
parameters and line amplitudes move?

HYPOTHESIS: chi2 drops or stays. [SII] and [OI] leave Cue, so U, n_H and Z move. The broad Balmer
fraction stays near model A's 0.18.

The line model is as in fit.py. Data, mask and start are those of the 10-13 rmatrix chain, with the
SFH hyperparameters at its max-L and one Powell start. Model A is rerun from the same start.
``check`` asserts that the free-line model with zero widths and no broad part reproduces model A.

RESULT: 2026-10-15, map.txt. chi2 on 5530 px: A 6058.8, free 5887.3, free_aur0.3 5896.0. The gain is
in [NeIII], [OII]3727, [OIII]4363, [OI] and [SII]. Broad f_b: H 0.167 (A 0.179), forbidden 0.09 to
0.21; sigma_b 87.9 and sigma_f 85.9 km/s. 12+log(O/H) goes from 7.96 (A) to 8.67 (free) and 8.61
(free_aur0.3), against direct 8.23, and C/O leaves its ceiling. Cue now over-predicts narrow [SII]
and [OI] and is too cold for [OIII]4363 and [OII]7325. U and n_H differ by up to 0.9 dex between the
two auroral settings.

INPUTS: the 10-13 <TID>_rmatrix_mask_..._pred.npz (start) and the 10-09 recoadd files that
model_setup.load_data reads. SEED: none.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-15_broadline_free_amplitude/map.py {check,A,free,free_aur0.3,report}``.

OUTPUT: map_<variant>.npz, map.txt, map_windows.png and map_windows_A_free.png in
results/2026-10-15_broadline_free_amplitude/.
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from model_setup import MISSING_LINES, TID, get_cue, load_data, make_obs, make_params

from hubersed.fitting.map_fits import map_fit
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style
from hubersed.sps.free_lines import FreeLineModel, add_free_line_params
from hubersed.sps.resolution_model import ResolutionMatrixModel

OUT = PATHS["RESULTS"] / "2026-10-15_broadline_free_amplitude"
START = (
    PATHS["RESULTS"]
    / "2026-10-13_nautilus_resolution_matrix"
    / f"{TID}_rmatrix_mask_er_broad_shared_split45_rest9000_seed0_pred.npz"
)
HYPERS = ("sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn")
SPLIT = 45.0
VARIANTS = {"A": None, "free": np.inf, "free_aur0.3": 0.3}
CUE_OH_ZERO = 8.71
WINDOWS = {
    "[OII]3727": (3718, 3740),
    "[NeIII]3869": (3860, 3878),
    "[OIII]4363": (4355, 4372),
    "Hbeta": (4850, 4876),
    "[OIII]5007": (4997, 5020),
    "[OI]6300": (6290, 6312),
    "Halpha+[NII]": (6540, 6595),
    "[SII]": (6705, 6745),
    "[OII]7325": (7312, 7340),
}


def build(variant, widths=None):
    """Return the model, observation, Cue source and start point of one variant.

    Parameters
    ----------
    variant : str
        A key of ``VARIANTS``.
    widths : tuple of float, optional
        For the free-line model, (strong, auroral, free) prior widths and the broad switch,
        replacing the defaults. Used by ``check``.
    """
    d = np.load(START, allow_pickle=True)
    best = dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))
    z, flux, unc, good = load_data(tuple(MISSING_LINES))
    assert np.isclose(z, float(d["z"]))
    params = make_params(z)
    for k in HYPERS:
        params[k]["isfree"] = False
        params[k]["init"] = float(best[k])
    if variant == "A":
        model = ResolutionMatrixModel(params)
    else:
        params = add_free_line_params(params, sigma_split=SPLIT, auroral_width=VARIANTS[variant])
        if widths is not None:
            for k, w in zip(
                ("eline_strong_width", "eline_auroral_width", "eline_free_width", "eline_broad"),
                widths,
                strict=True,
            ):
                params[k]["init"] = float(w)
        model = FreeLineModel(params)
        best["eline_sigma_broad_forb"] = best["eline_sigma_broad"]
    theta0 = np.array([best[k] for k in model.theta_labels()])
    return model, make_obs(flux, unc, good, "rmatrix"), get_cue(), theta0


def chi2(obs, spec, sel=None):
    """Return chi2 and the pixel count over the fitted pixels, optionally in ``sel``."""
    o = obs[0]
    m = o.mask if sel is None else o.mask & sel
    return float(np.sum(((o.flux - spec)[m] / o.uncertainty[m]) ** 2)), int(m.sum())


def check():
    """Assert that the free-line model with no free amplitudes equals model A at f_b = 0."""
    ma, obs, cue, ta = build("A")
    ta[ma.theta_labels().index("eline_fbroad")] = 0.0
    sa = ma.predict(ta, observations=obs, sps=cue)[0][0]
    mf, obs_f, _, tf = build("free", widths=(0.0, 0.0, 0.0, 0.0))
    sf = mf.predict(tf, observations=obs_f, sps=cue)[0][0]
    assert not mf.free_amplitudes, mf.free_amplitudes
    assert np.allclose(sf, sa, rtol=1e-8, atol=0), np.max(np.abs(sf / sa - 1))
    print(
        f"OK: no free amplitudes reproduces model A at f_b = 0 (max rel diff {np.max(np.abs(sf / sa - 1)):.1e})"
    )
    mf, obs_f, _, tf = build("free")
    t0 = time.time()
    sf = mf.predict(tf, observations=obs_f, sps=cue)[0][0]
    print(
        f"free at the start: one prediction {time.time() - t0:.2f} s, chi2 {chi2(obs_f, sf)[0]:.1f}"
    )
    for c in mf.free_amplitudes:
        print(
            f"  {c['name']:>12} {c['kind']:>6} amp {c['amp']:.4g} +- {c['sd']:.2g}, coverage {c['coverage']:.2f}"
        )


def run(variant, maxfev):
    """Run one MAP and save the start, the result and the optimizer info."""
    model, obs, cue, theta0 = build(variant)
    best, _ = map_fit(model, obs, cue, n_seeds=0, maxfev=maxfev, theta0=theta0, tag=variant)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / f"map_{variant}.npz",
        labels=np.array(model.theta_labels()),
        theta0=theta0,
        theta=best.x,
        fun=best.fun,
        nfev=best.nfev,
        success=best.success,
        maxfev=maxfev,
    )
    print(f"{variant}: -lnp {best.fun:.2f}, nfev {best.nfev}, success {best.success}")


def amplitude_lines(model):
    """Return report lines for the free amplitudes at the last prediction.

    q is the narrow flux over Cue's. f_b is the broad share of the ion's flux.
    """
    amps = {(c["name"], c["kind"]): c for c in model.free_amplitudes}
    out = [f"{'ion':>12} {'narrow q':>16} {'broad r':>16} {'f_b':>6}  flags"]
    for ion in dict.fromkeys(n for n, _ in amps):
        n, b = amps.get((ion, "narrow")), amps.get((ion, "broad"))
        flags = []
        if n is None:
            q, sq, cue = 1.0, 0.0, (b or n)["cue"]
        elif np.isfinite(n["width"]):
            q, sq, cue = n["amp"], n["sd"], n["cue"]
        else:
            q, sq, cue = n["amp"] / n["cue"], n["sd"] / n["cue"], n["cue"]
        qs = f"{q:7.3f} +- {sq:5.3f}" if n is not None else "    1 (tied)    "
        if n is not None and not n["seen"]:
            flags.append("narrow unseen")
        rs, fs = f"{'-':>16}", f"{'-':>6}"
        if b is not None:
            r, sr = b["amp"] / (q * cue), b["sd"] / (q * cue)
            rs, fs = f"{r:7.3f} +- {sr:5.3f}", f"{r / (1 + r):6.3f}"
            if b["amp"] < 0:
                flags.append("broad negative")
        if n is not None and n["amp"] < 0:
            flags.append("narrow negative")
        out.append(f"{ion:>12} {qs:>16} {rs:>16} {fs:>6}  {', '.join(flags)}")
    return out


def report():
    """Compare the MAP runs and write map.txt and map_windows.png."""
    out = [f"MAP fits of {TID}, rmatrix chain data and mask, SFH hyperparameters fixed at max-L."]
    states = {}
    for v in VARIANTS:
        path = OUT / f"map_{v}.npz"
        if not path.exists():
            continue
        r = np.load(path)
        model, obs, cue, _ = build(v)
        spec = model.predict(r["theta"], observations=obs, sps=cue)[0][0]
        states[v] = (r, model, obs, spec, float(model._ln_eline_penalty))
        out.append(
            f"{v}: -lnp {float(r['fun']):.2f}, nfev {int(r['nfev'])} of {int(r['maxfev'])}, "
            f"success {bool(r['success'])}, marginal term {states[v][4]:.2f}"
        )
    if not states:
        print("no runs")
        return
    cols = list(states)
    z = float(np.atleast_1d(states[cols[0]][1].params["zred"])[0])
    rest = states[cols[0]][2][0].wavelength / (1 + z)
    head = " ".join(f"{v:>14}" for v in cols)
    out += ["", f"chi2 (pixels) {'':>3} {head}"]
    for key, sel in [("all", None)] + [
        (k, (rest >= a) & (rest < b)) for k, (a, b) in WINDOWS.items()
    ]:
        vals = [chi2(states[v][2], states[v][3], sel) for v in cols]
        out.append(f"{key:>13} ({vals[0][1]:4d}) " + " ".join(f"{c:14.1f}" for c, _ in vals))
    labels = sorted({k for v in cols for k in states[v][1].theta_labels()}, key=str)
    out += ["", f"{'parameter':>22} {head}"]
    for k in labels:
        row = []
        for v in cols:
            lab = states[v][1].theta_labels()
            row.append(f"{states[v][0]['theta'][lab.index(k)]:14.4f}" if k in lab else f"{'-':>14}")
        out.append(f"{k:>22} " + " ".join(row))
    for v in cols:
        lab = states[v][1].theta_labels()
        th = states[v][0]["theta"]
        out.append(f"{v}: 12+log(O/H) {th[lab.index('gas_logz')] + CUE_OH_ZERO:.3f}")
    for v in cols:
        if v != "A":
            out += ["", f"{v} free amplitudes at the MAP"] + amplitude_lines(states[v][1])
    if "A" in states:
        lab = states["A"][1].theta_labels()
        out.append(
            f"\nA: f_b (all lines, shared) {states['A'][0]['theta'][lab.index('eline_fbroad')]:.3f}"
        )
    plot_windows({v: (states[v][2], states[v][3]) for v in cols}, z, OUT / "map_windows.png")
    if "A" in states and "free" in states:
        pair = {v: (states[v][2], states[v][3]) for v in ("A", "free")}
        plot_windows(pair, z, OUT / "map_windows_A_free.png")
    text = "\n".join(out) + "\n"
    print(text)
    (OUT / "map.txt").write_text(text)


def plot_windows(states, z, path):
    """Plot the data and each run's spectrum in the line windows, with chi below on -7 to 7."""
    use_apj_style()
    colors = {"A": "#b2182b", "free": "#2166ac", "free_aur0.3": "#4dac26"}
    fig, ax = plt.subplots(
        2,
        len(WINDOWS),
        figsize=(2.8 * len(WINDOWS), 5.0),
        gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05},
        sharex="col",
    )
    for i, (key, (a, b)) in enumerate(WINDOWS.items()):
        for k, (v, (obs, spec)) in enumerate(states.items()):
            o = obs[0]
            rest = o.wavelength / (1 + z)
            win = (rest >= a) & (rest < b)
            fit = o.mask[win]
            x = rest[win]
            if k == 0:
                ax[0, i].plot(
                    x, np.where(fit, o.flux[win], np.nan), color="0.2", lw=0.9, label="DESI"
                )
            c2, _ = chi2(obs, spec, win)
            ax[0, i].plot(x, np.where(fit, spec[win], np.nan), color=colors[v], lw=1.0,
                          label=rf"{v}, $\chi^2$ = {c2:.0f}")  # fmt: skip
            chi = (o.flux[win] - spec[win]) / o.uncertainty[win]
            ax[1, i].plot(x, np.where(fit, chi, np.nan), color=colors[v], lw=0.9)
        ax[0, i].set_title(key)
        ax[0, i].legend(frameon=False, fontsize=5)
        for h in (-2, 2):
            ax[1, i].axhline(h, color="0.5", ls="--", lw=0.6)
        ax[1, i].axhline(0, color="0.3", lw=0.6)
        ax[1, i].set_ylim(-7, 7)
        ax[1, i].set_xlabel(r"rest wavelength [$\AA$]")
    ax[0, 0].set_ylabel("flux [maggies]")
    ax[1, 0].set_ylabel(r"$\chi$")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("check", "report", *VARIANTS))
    ap.add_argument("--maxfev", type=int, default=40000)
    a = ap.parse_args()
    if a.step == "check":
        check()
    elif a.step == "report":
        report()
    else:
        run(a.step, a.maxfev)
