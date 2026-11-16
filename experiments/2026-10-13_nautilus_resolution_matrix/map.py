"""MAP fits of 39627770174637084 with the resolution matrix and the median LSF, with and without line masks.

Runs rmatrix and median from the 10-09 ER max-L point with each mask setting. Expected chi2 to drop
by the missing-line excess and the line results not to depend on the mask. Run from the repository
root, one process per run, as
``PYTHONPATH=<site with the C3K_ER fsps>:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=<SPS_HOME with the ER spectra> uv run --no-sync python
experiments/2026-10-13_nautilus_resolution_matrix/map.py {rmatrix,median}
{nomask,mask,mask_noPa16,mask_noH1112,mask_noPa1112,mask_noHeI} [--maxfev N]``, then ``... map.py report``.
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from model_setup import (
    ER_RESULTS,
    MISSING_LINES,
    TID,
    get_cue,
    load_data,
    make_model,
    make_obs,
    make_params,
)

from hubersed.fitting.map_fits import map_fit, sfh_from_theta
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

HYPERS = ("sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn")
OUT = PATHS["RESULTS"] / "2026-10-13_nautilus_resolution_matrix"
START = ER_RESULTS / f"{TID}_er_broad_shared_split45_rest9000_seed0_pred.npz"
# masked missing lines of each run; mask_noPa16 and mask_noH1112 find what drives sigma_smooth
MASKS = {
    "nomask": (),
    "mask": tuple(MISSING_LINES),
    "mask_noPa16": tuple(k for k in MISSING_LINES if k != "Pa16"),
    "mask_noH1112": tuple(k for k in MISSING_LINES if k not in ("H11", "H12")),
    "mask_noPa1112": tuple(k for k in MISSING_LINES if k not in ("Pa11", "Pa12")),
    "mask_noHeI": tuple(k for k in MISSING_LINES if not k.startswith("He I")),
}
RUNS = [(lsf, m) for m in MASKS for lsf in ("median", "rmatrix")]
MAGGIE_HZ_TO_1E17 = 3631e-23 / 1e-17
LINES = {
    "[OII]3726": "[O II] 3726",
    "[OII]3729": "[O II] 3729",
    "Hbeta": "Ba-beta 4861",
    "[OIII]5007": "[O III] 5007",
    "Halpha": "Ba-alpha 6563",
    "[NII]6584": "[N II] 6584",
    "[SII]6716": "[S II] 6716",
}
WINDOWS = {
    "[OIII]5007": (4995, 5022),
    "[OIII]4959": (4950, 4970),
    "Halpha": (6556, 6572),
    "Hbeta": (4845, 4880),
}
PLOT_WINDOWS = {
    "[OII]": (3718, 3740),
    "Hbeta": (4850, 4876),
    "[OIII]5007 + He I 5016": (4997, 5025),
    "Halpha+[NII]": (6540, 6595),
    "He I 7281": (7270, 7296),
    "Pa12": (8740, 8766),
}


def build(lsf, mask):
    """Return the model, observation, Cue source and start point of one run."""
    d = np.load(START, allow_pickle=True)
    best = dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))
    z, flux, unc, good = load_data(MASKS[mask])
    assert np.isclose(z, float(d["z"]))
    params = make_params(z)
    for k in HYPERS:
        params[k]["isfree"] = False
        params[k]["init"] = float(best[k])
    model = make_model(params, lsf)
    theta0 = np.array([best[k] for k in model.theta_labels()])
    return model, make_obs(flux, unc, good, lsf), get_cue(), theta0


def run(lsf, mask, maxfev):
    """Run one MAP and save the start, the result and the optimizer info."""
    model, obs, cue, theta0 = build(lsf, mask)
    t0 = time.time()
    pred0 = model.predict(theta0, observations=obs, sps=cue)[0][0]
    o = obs[0]
    chi2_0 = float(np.sum(((o.flux - pred0)[o.mask] / o.uncertainty[o.mask]) ** 2))
    print(f"{lsf} {mask}: one prediction {time.time() - t0:.2f} s, chi2 at start {chi2_0:.1f}")
    best, info = map_fit(model, obs, cue, n_seeds=0, maxfev=maxfev, theta0=theta0, tag=lsf)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / f"map_{lsf}_{mask}.npz",
        labels=np.array(model.theta_labels()),
        theta0=theta0,
        theta=best.x,
        fun=best.fun,
        nfev=best.nfev,
        success=best.success,
        maxfev=maxfev,
    )
    print(f"{lsf} {mask}: done, -lnp {best.fun:.2f}, nfev {best.nfev}, success {best.success}")


def describe(model, obs, cue, theta, masked_good):
    """Return the spectrum, chi2 parts, Cue line fluxes and recent sSFR at ``theta``."""
    spec = model.predict(theta, observations=obs, sps=cue)[0][0]
    o = obs[0]
    z = float(np.atleast_1d(model.params["zred"])[0])
    rest = o.wavelength / (1 + z)

    def c2(m):
        return float(np.sum(((o.flux - spec)[m] / o.uncertainty[m]) ** 2))

    parts = {"all own pixels": c2(o.mask), "masked pixel set": c2(o.mask & masked_good)}
    for key, (a, b) in WINDOWS.items():
        parts[key] = c2(o.mask & (rest >= a) & (rest < b))
    for name, lam in MISSING_LINES.items():
        parts[name] = c2(o.mask & (np.abs(rest - lam) < 4.0))
    names = np.char.strip(np.asarray(model.emline_info["name"]).astype(str))
    elum = model._eline_lum * model.flux_norm() / (1 + model._zred) * MAGGIE_HZ_TO_1E17
    fluxes = {k: float(elum[np.where(names == v)[0][0]]) for k, v in LINES.items()}
    return parts, fluxes, sfh_from_theta(model, theta), spec


def plot_windows(states, obs, z, masked_lines, path):
    """Plot the data and each run's spectrum in the line windows, with chi below on -7 to 7."""
    use_apj_style()
    o = obs[0]
    rest = o.wavelength / (1 + z)
    colors = {"median": "#b2182b", "rmatrix": "#e08214"}
    fig, ax = plt.subplots(
        2,
        len(PLOT_WINDOWS),
        figsize=(3.6 * len(PLOT_WINDOWS), 5.5),
        gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05},
        sharex="col",
    )
    for i, (key, (a, b)) in enumerate(PLOT_WINDOWS.items()):
        win = (rest >= a) & (rest < b)
        m = win & o.mask
        x = rest[win]
        fit = o.mask[win]
        ax[0, i].plot(x, np.where(fit, o.flux[win], np.nan), color="0.2", lw=0.9, label="DESI")
        for lsf, sp in states.items():
            c2 = float(np.sum(((o.flux[m] - sp[m]) / o.uncertainty[m]) ** 2))
            ax[0, i].plot(
                x,
                np.where(fit, sp[win], np.nan),
                color=colors[lsf],
                lw=1.0,
                label=rf"{lsf}, $\chi^2$ = {c2:.0f}",
            )
            chi = (o.flux[win] - sp[win]) / o.uncertainty[win]
            ax[1, i].plot(x, np.where(fit, chi, np.nan), color=colors[lsf], lw=0.9)
        for name in masked_lines:
            lam = MISSING_LINES[name]
            if a < lam < b:
                for row in ax[:, i]:
                    row.axvspan(lam - 4, lam + 4, color="0.9", zorder=0)
        ax[0, i].set_title(key)
        ax[0, i].legend(frameon=False, fontsize=6)
        for h in (-2, 2):
            ax[1, i].axhline(h, color="0.5", ls="--", lw=0.6)
        ax[1, i].axhline(0, color="0.3", lw=0.6)
        ax[1, i].set_ylim(-7, 7)
        ax[1, i].set_xlabel(r"rest wavelength [$\AA$]")
    ax[0, 0].set_ylabel("flux [maggies]")
    ax[1, 0].set_ylabel(r"$\chi$")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def report():
    """Compare the four MAP runs and write map.txt and the window figures."""
    runs = {
        r: np.load(OUT / f"map_{r[0]}_{r[1]}.npz")
        for r in RUNS
        if (OUT / f"map_{r[0]}_{r[1]}.npz").exists()
    }
    out = [
        f"MAP fits of {TID}, row-weighted recoadd with corrected ivar. SFH hyperparameters fixed at max-L."
    ]
    for (lsf, m), r in runs.items():
        out.append(
            f"{lsf} {m}: -lnp {float(r['fun']):.2f}, nfev {int(r['nfev'])} of {int(r['maxfev'])}, success {bool(r['success'])}"
        )
    masked_good = load_data(MASKS["mask"])[3]
    states, spectra, obs_of, z = {}, {m: {} for m in MASKS}, {}, None
    for (lsf, m), r in runs.items():
        model, obs, cue, _ = build(lsf, m)
        obs_of[m] = obs
        states[(lsf, m)] = describe(model, obs, cue, r["theta"], masked_good)
        spectra[m][lsf] = states[(lsf, m)][3]
        z = float(np.atleast_1d(model.params["zred"])[0])
    cols = list(states)
    head = " ".join(f"{lsf + ' ' + m:>16}" for lsf, m in cols)
    out += [
        "",
        "chi2 (each run on its own pixels; 'masked pixel set' is the same pixels for all)",
        f"{'':>16} {head}",
    ]
    for k in states[cols[0]][0]:
        out.append(f"{k:>16} " + " ".join(f"{states[c][0][k]:16.1f}" for c in cols))
    out += ["", "Cue line fluxes in 1e-17 erg/s/cm2", f"{'':>16} {head}"]
    for k in LINES:
        out.append(f"{k:>16} " + " ".join(f"{states[c][1][k]:16.1f}" for c in cols))
    out += ["", "log sSFR [1/yr] in the first three age bins"]
    for b in range(3):
        e = states[cols[0]][2]["edges_gyr"]
        out.append(
            f"{e[b] * 1e3:.2g}-{e[b + 1] * 1e3:.2g} Myr".rjust(16)
            + " "
            + " ".join(f"{np.log10(states[c][2]['ssfr'][b]):16.3f}" for c in cols)
        )
    labels = [str(x) for x in runs[cols[0]]["labels"]]
    out += ["", f"{'parameter':>17} {'max-L':>10} {head}"]
    for i, k in enumerate(labels):
        out.append(
            f"{k:>17} {runs[cols[0]]['theta0'][i]:10.4f} "
            + " ".join(f"{runs[c]['theta'][i]:16.4f}" for c in cols)
        )
    for m in MASKS:
        if spectra[m]:
            plot_windows(spectra[m], obs_of[m], z, MASKS[m], OUT / f"map_windows_{m}.png")
    text = "\n".join(out) + "\n"
    print(text)
    (OUT / "map.txt").write_text(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lsf", choices=("median", "rmatrix", "report"))
    ap.add_argument("mask", nargs="?", choices=tuple(MASKS), default="nomask")
    ap.add_argument("--maxfev", type=int, default=40000)
    a = ap.parse_args()
    if a.lsf == "report":
        report()
    else:
        run(a.lsf, a.mask, a.maxfev)
