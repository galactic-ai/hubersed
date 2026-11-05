"""Corner plots of the pocoMC posterior of the free-line fit, and its best point against the MAP.

Draws the weighted posterior of a poco.py state in three corner plots, with the map.py MAP point and
the pocoMC point of highest ln L + model.prior_product (among the 300 particles of highest ln L)
marked. The map.py line windows then show the data and both spectra with their chi.

INPUTS: pmc_final.state of poco.py copied from ls6, map_free.npz, the 10-13 _pred.npz. SEED: none.
For another target set FIT_TID and pass its 2026-10-18_candidate_map npz as --map.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix:experiments/2026-10-15_broadline_free_amplitude
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-15_broadline_free_amplitude/poco_plots.py --state <path>``.

OUTPUT: poco_corner_{gas_lines,stars_dust,sfh}.png, poco_windows.png and poco_best.txt next to
the state file.
"""

import argparse
from pathlib import Path

import astropy.units as u
import corner
import fit
import matplotlib.pyplot as plt
import numpy as np
from compare_poco import load_sampler, map_point
from map import WINDOWS, chi2

from hubersed.conversion import to_flambda
from hubersed.plotting.style import use_apj_style

GROUPS = {
    "gas_lines": [
        "gas_logz",
        "gas_logu",
        "gas_lognH",
        "gas_logno",
        "gas_logco",
        "eline_sigma",
        "eline_sigma_forb",
        "eline_sigma_broad",
        "eline_sigma_broad_forb",
        "eline_vbroad",
    ],
    "stars_dust": [
        "logzsol",
        "logmass",
        "dust2",
        "dust_ratio",
        "dust_index",
        "frac_nodust",
        "sigma_smooth",
        "sigma_reg",
        "tau_eq",
        "sigma_dyn",
        "tau_dyn",
    ],
    "sfh": [f"logsfr_ratios_{i}" for i in range(1, 10)],
}
COLORS = {"MAP": "#b2182b", "pocoMC best": "#1b7837", "posterior": "#2166ac"}
N_TOP = 300


def desi(wave, maggies):
    """Return maggies as DESI flux density, 1e-17 erg/s/cm^2/A."""
    return to_flambda(wave * u.AA, maggies * u.mgy).value


def corner_plot(theta, w, labels, points, group, path, variant):
    """Draw one corner plot of ``group`` with the marked ``points``, skipping fixed parameters."""
    names = [p for p in GROUPS[group] if p in labels]
    idx = [labels.index(p) for p in names]
    k = len(idx)
    fig = plt.figure(figsize=(2.4 * k, 2.4 * k))
    corner.corner(
        theta[:, idx],
        weights=w,
        fig=fig,
        color=COLORS["posterior"],
        labels=names,
        plot_datapoints=False,
        plot_density=False,
        smooth=1.0,
        bins=30,
        levels=(0.393, 0.865),
        max_n_ticks=3,
        label_kwargs={"fontsize": 16},
        hist_kwargs={"density": True},
        quantiles=(0.16, 0.5, 0.84),
    )
    for name, p in points.items():
        corner.overplot_lines(fig, p[idx], color=COLORS[name], lw=1.5)
        corner.overplot_points(fig, p[idx][None], color=COLORS[name], marker="s", ms=6)
    handles = [plt.Line2D([], [], color=c, lw=3) for c in COLORS.values()]
    fig.legend(handles, list(COLORS), loc="upper right", fontsize=16, frameon=False)
    fig.suptitle(f"{fit.TID} pocoMC {variant}, {group}", fontsize=20, y=1.01)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def windows_plot(obs, z, spectra, path):
    """Plot the data and each spectrum in the line windows in DESI units, with chi below."""
    o = obs[0]
    rest = o.wavelength / (1 + z)
    fig, ax = plt.subplots(
        2,
        len(WINDOWS),
        figsize=(2.8 * len(WINDOWS), 5.0),
        gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05},
        sharex="col",
    )
    for i, (key, (a, b)) in enumerate(WINDOWS.items()):
        win = (rest >= a) & (rest < b)
        fitted = o.mask[win]
        x = rest[win]
        data = desi(o.wavelength[win], o.flux[win])
        ax[0, i].plot(x, np.where(fitted, data, np.nan), color="0.2", lw=0.9, label="DESI")
        for name, spec in spectra.items():
            c2, _ = chi2(obs, spec, win)
            m = desi(o.wavelength[win], spec[win])
            ax[0, i].plot(x, np.where(fitted, m, np.nan), color=COLORS[name], lw=1.0,
                          label=rf"{name}, $\chi^2$ = {c2:.0f}")  # fmt: skip
            chi = (o.flux[win] - spec[win]) / o.uncertainty[win]
            ax[1, i].plot(x, np.where(fitted, chi, np.nan), color=COLORS[name], lw=0.9)
        ax[0, i].set_title(key)
        ax[0, i].legend(frameon=False, fontsize=5)
        for h in (-2, 2):
            ax[1, i].axhline(h, color="0.5", ls="--", lw=0.6)
        ax[1, i].axhline(0, color="0.3", lw=0.6)
        ax[1, i].set_ylim(-7, 7)
        ax[1, i].set_xlabel(r"rest wavelength [$\AA$]")
    ax[0, 0].set_ylabel(r"flux [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    ax[1, 0].set_ylabel(r"$\chi$")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    """Write the corner plots, the windows plot and the best-point summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--variant", default="free")
    parser.add_argument("--map", type=Path, default=None, help="MAP npz, default map_free.npz")
    args = parser.parse_args()
    use_apj_style()

    s = load_sampler(args.state)
    usamp, w, logl, _ = s.posterior()
    w = w / w.sum()
    model, obs, cue = fit.process_state(args.variant)
    labels = list(model.theta_labels())
    theta = np.array([model.prior_transform(ui) for ui in usamp])

    top = np.argsort(logl)[-N_TOP:]
    lnp = np.array([logl[i] + model.prior_product(theta[i]) for i in top])
    best = theta[top[np.argmax(lnp)]]
    theta_map = map_point(model, args.map)
    lnl_map = fit.loglike(theta_map, args.variant)
    lnp_map = lnl_map + model.prior_product(theta_map)
    points = {"MAP": theta_map, "pocoMC best": best}

    spectra = {k: model.predict(p, observations=obs, sps=cue)[0][0] for k, p in points.items()}
    z = float(np.atleast_1d(model.params["zred"])[0])
    lines = [
        f"{args.state}",
        f"ln L: MAP {lnl_map:.2f}, pocoMC best {logl[top[np.argmax(lnp)]]:.2f}, "
        f"pocoMC max {np.max(logl):.2f}",
        f"ln L + ln prior: MAP {lnp_map:.2f}, pocoMC best {np.max(lnp):.2f}",
    ]
    for k, spec in spectra.items():
        c2, n = chi2(obs, spec)
        lines.append(f"{k}: chi2 {c2:.1f} on {n} px, reduced {c2 / n:.3f}")
    lines += ["", f"{'parameter':>24} {'MAP':>10} {'pocoMC best':>12}"]
    lines += [f"{k:>24} {theta_map[i]:10.4g} {best[i]:12.4g}" for i, k in enumerate(labels)]
    text = "\n".join(lines)
    print(text)
    out = args.state.parent
    (out / "poco_best.txt").write_text(text + "\n")

    for group in GROUPS:
        corner_plot(theta, w, labels, points, group, out / f"poco_corner_{group}.png", args.variant)
    windows_plot(obs, z, spectra, out / "poco_windows.png")


if __name__ == "__main__":
    main()
