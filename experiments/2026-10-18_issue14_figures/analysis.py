"""Figures of 39627770174637084 for issue #14: corner plot, residuals by run, and line windows.

The residual and window figures compare three fits, each against the data it saw.
1. MILES + Cue, 2026-09-24_nautilus_tauin. DR1 coadd, MILES window, not degraded.
2. C3K HR + Cue, 2026-09-25_nautilus_tauin_c3k_hr. The same data degraded to 43.6 km/s.
3. C3K ER + free-line broad model + Cue + resolution matrix + recoadd ivar, the pocoMC free_dust run
   of 2026-10-15_broadline_free_amplitude.
Each row's best fit is its maximum-likelihood point (rows 1 and 2) or the highest-posterior pocoMC
particle (row 3). Row 1 comes from 2026-10-18_issue14_miles.py, run first in the 09-24 environment,
because the current code shifts that model by 6.6%. Row 2 is the saved sp_best of 09-25.

RESULT: 2026-10-18. chi2_red on own data: MILES 2.52 and C3K HR 1.77 (4448 px), pocoMC 1.05
(5530 px). Rows 1-2 use the DR1 coadd ivar and row 3 the recoadd ivar.

COMMAND: from the repository root on the Mac, after 2026-10-18_issue14_miles.py,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-15_broadline_free_amplitude:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-18_issue14_figures.py``.

OUTPUT: corner_sfh.png, residuals_by_run.png and windows_by_run.png in
results/2026-10-18_issue14_figures/.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from hubersed.paths import PATHS

TID = 39627770174637084
EXP = Path(__file__).resolve().parents[1]  # the experiments folder
RES = PATHS["RESULTS"]
OUT = RES / "2026-10-18_issue14_figures"
PLOTDATA = RES / "2026-09-25_nautilus_tauin_c3k_hr"


def load_module(name, path):
    """Import the file ``path`` as a module called ``name``."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chi2_red(flux, unc, model, mask, n_free):
    """Return chi2 over the masked pixels divided by (pixels - free parameters), and the pixels."""
    n = int(mask.sum())
    chi2 = float(np.sum(((flux - model)[mask] / unc[mask]) ** 2))
    return chi2 / (n - n_free), n


def c3k_hr_row():
    """Return the 09-25 C3K HR data (degraded, MILES window) and its saved max-L spectrum."""
    f25 = load_module("fit_0925", EXP / "2026-09-25_nautilus_tauin_c3k_hr" / "fit.py")
    pd = np.load(PLOTDATA / f"{TID}_C3K_plotdata.npz")
    z, flux, unc, good = f25.load_data(TID, "miles")
    assert np.array_equal(good, pd["good"])
    return dict(flux=flux, unc=unc, mask=good, model=pd["sp_best"], z=z, n_free=len(pd["labels"]))


STATE = (
    RES
    / "2026-10-15_broadline_free_amplitude"
    / f"poco_{TID}_free_dust_neff1024"
    / "pmc_final.state"
)
VARIANT = "free_dust"
CORNER = [
    "logmass",
    "logzsol",
    "dust2",
    "dust_ratio",
    "dust_index",
    "frac_nodust",
    "sigma_smooth",
    "gas_logz",
    "gas_logu",
    "gas_lognH",
    "gas_logno",
    "gas_logco",
    "eline_sigma",
    "eline_sigma_broad",
]
RUNS = {
    "miles": ("MILES + Cue", "#4393c3"),
    "c3k_hr": ("C3K HR + Cue (degraded)", "#e08214"),
    "poco": ("C3K ER + free-line broad + Cue + R matrix + new ivar", "#b2182b"),
}
N_TOP = 300


def wquantile(x, w, q):
    """Return weighted quantiles of ``x`` with weights ``w`` that sum to 1."""
    o = np.argsort(x)
    return np.interp(q, np.cumsum(w[o]), x[o])


def axis_range(x, w, q=(0.001, 0.999), pad=0.05):
    """Return the central 99.8% of the posterior plus 5% padding."""
    lo, hi = wquantile(x, w, np.array(q))
    d = (hi - lo) * pad or 1e-3
    return (lo - d, hi + d)


def desi(wave, maggies):
    """Return maggies as DESI flux density, 1e-17 erg/s/cm^2/A."""
    import astropy.units as u

    from hubersed.conversion import to_flambda

    return to_flambda(wave * u.AA, maggies * u.mgy).value


def poco_row():
    """Return the pocoMC posterior, its highest-posterior particle and row 3 of the comparison."""
    import fit
    from compare_poco import load_sampler

    assert hasattr(fit, "process_state"), "put the 10-15 folder before the 10-13 one on PYTHONPATH"
    s = load_sampler(STATE)
    usamp, w, logl, _ = s.posterior()
    w = w / w.sum()
    model, obs, cue = fit.process_state(VARIANT)
    theta = np.array([model.prior_transform(u) for u in usamp])
    top = np.argsort(logl)[-N_TOP:]
    lnp = np.array([logl[i] + model.prior_product(theta[i]) for i in top])
    best = theta[top[np.argmax(lnp)]]
    spec = model.predict(best, observations=obs, sps=cue)[0][0]
    o = obs[0]
    z = float(np.atleast_1d(model.params["zred"])[0])
    row = dict(
        wave=np.asarray(o.wavelength, float),
        flux=np.asarray(o.flux, float),
        unc=np.asarray(o.uncertainty, float),
        mask=np.asarray(o.mask, bool),
        model=np.asarray(spec, float),
        z=z,
        n_free=model.ndim,
    )
    return model, theta, w, best, row


def load_rows(poco):
    """Return the three rows of the comparison, each with its own data, mask and best fit."""
    from hubersed.fitting.chi2 import WAVE_OBS

    d = np.load(OUT / "miles_0924.npz")
    rows = {"miles": {k: d[k] for k in ("flux", "unc", "mask", "model", "z", "n_free")}}
    rows["c3k_hr"] = c3k_hr_row()
    for run in ("miles", "c3k_hr"):
        rows[run]["wave"] = WAVE_OBS
    rows["poco"] = poco
    assert np.allclose(rows["poco"]["wave"], WAVE_OBS)
    for r in rows.values():
        r["mask"] = r["mask"] & np.isfinite(r["unc"]) & (r["unc"] > 0)
        r["chi"] = np.where(r["mask"], (r["flux"] - r["model"]) / r["unc"], np.nan)
        r["rest"] = r["wave"] / (1 + float(r["z"]))
    return rows


def corner_figure(model, theta, w, best, path):
    """Draw the corner plot with value +- error titles and the SFH posterior and MAP at top right."""
    import corner
    import matplotlib.pyplot as plt

    from hubersed.fitting.map_fits import sfh_from_theta

    labels = list(model.theta_labels())
    idx = [labels.index(p) for p in CORNER]
    k = len(idx)
    fig = plt.figure(figsize=(2.7 * k, 2.7 * k))
    corner.corner(
        theta[:, idx],
        weights=w,
        fig=fig,
        labels=CORNER,
        truths=best[idx],
        truth_color="C1",
        range=[axis_range(theta[:, i], w) for i in idx],
        plot_datapoints=False,
        fill_contours=True,
        smooth=1.0,
        bins=30,
        levels=(0.393, 0.865),  # 1 and 2 sigma for a 2-D Gaussian
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        max_n_ticks=3,
        label_kwargs={"fontsize": 22},
        title_kwargs={"fontsize": 14},
        labelpad=0.12,
    )
    axes = np.array(fig.axes).reshape(k, k)
    for j, i in enumerate(idx):
        ax = axes[j, j]
        for edge in np.atleast_1d(model.config_dict[labels[i]]["prior"].range):
            if ax.get_xlim()[0] <= edge <= ax.get_xlim()[1]:
                ax.axvline(edge, color="0.4", ls="--", lw=1)
    for ax in axes.ravel():
        ax.tick_params(labelsize=13)

    rng = np.random.default_rng(0)
    sfhs = [sfh_from_theta(model, theta[i]) for i in rng.choice(len(w), size=2000, p=w)]
    edges = sfhs[0]["edges_gyr"]
    sfh_map = sfh_from_theta(model, best)
    left, width = 0.56, 0.40
    for n, (key, ylab, log) in enumerate(
        [("ssfr", r"sSFR [yr$^{-1}$]", True), ("cmf", "cumulative mass fraction", False)]
    ):
        ax = fig.add_axes([left, 0.80 - n * 0.17, width, 0.15])
        vals = np.array([s[key] for s in sfhs])
        lo, med, hi = np.percentile(vals, [16, 50, 84], axis=0)
        ax.stairs(hi, edges, baseline=lo, fill=True, color="0.6", alpha=0.5, lw=0, label="16-84%")
        ax.stairs(med, edges, color="k", lw=2, label="posterior median")
        ax.stairs(sfh_map[key], edges, color="C1", ls="--", lw=2, label="MAP")
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
            ax.legend(frameon=False, fontsize=16, loc="lower left")
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("lookback time [Gyr]", fontsize=20)
        ax.set_ylabel(ylab, fontsize=20)
        ax.tick_params(labelsize=15)
    fig.suptitle(
        f"{TID} pocoMC {VARIANT}: posterior, MAP (orange) and SFH", fontsize=28, x=0.76, y=0.985
    )
    fig.savefig(path, dpi=80)
    plt.close(fig)
    print(f"wrote {path}")


def residual_figure(rows, path):
    """Draw the data and the pocoMC best fit, then the chi of each run from -5 to 5."""
    import matplotlib.pyplot as plt

    p = rows["poco"]
    fig, ax = plt.subplots(
        4,
        1,
        figsize=(16, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 1], "hspace": 0.06},
    )
    ax[0].plot(
        p["rest"], np.where(p["mask"], desi(p["wave"], p["flux"]), np.nan), color="0.2", lw=0.6,
        label="DESI (recoadd)",
    )  # fmt: skip
    ax[0].plot(
        p["rest"], np.where(p["mask"], desi(p["wave"], p["model"]), np.nan), color=RUNS["poco"][1],
        lw=0.8, label="pocoMC MAP, " + RUNS["poco"][0],
    )  # fmt: skip
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
    ax[0].legend(frameon=False, fontsize=13, loc="upper right")
    top = np.nanpercentile(desi(p["wave"], p["flux"])[p["mask"]], 99.5)
    ax[0].set_ylim(0, 1.15 * top)
    for a, run in zip(ax[1:], RUNS, strict=True):
        r = rows[run]
        red, n = chi2_red(r["flux"], r["unc"], r["model"], r["mask"], int(r["n_free"]))
        a.plot(r["rest"], r["chi"], color=RUNS[run][1], lw=0.5)
        for h in (-2, 2):
            a.axhline(h, color="0.3", ls=":", lw=1)
        a.axhline(0, color="0.3", lw=0.6)
        a.set_ylim(-5, 5)
        a.set_ylabel(r"$\chi$", fontsize=14)
        a.text(
            0.005, 0.88, rf"{RUNS[run][0]}: $\chi^2_\nu$ = {red:.2f} ({n} px)",
            transform=a.transAxes, fontsize=12, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
        )  # fmt: skip
    ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def windows_figure(rows, path):
    """Draw the line windows with the data, each run's best fit and chi, and chi2/N per window."""
    import matplotlib.pyplot as plt
    from map import WINDOWS

    p = rows["poco"]
    fig, ax = plt.subplots(
        2,
        len(WINDOWS),
        figsize=(3.2 * len(WINDOWS), 6.0),
        gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05},
        sharex="col",
    )
    for i, (key, (a, b)) in enumerate(WINDOWS.items()):
        win = (p["rest"] >= a) & (p["rest"] < b)
        x = p["rest"][win]
        data = desi(p["wave"][win], p["flux"][win])
        ax[0, i].plot(x, np.where(p["mask"][win], data, np.nan), color="0.2", lw=1.0)
        handles = []
        for run, (_, color) in RUNS.items():
            r = rows[run]
            w = (r["rest"] >= a) & (r["rest"] < b)
            m = w & r["mask"]
            if not m.any():
                continue
            c2n = float(np.sum(r["chi"][m] ** 2) / m.sum())
            (h,) = ax[0, i].plot(
                r["rest"][w], np.where(r["mask"][w], desi(r["wave"][w], r["model"][w]), np.nan),
                color=color, lw=1.1, label=rf"$\chi^2/N$ = {c2n:.1f}",
            )  # fmt: skip
            handles.append(h)
            ax[1, i].plot(r["rest"][w], r["chi"][w], color=color, lw=0.9)
        ax[0, i].set_title(key, fontsize=14)
        lo, hi = ax[0, i].get_ylim()
        ax[0, i].set_ylim(lo, lo + 1.45 * (hi - lo))  # room for the chi2 legend above the lines
        ax[0, i].legend(handles=handles, frameon=False, fontsize=10, loc="upper left")
        for h in (-2, 2):
            ax[1, i].axhline(h, color="0.3", ls=":", lw=0.8)
        ax[1, i].axhline(0, color="0.3", lw=0.6)
        ax[1, i].set_ylim(-5, 5)
        ax[1, i].set_xlabel(r"rest [$\AA$]", fontsize=12)
    ax[0, 0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ cgs $\AA^{-1}$]", fontsize=13)
    ax[1, 0].set_ylabel(r"$\chi$", fontsize=13)
    handles = [plt.Line2D([], [], color="0.2", lw=2)] + [
        plt.Line2D([], [], color=c, lw=3) for _, c in RUNS.values()
    ]
    names = ["DESI recoadd (the C3K HR fit saw a degraded copy)"] + [n for n, _ in RUNS.values()]
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=4, fontsize=13,
               frameon=False)  # fmt: skip
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def figures():
    """Write the corner, residual and window figures."""
    from hubersed.plotting.style import use_apj_style

    use_apj_style()
    model, theta, w, best, poco = poco_row()
    corner_figure(model, theta, w, best, OUT / "corner_sfh.png")
    rows = load_rows(poco)
    residual_figure(rows, OUT / "residuals_by_run.png")
    windows_figure(rows, OUT / "windows_by_run.png")


def main():
    """Write the figures."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    figures()


if __name__ == "__main__":
    main()
