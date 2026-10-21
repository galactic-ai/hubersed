"""QUESTION: Do nautilus (#14, converged) and pocoMC agree on the free-variant posterior of 39627770174637084?
HYPOTHESIS: None registered before this analysis.
INPUTS: the nautilus .h5 of 2026-10-15_broadline_free_amplitude (job 3513585, copied from ls6 on 2026-10-21)
and poco_<TID>_free_neff512_ls6/pmc_final.state. Both use the free variant (29 parameters, frac_nodust 0).
The best point of each run is the highest ln L + ln prior among its 300 points of highest ln L. Styling follows
2026-10-18_issue14_figures.
SEED: 0 for the SFH draws.
COMMAND: from the repository root on the Mac with the C3K_ER build,
PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-15_broadline_free_amplitude:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-19_candidate_poco_figures/nautilus_vs_poco.py
RESULT: pending.
FIGURES: nautilus_vs_poco_corner_sfh.png and nautilus_vs_poco_residuals.png, with nautilus_vs_poco_summary.txt,
in results/2026-10-19_candidate_poco_figures/.
"""

import numpy as np

from hubersed.paths import PATHS

TID = 39627770174637084
VARIANT = "free"
RES = PATHS["RESULTS"] / "2026-10-15_broadline_free_amplitude"
POCO_STATE = RES / f"poco_{TID}_free_neff512_ls6" / "pmc_final.state"
OUT = PATHS["RESULTS"] / "2026-10-19_candidate_poco_figures"
CORNER = [
    "logmass", "logzsol", "dust2", "dust_ratio", "dust_index", "sigma_smooth", "gas_logz",
    "gas_logu", "gas_lognH", "gas_logno", "gas_logco", "eline_sigma", "eline_sigma_broad",
]  # fmt: skip
RUNS = {"nautilus": "#2166ac", "pocoMC": "#b2182b"}
N_TOP = 300


def wquantile(x, w, q):
    """Return weighted quantiles of ``x`` with weights ``w`` that sum to 1."""
    o = np.argsort(x)
    return np.interp(q, np.cumsum(w[o]), x[o])


def best_point(model, theta, logl):
    """Return the highest ln L + ln prior point among the N_TOP points of highest ln L."""
    top = np.argsort(logl)[-N_TOP:]
    lnp = np.array([logl[i] + model.prior_product(theta[i]) for i in top])
    return theta[top[np.argmax(lnp)]], float(logl[top[np.argmax(lnp)]])


def load_poco(model):
    """Return theta, weights, ln L and a summary of the pocoMC free run."""
    from compare_poco import load_sampler

    s = load_sampler(POCO_STATE)
    u, w, logl, _ = s.posterior()
    w = w / w.sum()
    theta = np.array([model.prior_transform(x) for x in u])
    info = f"pocoMC: iteration {s.t}, calls {s.calls}, log Z {float(s.logz):.2f}"
    return theta, w, np.asarray(logl), info + f", ESS {1 / np.sum(w**2):.0f}"


def load_nautilus(model, path):
    """Return theta, weights, ln L and a summary of the nautilus run. Only reads the file."""
    from nautilus import Sampler

    s = Sampler(lambda u: u, lambda x: 0.0, n_dim=model.ndim, n_live=2000,
                filepath=str(path), resume=True)  # fmt: skip
    assert s.explored and s.discard_exploration
    u, log_w, logl = s.posterior()
    keep = log_w > log_w.max() - 30  # lighter points add nothing to any statistic
    theta = np.array([model.prior_transform(x) for x in u[keep]])
    w = np.exp(log_w[keep] - log_w[keep].max())
    w /= w.sum()
    info = (
        f"nautilus: calls {s.n_like}, log Z {s.log_z:.2f}, N_eff {s.n_eff:.0f}, kept {keep.sum()}"
    )
    return theta, w, logl[keep], info


def corner_figure(model, runs, path):
    """Overlay both posteriors with filled contours, two-run titles and the SFH panels."""
    import corner
    import matplotlib.pyplot as plt

    from hubersed.fitting.map_fits import sfh_from_theta

    labels = list(model.theta_labels())
    names = [p for p in CORNER if p in labels]
    idx = [labels.index(p) for p in names]
    k = len(idx)
    rng_ax = []
    for i in idx:
        lo = min(wquantile(r["theta"][:, i], r["w"], 0.001) for r in runs.values())
        hi = max(wquantile(r["theta"][:, i], r["w"], 0.999) for r in runs.values())
        d = (hi - lo) * 0.05 or 1e-3
        rng_ax.append((lo - d, hi + d))
    fig = plt.figure(figsize=(2.7 * k, 2.7 * k))
    for name, r in runs.items():
        corner.corner(
            r["theta"][:, idx], weights=r["w"], fig=fig, color=RUNS[name], labels=names,
            range=rng_ax, plot_datapoints=False, plot_density=False, fill_contours=True,
            smooth=1.0, bins=30,
            levels=(0.393, 0.865),  # 1 and 2 sigma for a 2-D Gaussian
            max_n_ticks=3, label_kwargs={"fontsize": 22}, labelpad=0.12,
            hist_kwargs={"density": True, "lw": 2},
        )  # fmt: skip
        corner.overplot_lines(fig, r["best"][idx], color=RUNS[name], ls="--", lw=1.5)
    axes = np.array(fig.axes).reshape(k, k)
    for j, i in enumerate(idx):
        ax = axes[j, j]
        for edge in np.atleast_1d(model.config_dict[labels[i]]["prior"].range):
            if ax.get_xlim()[0] <= edge <= ax.get_xlim()[1]:
                ax.axvline(edge, color="0.4", ls=":", lw=1)
        for n, (name, r) in enumerate(runs.items()):
            q = wquantile(r["theta"][:, i], r["w"], np.array([0.16, 0.5, 0.84]))
            ax.text(0.5, 1.04 + 0.16 * n, f"{q[1]:.3f} -{q[1] - q[0]:.3f} +{q[2] - q[1]:.3f}",
                    color=RUNS[name], fontsize=14, ha="center", transform=ax.transAxes)  # fmt: skip
    for ax in axes.ravel():
        ax.tick_params(labelsize=13)

    rng = np.random.default_rng(0)
    for n, (key, ylab, log) in enumerate(
        [("ssfr", r"sSFR [yr$^{-1}$]", True), ("cmf", "cumulative mass fraction", False)]
    ):
        ax = fig.add_axes([0.56, 0.80 - n * 0.17, 0.40, 0.15])
        for name, r in runs.items():
            draws = rng.choice(len(r["w"]), size=2000, p=r["w"])
            sfhs = [sfh_from_theta(model, r["theta"][i]) for i in draws]
            edges = sfhs[0]["edges_gyr"]
            lo, med, hi = np.percentile([s[key] for s in sfhs], [16, 50, 84], axis=0)
            ax.stairs(hi, edges, baseline=lo, fill=True, color=RUNS[name], alpha=0.3, lw=0)
            ax.stairs(med, edges, color=RUNS[name], lw=2, label=f"{name} median, 16-84%")
            ax.stairs(sfh_from_theta(model, r["best"])[key], edges, color=RUNS[name], ls="--",
                      lw=1.5, label=f"{name} best")  # fmt: skip
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
            ax.legend(frameon=False, fontsize=15, loc="lower left")
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("lookback time [Gyr]", fontsize=20)
        ax.set_ylabel(ylab, fontsize=20)
        ax.tick_params(labelsize=15)
    handles = [plt.Line2D([], [], color=c, lw=6, alpha=0.6) for c in (RUNS[n] for n in runs)]
    fig.legend(handles, list(runs), loc="upper left",
               bbox_to_anchor=(0.56, 0.62), fontsize=20, frameon=False)  # fmt: skip
    fig.suptitle(f"{TID}, variant {VARIANT}: nautilus (blue) and pocoMC (red), best points dashed",
                 fontsize=28, x=0.76, y=0.985)  # fmt: skip
    fig.savefig(path, dpi=80)
    plt.close(fig)
    print(f"wrote {path}")


def residual_figure(obs, z, runs, n_free, path):
    """Draw data and both best fits, then each run's chi with its chi2/(N - free parameters)."""
    import astropy.units as u
    import matplotlib.pyplot as plt

    from hubersed.conversion import to_flambda

    o = obs[0]
    wave = np.asarray(o.wavelength, float)
    flux, unc = np.asarray(o.flux, float), np.asarray(o.uncertainty, float)
    mask = np.asarray(o.mask, bool) & np.isfinite(unc) & (unc > 0)
    rest = wave / (1 + z)

    def desi(m):
        return to_flambda(wave * u.AA, m * u.mgy).value

    n = len(runs)
    fig, ax = plt.subplots(n + 1, 1, figsize=(16, 4.5 + 2.2 * n), sharex=True,
                           gridspec_kw={"height_ratios": [3] + [1] * n, "hspace": 0.06})  # fmt: skip
    ax[0].plot(
        rest, np.where(mask, desi(flux), np.nan), color="0.2", lw=0.6, label="DESI (recoadd)"
    )
    for a, (name, r) in zip(ax[1:], runs.items(), strict=True):
        chi = np.where(mask, (flux - r["spec"]) / unc, np.nan)
        red = float(np.nansum(chi**2)) / (mask.sum() - n_free)
        r["chi2_nu"] = red
        ax[0].plot(rest, np.where(mask, desi(r["spec"]), np.nan), color=RUNS[name], lw=0.8,
                   label=f"{name} best")  # fmt: skip
        a.plot(rest, chi, color=RUNS[name], lw=0.5)
        for h in (-2, 2):
            a.axhline(h, color="0.3", ls=":", lw=1)
        a.axhline(0, color="0.3", lw=0.6)
        a.set_ylim(-5, 5)
        a.set_ylabel(r"$\chi$", fontsize=14)
        beyond = int(np.sum(np.abs(chi[mask]) > 5))
        a.text(0.005, 0.88, rf"{name}: $\chi^2_\nu$ = {red:.3f} ({mask.sum()} px, "
               rf"{n_free} free, {beyond} px beyond $\pm$5)", transform=a.transAxes, fontsize=12,
               va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))  # fmt: skip
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
    ax[0].legend(frameon=False, fontsize=13, loc="upper right")
    ax[0].set_title(f"{TID}, z = {z:.5f}, variant {VARIANT}", fontsize=15)
    ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    """Load both runs, print the summary and draw the two figures."""
    import fit

    from hubersed.plotting.style import use_apj_style

    assert hasattr(fit, "process_state"), "put the 10-15 folder before the 10-13 one on PYTHONPATH"
    use_apj_style()
    model, obs, cue = fit.process_state(VARIANT)
    z = float(np.atleast_1d(model.params["zred"])[0])
    loaders = {"nautilus": lambda: load_nautilus(model, RES / f"{fit.stem(VARIANT, 2000)}.h5"),
               "pocoMC": lambda: load_poco(model)}  # fmt: skip
    runs, lines = {}, []
    for name, load in loaders.items():
        if name == "nautilus" and not (RES / f"{fit.stem(VARIANT, 2000)}.h5").exists():
            print(f"skipping nautilus, no {fit.stem(VARIANT, 2000)}.h5 in {RES}")
            continue
        theta, w, logl, info = load()
        best, best_logl = best_point(model, theta, logl)
        spec = np.asarray(model.predict(best, observations=obs, sps=cue)[0][0], float)
        runs[name] = dict(theta=theta, w=w, best=best, spec=spec)
        lines.append(f"{info}, max ln L {logl.max():.2f}, best ln L {best_logl:.2f}")
    residual_figure(obs, z, runs, model.ndim, OUT / "nautilus_vs_poco_residuals.png")
    for name, r in runs.items():
        lines.append(f"{name}: chi2_nu {r['chi2_nu']:.3f}")
    corner_figure(model, runs, OUT / "nautilus_vs_poco_corner_sfh.png")
    text = "\n".join(lines)
    print(text)
    (OUT / "nautilus_vs_poco_summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
