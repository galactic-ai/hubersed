"""pocoMC posteriors and best fits of the follow-up batch targets (#17), as 2026-10-31_poco_robust6.

QUESTION: what do C3K ER continuum fits give for the follow-up outliers that pass scrutiny
(2026-10-31_quiescent_qa), and where do the max-likelihood models miss the data?
HYPOTHESIS: none registered.
INPUTS: <TARGETID>/poco_neff1024/pmc_final.state and <TARGETID>_poco_maxl.npz (predict.py) for each
target, copied from stampede3; the model of fit.py for the prior transform (no FSPS). A_V and E(B-V)
from the diffuse KC13 curve as in 2026-10-31_poco_c3k_er_dark_39633433328091196/analysis.py, whose
corner style and plots this repeats without the MAP.
SEED: 0, for 20000 prior draws and 2000 sSFR draws.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_poco_quiescent/analysis.py --tid <TARGETID>``.
RESULT: 2026-11-01, 39633301257850206: ESS 4555, log Z 90445.08 +- 0.84; max-likelihood chi2/N 1.018 on
5043 px with no structured residual. Medians: logmass 11.28, logzsol -0.17, afe +0.31, E(B-V) 0.17,
A_V 0.50, dust_index -0.30 (posterior.txt).
FIGURES: results/2026-10-31_poco_quiescent/<TARGETID>/ posterior.txt, prior_edges.txt, best_fit.png,
corner.png, corner_stars_dust.png, corner_sfh.png, corner_all_ssfr.png.
"""

# %%
import argparse
import importlib.util
from pathlib import Path

import astropy.units as u
import corner
import dill
import matplotlib.pyplot as plt
import numpy as np
import pocomc as pc
from scipy.signal import medfilt

from hubersed.conversion import to_flambda
from hubersed.fitting.map_fits import sfh_from_theta
from hubersed.plotting.style import use_apj_style

HERE = Path(__file__).resolve().parent
N_PRIOR = 20000
CORNER = ["dust2", "dust_index", "A_V", "E(B-V)", "logzsol", "afe"]
TABLE = [
    "dust2", "dust_index", "dust_ratio", "A_V", "E(B-V)", "logzsol", "afe", "logmass",
    "sigma_smooth", "sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn",
] + [f"logsfr_ratios_{i}" for i in range(1, 10)]  # fmt: skip
RED = "#b2182b"


def kc13_curve(lam, delta):
    """Return the FSPS dust_type 4 curve, tau(lam) / tau_V, for rest ``lam`` in A."""
    lam = np.asarray(lam, float)
    delta = np.atleast_1d(np.asarray(delta, float))[:, None]
    x = 1e4 / lam
    blue = 1.17 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3) + 1.78
    red = 1.17 * (-1.857 + 1.04 * x) + 1.78
    cal = np.clip(np.where(lam < 6300.0, blue, red) / 0.44 / 4.05, 0.0, None)
    eb = 0.85 - 1.9 * delta
    drude = eb * (lam * 350.0) ** 2 / ((lam**2 - 2175.0**2) ** 2 + (lam * 350.0) ** 2)
    return (cal + drude / 4.05) * (lam / 5500.0) ** delta


def attenuation(lam, dust2, delta):
    """Return A_lambda in mag, shape (len(dust2), len(lam)), for diffuse depth ``dust2``."""
    dust2 = np.atleast_1d(np.asarray(dust2, float))[:, None]
    return 2.5 * np.log10(np.e) * dust2 * kc13_curve(lam, delta)


def wquantile(x, w, q):
    """Return the weighted quantiles ``q`` of ``x``."""
    order = np.argsort(x)
    cdf = (np.cumsum(w[order]) - 0.5 * w[order]) / np.sum(w)
    return np.interp(q, cdf, x[order])


def columns(model, theta):
    """Return a dict of named columns, with A_V and E(B-V) added."""
    labels = list(model.theta_labels())
    cols = {k: theta[:, i] for i, k in enumerate(labels)}
    ab, av = attenuation([4400.0, 5500.0], cols["dust2"], cols["dust_index"]).T
    cols["A_V"], cols["E(B-V)"] = av, ab - av
    return cols


def main(tid):
    """Write the table, prior edges, corners and best fit of one target."""
    spec = importlib.util.spec_from_file_location("quiescent_fit", HERE / "fit.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    poco = wrapper.load_poco(tid)
    out = poco.OUT
    model = poco.state(with_sps=False)[0]
    s = pc.Sampler.__new__(pc.Sampler)
    with open(out / "poco_neff1024" / "pmc_final.state", "rb") as f:
        s.__dict__.update(dill.load(f))
    usamp, w, logl, _ = s.posterior()
    w = w / w.sum()
    ess = 1.0 / np.sum(w**2)
    logz, logz_err = s.evidence()
    theta_post = np.array([model.prior_transform(x) for x in usamp])
    post = columns(model, theta_post)
    rng = np.random.default_rng(0)
    prior = columns(model, np.array([model.prior_transform(x)
                                     for x in rng.uniform(size=(N_PRIOR, model.ndim))]))  # fmt: skip

    def prior_range(name):
        if name in model.config_dict:
            lo, hi = np.ravel(model.config_dict[name]["prior"].range)[:2]
            return float(lo), float(hi)
        return float(prior[name].min()), float(prior[name].max())

    # table
    q = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
    lines = [f"{tid}: {len(w)} samples, ESS {ess:.0f}, log Z {logz:.3f} +- {logz_err:.3f}, "
             f"max lnlike {logl.max():.3f}", "",
             f"{'parameter':>16} {'prior lo':>9} {'prior hi':>9} " + " ".join(
                 f"{'q' + format(100 * x, 'g'):>9}" for x in q)]  # fmt: skip
    for p in TABLE:
        lo, hi = prior_range(p)
        lines.append(
            f"{p:>16} {lo:9.3f} {hi:9.3f} "
            + " ".join(f"{v:9.4f}" for v in wquantile(post[p], w, q))
        )
    (out / "posterior.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # prior edges
    labels = list(model.theta_labels())
    names = [p for p in labels if not p.startswith("logsfr_ratios")]
    edge = [f"{'parameter':>16} {'prior lo':>9} {'prior hi':>9} {'q0.5%':>9} {'median':>9} "
            f"{'q99.5%':>9} {'lo dist':>7} {'hi dist':>7}"]  # fmt: skip
    for p in names:
        lo, hi = prior_range(p)
        a, m, b = wquantile(post[p], w, np.array([0.005, 0.5, 0.995]))
        edge.append(f"{p:>16} {lo:9.3g} {hi:9.3g} {a:9.4g} {m:9.4g} {b:9.4g} "
                    f"{(a - lo) / (hi - lo):7.3f} {(hi - b) / (hi - lo):7.3f}")  # fmt: skip
    edge += [
        "",
        "dist = distance of the 0.5% or 99.5% quantile to the bound, in units of prior width.",
    ]
    (out / "prior_edges.txt").write_text("\n".join(edge) + "\n")

    # corners, in the style of the 10-31 dark fit
    use_apj_style()

    def axis_range(x, qq=(0.001, 0.999), pad=0.05):
        lo, hi = wquantile(x, w, np.array(qq))
        d = (hi - lo) * pad or 1e-3
        return (lo - d, hi + d)

    def red_corner(cols, path, inset=False):
        k = len(cols)
        fig = plt.figure(figsize=(2.8 * k, 2.8 * k))
        corner.corner(np.column_stack([post[p] for p in cols]), weights=w, fig=fig, labels=cols,
                      color=RED, range=[axis_range(post[p]) for p in cols], plot_datapoints=False,
                      fill_contours=True, smooth=1.0, bins=30, levels=(0.393, 0.865),
                      quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3f", max_n_ticks=3,
                      label_kwargs={"fontsize": 34}, title_kwargs={"fontsize": 30}, labelpad=0.3,
                      hist_kwargs={"color": RED, "lw": 2})  # fmt: skip
        axes = np.array(fig.axes).reshape(k, k)
        for j, p in enumerate(cols):
            if p not in model.config_dict:
                continue
            for e in prior_range(p):
                if axes[j, j].get_xlim()[0] <= e <= axes[j, j].get_xlim()[1]:
                    axes[j, j].axvline(e, color="k", ls="--", lw=2)
        for ax in axes.ravel():
            ax.tick_params(labelsize=22)
        for ax in axes[-1]:
            ax.xaxis.label.set_rotation(35)
            ax.xaxis.label.set_ha("right")
        for ax in axes[:, 0]:
            ax.yaxis.label.set_rotation(55)
            ax.yaxis.label.set_va("top")
        if inset:
            r = np.random.default_rng(0)
            sfhs = [sfh_from_theta(model, theta_post[i]) for i in r.choice(len(w), size=2000, p=w)]
            edges = sfhs[0]["edges_gyr"]
            lo16, med50, hi84 = np.percentile([sf["ssfr"] for sf in sfhs], [16, 50, 84], axis=0)
            ax = fig.add_axes([0.53, 0.62, 0.44, 0.33])
            ax.stairs(hi84, edges, baseline=lo16, fill=True, color=RED, alpha=0.35, lw=0,
                      label=r"16--84\%")  # fmt: skip
            ax.stairs(med50, edges, color=RED, lw=3, label="median")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("lookback time [Gyr]", fontsize=44)
            ax.set_ylabel(r"sSFR [yr$^{-1}$]", fontsize=44)
            ax.tick_params(labelsize=34)
            ax.legend(frameon=False, fontsize=38, loc="lower left")
            ax.set_title(f"{tid}, pocoMC C3K ER", fontsize=46)
        fig.savefig(path, dpi=60, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")

    red_corner(CORNER, out / "corner.png")
    red_corner(["logzsol", "afe", "logmass", "dust2", "dust_ratio", "dust_index", "sigma_smooth",
                "sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn"], out / "corner_stars_dust.png")  # fmt: skip
    red_corner([f"logsfr_ratios_{i}" for i in range(1, 10)], out / "corner_sfh.png")
    red_corner(names, out / "corner_all_ssfr.png", inset=True)

    # data with the max-likelihood model, then chi
    d = np.load(out / f"{tid}_poco_maxl.npz")
    g = d["good"]
    rest = d["wave"] / (1 + float(d["z"]))

    def flam(x):
        return to_flambda(np.asarray(d["wave"], float) * u.AA, np.asarray(x, float) * u.mgy).value

    data = flam(d["flux"])
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.where(g, (d["flux"] - d["sp_best"]) / d["unc"], np.nan)
    med = np.full(data.size, np.nan)
    med[g] = medfilt(data[g], 9)
    fig, ax = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})  # fmt: skip
    ax[0].plot(
        rest, np.where(g, data, np.nan), color="0.75", lw=0.5, label="DESI SV3 bright (DR1 coadd)"
    )
    ax[0].plot(rest, med, color="k", lw=0.8, label="median filter, 9 pixels")
    ax[0].plot(rest, np.where(g, flam(d["sp_best"]), np.nan), color=RED, lw=0.8,
               label="C3K ER pocoMC max-likelihood")  # fmt: skip
    ax[1].plot(rest, chi, color=RED, lw=0.5)
    for h in (-2, 2):
        ax[1].axhline(h, color="0.3", ls="--", lw=1)
    ax[1].axhline(0, color="0.3", lw=0.6)
    ax[1].set_ylim(-5, 5)
    ax[1].set_ylabel(r"$\chi$", fontsize=14)
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
    ax[0].set_title(f"{tid}  z {float(d['z']):.4f}", fontsize=14)
    ax[0].legend(frameon=False, fontsize=12, loc="upper right")
    ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
    fig.savefig(out / "best_fit.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    c = chi[g]
    print(f"max-likelihood: chi2 {np.sum(c**2):.1f} on {g.sum()} px, chi2/N {np.mean(c**2):.3f}; "
          f"wrote {out / 'best_fit.png'}")  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tid", type=int, required=True)
    main(ap.parse_args().tid)
