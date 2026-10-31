"""pocoMC posterior of the C3K ER fit of the SV3 dark spectrum of 39633433328091196: dust (#17).

QUESTION: what are dust2, dust_index, A_V and E(B-V) once logzsol, afe, the SFH and its
hyperparameters are marginalised? Does the posterior exclude dust2 = 0, and how does dust trade
against logzsol and afe?
HYPOTHESIS: none registered.
INPUTS: pmc_final.state in poco_neff1024_forktest (iterations 1-21 from skx-dev job 3664872, the rest
from spr job 3664976 after the resume), copied from stampede3. The model of fit.py for the prior
transform (no FSPS). The dark MAP <TID>_map.npz of 2026-10-30_continuum_39633433328091196_dark_map.
A_V = A(5500 A), E(B-V) = A(4400) - A(5500), from the diffuse KC13 curve as in
2026-10-29_c3k_vs_miles_dust/analysis.py; the birth cloud is left out.
SEED: 0, for 20000 draws of the prior.
COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-31_poco_c3k_er_dark_39633433328091196/analysis.py``.
RESULT: 2026-10-31, posterior.txt. ESS 4506, log Z 85365.236 +- 0.145. E(B-V) 0.124 (68% 0.114-0.136),
A_V 0.530 (0.410-0.657), dust2 0.488 (0.378-0.606), dust_index +0.07 (-0.25 to +0.27, piled toward the
+0.4 prior edge). dust2 = 0 is excluded: 0.5% quantile 0.26, P(A_V < 0.05) 0 of 5203 against 0.028 in
the prior. A_V follows dust_index (r +0.92) and logmass (+0.89); E(B-V) is the tight quantity. Dust
barely trades with logzsol (r +0.19) or afe (-0.33). The MAP sits low: E(B-V) 0.098 at the 0.3rd
percentile, logzsol 0.148 at the 2.3rd. Assumptions: one dark exposure, C3K ER, #14 hyperparameter
priors, dust_index prior edge at +0.4, monochromatic E(B-V), diffuse dust only. best_fit.png:
max-likelihood sample chi2 4768.7 against the dark MAP's 4791.3 on 4538 px; the two models overlap.
prior_edges.txt: tau_eq, sigma_dyn and tau_dyn reach their lower prior bounds, dust_index its upper
one. The MAP's logsfr_ratios 2 and 3 lie beyond every posterior sample (pct 0.0 and 100.0),
and 1, 4, 5 and 7 outside the 95% range (pct 97.7, 2.2, 98.0, 1.2).
FIGURES: results/2026-10-31_poco_c3k_er_dark_39633433328091196/ posterior.txt, attenuation.png,
best_fit.png from <TID>_poco_maxl.npz of predict.py (max-likelihood sample), prior_edges.txt, and the
corners corner.png (dust), corner_stars_dust.png, corner_sfh.png and corner_all_ssfr.png (with the
sSFR posterior from 2000 draws, seed 0), all in the style of corner_sfh.py of 10-19: axes the central
99.8% of the posterior plus 5%, prior bounds dashed, no points marked.
"""

# %%
import importlib.util
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import numpy as np
import pocomc as pc

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

HERE = Path(__file__).resolve().parent
RES = PATHS["RESULTS"]
OUT = RES / HERE.name
STATE = OUT / "poco_neff1024_forktest" / "pmc_final.state"
DARK = RES / "2026-10-30_continuum_39633433328091196_dark_map"
N_PRIOR = 20000
AV_CUT = 0.05  # mag, for prior and posterior mass below it
# lnlike of the dark MAP point (hyperparameters fixed at the HR max-L) under this likelihood, from
# the check step of fit.py on stampede3 (logs of jobs 3664748 and 3664872)
MAP_LNL = 85382.223465
CORNER = ["dust2", "dust_index", "A_V", "E(B-V)", "logzsol", "afe"]
TABLE = [
    "dust2", "dust_index", "dust_ratio", "A_V", "E(B-V)", "logzsol", "afe", "logmass",
    "sigma_smooth", "sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn",
] + [f"logsfr_ratios_{i}" for i in range(1, 10)]  # fmt: skip
RED = "#b2182b"
BLUE = "#2166ac"


def load_fit():
    """Return fit.py of this folder, loaded by path."""
    spec = importlib.util.spec_from_file_location("fitpoco", HERE / "fit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sampler(path):
    """Return a pocoMC Sampler holding the state in ``path``, without its pool."""
    sampler = pc.Sampler.__new__(pc.Sampler)
    with open(path, "rb") as f:
        sampler.__dict__.update(dill.load(f))
    return sampler


def kc13_curve(lam, delta):
    """Return the FSPS dust_type 4 curve, tau(lam) / tau_V, for rest ``lam`` in A.

    As 2026-10-29_c3k_vs_miles_dust/analysis.py: the Calzetti curve over 4.05, plus a 2175 A
    Drude bump of strength 0.85 - 1.9 delta, times (lam / 5500)^delta.
    """
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


def wcorr(x, y, w):
    """Return the weighted Pearson correlation of ``x`` and ``y``."""
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(
        np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)
    )


def columns(model, theta):
    """Return a dict of named columns, with A_V and E(B-V) added."""
    labels = list(model.theta_labels())
    cols = {k: theta[:, i] for i, k in enumerate(labels)}
    ab, av = attenuation([4400.0, 5500.0], cols["dust2"], cols["dust_index"]).T
    cols["A_V"], cols["E(B-V)"] = av, ab - av
    return cols


# %% posterior, prior draws and the dark MAP
fit = load_fit()
model = fit.state(with_sps=False)[0]
s = load_sampler(STATE)
usamp, w, logl, _ = s.posterior()
w = w / w.sum()
ess = 1.0 / np.sum(w**2)
logz, logz_err = s.evidence()
theta_post = np.array([model.prior_transform(u) for u in usamp])
post = columns(model, theta_post)
rng = np.random.default_rng(0)
prior = columns(
    model, np.array([model.prior_transform(u) for u in rng.uniform(size=(N_PRIOR, model.ndim))])
)
m = np.load(DARK / f"{fit.base.TID}_map.npz")
mp = dict(zip([str(x) for x in m["labels"]], m["best"], strict=True))
ab, av = attenuation([4400.0, 5500.0], mp["dust2"], mp["dust_index"])[0]
mp["A_V"], mp["E(B-V)"] = av, ab - av
assert abs(mp["A_V"] - 0.423) < 0.001 and abs(mp["E(B-V)"] - 0.098) < 0.001, (av, ab - av)


# %% table
def prior_range(name):
    """Return the prior bounds of a free parameter, or the range of the prior draws if derived."""
    if name in model.config_dict:
        lo, hi = np.ravel(model.config_dict[name]["prior"].range)[:2]
        return float(lo), float(hi)
    return float(prior[name].min()), float(prior[name].max())


Q = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
lines = [
    f"pmc_final.state: {len(w)} samples after pocoMC's weight trimming, ESS {ess:.0f}, "
    f"log Z {logz:.3f} +- {logz_err:.3f}",
    f"max lnlike of the posterior samples {logl.max():.3f}; dark MAP point (hyperparameters fixed "
    f"at the HR max-L) {MAP_LNL:.3f}; difference {logl.max() - MAP_LNL:+.3f}",
    "",
    f"{'parameter':>13} {'prior lo':>9} {'prior hi':>9} {'q2.5':>9} {'q16':>9} {'median':>9} "
    f"{'q84':>9} {'q97.5':>9} {'MAP':>9} {'MAP pct':>7}",
]
for k in TABLE:
    lo, hi = prior_range(k)
    qs = wquantile(post[k], w, Q)
    if k in mp:
        pct = 100 * np.sum(w[post[k] <= mp[k]])
        mcell = f"{mp[k]:9.4f} {pct:7.1f}"
    else:
        mcell = f"{'fixed':>9} {'':>7}"
    lines.append(f"{k:>13} {lo:9.3f} {hi:9.3f} " + " ".join(f"{q:9.4f}" for q in qs) + f" {mcell}")
lines += [
    "Quantiles are weighted. MAP pct = posterior weight at or below the MAP value. A_V and E(B-V)",
    "prior bounds are the range of the prior draws. Hyperparameters were fixed in the MAP.",
    "",
    "dust2 = 0:",
]
for k in ("dust2", "A_V"):
    q005, q025 = wquantile(post[k], w, np.array([0.005, 0.025]))
    lines.append(f"  {k}: posterior 0.5% quantile {q005:.4f}, 2.5% quantile {q025:.4f}")
pp = np.sum(w[post["A_V"] < AV_CUT])
pr = np.mean(prior["A_V"] < AV_CUT)
lines += [
    f"  P(A_V < {AV_CUT} mag): posterior {pp:.2e} (of {len(w)} samples, ESS {ess:.0f}), "
    f"prior {pr:.4f} ({N_PRIOR} draws, seed 0)",
    "",
    "weighted correlation with:     logzsol      afe  logmass  dust_index",
]
for k in ("dust2", "A_V", "E(B-V)"):
    r = [wcorr(post[k], post[j], w) for j in ("logzsol", "afe", "logmass", "dust_index")]
    lines.append(f"  {k:>8} " + " ".join(f"{x:+9.3f}" for x in r))
text = "\n".join(lines) + "\n"
(OUT / "posterior.txt").write_text(text)
print(text)

# %% corners in one style: red filled 1 and 2 sigma contours, median and 16-84% titles, axes the
# central 99.8% of the posterior plus 5%, prior bounds dashed where they fall inside
import corner  # noqa: E402

from hubersed.fitting.map_fits import sfh_from_theta  # noqa: E402

use_apj_style()
labels = list(model.theta_labels())


def axis_range(x, q=(0.001, 0.999), pad=0.05):
    """Return the central 99.8% of the posterior plus 5% padding."""
    lo, hi = wquantile(x, w, np.array(q))
    d = (hi - lo) * pad or 1e-3
    return (lo - d, hi + d)


def red_corner(names, path, inset=False):
    """Draw the corner of the posterior columns ``names``, with the sSFR posterior if ``inset``."""
    k = len(names)
    fig = plt.figure(figsize=(2.8 * k, 2.8 * k))
    corner.corner(np.column_stack([post[p] for p in names]), weights=w, fig=fig, labels=names,
                  color=RED, range=[axis_range(post[p]) for p in names], plot_datapoints=False,
                  fill_contours=True, smooth=1.0, bins=30, levels=(0.393, 0.865),
                  quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3f", max_n_ticks=3,
                  label_kwargs={"fontsize": 34}, title_kwargs={"fontsize": 30}, labelpad=0.3,
                  hist_kwargs={"color": RED, "lw": 2})  # fmt: skip
    axes = np.array(fig.axes).reshape(k, k)
    for j, p in enumerate(names):
        if p not in model.config_dict:  # derived, or an element of the logsfr_ratios vector
            continue
        ax = axes[j, j]
        for edge in prior_range(p):
            if ax.get_xlim()[0] <= edge <= ax.get_xlim()[1]:
                ax.axvline(edge, color="k", ls="--", lw=2)
    for ax in axes.ravel():
        ax.tick_params(labelsize=22)
    for ax in axes[-1]:
        ax.xaxis.label.set_rotation(35)
        ax.xaxis.label.set_ha("right")
    for ax in axes[:, 0]:
        ax.yaxis.label.set_rotation(55)
        ax.yaxis.label.set_va("top")
    if inset:
        rng = np.random.default_rng(0)
        sfhs = [sfh_from_theta(model, theta_post[i]) for i in rng.choice(len(w), size=2000, p=w)]
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
        ax.set_title(f"{fit.base.TID}, pocoMC C3K ER dark", fontsize=46)
    fig.savefig(path, dpi=60, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


red_corner(CORNER, OUT / "corner.png")
red_corner(["logzsol", "afe", "logmass", "dust2", "dust_ratio", "dust_index", "sigma_smooth",
            "sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn"], OUT / "corner_stars_dust.png")  # fmt: skip
red_corner([f"logsfr_ratios_{i}" for i in range(1, 10)], OUT / "corner_sfh.png")
red_corner(
    [p for p in labels if not p.startswith("logsfr_ratios")], OUT / "corner_all_ssfr.png", True
)

# %% prior edges, as corner_sfh.py of 10-19
names = [p for p in labels if not p.startswith("logsfr_ratios")]
edge_lines = [
    f"{'parameter':>16} {'prior lo':>9} {'prior hi':>9} {'q0.5%':>9} {'median':>9} "
    f"{'q99.5%':>9} {'lo dist':>7} {'hi dist':>7}"
]
for p in names:
    lo, hi = prior_range(p)
    a, med_, b = wquantile(post[p], w, np.array([0.005, 0.5, 0.995]))
    edge_lines.append(f"{p:>16} {lo:9.3g} {hi:9.3g} {a:9.4g} {med_:9.4g} {b:9.4g} "
                      f"{(a - lo) / (hi - lo):7.3f} {(hi - b) / (hi - lo):7.3f}")  # fmt: skip
edge_lines += [
    "",
    "dist = distance of the 0.5% or 99.5% quantile to the bound, in units of prior width.",
]
(OUT / "prior_edges.txt").write_text("\n".join(edge_lines) + "\n")
print("\n".join(edge_lines))

# %% attenuation curve: posterior median and 68% band against the MAP
lam = np.linspace(3000.0, 7500.0, 451)
curves = attenuation(lam, post["dust2"], post["dust_index"])
band = np.array([wquantile(curves[:, j], w, np.array([0.16, 0.5, 0.84])) for j in range(lam.size)])
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(lam, band[:, 0], band[:, 2], color=RED, alpha=0.3, lw=0, label=r"posterior 16-84\%")
ax.plot(lam, band[:, 1], color=RED, lw=1.5, label="posterior median")
ax.plot(lam, attenuation(lam, mp["dust2"], mp["dust_index"])[0], color=BLUE, ls="--", lw=1.5,
        label=f"dark MAP (dust2 {mp['dust2']:.3f}, dust_index {mp['dust_index']:+.2f})")  # fmt: skip
ax.set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
ax.set_ylabel(r"$A_\lambda$ [mag]", fontsize=14)
ax.legend(frameon=False, fontsize=11)
fig.savefig(OUT / "attenuation.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'attenuation.png'}")


# %% data with the max-likelihood pocoMC model and the dark MAP, then the chi of each
import astropy.units as u  # noqa: E402
from scipy.signal import medfilt  # noqa: E402

from hubersed.conversion import to_flambda  # noqa: E402


def flam(d, x):
    """Return maggies on the fit's wavelength grid in 1e-17 erg/s/cm^2/A."""
    return to_flambda(np.asarray(d["wave"], float) * u.AA, np.asarray(x, float) * u.mgy).value


def chi(d):
    """Return the chi of a fit on its own data, NaN on unfitted pixels."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d["good"], (d["flux"] - d["sp_best"]) / d["unc"], np.nan)


fits = {
    "pocoMC max-likelihood": (np.load(OUT / f"{fit.base.TID}_poco_maxl.npz"), RED),
    "dark MAP": (m, BLUE),
}
a = fits["pocoMC max-likelihood"][0]
assert np.array_equal(a["good"], m["good"]) and np.allclose(
    a["flux"][a["good"]], m["flux"][m["good"]]
)
rest = a["wave"] / (1 + float(a["z"]))
g = a["good"]
data = flam(a, a["flux"])
fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.06})  # fmt: skip
ax[0].plot(rest, np.where(g, data, np.nan), color="0.75", lw=0.5, label="DESI SV3 dark (DR1 coadd)")
med = np.full(data.size, np.nan)
med[g] = medfilt(data[g], 9)
ax[0].plot(rest, med, color="k", lw=0.8, label="median filter, 9 pixels")
for (name, (d, color)), a_chi in zip(fits.items(), ax[1:], strict=True):
    ax[0].plot(rest, np.where(g, flam(d, d["sp_best"]), np.nan), color=color, lw=0.8,
               label=f"C3K ER {name}")  # fmt: skip
    a_chi.plot(rest, chi(d), color=color, lw=0.5)
    for h in (-2, 2):
        a_chi.axhline(h, color="0.3", ls="--", lw=1)
    a_chi.axhline(0, color="0.3", lw=0.6)
    a_chi.set_ylim(-5, 5)
    a_chi.set_ylabel(r"$\chi$", fontsize=14)
ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
ax[0].set_ylim(bottom=0)
ax[0].legend(frameon=False, fontsize=12, loc="upper right")
ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
fig.savefig(OUT / "best_fit.png", dpi=130, bbox_inches="tight")
plt.close(fig)
for name, (d, _) in fits.items():
    c = chi(d)[g]
    print(f"{name}: chi2 {np.sum(c**2):.1f} on {g.sum()} px")
print(f"wrote {OUT / 'best_fit.png'}")
