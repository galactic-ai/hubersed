"""Does a PSB SFH fit gallery105 22, 51 and 69 better than the current stochastic SFH?

QUESTION: for each target, how do model A (current stochastic SFH) and model B (prospect's PSB SFH)
compare in max-likelihood chi2, log evidence and SFH posterior?
HYPOTHESIS: none registered. Good fits under both mean rarity or data; bad under A and good under B
mean the SFH prior cannot reach them; bad under both mean model failure.
INPUTS: results/2026-11-14_poco_psb_sfh/<TID>/<model>/{fit.h5, maxl.npz} (fit.py, predict.py).
SEED: 0 (posterior draws).
COMMAND: ``PYTHONPATH=experiments/2026-10-15_broadline_free_amplitude uv run --no-sync python
experiments/2026-11-14_poco_psb_sfh/analysis.py``.
RESULT: 2026-11-14, fits.txt. Both fit 22, 51, 69 well (chi2/N 0.994-1.098), ln Z(B) - ln Z(A) 3.98
+- 1.01, 0.43 +- 0.71, -0.05 +- 0.49, so their flags are rarity or data. 99 fits with Cue (1.031).
2026-11-20: model L chi2/N of gallery105 28, 34, 80, 25, 64 is 1.486, 1.426, 1.417, 1.091, 1.107;
L_recoadd 1.299, 1.159, 1.244, 1.059, 1.091 (ln Z not compared, different data).
Assumptions: continuum-only C3K ER, no line mask, SV3 DR1 coadd, B's cut prior; 51 has a Gaia match.
FIGURES: results/2026-11-14_poco_psb_sfh/ <TID>/<model>/best_fit.png, sfh.png, sfh_99.png,
corner_99_AC.png, corner_99_AC_AC0.png, fit_<nn>_<model>.png. For L, sfh_em5.png. For L and
L_recoadd, fit_<nn>_<model>.png, windows_<nn>_<model>.png, corner_<nn>_<model>_{stars_dust,gas,all_ssfr}.png.
"""

# %%
import importlib.util
import sys
from pathlib import Path

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

from hubersed.conversion import to_flambda
from hubersed.plotting.style import use_apj_style

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fitpsb", HERE / "fit.py")
fit = importlib.util.module_from_spec(_spec)
sys.modules["fitpsb"] = fit
_spec.loader.exec_module(fit)
OUT = fit.OUT
NAMES = {
    "39627800440737892": "22",
    "39628454575998103": "51",
    "39633145275876182": "69",
    "39633421919584782": "99",
    "39627818010675704": "28",
    "39627854832472157": "34",
    "39633318999755849": "80",
    "39627806493117457": "25",
    "39633127609469239": "64",
}  # numbers from results/2026-11-13_gallery105/gallery105.tsv
EM5 = list(NAMES)[4:]  # the emission-line outliers, models L and L_recoadd only
PSB4 = list(NAMES)[:4]
# all but A and B only for 99; L and L_recoadd only for the emission-line outliers
MODELS = ("A", "B", "AC", "BC", "AC0", "AC_main", "BC_main", "L", "L_recoadd")
L_MODELS = ("L", "L_recoadd")
DATA_LABEL = {"sv3": "DESI DR1 SV3 coadd", "recoadd": "DESI SV3 recoadd"}


def models_of(tid):
    """Return the models of ``tid`` with a predict.py output."""
    return [m for m in MODELS if (OUT / tid / m / "maxl.npz").is_file()]


def model_of(name, d):
    """Return the fit.py model of ``name`` for maxl.npz ``d``; L models take their sigma_smooth floor."""
    floor = float(d["sigma_floor"]) if name in L_MODELS else None
    return fit.make_model(name, float(d["z"]), floor)


RED = "#b2182b"
N_DRAW = 2000
REST_BALMER = (3700.0, 4400.0)
F15_T, PEAK_MAX = 1.5, 2.0  # Gyr, as sfh.py


def wquantile(x, w, q):
    """Return the weighted quantiles ``q`` of ``x``."""
    order = np.argsort(x)
    cdf = (np.cumsum(w[order]) - 0.5 * w[order]) / np.sum(w)
    return np.interp(q, cdf, x[order])


def sfh_stats(model, theta):
    """Return the mass fraction formed within F15_T Gyr and the SFR-peak lookback time in Gyr."""
    model.set_parameters(theta)
    edges = 10 ** np.asarray(model.params["agebins"], float) / 1e9
    mass = np.asarray(model.params["mass"], float)
    lo, hi = edges[:, 0], edges[:, 1]
    f15 = np.sum(mass * np.clip(np.minimum(hi, F15_T) - lo, 0, None) / (hi - lo)) / mass.sum()
    recent = lo < PEAK_MAX
    k = np.flatnonzero(recent)[np.argmax((mass / (hi - lo))[recent])]
    return f15, 0.5 * (lo[k] + hi[k])


def best_fit_figure(d, path, title):
    """Draw data, max-likelihood model and chi as 2026-11-01_poco_repeats/analysis.py."""
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
    ax[0].plot(rest, np.where(g, data, np.nan), color="0.75", lw=0.5,
               label="DESI SV3 bright (DR1 coadd)")  # fmt: skip
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
    ax[0].set_title(title, fontsize=14)
    ax[0].legend(frameon=False, fontsize=12, loc="upper right")
    ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# %%
use_apj_style()
q = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
lines, logz = [], {}
for tid, nn in NAMES.items():
    for name in models_of(tid):
        folder = OUT / tid / name
        d = np.load(folder / "maxl.npz")
        with h5py.File(folder / "fit.h5", "r") as h:
            theta = h["sampling/unstructured_chain"][:]
            w = h["sampling/weights"][:]
            lnl = h["sampling/lnlike"][:]
        w = w / w.sum()
        model = model_of(name, d)
        labels = list(model.theta_labels())
        assert labels == [str(x) for x in d["labels"]], "labels differ from predict.py"
        g = d["good"]
        chi = (d["flux"] - d["sp_best"])[g] / d["unc"][g]
        rest = (d["wave"] / (1 + float(d["z"])))[g]
        bal = (rest >= REST_BALMER[0]) & (rest < REST_BALMER[1])
        logz[(tid, name)] = (float(d["logz"]), float(d["logz_err"]))
        lines += ["", f"{nn} ({tid}) model {name}: {len(w)} samples, ESS {1 / np.sum(w**2):.0f}, "
                  f"log Z {float(d['logz']):.3f} +- {float(d['logz_err']):.3f}, max lnlike "
                  f"{lnl.max():.3f} (maxl.npz {float(d['lnlike']):.3f}); chi2/N {np.mean(chi**2):.3f} "
                  f"on {g.sum()} px, {np.mean(chi[bal] ** 2):.3f} on {bal.sum()} px at rest "
                  f"{REST_BALMER[0]:.0f}-{REST_BALMER[1]:.0f} A"]  # fmt: skip
        lines.append(f"  {'parameter':>20} {'prior lo':>9} {'prior hi':>9} "
                     + " ".join(f"{'q' + format(100 * x, 'g'):>9}" for x in q)
                     + f" {'lo dist':>7} {'hi dist':>7}")  # fmt: skip
        for j, p in enumerate(labels):
            # only priors with hard bounds (mini, maxi); StudentT and the hyperparameter MVN have none
            pp = model.config_dict[p]["prior"].params if p in model.config_dict else {}
            lo, hi = (float(pp["mini"]), float(pp["maxi"])) if "mini" in pp else (np.nan, np.nan)
            a, b = wquantile(theta[:, j], w, np.array([0.005, 0.995]))
            span = hi - lo if np.isfinite(hi - lo) else np.nan
            lines.append(f"  {p:>20} {lo:9.3g} {hi:9.3g} "
                         + " ".join(f"{v:9.4f}" for v in wquantile(theta[:, j], w, q))
                         + f" {(a - lo) / span:7.3f} {(hi - b) / span:7.3f}")  # fmt: skip
        rng = np.random.default_rng(0)
        stats = np.array([sfh_stats(model, theta[i]) for i in rng.choice(len(w), N_DRAW, p=w)])
        for k, lab in enumerate((f"mass fraction formed < {F15_T} Gyr", "SFR-peak lookback (Gyr)")):
            p16, p50, p84 = np.percentile(stats[:, k], [16, 50, 84])
            lines.append(f"  {lab}: {p50:.3f} (16-84% {p16:.3f}-{p84:.3f})")
        best_fit_figure(
            d, folder / "best_fit.png", f"{nn} {tid}  z {float(d['z']):.4f}  model {name}"
        )
lines += ["", "ln Z differences (B and BC include their cut prior, about -0.35):"]
for tid, nn in NAMES.items():
    for b, a in (("B", "A"), ("BC", "AC"), ("AC", "A"), ("BC", "B"), ("AC", "AC0"),
                 ("BC_main", "AC_main")):  # fmt: skip
        if (tid, a) in logz and (tid, b) in logz:
            (za, ea), (zb, eb) = logz[(tid, a)], logz[(tid, b)]
            lines.append(f"  {nn} ({tid}) {b} - {a}: {zb - za:.2f} +- {np.hypot(ea, eb):.2f}")
lines.append("")
lines.append("dist = distance of the 0.5% or 99.5% quantile to the bound, in prior widths.")
text = "\n".join(lines)
(OUT / "fits.txt").write_text(text + "\n")
print(text)


# %% logzsol degeneracies: Spearman rank correlation over 2000 weighted draws (seed 0)
from scipy.stats import spearmanr  # noqa: E402

deg = ["Spearman correlation of logzsol with each quantity over 2000 posterior draws (seed 0)"]
for tid, nn in NAMES.items():
    for name in [m for m in models_of(tid) if m not in L_MODELS]:  # PSB models only
        folder = OUT / tid / name
        d = np.load(folder / "maxl.npz")
        with h5py.File(folder / "fit.h5", "r") as h:
            theta = h["sampling/unstructured_chain"][:]
            w = h["sampling/weights"][:]
        w = w / w.sum()
        model = fit.make_model(name, float(d["z"]))
        labels = list(model.theta_labels())
        idx = np.random.default_rng(0).choice(len(w), N_DRAW, p=w)
        f15 = np.array([sfh_stats(model, theta[i])[0] for i in idx])
        z = theta[idx, labels.index("logzsol")]
        cols = {"f(<1.5 Gyr)": f15}
        cols.update(
            {
                p: theta[idx, labels.index(p)]
                for p in ("afe", "dust2", "dust_index", "logmass")
                if p in labels
            }
        )
        deg.append(f"{nn} model {name}: logzsol median {np.median(z):.3f}, share > 0.45 "
                   f"{np.mean(z > 0.45):.3f}; " + ", ".join(
                       f"{k} {spearmanr(z, v).statistic:+.2f}" for k, v in cols.items()))  # fmt: skip
(OUT / "degeneracy.txt").write_text("\n".join(deg) + "\n")
print("\n".join(deg))


# %% SFH figure: sSFR = SFR / (0.6 x mass formed) as hubersed.fitting.map_fits.sfh_from_theta, on a
# common log lookback-time grid since model B's bins move between draws
TGRID = np.logspace(-3, np.log10(14.0), 400)  # Gyr
STYLE = {
    "A": dict(color=RED, label="A (stochastic)"),
    "B": dict(color="#2166ac", label="B (PSB)"),
    "AC": dict(color=RED, ls="--", label="A + Cue"),
    "BC": dict(color="#2166ac", ls="--", label="B + Cue"),
    "AC0": dict(color="#f4a582", ls=":", label="A + Cue, afe 0"),
    "AC_main": dict(color="#b35806", ls="-.", label="A + Cue, main coadd"),
    "BC_main": dict(color="#542788", ls="-.", label="B + Cue, main coadd"),
    "L": dict(color=RED, label="L (stochastic + Cue + free lines)"),
    "L_recoadd": dict(color=RED, label="L, SV3 recoadd"),
}
SFH_MODELS = ("A", "B", "AC", "BC")  # sfh.png; 99's Cue variants go to sfh_99.png


def ssfr_grid(model, theta):
    """Return sSFR (1/yr) of ``theta`` on TGRID, NaN outside the age bins."""
    model.set_parameters(theta)
    edges = 10 ** np.asarray(model.params["agebins"], float) / 1e9
    mass = np.asarray(model.params["mass"], float)
    ssfr = mass / ((edges[:, 1] - edges[:, 0]) * 1e9) / (0.6 * mass.sum())
    k = np.searchsorted(edges[:, 1], TGRID)
    out = np.full(TGRID.size, np.nan)
    ok = (TGRID >= edges[0, 0]) & (k < mass.size)
    out[ok] = ssfr[k[ok]]
    return out


def ssfr_band(tid, name):
    """Return z and the 16/50/84 percentiles of sSFR on TGRID for one fit (2000 draws, seed 0)."""
    folder = OUT / tid / name
    d = np.load(folder / "maxl.npz")
    with h5py.File(folder / "fit.h5", "r") as h:
        theta = h["sampling/unstructured_chain"][:]
        w = h["sampling/weights"][:]
    w = w / w.sum()
    model = model_of(name, d)
    idx = np.random.default_rng(0).choice(len(w), N_DRAW, p=w)
    grid = np.array([ssfr_grid(model, theta[i]) for i in idx])
    return float(d["z"]), np.nanpercentile(grid, [16, 50, 84], axis=0)


def draw(ax, tid, names):
    """Draw the sSFR bands of ``names`` for ``tid`` on ``ax``."""
    for name in names:
        z, (lo, med, hi) = ssfr_band(tid, name)
        alpha = 0.25 if name in ("A", "B", "L") else 0.12
        ax.fill_between(TGRID, lo, hi, color=STYLE[name]["color"], alpha=alpha, lw=0)
        ax.plot(TGRID, med, lw=2, **STYLE[name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{NAMES[tid]}  {tid}  z {z:.3f}", fontsize=13)
    ax.set_ylabel(r"sSFR [yr$^{-1}$]", fontsize=13)


fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
for ax, tid in zip(axes.ravel(), PSB4, strict=True):
    names = [m for m in models_of(tid) if m in SFH_MODELS]
    draw(ax, tid, names)
    if len(names) > 2:
        ax.legend(frameon=False, fontsize=10)
for ax in axes[-1]:
    ax.set_xlabel("lookback time [Gyr]", fontsize=13)
axes[0, 0].legend(frameon=False, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "sfh.png", dpi=120)
plt.close(fig)
print(f"wrote {OUT / 'sfh.png'}")

# the emission-line outliers' model L, five panels of a 2 x 3 grid
if any("L" in models_of(t) for t in EM5):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for ax, tid in zip(axes.ravel(), EM5, strict=False):
        if "L" in models_of(tid):
            draw(ax, tid, ["L"])
    axes[1, 2].set_axis_off()
    axes[0, 2].tick_params(labelbottom=True)  # its lower neighbour is off
    for ax in (axes[0, 2], axes[1, 0], axes[1, 1]):
        ax.set_xlabel("lookback time [Gyr]", fontsize=13)
    axes[0, 0].legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "sfh_em5.png", dpi=120)
    plt.close(fig)
    print(f"wrote {OUT / 'sfh_em5.png'}")

# 99's Cue variants: SV3 A and B, afe fixed at 0, and the main coadd
TID99 = "39633421919584782"
fig, ax = plt.subplots(figsize=(8, 5.5))
draw(ax, TID99, [m for m in ("AC", "BC", "AC0", "AC_main", "BC_main") if m in models_of(TID99)])
ax.set_xlabel("lookback time [Gyr]", fontsize=13)
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "sfh_99.png", dpi=120)
plt.close(fig)
print(f"wrote {OUT / 'sfh_99.png'}")


# %% where does free afe help 99: chi2 of AC0 (afe 0) minus AC (afe free) by rest-frame region at the
# max-likelihood points; region edges are rough feature windows (vacuum, from memory)
REGIONS = {
    "blue end 3550-3900": (3550, 3900),
    "Balmer/4000 3900-4400": (3900, 4400),
    "Hbeta 4840-4890": (4840, 4890),
    "Mg b 5150-5210": (5150, 5210),
    "Fe 5250-5350": (5250, 5350),
    "Na D 5880-5910": (5880, 5910),
    "Halpha+[NII] 6540-6600": (6540, 6600),
    "Ca triplet 8480-8680": (8480, 8680),
    "all": (0, 1e5),
}
if all((OUT / TID99 / m / "maxl.npz").is_file() for m in ("AC", "AC0")):
    chi2 = {}
    for m in ("AC", "AC0"):
        d = np.load(OUT / TID99 / m / "maxl.npz")
        g = d["good"]
        rest = d["wave"] / (1 + float(d["z"]))
        chi2[m] = (np.where(g, ((d["flux"] - d["sp_best"]) / d["unc"]) ** 2, 0.0), rest, g)
    lines = [
        "99: chi2(AC0, afe 0) - chi2(AC, afe free) at the max-likelihood points, by rest region"
    ]
    for lab, (lo, hi) in REGIONS.items():
        s = (chi2["AC"][1] >= lo) & (chi2["AC"][1] < hi) & chi2["AC"][2]
        dc = float(chi2["AC0"][0][s].sum() - chi2["AC"][0][s].sum())
        lines.append(f"  {lab:24} {int(s.sum()):5d} px  dchi2 {dc:+8.2f}")
    (OUT / "afe_regions.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# %% 99 corners in the style of 2026-11-01_poco_repeats/analysis.py red_corner: AC alone with afe, and
# AC (red) over AC0 (orange) on their shared parameters; prior bounds dashed where in range
import corner  # noqa: E402

CORNER_99 = ["afe", "dust2", "dust_index", "logzsol", "logmass", "gas_logz"]


def posterior(name, tid=TID99):
    """Return the model, samples and normalised weights of a fit (99 by default)."""
    d = np.load(OUT / tid / name / "maxl.npz")
    with h5py.File(OUT / tid / name / "fit.h5", "r") as h:
        theta = h["sampling/unstructured_chain"][:]
        w = h["sampling/weights"][:]
    return model_of(name, d), theta, w / w.sum()


def corner_fig(names, cols, path, colors, tid=TID99, inset=False):
    """Draw overlaid corners of ``names`` on ``cols`` and save to ``path``.

    Parameters
    ----------
    inset : bool
        Add the sSFR median and 16-84% band of each fit (ssfr_band, as sfh.png) top right.
    """
    fig = None
    posts = [posterior(n, tid) for n in names]
    rng_ = []
    for p in cols:
        los, his = [], []
        for model, theta, w in posts:
            x = theta[:, list(model.theta_labels()).index(p)]
            lo, hi = wquantile(x, w, np.array([0.001, 0.999]))
            los.append(lo)
            his.append(hi)
        pad = 0.05 * (max(his) - min(los))
        rng_.append((min(los) - pad, max(his) + pad))
    for (model, theta, w), c in zip(posts, colors, strict=True):
        labels = list(model.theta_labels())
        x = np.column_stack([theta[:, labels.index(p)] for p in cols])
        fig = corner.corner(x, weights=w, fig=fig, labels=cols, color=c, range=rng_,
                            plot_datapoints=False, fill_contours=True, smooth=1.0, bins=30,
                            levels=(0.393, 0.865), hist_kwargs={"color": c, "lw": 2},
                            label_kwargs={"fontsize": 16})  # fmt: skip
    axes = np.array(fig.axes).reshape(len(cols), len(cols))
    model = posts[0][0]
    for j, p in enumerate(cols):
        pp = model.config_dict[p]["prior"].params
        for e in (pp.get("mini"), pp.get("maxi")):
            if e is not None and rng_[j][0] <= e <= rng_[j][1]:
                axes[j, j].axvline(e, color="k", ls="--", lw=1.5)
    for c, n in zip(colors, names, strict=True):
        axes[0, -1].plot([], [], color=c, lw=3, label=STYLE[n]["label"])
    if inset:  # legend below the inset, as ladder_corner.py
        fig.legend(*axes[0, -1].get_legend_handles_labels(), frameon=False, fontsize=14,
                   loc="upper right", bbox_to_anchor=(0.98, 0.60))  # fmt: skip
        ax = fig.add_axes([0.53, 0.66, 0.44, 0.31])
        for n, c in zip(names, colors, strict=True):
            _, (lo, med, hi) = ssfr_band(tid, n)
            ax.fill_between(TGRID, lo, hi, color=c, alpha=0.25, lw=0)
            ax.plot(TGRID, med, color=c, lw=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("lookback time [Gyr]", fontsize=24)
        ax.set_ylabel(r"sSFR [yr$^{-1}$]", fontsize=24)
        ax.tick_params(labelsize=18)
        ax.set_title(r"median and 16--84\%", fontsize=22)
    else:
        axes[0, -1].legend(frameon=False, fontsize=14, loc="upper right")
    fig.savefig(path, dpi=70, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


corner_fig(["AC"], CORNER_99, OUT / "corner_99_AC.png", [RED])
corner_fig(["AC", "AC0"], CORNER_99[1:], OUT / "corner_99_AC_AC0.png", [RED, "#f4a582"])


# %% total metallicity [Z/H] = logzsol + ([Z/H] - [Fe/H])(logzsol, afe) from the MIST2 grid
# (2026-11-01_poco_repeats/mist2_zh.txt), taking logzsol as [Fe/H] in the afe-enabled C3K ER fits;
# [Fe/H] and afe trade along constant [Z/H]
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

mz = np.loadtxt(OUT.parent / "2026-11-01_poco_repeats" / "mist2_zh.txt", skiprows=1, max_rows=85)
FEH, AFE = np.unique(mz[:, 0]), np.unique(mz[:, 1])
DZ = np.full((FEH.size, AFE.size), np.nan)
for row in mz:
    DZ[np.searchsorted(FEH, row[0]), np.searchsorted(AFE, row[1])] = row[5]
assert np.isfinite(DZ).all(), "MIST2 grid is not complete"
dz_of = RegularGridInterpolator((FEH, AFE), DZ)


def q3(x, w):
    """Return the weighted 16/50/84 percentiles of ``x`` as text."""
    return " ".join(f"{v:+.3f}" for v in wquantile(x, w, np.array([0.16, 0.5, 0.84])))


zh = [
    "[Z/H] = logzsol + MIST2 ([Z/H] - [Fe/H]) at (logzsol, afe); 16/50/84% of the weighted samples"
]
for tid, nn in NAMES.items():
    for name in [m for m in models_of(tid) if m not in L_MODELS]:  # PSB models only
        d = np.load(OUT / tid / name / "maxl.npz")
        with h5py.File(OUT / tid / name / "fit.h5", "r") as h:
            theta = h["sampling/unstructured_chain"][:]
            w = h["sampling/weights"][:]
        w = w / w.sum()
        labels = list(fit.make_model(name, float(d["z"])).theta_labels())
        feh = theta[:, labels.index("logzsol")]
        afe = theta[:, labels.index("afe")] if "afe" in labels else np.zeros_like(feh)
        tot = feh + dz_of(np.column_stack([feh, afe]))
        zh.append(f"  {nn} {name:8} logzsol {q3(feh, w)}  afe {q3(afe, w)}  [Z/H] {q3(tot, w)}")
(OUT / "total_z.txt").write_text("\n".join(zh) + "\n")
print("\n".join(zh))


# %% fit figures in the design of 2026-11-01_poco_repeats/ladder_plot.py (top panel from zero, 9 px
# median filter, DR10 cutout with the 1.5'' fibre top right, one chi panel), with PSB markings as in the
# literature: Balmer series H-alpha to H-theta (orange) and the
# strong forbidden lines (purple) as ticks (Pawlik+18), Ca II H&K ticks (Wild+07), and the HdA band (dark
# grey) and its continua (light grey) shaded (Setton+20). Rest vacuum wavelengths; line values from
# memory, HdA bands from Worthey & Ottaviani 1997 moved from air to vacuum as indices.py.
from matplotlib.patches import Circle  # noqa: E402

from hubersed.sps.utils import _airtovac  # noqa: E402

_lspec = importlib.util.spec_from_file_location(
    "ladder_plot_1101", HERE.parent / "2026-11-01_poco_repeats" / "ladder_plot.py"
)
ladder = importlib.util.module_from_spec(_lspec)
_lspec.loader.exec_module(ladder)
ladder.CUTOUTS = OUT / "cutouts"  # cache the cutouts here, not in 11-01's results
BALMER = {r"H$\alpha$+[NII]": 6564.61, r"H$\beta$": 4862.68, r"H$\gamma$": 4341.69, r"H$\delta$": 4102.89,
          r"H$\epsilon$+Ca II H": 3971.20, r"H$\zeta$": 3890.17, r"H$\eta$": 3836.48, r"H$\theta$": 3798.98}  # fmt: skip
CAII = {"Ca II K": 3934.78, "": 3969.59}  # Ca II H labelled with H-epsilon
FORBIDDEN = {  # label: (label position, tick wavelengths)
    "[OII]": (3728.5, (3727.09, 3729.88)),
    "[OIII]": (4984.3, (4960.30, 5008.24)),
    "": (6585.27, (6549.86, 6585.27)),  # [NII], labelled with H-alpha
    "[SII]": (6725.5, (6718.29, 6732.67)),
}
HDA_BANDS = [(_airtovac(a), _airtovac(b)) for a, b in
             ((4083.50, 4122.25), (4041.60, 4079.75), (4128.50, 4161.00))]  # fmt: skip
ORANGE, PURPLE, GREEN = "#e66101", "#5e3c99", "#1b7837"
FIG_FITS = {  # (TID, model): chi panel label
    (TID99, "AC"): "C3K ER + afe + R matrix, stochastic SFH + Cue (DR1 SV3 coadd)",
    ("39627800440737892", "A"): "C3K ER + afe + R matrix, stochastic SFH (DR1 SV3 coadd)",
    ("39628454575998103", "A"): "C3K ER + afe + R matrix, stochastic SFH (DR1 SV3 coadd)",
    ("39633145275876182", "A"): "C3K ER + afe + R matrix, stochastic SFH (DR1 SV3 coadd)",
} | {(tid, "L"): "C3K ER + R matrix, stochastic SFH + Cue + free lines (DR1 SV3 coadd)" for tid in EM5}  # fmt: skip
FIG_FITS |= {(tid, "L_recoadd"): "C3K ER + R matrix, stochastic SFH + Cue + free lines (SV3 recoadd)"
             for tid in EM5}  # fmt: skip
# model L's ticks: EMLINES of 2026-11-07_emline_ladder_39627770174637084/analysis.py (rest vacuum)
EMLINES = {  # label: (label position, tick wavelengths)
    "[OII]": (3728.5, (3727.09, 3729.88)),
    "[NeIII]": (3869.86, (3869.86,)),
    r"H$\delta$": (4102.89, (4102.89,)),
    r"H$\gamma$": (4341.69, (4341.69, 4364.44)),
    r"H$\beta$": (4862.68, (4862.68,)),
    "[OIII]": (4984.3, (4960.30, 5008.24)),
    "HeI": (5877.25, (5877.25,)),
    "[OI]": (6302.05, (6302.05,)),
    r"H$\alpha$+[NII]": (6564.61, (6549.86, 6564.61, 6585.27)),
    "[SII]": (6725.5, (6718.29, 6732.67)),
    "[ArIII]": (7137.77, (7137.77,)),
    "[OII]7325": (7327.1, (7321.94, 7332.21)),
}


def fit_figure(tid, name, label):
    """Draw the fit figure of ``tid`` and model ``name`` with the PSB markings and its cutout."""
    d = np.load(OUT / tid / name / "maxl.npz")
    g = d["good"]
    rest = d["wave"] / (1 + float(d["z"]))

    def fl(x):
        return np.where(g, to_flambda(d["wave"] * u.AA, np.asarray(x, float) * u.mgy).value, np.nan)

    data = fl(d["flux"])
    med = np.full(data.size, np.nan)
    med[g] = medfilt(data[g], 9)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.where(g, (d["flux"] - d["sp_best"]) / d["unc"], np.nan)
    fig, ax = plt.subplots(2, 1, figsize=(14, 5.6), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})  # fmt: skip
    ax[0].plot(rest, data, color="0.6", lw=0.6, label="DESI Spectrum")
    ax[0].plot(rest, med, color="k", lw=0.5, label="Median Filter")
    ax[0].plot(rest, fl(d["sp_best"]), color=RED, lw=0.9, label="Model Fit")
    ax[0].set_ylim(bottom=0)
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
    ax[0].legend(frameon=False, fontsize=16, loc="lower center", bbox_to_anchor=(0.5, 1.14), ncol=3)
    ax[1].plot(rest, chi, color=RED, lw=0.5)
    for h in (-2, 2):
        ax[1].axhline(h, color="0.3", ls="--", lw=1)
    ax[1].axhline(0, color="0.3", lw=0.6)
    ax[1].set_ylim(-5, 5)
    ax[1].set_ylabel(r"$\chi$", fontsize=14)
    ax[1].text(0.01, 0.93, label, transform=ax[1].transAxes, ha="left", va="top", fontsize=11,
               bbox=dict(facecolor="white", edgecolor=RED, boxstyle="round,pad=0.25"))  # fmt: skip
    lo, hi = rest[g].min(), rest[g].max()
    ax[1].set_xlim(lo, hi)
    trans = ax[0].get_xaxis_transform()
    if name in L_MODELS:  # the emline ladder's ticks, no index bands
        for n, (m, ws) in EMLINES.items():
            for w in ws:
                ax[0].axvline(w, ymin=0.96, ymax=1.0, color="tab:blue", lw=1.2)
            if lo < m < hi:
                ax[0].text(m, 1.01, n, transform=trans, rotation=90, fontsize=9, ha="center",
                           va="bottom", color="tab:blue")  # fmt: skip
    else:
        for k, (a, b) in enumerate(HDA_BANDS):
            for axx in ax:
                axx.axvspan(a, b, color="0.35" if k == 0 else "0.75", alpha=0.35, lw=0)
        ticks = [(n, (w,), w, ORANGE) for n, w in BALMER.items()]
        ticks += [(n, (w,), w, GREEN) for n, w in CAII.items()]
        ticks += [(n, ws, m, PURPLE) for n, (m, ws) in FORBIDDEN.items()]
        for n, ws, m, c in ticks:
            for w in ws:
                ax[0].axvline(w, ymin=0.95, ymax=1.0, color=c, lw=1.2)
            if n and lo < m < hi:
                ax[0].text(m, 1.01, n, transform=trans, rotation=90, fontsize=8, ha="center",
                           va="bottom", color=c)  # fmt: skip
    ins = ax[0].inset_axes([0.885, 0.64, 0.11, 0.34])
    img = ladder.cutout(int(tid))
    ins.imshow(img, origin="upper")
    cc = ((img.width - 1) / 2, (img.height - 1) / 2)
    ins.add_patch(
        Circle(cc, 0.5 * ladder.FIBRE_DIAM / ladder.PIXSCALE, fill=False, ec="lime", lw=1.0)
    )
    ins.set_axis_off()
    ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
    path = OUT / f"fit_{NAMES[tid]}_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}; chi2/N {np.nanmean(chi**2):.3f} on {g.sum()} px")


for (tid, name), label in FIG_FITS.items():
    if (OUT / tid / name / "maxl.npz").is_file():
        fit_figure(tid, name, label)


# %% model L line windows: chi2/N at the max-likelihood point in the windows of
# 2026-10-15_broadline_free_amplitude/map.py (rest vacuum A), and one figure per target in the design
# of the emline ladder windows.png (2026-10-18_issue14_figures windows_figure)
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


def windows_figure(tid, name, label):
    """Draw data, max-likelihood model and chi in each line window; return chi2/N and N per window."""
    d = np.load(OUT / tid / name / "maxl.npz")
    g = d["good"]
    rest = d["wave"] / (1 + float(d["z"]))

    def fl(x):
        return np.where(g, to_flambda(d["wave"] * u.AA, np.asarray(x, float) * u.mgy).value, np.nan)

    data, mod = fl(d["flux"]), fl(d["sp_best"])
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.where(g, (d["flux"] - d["sp_best"]) / d["unc"], np.nan)
    fig, ax = plt.subplots(2, len(WINDOWS), figsize=(3.2 * len(WINDOWS), 6.0), sharex="col",
                           gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05})  # fmt: skip
    stats = {}
    for i, (key, (a, b)) in enumerate(WINDOWS.items()):
        win = (rest >= a) & (rest < b)
        m = win & g
        stats[key] = (float(np.mean(chi[m] ** 2)) if m.any() else np.nan, int(m.sum()))
        ax[0, i].plot(rest[win], data[win], color="0.2", lw=1.0)
        if m.any():
            ax[0, i].plot(rest[win], mod[win], color=RED, lw=1.1,
                          label=rf"$\chi^2/N$ = {stats[key][0]:.1f}")  # fmt: skip
            ax[1, i].plot(rest[win], chi[win], color=RED, lw=0.9)
        ax[0, i].set_title(key, fontsize=14)
        ylo, yhi = ax[0, i].get_ylim()
        ax[0, i].set_ylim(ylo, ylo + 1.45 * (yhi - ylo))  # room for the chi2 legend above the lines
        if m.any():
            ax[0, i].legend(frameon=False, fontsize=10, loc="upper left")
        for h in (-2, 2):
            ax[1, i].axhline(h, color="0.3", ls="--", lw=0.8)
        ax[1, i].axhline(0, color="0.3", lw=0.6)
        ax[1, i].set_ylim(-5, 5)
        ax[1, i].set_xlabel(r"rest wavelength [$\AA$]", fontsize=12)
    ax[0, 0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ cgs $\AA^{-1}$]", fontsize=13)
    ax[1, 0].set_ylabel(r"$\chi$", fontsize=13)
    handles = [plt.Line2D([], [], color="0.2", lw=2), plt.Line2D([], [], color=RED, lw=3)]
    fig.legend(handles, [DATA_LABEL[fit.split_name(name)[1]], label], loc="lower center",
               bbox_to_anchor=(0.5, 0.915), ncol=2, fontsize=13, frameon=False)  # fmt: skip
    path = OUT / f"windows_{NAMES[tid]}_{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return stats


rows = []
for tid in EM5:
    for name in [m for m in L_MODELS if m in models_of(tid)]:
        st = windows_figure(tid, name, FIG_FITS[(tid, name)])
        rows.append(f"{NAMES[tid]:>3} {tid:>18} {name:>9} "
                    + " ".join(f"{c:7.2f} ({n:2d})" for c, n in st.values()))  # fmt: skip
if rows:
    win_text = "\n".join(
        ["models L and L_recoadd chi2/N (N px) at the max-likelihood point in the line windows",
         "(rest vacuum A, 2026-10-15_broadline_free_amplitude/map.py WINDOWS)",
         f"{'nn':>3} {'TARGETID':>18} {'model':>9} " + " ".join(f"{k:>12}" for k in WINDOWS)]
        + rows
    )  # fmt: skip
    (OUT / "windows_em5.txt").write_text(win_text + "\n")
    print(win_text)


# %% model L corners as corner_99_AC_AC0.png (corner_fig): stars and dust (the ladder_corner_stars_dust
# set of 2026-11-01_poco_repeats/ladder_corner.py; L has no afe), Cue gas and line widths, and every
# parameter but logsfr_ratios with the sSFR inset. Samples and weights from fit.h5.
STARS_L = ["logzsol", "logmass", "dust2", "dust_ratio", "dust_index", "sigma_smooth", "sigma_reg",
           "tau_eq", "sigma_dyn", "tau_dyn"]  # fmt: skip
GAS = ["gas_logz", "gas_logu", "gas_lognH", "gas_logno", "gas_logco", "eline_sigma", "r_broad",
       "r_broad_forb", "eline_vbroad"]  # fmt: skip
for tid in EM5:
    for name in [m for m in L_MODELS if m in models_of(tid)]:
        labels = list(posterior(name, tid)[0].theta_labels())
        every = [p for p in labels if not p.startswith("logsfr_ratios")]
        stem = f"corner_{NAMES[tid]}_{name}"
        corner_fig([name], STARS_L, OUT / f"{stem}_stars_dust.png", [RED], tid=tid)
        corner_fig([name], GAS, OUT / f"{stem}_gas.png", [RED], tid=tid)
        corner_fig([name], every, OUT / f"{stem}_all_ssfr.png", [RED], tid=tid, inset=True)
