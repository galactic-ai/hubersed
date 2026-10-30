"""MILES against C3K dust for 39633433328091196: C. Hahn's four tests (issue #17, 2026-10-29).

QUESTION: (1, 2) at the same parameters, how do the MILES and C3K ER models differ? (3) how do
the attenuation curves of the MILES and C3K ER best points compare? (4) are dust2 and stellar
mass degenerate?
HYPOTHESIS: none registered.
INPUTS: (3) the best points of the C3K ER MAP (icx job 3657895) and the 10-08 MILES nautilus
max-L, until the MILES MAP of 2026-10-29_nodust_map_39633433328091196 exists. (4) the weighted
posterior points of the 10-08 nautilus runs miles_miles_neboff_nodeg, c3k_hr_full_neboff_afezero
and c3k_hr_full_neboff (pred npz, md5 of the MILES one equal to /work; the two HR ones copied from
/work 2026-10-29), and the c3k_hr_full_neboff_afefree checkpoint (no pred npz; points read as in
hypers.py). (1, 2) spectra_er.npz and spectra_miles.npz of spectra.py, when copied.
SEED: none.
COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-29_c3k_vs_miles_dust/analysis.py``.
RESULT: pending.
FIGURES: results/2026-10-29_c3k_vs_miles_dust/curves.png, degeneracy.png, degeneracy.txt and,
once the spectra exist, spectra.png.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39633433328091196
OUT = PATHS["RESULTS"] / "2026-10-29_c3k_vs_miles_dust"
RUNS10 = PATHS["RESULTS"] / "2026-10-08_continuum_outlier"
MAPDIR = PATHS["RESULTS"] / "2026-10-29_continuum_39633433328091196_map"
POSTERIORS = {
    "MILES": (RUNS10 / f"{TID}_miles_miles_neboff_nodeg_pred.npz", "#4393c3"),
    "C3K HR, afe 0": (RUNS10 / f"{TID}_c3k_hr_full_neboff_afezero_pred.npz", "#e08214"),
    "C3K HR, no afe": (RUNS10 / f"{TID}_c3k_hr_full_neboff_pred.npz", "#8c510a"),
    "C3K HR, afe free": (RUNS10 / f"{TID}_c3k_hr_full_neboff_afefree_seed0.h5", "#7b3294"),
}
BEST = {
    "C3K ER MAP": (MAPDIR / f"{TID}_map.npz", "#b2182b"),
    "MILES max-L": (POSTERIORS["MILES"][0], "#4393c3"),
}
KEEP_WEIGHT = 0.9999  # posterior weight kept from a checkpoint read without a pred npz
OUT.mkdir(parents=True, exist_ok=True)


def kc13_curve(lam, delta):
    """Return the FSPS dust_type 4 curve, tau(lam) / tau_V, for rest ``lam`` in A.

    As attn_curve.f90:133-154 of the python-fsps fork in uv.lock: the Calzetti curve over 4.05,
    plus a 2175 A Drude bump of strength 0.85 - 1.9 delta, times (lam / 5500)^delta.
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
    """Return A_lambda in mag for diffuse optical depth ``dust2`` and slope ``delta``."""
    return (
        2.5
        * np.log10(np.e)
        * np.atleast_1d(np.asarray(dust2, float))[:, None]
        * kc13_curve(lam, delta)
    )


def best(path):
    """Return the saved best point of a fit as a dict of label to value."""
    d = np.load(path)
    return dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))


# check against FSPS itself (reddening.py): C3K ER MAP A_V 0.447 and E(B-V) 0.064
p = best(BEST["C3K ER MAP"][0])
a_b, a_v = attenuation([4400.0, 5500.0], p["dust2"], p["dust_index"])[0]
assert abs(a_v - 0.447) < 0.002 and abs(a_b - a_v - 0.064) < 0.002, (a_v, a_b - a_v)

# %% test 3: attenuation curves of the two best points
use_apj_style()
lam = np.linspace(3000.0, 7500.0, 451)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for name, (path, color) in BEST.items():
    p = best(path)
    a = attenuation(lam, p["dust2"], p["dust_index"])[0]
    av = attenuation([5500.0], p["dust2"], p["dust_index"])[0, 0]
    label = f"{name}: dust2 {p['dust2']:.3f}, dust_index {p['dust_index']:+.2f}"
    ax[0].plot(lam, a, color=color, lw=1.5, label=label)
    ax[1].plot(lam, a / av, color=color, lw=1.5)
ax[1].plot(lam, kc13_curve(lam, 0.0)[0] / kc13_curve([5500.0], 0.0)[0, 0], color="0.4", ls="--",
           lw=1, label="Calzetti (dust_index 0)")  # fmt: skip
ax[0].set_ylabel(r"$A_\lambda$ [mag]", fontsize=13)
ax[1].set_ylabel(r"$A_\lambda / A_V$", fontsize=13)
for a in ax:
    a.set_xlabel(r"rest wavelength [$\AA$]", fontsize=13)
ax[0].legend(frameon=False, fontsize=10)
ax[1].legend(frameon=False, fontsize=10)
fig.savefig(OUT / "curves.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'curves.png'}")


# %% test 4: dust2 against stellar mass in the nautilus posteriors
def wquantiles(x, w, q=(0.16, 0.5, 0.84)):
    """Return weighted quantiles of ``x``."""
    i = np.argsort(x)
    c = np.cumsum(w[i]) / w.sum()
    return np.interp(q, c, x[i])


def wcorr(x, y, w):
    """Return the weighted Pearson correlation of ``x`` and ``y``."""
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(
        np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)
    )


lines = ["weighted posterior median and 16-84% range; weighted Pearson r", ""]
post = {}


def read_posterior(path):
    """Return the labels, weighted points, log weights and N_eff of a 10-08 nautilus run.

    A pred npz holds them. For a checkpoint without one (afefree), the points that carry weight
    go through the prior transform of the 10-08 make_model, as in
    2026-10-29_continuum_39633433328091196_map/hypers.py; the likelihood is never called.
    """
    if path.suffix == ".npz":
        d = np.load(path)
        return [str(x) for x in d["labels"]], d["points"], d["log_w"], float(d["n_eff"])
    import importlib.util

    from nautilus import Sampler

    spec = importlib.util.spec_from_file_location(
        "fit_1008", Path(__file__).resolve().parents[1] / "2026-10-08_continuum_outlier" / "fit.py"
    )
    f08 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f08)
    z = float(np.load(MAPDIR / "hr_maxl.npz")["z"])
    model = f08.make_model(z, False, "free")
    # vectorized identity prior: posterior() returns unit-cube points without a Python loop
    s = Sampler(lambda u: u, lambda x: 0.0, n_dim=model.ndim, n_live=1000, filepath=str(path),
                resume=True, vectorized=True)  # fmt: skip
    assert s.explored, path
    unit, log_w, _ = s.posterior()
    # only the heaviest points holding KEEP_WEIGHT of the weight go through the slow transform
    order = np.argsort(log_w)[::-1]
    w = np.exp(log_w[order] - log_w.max())
    n = int(np.searchsorted(np.cumsum(w) / w.sum(), KEEP_WEIGHT)) + 1
    keep = order[:n]
    print(f"{path.name}: {len(unit)} points, {n} hold {KEEP_WEIGHT} of the weight", flush=True)
    pts = np.array([model.prior_transform(u) for u in unit[keep]])
    return list(model.theta_labels()), pts, log_w[keep], float(s.n_eff)


for name, (path, _) in POSTERIORS.items():
    labels, pts, log_w, n_eff = read_posterior(path)
    w = np.exp(log_w - log_w.max())
    col = {k: pts[:, labels.index(k)] for k in ("dust2", "dust_index", "logmass", "logzsol")}
    a_b, a_v = attenuation([4400.0, 5500.0], col["dust2"], col["dust_index"]).T
    col["A_V"], col["E(B-V)"] = a_v, a_b - a_v
    post[name] = (col, w)
    lines.append(f"{name}: {len(w)} points, N_eff {n_eff:.0f}")
    for k in ("dust2", "dust_index", "A_V", "E(B-V)", "logmass", "logzsol"):
        lo, med, hi = wquantiles(col[k], w)
        lines.append(f"  {k:>10} {med:8.3f} [{lo:8.3f}, {hi:8.3f}]")
    lines.append(f"  edge: weight with dust_index > 0.38 {w[col['dust_index'] > 0.38].sum() / w.sum():.2f}, "
                 f"< -2.48 {w[col['dust_index'] < -2.48].sum() / w.sum():.2f}")  # fmt: skip
    for x, y in (("dust2", "logmass"), ("A_V", "logmass"), ("dust2", "dust_index"),
                 ("A_V", "dust_index")):  # fmt: skip
        lines.append(f"  r({x}, {y}) {wcorr(col[x], col[y], w):+.2f}")
    lines.append("")
text = "\n".join(lines)
(OUT / "degeneracy.txt").write_text(text + "\n")
print(text)

pairs = (("A_V", "logmass"), ("dust_index", "A_V"), ("dust_index", "E(B-V)"))
fig, ax = plt.subplots(1, len(pairs), figsize=(15, 4.6))
for name, (_, color) in POSTERIORS.items():
    col, w = post[name]
    keep = w > 1e-3 * w.max()  # points that carry weight, for a readable scatter
    for a, (x, y) in zip(ax, pairs, strict=True):
        a.scatter(col[x][keep], col[y][keep], s=4, alpha=0.25, color=color, lw=0,
                  label=name if a is ax[0] else None)  # fmt: skip
for a, (x, y) in zip(ax, pairs, strict=True):
    a.set_xlabel(x, fontsize=13)
    a.set_ylabel(y, fontsize=13)
ax[0].legend(frameon=False, fontsize=10, markerscale=3)
fig.savefig(OUT / "degeneracy.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'degeneracy.png'}")


# %% test 4: dust2 against logmass, weighted 68% and 95% credible contours
def credible_levels(h, fracs=(0.95, 0.68)):
    """Return the histogram values that enclose ``fracs`` of the total weight, in rising order."""
    v = np.sort(h.ravel())[::-1]
    c = np.cumsum(v) / v.sum()
    return [v[np.searchsorted(c, f)] for f in fracs]


N_BINS = 50
fig, ax = plt.subplots(figsize=(6.5, 5))
for name, (_, color) in POSTERIORS.items():
    col, w = post[name]
    h, xe, ye = np.histogram2d(col["dust2"], col["logmass"], bins=N_BINS, weights=w)
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    ax.contour(xc, yc, h.T, levels=credible_levels(h), colors=color, linewidths=(1.0, 1.8))
    ax.plot([], [], color=color, label=name)
ax.set_xlabel("dust2", fontsize=13)
ax.set_ylabel("logmass", fontsize=13)
ax.legend(frameon=False, fontsize=10, loc="lower right")
fig.savefig(OUT / "dust2_logmass.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'dust2_logmass.png'}")


# %% test 1: MILES and C3K ER models at the MILES best point (10-08 nautilus max-L, until the
# MILES MAP exists). MILES at its own point is its saved best spectrum; C3K ER there is
# spectra_er.npz (afe 0, MILES SFH hyperparameters). Each chi is on that library's own data.
def flam(wave, maggies):
    """Return maggies on ``wave`` in 1e-17 erg/s/cm^2/A."""
    import astropy.units as u

    from hubersed.conversion import to_flambda

    return to_flambda(np.asarray(wave, float) * u.AA, np.asarray(maggies, float) * u.mgy).value


def comparison_figure(path, models, ratio_label):
    """Draw data and two models, the chi of each on its own data, and their ratio."""
    from scipy.signal import medfilt

    ref = models[0][1]
    rest = ref["wave"] / (1 + float(ref["z"]))
    g0 = ref["good"]
    data = flam(ref["wave"], ref["flux"])
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1, 1, 1], "hspace": 0.06})  # fmt: skip
    ax[0].plot(rest, np.where(g0, data, np.nan), color="0.75", lw=0.5, label="DESI SV3 (DR1 coadd)")
    med = np.full(data.size, np.nan)
    med[g0] = medfilt(data[g0], 9)
    ax[0].plot(rest, med, color="k", lw=0.8, label="median filter, 9 pixels")
    for (label, d, color), a in zip(models, ax[1:3], strict=True):
        g = d["good"]
        ax[0].plot(rest, np.where(g, flam(d["wave"], d["spec"]), np.nan), color=color, lw=0.8,
                   label=label)  # fmt: skip
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(g, (d["flux"] - d["spec"]) / d["unc"], np.nan)
        a.plot(rest, c, color=color, lw=0.5)
        for h in (-2, 2):
            a.axhline(h, color="0.3", ls="--", lw=1)
        a.axhline(0, color="0.3", lw=0.6)
        a.set_ylim(-5, 5)
        a.set_ylabel(r"$\chi$", fontsize=14)
    (_, d1, _), (_, d2, _) = models
    both = d1["good"] & d2["good"]
    ax[3].plot(rest, np.where(both, d1["spec"] / d2["spec"], np.nan), color="k", lw=0.6)
    ax[3].axhline(1, color="0.3", lw=0.6)
    ax[3].set_ylabel(ratio_label, fontsize=12)
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
    ax[0].set_ylim(bottom=0)
    ax[0].legend(frameon=False, fontsize=12, loc="upper right")
    ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


m = np.load(POSTERIORS["MILES"][0])
er_map = np.load(MAPDIR / f"{TID}_map.npz")
s_er = np.load(OUT / "spectra_er.npz")
assert np.allclose(m["wave"], er_map["wave"]) and np.isclose(float(m["z"]), float(er_map["z"]))
miles_at_miles = dict(wave=m["wave"], z=m["z"], flux=m["flux"], unc=m["unc"], good=m["good"],
                      spec=m["sp_best"])  # fmt: skip
er_at_miles = dict(wave=er_map["wave"], z=s_er["z"], flux=s_er["flux"], unc=s_er["unc"],
                   good=s_er["good"], spec=s_er["spec_at_miles"])  # fmt: skip
for label, d in (("MILES", miles_at_miles), ("C3K ER", er_at_miles)):
    g = d["good"]
    chi2 = np.sum(((d["flux"][g] - d["spec"][g]) / d["unc"][g]) ** 2)
    print(f"test 1, {label} at the MILES point: chi2 {chi2:.1f} on {g.sum()} px")
both = miles_at_miles["good"] & er_at_miles["good"]
for label, d in (("MILES", miles_at_miles), ("C3K ER", er_at_miles)):
    chi2 = np.sum(((d["flux"][both] - d["spec"][both]) / d["unc"][both]) ** 2)
    print(f"test 1, {label} at the MILES point, common {both.sum()} px: chi2 {chi2:.1f}")
use_apj_style()
comparison_figure(
    OUT / "spectra_at_miles_point.png",
    [
        ("C3K ER at the MILES point", er_at_miles, "#b2182b"),
        ("MILES at its own point (max-L)", miles_at_miles, "#4393c3"),
    ],  # fmt: skip
    "C3K ER / MILES",
)


# %% test 2: MILES and C3K ER models at the C3K ER MAP point (icx job 3657895). C3K ER at its own
# point is its saved best spectrum; MILES there is spectra_miles.npz (afe dropped, the ER fixed SFH
# hyperparameters). Each chi is on that library's own data.
s_mi = np.load(OUT / "spectra_miles.npz")
er_at_er = dict(wave=er_map["wave"], z=er_map["z"], flux=er_map["flux"], unc=er_map["unc"],
                good=er_map["good"], spec=er_map["sp_best"])  # fmt: skip
miles_at_er = dict(wave=m["wave"], z=s_mi["z"], flux=s_mi["flux"], unc=s_mi["unc"],
                   good=s_mi["good"], spec=s_mi["spec_at_er"])  # fmt: skip
both = er_at_er["good"] & miles_at_er["good"]
for label, d in (("C3K ER", er_at_er), ("MILES", miles_at_er)):
    g = d["good"]
    chi2 = np.sum(((d["flux"][g] - d["spec"][g]) / d["unc"][g]) ** 2)
    chi2c = np.sum(((d["flux"][both] - d["spec"][both]) / d["unc"][both]) ** 2)
    print(f"test 2, {label} at the C3K ER point: chi2 {chi2:.1f} on {g.sum()} px, "
          f"{chi2c:.1f} on the common {both.sum()} px")  # fmt: skip
comparison_figure(
    OUT / "spectra_at_er_point.png",
    [
        ("C3K ER at its own point (MAP)", er_at_er, "#b2182b"),
        ("MILES at the C3K ER point", miles_at_er, "#4393c3"),
    ],  # fmt: skip
    "C3K ER / MILES",
)
