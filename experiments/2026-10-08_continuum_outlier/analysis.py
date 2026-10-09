"""QUESTION: Do the vanilla models fit the continuum outlier 39627757533007793, and what do the
library (MILES or C3K_HR), the window, the resolution handling and nebular emission change?
HYPOTHESIS: The vanilla MILES fits leave structured residuals at age and abundance features, and
C3K_HR with degraded data reduces them. The predictions registered before this analysis are in
knowledge/_evidence/2026-10-08_toggle_physics/README.md, section 4.
INPUTS: DESI DR1 spectrum from hubersed.io.desi.load_spectrum. The <run>_pred.npz files that
predict.py writes from the checkpoints results/2026-10-08_continuum_outlier/<run>_seed0.h5.
Index definitions from $SPS_HOME/data/allindices.dat.
SEED: 0 for nautilus, 0 for the 100 posterior draws in predict.py, 0 here.
COMMAND: on an ls6 compute node, run predict.py once per converged run (MILES runs with the MILES
build first on PYTHONPATH, as in run.sh), then
uv run python experiments/2026-10-08_continuum_outlier/analysis.py --tid 39627757533007793
RESULT: All eight runs converged (N_eff 2001-2028, 2026-10-08). Nebular off wins every pair:
Delta log Z (on - off) is -6.74 (1a-1b), -6.56 (1c-1d), -11.32 (2a) and -9.78 (2b). Every run piles
95-98% of the mass into the oldest bin, with mass-weighted age 7.51-7.68 Gyr against a cap of
7.81. MILES puts logzsol at +0.27, above the +0.20 edge of the MILES spectra, and dust2 at its
floor. C3K gives logzsol +0.12 to +0.16, dust2 0.12-0.16 and 1-1.4% of the mass younger than
1 Gyr. sigma_smooth is 264-267 km/s for degraded MILES, 250-254 for undegraded MILES and 287-291
for C3K. With the native noise (1c, 1d) the max-L chi2_nu is 1.58. In the C3K runs nebular
emission pushes sigma_dyn to about 0.89 (prior LogU 0.01-1). In all eight runs the data have
stronger Mgb (+2.5 to +3.0 sigma), weaker Fe5335 (-2.2 to -3.0), stronger HdeltaA (+2.1 to +3.0)
and a weaker Dn4000 (-0.9 to -2.0) than the model. No switch removes that pattern. Fe5270 and
NaD have masked pixels. The gas parameters stay at their priors, except eline_sigma (median 218-221).
FIGURES: results/2026-10-08_continuum_outlier/<tid>/. Per run spectrum_<run>.png,
spectrum_zoom_<run>.png, corner_phys_<run>.png, corner_sfh_<run>.png and sfh_<run>.png. Across runs
residuals.png, zoom.png, chi2_observed.png, corner_<pair>.png and cmf.png.
"""

# %%
import argparse
import os
from pathlib import Path

import astropy.units as u
import corner
import matplotlib.pyplot as plt
import numpy as np
import spender
from fit import make_model
from matplotlib.lines import Line2D

from hubersed.conversion import to_flambda
from hubersed.fitting.map_fits import sfh_from_theta
from hubersed.paths import PATHS
from hubersed.plotting.sfh import sfh_figure
from hubersed.plotting.spectra import plot_residual, residual_chi, spectrum_figure

# parse_known_args, so the cells also run in an interactive session
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tid", type=int, default=39627757533007793)
TID = parser.parse_known_args()[0].tid
OUT = PATHS["RESULTS"] / "2026-10-08_continuum_outlier"
FIG = OUT / str(TID)
FIG.mkdir(parents=True, exist_ok=True)
RUNS = {
    "1a": "miles_miles_nebon",
    "1b": "miles_miles_neboff",
    "1c": "miles_miles_nebon_nodeg",
    "1d": "miles_miles_neboff_nodeg",
    "2a-on": "c3k_hr_miles_nebon",
    "2a-off": "c3k_hr_miles_neboff",
    "2b-on": "c3k_hr_full_nebon",
    "2b-off": "c3k_hr_full_neboff",
    "3-zero": "c3k_hr_full_neboff_afezero",
    "3-free": "c3k_hr_full_neboff_afefree",
}
# Delta log Z is valid only within these pairs, because each pair fits identical data.
PAIRS = [("1a", "1b"), ("1c", "1d"), ("2a-on", "2a-off"), ("2b-on", "2b-off"), ("3-free", "3-zero")]
# Posterior overlays. Only the first is a same-data comparison.
OVERLAYS = {
    "nebular": ["1a", "1b"],
    "degrade": ["1b", "1d"],
    "library": ["1b", "2a-off"],
    "window": ["2a-off", "2b-off"],
    "alpha": ["3-zero", "3-free"],
}
COLOR = dict(zip(RUNS, [f"C{i}" for i in range(len(RUNS))], strict=True))
GEN = np.random.default_rng(0)

# %% load what predict.py saved
run = {}
for key, stem in RUNS.items():
    f = OUT / f"{TID}_{stem}_pred.npz"
    if not f.exists():
        print(f"{key}: no {f.name}")
        continue
    d = dict(np.load(f))
    d["labels"] = [str(x) for x in d["labels"]]
    d["w"] = np.exp(d["log_w"] - d["log_w"].max())
    d["w"] /= d["w"].sum()
    d["nebular"] = "nebon" in stem
    d["afe"] = stem.split("_afe")[1] if "_afe" in stem else "none"
    run[key] = d
z = float(next(iter(run.values()))["z"])
model = {(d["nebular"], d["afe"]): make_model(z, d["nebular"], d["afe"]) for d in run.values()}


def wquantile(x, w, q):
    """Weighted quantiles of samples x (weights w, summing to 1)."""
    o = np.argsort(x)
    return np.interp(q, np.cumsum(w[o]), x[o])


def chi(d, sp):
    """Residual (data - model) / uncertainty on the fitted pixels, NaN elsewhere."""
    return np.where(d["good"], (d["flux"] - sp) / d["unc"], np.nan)


def running_median(x, n):
    """Running median over n samples, centred, shorter at the ends."""
    h = n // 2
    return np.array([np.median(x[max(0, j - h) : j + h + 1]) for j in range(len(x))])


# %% evidence, fit quality, and Delta log Z where it is valid
for key, d in run.items():
    c = chi(d, d["sp_best"])
    npix = int(d["good"].sum())
    print(
        f"{key:7s} log Z {float(d['log_z']):10.2f}  N_eff {float(d['n_eff']):5.0f}  "
        f"calls {int(d['n_like']):8d}  max lnL {d['log_l'].max():10.2f}  npix {npix}  "
        f"max-L chi2_nu {np.nansum(c**2) / (npix - len(d['labels'])):.3f}"
    )
for a, b in PAIRS:
    if a in run and b in run:
        print(f"Delta log Z {a} - {b} = {float(run[a]['log_z']) - float(run[b]['log_z']):+.2f}")

# %% alpha: the afe posterior, Savage-Dickey at afe = 0, and [Z/H]
# 3-zero and 3-free share data and model and differ only in afe, so they are nested and
# B(zero : free) = p(afe = 0 | data) / p(afe = 0), with the TopHat prior density 1 / 0.8 at 0.
# The posterior density at 0 is a weighted histogram over the central bins. With alpha on,
# logzsol is [Fe/H], and [Z/H] ~ [Fe/H] + 0.75 [alpha/Fe] (Vazdekis et al. 2015).
if "3-free" in run:
    d = run["3-free"]
    afe = d["points"][:, d["labels"].index("afe")]
    feh = d["points"][:, d["labels"].index("logzsol")]
    print("afe 16/50/84:", np.round(wquantile(afe, d["w"], [0.16, 0.5, 0.84]), 3))
    print("logzsol ([Fe/H]) 16/50/84:", np.round(wquantile(feh, d["w"], [0.16, 0.5, 0.84]), 3))
    print("[Z/H] 16/50/84:", np.round(wquantile(feh + 0.75 * afe, d["w"], [0.16, 0.5, 0.84]), 3))
    for h in [0.025, 0.05]:
        dens = d["w"][np.abs(afe) < h].sum() / (2 * h)
        print(
            f"posterior density at afe 0 (+-{h}) {dens:.3f}, ln B(zero:free) {np.log(dens / 1.25):+.2f}"
        )
    print("afe weight within 0.02 of the 0.6 edge:", round(d["w"][afe > 0.58].sum(), 3))

# %% derived quantities from 4000 weighted draws per run
# Mass-weighted age uses the bin midpoints in linear time, the SFR being constant in each bin.
# The oldest bin caps it (4.59-11.02 Gyr at this redshift, so about 7.8 Gyr).
for d in run.values():
    m = model[d["nebular"], d["afe"]]
    idx = GEN.choice(len(d["w"]), size=4000, p=d["w"])
    mwa, old, young = [], [], []
    for theta in d["points"][idx]:
        s = sfh_from_theta(m, theta)
        e = s["edges_gyr"]
        frac = np.diff(np.concatenate([[0.0], s["cmf"]]))
        mwa.append(frac @ (0.5 * (e[:-1] + e[1:])))
        old.append(frac[-1])
        young.append(frac[e[1:] <= 1.0].sum())
    d["derived"] = {
        "MWA_gyr": np.array(mwa),
        "f_oldest": np.array(old),
        "f_lt1gyr": np.array(young),
    }
edges = s["edges_gyr"]
print(
    f"oldest bin {edges[-2]:.2f}-{edges[-1]:.2f} Gyr, MWA cap {0.5 * (edges[-2] + edges[-1]):.2f}"
)

# %% posterior table, with the weight within 2% of a prior edge
phys = list(dict.fromkeys(lab for d in run.values() for lab in d["labels"]))
phys = [lab for lab in phys if not lab.startswith("logsfr_ratios")]
print(f"{'parameter':14s}" + "".join(f"{k:>24s}" for k in run))
for lab in phys:
    row = f"{lab:14s}"
    for d in run.values():
        if lab not in d["labels"]:
            row += f"{'':>24s}"
            continue
        x = d["points"][:, d["labels"].index(lab)]
        lo, mid, hi = wquantile(x, d["w"], [0.16, 0.5, 0.84])
        a, b = model[d["nebular"], d["afe"]].config_dict[lab]["prior"].range
        edge = d["w"][(x < a + 0.02 * (b - a)) | (x > b - 0.02 * (b - a))].sum()
        flag = f"{edge:.0%}" if edge > 0.05 else ""
        row += f"{mid:9.3f} [{lo:7.3f},{hi:7.3f}]{flag:>3s}"[:24].rjust(24)
    print(row)
for name in ["MWA_gyr", "f_oldest", "f_lt1gyr"]:
    row = f"{name:14s}"
    for d in run.values():
        lo, mid, hi = np.percentile(d["derived"][name], [16, 50, 84])
        row += f"{mid:9.3f} [{lo:7.3f},{hi:7.3f}]".rjust(24)
    print(row)

# %% registered predictions (README section 4), as numbers to judge by hand


def med(key, lab):
    """Weighted median of one parameter, or of a derived quantity, in one run."""
    d = run[key]
    if lab in d.get("derived", {}):
        return float(np.median(d["derived"][lab]))
    return float(wquantile(d["points"][:, d["labels"].index(lab)], d["w"], 0.5))


checks = {
    "P-dust: dust2 near 0, dust_index near +0.4 (alpha off)": [
        (k, med(k, "dust2"), med(k, "dust_index")) for k in run
    ],
    "P-age: MWA at the cap, mass in the oldest bin": [
        (k, med(k, "MWA_gyr"), med(k, "f_oldest")) for k in run
    ],
    "P-lib: C3K younger or lower logzsol than MILES (1b vs 2a-off)": [
        (k, med(k, "logzsol"), med(k, "MWA_gyr")) for k in ["1b", "2a-off"] if k in run
    ],
    "P-win: 2b higher dust2 and/or logzsol, same sigma_smooth": [
        (k, med(k, "dust2"), med(k, "logzsol"), med(k, "sigma_smooth"))
        for k in ["2a-off", "2b-off"]
        if k in run
    ],
    "P-res: 1c/1d sigma_smooth 6-9 km/s below 1a/1b": [
        (k, med(k, "sigma_smooth")) for k in ["1a", "1b", "1c", "1d"] if k in run
    ],
}
for name, rows in checks.items():
    print(name)
    for r in rows:
        print("   ", r[0], " ".join(f"{v:8.3f}" for v in r[1:]))

# P-neb: shift of each shared parameter between nebular on and off, in units of the joint sigma
for a, b in PAIRS:
    if a in run and b in run:
        shifts = []
        for lab in run[b]["labels"]:
            xa = run[a]["points"][:, run[a]["labels"].index(lab)]
            xb = run[b]["points"][:, run[b]["labels"].index(lab)]
            qa, qb = (
                wquantile(xa, run[a]["w"], [0.16, 0.5, 0.84]),
                wquantile(xb, run[b]["w"], [0.16, 0.5, 0.84]),
            )
            sig = np.hypot(qa[2] - qa[0], qb[2] - qb[0]) / 2
            shifts.append((abs(qa[1] - qb[1]) / sig, lab))
        print(f"P-neb {a} vs {b}: largest shifts", sorted(shifts)[-3:])

# %% gas parameters in the nebular-on runs, posterior width over prior width
for key, d in run.items():
    if not d["nebular"]:
        continue
    for lab in [x for x in d["labels"] if x.startswith(("gas_", "eline_"))]:
        x = d["points"][:, d["labels"].index(lab)]
        lo, mid, hi = wquantile(x, d["w"], [0.16, 0.5, 0.84])
        a, b = model[True, d["afe"]].config_dict[lab]["prior"].range
        print(
            f"{key:6s} {lab:14s} median {mid:8.3f}  prior [{a}, {b}]  16-84 width / prior width {(hi - lo) / (b - a):.2f}"
        )

# %% spectral indices of the data and of the model, on each run's own data
# Lick indices in Angstrom (type 2) or magnitudes (type 1), and Dn4000 as a flux ratio in f_nu
# (type 3). The data noise comes from 200 Gaussian draws of the fit uncertainty, which the
# degraded runs overstate per pixel (the original ivar is kept), so the data error is too large.
table = np.genfromtxt(
    Path(os.environ["SPS_HOME"]) / "data" / "allindices.dat",
    dtype=None,
    encoding=None,
    comments="#",
)
INDEX = {
    str(r[-1]): (r[0], r[1], r[2], r[3], r[4], r[5], int(r[6]))
    for r in table
    if str(r[-1])
    in {
        "Dn4000",
        "HdeltaA",
        "CN1",
        "Ca4227",
        "G4300",
        "C4668",
        "Hbeta",
        "Mgb",
        "Fe5270",
        "Fe5335",
        "NaD",
        "TiO1",
        "TiO2",
    }
}


def index(rest, fnu, good, f0, f1, b0, b1, r0, r1, kind):
    """One index on a rest-frame spectrum in f_nu units, NaN if a band has an unfitted pixel."""
    band = {
        k: (rest >= lo) & (rest <= hi)
        for k, (lo, hi) in dict(f=(f0, f1), b=(b0, b1), r=(r0, r1)).items()
    }
    if kind == 3:
        if not (good[band["b"]].all() and good[band["r"]].all()):
            return np.nan
        return fnu[band["r"]].mean() / fnu[band["b"]].mean()
    if not all(good[m].all() and m.any() for m in band.values()):
        return np.nan
    flam = fnu / rest**2
    cb, cr = flam[band["b"]].mean(), flam[band["r"]].mean()
    xb, xr = 0.5 * (b0 + b1), 0.5 * (r0 + r1)
    w = rest[band["f"]]
    cont = cb + (cr - cb) * (w - xb) / (xr - xb)
    ratio = flam[band["f"]] / cont
    if kind == 2:
        return np.trapezoid(1 - ratio, w)
    return -2.5 * np.log10(np.trapezoid(ratio, w) / (w[-1] - w[0]))


print(f"{'index':8s}" + "".join(f"{k:>22s}" for k in run))
for name, bands in INDEX.items():
    row = f"{name:8s}"
    for d in run.values():
        rest = d["wave"] / (1 + z)
        data = index(rest, d["flux"], d["good"], *bands)
        if np.isnan(data):
            row += f"{'':>22s}"
            continue
        noisy = d["flux"] + GEN.normal(0, 1, (200, d["flux"].size)) * np.where(
            d["good"], d["unc"], 0
        )
        noise = np.std([index(rest, f, d["good"], *bands) for f in noisy])
        mod = np.array([index(rest, s, d["good"], *bands) for s in d["draw_spec"]])
        sig = np.hypot(noise, np.std(mod))
        row += f"{data:7.3f} {np.median(mod):7.3f} {(data - np.median(mod)) / sig:+5.1f}s".rjust(22)
    print(row)
print("each cell: data, model median, (data - model) / sigma")

# %% unfitted pixels inside the index bands, in observed wavelength, for the widest window
d = run[max(run, key=lambda k: run[k]["good"].sum())]
rest = d["wave"] / (1 + z)
for name, (f0, f1, b0, b1, r0, r1, _) in INDEX.items():
    inb = (
        ((rest >= f0) & (rest <= f1))
        | ((rest >= b0) & (rest <= b1))
        | ((rest >= r0) & (rest <= r1))
    )
    bad = d["wave"][inb & ~d["good"]]
    if bad.size:
        print(
            f"{name:8s} {bad.size:3d} unfitted pixels, observed {bad.min():.1f}-{bad.max():.1f} A"
        )

# %% residual ladder: chi of the max-L model, a 25-pixel running median, offset per run
fig, ax = plt.subplots(figsize=(12, 1.2 + 0.9 * len(run)))
for i, (key, d) in enumerate(run.items()):
    rest = d["wave"] / (1 + z)
    c = chi(d, d["sp_best"])
    k = np.isfinite(c)
    ax.plot(rest[k], running_median(c[k], 25) - 3 * i, color=COLOR[key], lw=0.8)
    ax.axhline(-3 * i, color="0.7", lw=0.5)
    ax.text(rest[k][0] - 40, -3 * i, key, ha="right", va="center", color=COLOR[key])
for w in [3934, 3969, 4102, 4300, 4861, 5175, 5270, 5335, 5893, 6563, 8190]:
    ax.axvline(w, color="0.85", lw=0.6, zorder=0)
ax.set_xlabel(r"rest wavelength [$\AA$]")
ax.set_ylabel(r"running median of $\chi$, offset by 3 per run")
ax.set_yticks([])
fig.savefig(FIG / "residuals.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% zooms: data over max-L model in percent, a 9-pixel running median, per run
# Shaded spans are the feature bands of the indices above.
REGIONS = [(3800, 4200), (4800, 5400), (5850, 5950), (8100, 8300)]
fig, axes = plt.subplots(1, len(REGIONS), figsize=(14, 3.5))
for a, (lo, hi) in zip(axes, REGIONS, strict=True):
    for f0, f1, *_ in INDEX.values():
        if lo < f0 < hi:
            a.axvspan(f0, f1, color="0.9", lw=0)
    for key, d in run.items():
        rest = d["wave"] / (1 + z)
        k = d["good"] & (rest > lo) & (rest < hi)
        if k.any():
            ratio = 100 * (d["flux"][k] / d["sp_best"][k] - 1)
            a.plot(rest[k], running_median(ratio, 9), color=COLOR[key], lw=0.8, label=key)
    a.axhline(0, color="0.5", lw=0.5)
    a.set_xlabel(r"rest wavelength [$\AA$]")
axes[0].set_ylabel("data / max-L model - 1 [%]")
axes[0].legend(frameon=False, fontsize="small", ncol=2)
fig.savefig(FIG / "zoom.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% corner overlays of the shared physical parameters
for tag, keys in OVERLAYS.items():
    keys = [k for k in keys if k in run]
    if len(keys) < 2:
        continue
    labs = [lab for lab in phys if all(lab in run[k]["labels"] for k in keys)]
    rng = []
    for lab in labs:
        lo = min(
            wquantile(run[k]["points"][:, run[k]["labels"].index(lab)], run[k]["w"], 0.001)
            for k in keys
        )
        hi = max(
            wquantile(run[k]["points"][:, run[k]["labels"].index(lab)], run[k]["w"], 0.999)
            for k in keys
        )
        pad = (hi - lo) * 0.05 or 1e-3
        rng.append((lo - pad, hi + pad))
    fig = None
    for k in keys:
        d = run[k]
        fig = corner.corner(
            d["points"][:, [d["labels"].index(lab) for lab in labs]],
            weights=d["w"],
            fig=fig,
            color=COLOR[k],
            labels=labs,
            range=rng,
            plot_datapoints=False,
            plot_density=False,
            smooth=1.0,
            bins=30,
            levels=(0.393, 0.865),  # 1 and 2 sigma for a 2-D Gaussian
            label_kwargs={"fontsize": 9},
            hist_kwargs={"density": True},
        )
    fig.legend(
        handles=[Line2D([], [], color=COLOR[k], label=k) for k in keys],
        loc="upper right",
        frameon=False,
    )
    fig.savefig(FIG / f"corner_{tag}.png", dpi=110)
    plt.close(fig)

# %% cumulative mass fraction, median and 16-84% per run
fig, ax = plt.subplots(figsize=(6, 4))
for key, d in run.items():
    m = model[d["nebular"], d["afe"]]
    idx = GEN.choice(len(d["w"]), size=1000, p=d["w"])
    cmf = np.array([sfh_from_theta(m, d["points"][i])["cmf"] for i in idx])
    lo, mid, hi = np.percentile(cmf, [16, 50, 84], axis=0)
    t = edges[1:]
    ax.step(t, mid, where="post", color=COLOR[key], label=key)
    ax.fill_between(t, lo, hi, step="post", color=COLOR[key], alpha=0.15, lw=0)
ax.set_xscale("log")
ax.set_xlabel("lookback time [Gyr]")
ax.set_ylabel("mass formed after this time")
ax.legend(frameon=False, fontsize="small")
fig.savefig(FIG / "cmf.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% where the misfit sits, in the undegraded runs, whose uncertainty is the DESI noise
# DESI arms from Guy et al. 2023 (arXiv:2209.14482, instrument.tex:22): b 3600-5930, r 5600-7720
# and z 7470-9800 A. Pixels in an overlap count for both arms. Sky lines are spender's list
# (data/sky-lines.txt, in nm). The fit masks none of them, only pixels with no inverse variance.
# Each mean chi2 should be 1 within sqrt(2 / n).
ARMS = {"b": (3600, 5930), "r": (5600, 7720), "z": (7470, 9800)}
sky = np.genfromtxt(
    Path(spender.__file__).parent / "data" / "sky-lines.txt",
    names=["wavelength", "intensity", "name", "status"],
    dtype=None,
    encoding=None,
)
features = {k: v for k, v in INDEX.items() if v[-1] != 3}
for key in [k for k in ["1c", "1d"] if k in run]:
    d = run[key]
    w, g = d["wave"], d["good"]
    rest = w / (1 + z)
    c2 = chi(d, d["sp_best"]) ** 2
    near = {}
    for lo, hi in [(2, np.inf), (0.5, 2)]:
        lines = sky["wavelength"][(sky["intensity"] > lo) & (sky["intensity"] <= hi)] * 10
        near[lo] = np.any(np.abs(w[:, None] - lines[None, :]) < 2.0, axis=1)
    feat = np.zeros_like(g)
    for f0, f1, *_ in features.values():
        feat |= (rest >= f0) & (rest <= f1)
    sets = {f"arm {a}": (w >= lo) & (w <= hi) for a, (lo, hi) in ARMS.items()}
    sets |= {
        "within 2 A of a sky line, intensity > 2": near[2],
        "within 2 A of a sky line, intensity 0.5-2": near[0.5] & ~near[2],
        "away from sky lines": ~near[2] & ~near[0.5],
        "index feature bands": feat,
        "away from sky lines, outside feature bands": ~near[2] & ~near[0.5] & ~feat,
    }
    print(f"\n{key}: all fitted pixels, n {g.sum()}, mean chi2 {np.nanmean(c2[g]):.3f}")
    for label, s in sets.items():
        s = s & g
        print(
            f"  {label:45s} n {s.sum():5d}  mean chi2 {np.nanmean(c2[s]):.3f} "
            f"+- {np.sqrt(2 / max(s.sum(), 1)):.3f}  share of chi2 {np.nansum(c2[s]) / np.nansum(c2[g]):.2f}"
        )

# mean chi2 in 100 A bins of observed wavelength, with the arm edges and the bright sky lines
fig, ax = plt.subplots(figsize=(11, 3.5))
for key in [k for k in ["1c", "1d"] if k in run]:
    d = run[key]
    c2 = chi(d, d["sp_best"]) ** 2
    edges_w = np.arange(3600, 9900, 100)
    k = np.digitize(d["wave"], edges_w)
    m = [
        np.nanmean(c2[(k == i) & d["good"]]) if ((k == i) & d["good"]).sum() > 20 else np.nan
        for i in range(1, len(edges_w))
    ]
    ax.stairs(m, edges_w, color=COLOR[key], label=key)
for lo, hi in ARMS.values():
    ax.axvspan(lo, hi, color="0.5", alpha=0.06, lw=0)
for lw_ in sky["wavelength"][sky["intensity"] > 2] * 10:
    ax.axvline(lw_, ymin=0, ymax=0.05, color="0.4", lw=0.5)
ax.axhline(1, color="0.5", lw=0.6, ls="--")
ax.set_xlabel(r"observed wavelength [$\AA$]")
ax.set_ylabel(r"mean $\chi^2$ per pixel")
ax.legend(frameon=False)
secax = ax.secondary_xaxis("top", functions=(lambda x: x / (1 + z), lambda x: x * (1 + z)))
secax.set_xlabel(r"rest wavelength [$\AA$]")
fig.savefig(FIG / "chi2_observed.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% spectrum per run: the data the run fitted, its max-L model and a 16-84% band, chi below
# Chi uses the fit uncertainty, the original DESI ivar kept after degrading, so it understates
# the residual of the degraded runs. This cell sets the ApJ style, so it comes last.
for key, d in run.items():
    rest = d["wave"] / (1 + z)

    def flam(maggies, d=d):
        """Convert maggies to DESI f_lambda units, NaN outside the fitted pixels."""
        f = to_flambda(d["wave"] * u.AA, np.asarray(maggies, float) * u.mgy).value
        return np.where(d["good"], f, np.nan)

    lo, hi = np.percentile(d["draw_spec"], [16, 84], axis=0)
    fig, ax = spectrum_figure(
        d["wave"],
        z=z,
        figsize=(11, 6),
        data=flam(d["flux"]),
        unc=flam(d["unc"]),
        band_kw={},
        data_kw={"label": f"DESI spectrum, as fitted in {key}"},
        models=[
            {"flux": flam(d["sp_best"]), "color": "#b2182b", "lw": 0.8, "label": f"{key} max-L"}
        ],
    )
    ax[0].fill_between(rest, flam(lo), flam(hi), color="#b2182b", alpha=0.3, lw=0)
    plot_residual(
        ax[1],
        d["wave"],
        z=z,
        chi=residual_chi(d["flux"], d["sp_best"], d["unc"], d["good"]),
        lw=0.5,
    )
    ax[1].set_ylim(-6, 6)
    ax[1].set_xlim(rest[d["good"]].min(), rest[d["good"]].max())
    # the noisy blue end would otherwise set the flux axis
    ax[0].set_ylim(0, 1.3 * np.nanpercentile(flam(d["sp_best"]), 99.5))
    fig.savefig(FIG / f"spectrum_{key}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # zooms on Dn4000 and Hdelta, on Hbeta to Fe5335, and on Halpha, where the run has pixels
    regions = [(3820, 4180), (4800, 5400), (6520, 6610)]
    regions = [r for r in regions if (d["good"] & (rest > r[0]) & (rest < r[1])).sum() > 10]
    fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 3.5), squeeze=False)
    for a, (w_lo, w_hi) in zip(axes[0], regions, strict=True):
        k = (rest > w_lo) & (rest < w_hi)
        a.plot(rest[k], flam(d["flux"])[k], color="0.3", lw=0.7, label="DESI spectrum")
        a.fill_between(rest[k], flam(lo)[k], flam(hi)[k], color="#b2182b", alpha=0.3, lw=0)
        a.plot(rest[k], flam(d["sp_best"])[k], color="#b2182b", lw=0.8, label=f"{key} max-L")
        a.set_xlabel(r"rest wavelength [$\AA$]")
    axes[0, 0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    axes[0, 0].legend(frameon=False, fontsize="small")
    fig.savefig(FIG / f"spectrum_zoom_{key}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# %% per run: corner plots of the physical and the SFH parameters, and the SFH posterior
# As in 2026-09-24_nautilus_tauin, with dashed lines at prior edges that fall inside an axis.


def axis_range(x, w, q=(0.001, 0.999), pad=0.05):
    """Central 99.8% of the posterior plus 5% padding."""
    lo, hi = wquantile(x, w, np.array(q))
    pad = (hi - lo) * pad or 1e-3
    return (lo - pad, hi + pad)


for key, d in run.items():
    m = model[d["nebular"], d["afe"]]
    labels, pts, w = d["labels"], d["points"], d["w"]
    groups = {
        "phys": [i for i, lab in enumerate(labels) if not lab.startswith("logsfr_ratios")],
        "sfh": [i for i, lab in enumerate(labels) if lab.startswith("logsfr_ratios")],
    }
    for tag, idx in groups.items():
        fig = corner.corner(
            pts[:, idx],
            weights=w,
            labels=[labels[i] for i in idx],
            range=[axis_range(pts[:, i], w) for i in idx],
            plot_datapoints=False,
            fill_contours=True,
            smooth=1.0,
            bins=30,
            levels=(0.393, 0.865),  # 1 and 2 sigma for a 2-D Gaussian
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3f",
            label_kwargs={"fontsize": 9},
            title_kwargs={"fontsize": 8},
        )
        if tag == "phys":
            axes = np.array(fig.axes).reshape(len(idx), len(idx))
            for j, i in enumerate(idx):
                ax = axes[j, j]
                for edge in np.atleast_1d(m.config_dict[labels[i]]["prior"].range):
                    if ax.get_xlim()[0] <= edge <= ax.get_xlim()[1]:
                        ax.axvline(edge, color="0.4", ls="--", lw=1)
        fig.suptitle(f"{TID} {key}", fontsize=12)
        fig.savefig(FIG / f"corner_{tag}_{key}.png", dpi=110)
        plt.close(fig)

    # SFH: median and 16-84% per bin from 2000 weighted draws
    sfhs = [sfh_from_theta(m, pts[i]) for i in GEN.choice(len(w), size=2000, p=w)]
    edges = sfhs[0]["edges_gyr"]
    ssfr_lo, ssfr_med, ssfr_hi = np.percentile([s["ssfr"] for s in sfhs], [16, 50, 84], axis=0)
    cmf_lo, cmf_med, cmf_hi = np.percentile([s["cmf"] for s in sfhs], [16, 50, 84], axis=0)
    fig, ax = sfh_figure(
        edges,
        ssfr_med,
        cmf_med,
        ssfr_kwargs={"label": f"{key} median"},
        cmf_kwargs={"color": "C3"},
    )
    ax[0].stairs(ssfr_hi, edges, baseline=ssfr_lo, fill=True, color="C3", alpha=0.25, lw=0)
    ax[1].stairs(cmf_hi, edges, baseline=cmf_lo, fill=True, color="C3", alpha=0.25, lw=0)
    ax[0].legend(frameon=True, fontsize="small", loc="lower left")
    fig.savefig(FIG / f"sfh_{key}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# %%
