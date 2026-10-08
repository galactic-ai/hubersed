"""QUESTION: with the narrow/broad split at 45 km/s and the [OI]/[SII] sky pixels unmasked, does the
broad Balmer width leave the rail it hit at the 100 km/s split (37084, 2026-09-26), does gas_lognH leave
its floor once both [SII] lines are fitted, and what do the line residuals look like in both objects?
HYPOTHESIS: sigma_b settles below 100 km/s (EmFit outflow sigma ~59 / 73 km/s); gas_lognH moves off
1.0 in 37084 now that [SII]6731 is in the fit; eline_sigma_forb and gas_logco stay railed (one forbidden
width cannot serve [OIII] and [NII]/[SII]; design note 6.2, 6.7).
INPUTS: results/2026-09-30_nautilus_c3k_broadline_split45/<tid>_broad_shared_split45_unmask_rest9000_seed0.h5
(runs A = 37084, C = 26597, both forbidden_broad="shared"); for comparison
results/2026-09-26_nautilus_c3k_broadline/run2/39627770174637084_broad_miles_seed0.h5 and
results/2026-09-28_nautilus_c3k_broadline_shared/39628357133926597_broad_full_rest9000_seed0_plotdata.npz.
SEED: none needed (no random draws).
COMMAND: uv run python experiments/2026-09-30_nautilus_c3k_broadline_split45/analysis.py
RESULT: see analysis_summary.txt next to the figures.
FIGURES: params_table.txt, corner_lines_<tid>.png, line_windows.png, spectrum_<tid>.png
"""

# %%
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import astropy.units as u
import corner
import matplotlib.pyplot as plt
import numpy as np
from nautilus import Sampler

from hubersed.conversion import to_flambda
from hubersed.fitting.chi2 import WAVE_OBS
from hubersed.fitting.map_fits import chi2_parts, get_sps
from hubersed.paths import PATHS
from hubersed.plotting.spectra import plot_residual, residual_chi, spectrum_figure

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit import load_data, loglike, make_model, make_obs  # noqa: E402

OUT = PATHS["RESULTS"] / "2026-09-30_nautilus_c3k_broadline_split45"
RUNS = {"A": 39627770174637084, "C": 39628357133926597}
KEY = [
    "eline_sigma",
    "eline_sigma_forb",
    "eline_fbroad",
    "eline_sigma_broad",
    "eline_vbroad",
    "gas_logz",
    "gas_logu",
    "gas_lognH",
    "gas_logno",
    "gas_logco",
    "dust2",
    "dust_ratio",
    "dust_index",
    "logzsol",
    "logmass",
    "sigma_smooth",
]
LINE_PARS = [
    "eline_sigma",
    "eline_sigma_forb",
    "eline_fbroad",
    "eline_sigma_broad",
    "eline_vbroad",
    "gas_lognH",
    "gas_logno",
    "gas_logco",
    "gas_logu",
]
# rest-frame (vacuum) windows for the line residual figure
WINDOWS = {
    "Hbeta": (4845, 4880),
    "[OIII]5007": (4995, 5022),
    "[OI]6300": (6292, 6312),
    "Halpha+[NII]": (6540, 6595),
    "[SII]": (6708, 6742),
}


def load_module(path, name):
    """Import a fit.py from another experiment under its own module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wquantile(x, w, q):
    """Weighted quantiles of samples x (weights w, summing to 1)."""
    o = np.argsort(x)
    return np.interp(q, np.cumsum(w[o]), x[o])


def read_checkpoint(path, model):
    """Posterior points (physical), weights, max-L point and summary from a nautilus checkpoint."""
    s = Sampler(
        lambda x: x, lambda x: 0.0, n_dim=model.ndim, n_live=1000, filepath=str(path), resume=True
    )
    cube, log_w, log_l = s.posterior()
    # keep the highest-weight points holding 99.99% of the posterior mass
    o = np.argsort(log_w)[::-1]
    keep = o[
        : np.searchsorted(
            np.cumsum(np.exp(log_w[o] - log_w.max())) / np.exp(log_w - log_w.max()).sum(), 0.9999
        )
        + 1
    ]
    w = np.exp(log_w[keep] - log_w[keep].max())
    return dict(
        points=np.array([model.prior_transform(x) for x in cube[keep]]),
        w=w / w.sum(),
        labels=list(model.theta_labels()),
        best=model.prior_transform(cube[np.argmax(log_l)]),
        max_lnl=float(log_l.max()),
        n_eff=float(s.n_eff),
        log_z=float(s.log_z),
        n_like=int(s.n_like),
        prior={
            lab: np.ravel(model.config_dict[lab]["prior"].range)
            for lab in model.theta_labels()
            if lab in model.config_dict
        },
    )


def column(p, lab):
    return p["points"][:, p["labels"].index(lab)]


# %% posteriors of the new runs (A, C) and the earlier ones they are compared with
post, models, data = {}, {}, {}
for tag, tid in RUNS.items():
    cfg = json.loads((OUT / f"{tid}_broad_shared_split45_unmask_rest9000_seed0.json").read_text())
    z, flux, unc, good, n_un = load_data(tid, cfg["rest_max"], unmask=True)
    m = make_model(z, cfg["forbidden_broad"], cfg["sigma_split"])
    assert list(m.theta_labels()) == cfg["labels"]
    models[tag], data[tag] = m, (z, flux, unc, good)
    post[tag] = read_checkpoint(OUT / f"{tid}_broad_shared_split45_unmask_rest9000_seed0.h5", m)
    p = post[tag]
    print(
        f"{tag} ({tid}): N_eff {p['n_eff']:.0f}, calls {p['n_like']}, log Z {p['log_z']:.2f}, "
        f"max lnL {p['max_lnl']:.2f}, {len(p['w'])} points kept, {n_un} sky pixels unmasked"
    )

f26 = load_module(PATHS["ROOT"] / "experiments/2026-09-26_nautilus_c3k_broadline/fit.py", "fit0926")
z26, *_ = f26.load_data(RUNS["A"], "miles")
m26 = f26.make_model(z26)
post["A_0926"] = read_checkpoint(
    PATHS["RESULTS"]
    / "2026-09-26_nautilus_c3k_broadline/run2/39627770174637084_broad_miles_seed0.h5",
    m26,
)
d28 = np.load(
    PATHS["RESULTS"] / "2026-09-28_nautilus_c3k_broadline_shared/"
    "39628357133926597_broad_full_rest9000_seed0_plotdata.npz"
)
post["C_0928"] = dict(points=d28["points"], w=d28["w"] / d28["w"].sum(), labels=list(d28["labels"]))
f28 = load_module(
    PATHS["ROOT"] / "experiments/2026-09-28_nautilus_c3k_broadline_shared/fit.py", "fit0928"
)
post["C_0928"]["prior"] = {
    lab: np.ravel(c["prior"].range)
    for lab, c in f28.make_params(float(d28["z"])).items()
    if isinstance(c, dict) and c.get("isfree") and c.get("prior") is not None
}

# %% parameter table: 16/50/84, with a flag where the median sits within 2% of a prior edge
NAMES = {
    "A_0926": "37084 09-26 (split 100, MILES, masked)",
    "A": "37084 09-30 A (split 45, unmasked)",
    "C_0928": "26597 09-28 (split 50, masked)",
    "C": "26597 09-30 C (split 45, unmasked)",
}
lines = []
for pair in [("A_0926", "A"), ("C_0928", "C")]:
    lines.append(f"\n{'parameter':18s}" + "".join(f"{NAMES[r]:>44s}" for r in pair))
    for lab in KEY:
        row = f"{lab:18s}"
        for r in pair:
            p = post[r]
            if lab not in p["labels"]:
                row += f"{'-':>44s}"
                continue
            a, mid, b = wquantile(column(p, lab), p["w"], [0.16, 0.5, 0.84])
            lo, hi = p["prior"].get(lab, (np.nan, np.nan))[:2]
            rail = ""
            if np.isfinite(lo) and (mid - lo) < 0.02 * (hi - lo):
                rail = " FLOOR"
            elif np.isfinite(hi) and (hi - mid) < 0.02 * (hi - lo):
                rail = " CEIL"
            row += f"{mid:12.3f} [{a:9.3f},{b:9.3f}]{rail:6s}".rjust(44)
        lines.append(row)
table = "\n".join(lines)
print(table)
(OUT / "params_table.txt").write_text(table + "\n")

# %% corner plots of the line and gas parameters, new run vs the earlier run of the same object
for new, old, tid in [("A", "A_0926", RUNS["A"]), ("C", "C_0928", RUNS["C"])]:
    labs = [lab for lab in LINE_PARS if lab in post[new]["labels"] and lab in post[old]["labels"]]
    rng = []
    for lab in labs:
        a = min(wquantile(column(post[r], lab), post[r]["w"], 0.001) for r in (new, old))
        b = max(wquantile(column(post[r], lab), post[r]["w"], 0.999) for r in (new, old))
        d = (b - a) * 0.05 or 1e-3
        rng.append((a - d, b + d))
    fig = None
    for r, color in [(old, "0.5"), (new, "C3")]:
        fig = corner.corner(
            np.column_stack([column(post[r], lab) for lab in labs]),
            weights=post[r]["w"],
            fig=fig,
            color=color,
            labels=labs,
            range=rng,
            plot_datapoints=False,
            plot_density=False,
            smooth=1.0,
            bins=30,
            levels=(0.393, 0.865),
            label_kwargs={"fontsize": 9},
            hist_kwargs={"density": True},
        )
    axes = np.array(fig.axes).reshape(len(labs), len(labs))
    for j, lab in enumerate(labs):
        for r in (new, old):
            for edge in np.atleast_1d(post[r]["prior"].get(lab, [])):
                if axes[j, j].get_xlim()[0] <= edge <= axes[j, j].get_xlim()[1]:
                    axes[j, j].axvline(edge, color="0.3", ls="--", lw=0.8)
    fig.legend(
        handles=[
            plt.Line2D([], [], color="0.5", label=NAMES[old]),
            plt.Line2D([], [], color="C3", label=NAMES[new]),
        ],
        loc="upper right",
        frameon=False,
    )
    fig.savefig(OUT / f"corner_lines_{tid}.png", dpi=110)
    plt.close(fig)

# %% max-L spectra: total model and the same point with the broad component switched off
cue = get_sps(zero_library_resolution=False)["cue"]
spec = {}
for tag, tid in RUNS.items():
    z, flux, unc, good = data[tag]
    m, p = models[tag], post[tag]
    lnl = loglike(p["best"], tid, "shared", 45.0, True)
    obs = make_obs(flux, unc, good)
    sp, stats = chi2_parts(m, p["best"], obs, cue, np.zeros_like(good))
    nb = p["best"].copy()
    nb[p["labels"].index("eline_fbroad")] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp0, _ = chi2_parts(m, nb, obs, cue, np.zeros_like(good))
    chi = residual_chi(flux, sp, unc, good)
    spec[tag] = dict(sp=sp, sp_narrow=sp0, chi=chi, chi2_red=stats["chi2_red"])
    print(
        f"{tag}: max-L lnL recomputed {lnl:.2f} vs checkpoint {p['max_lnl']:.2f}; chi2_nu {stats['chi2_red']:.3f}"
    )


def flam(maggies, good):
    f = to_flambda(WAVE_OBS * u.AA, np.asarray(maggies, float) * u.mgy).value
    return np.where(good, f, np.nan)


# %% line windows: data, max-L model, narrow-only model, chi residuals; one row pair per object
summary = []
fig, axes = plt.subplots(
    4, len(WINDOWS), figsize=(16, 8.5), gridspec_kw=dict(height_ratios=[3, 1.2, 3, 1.2])
)
for k, (tag, tid) in enumerate(RUNS.items()):
    z, flux, unc, good = data[tag]
    rest = WAVE_OBS / (1 + z)
    s = spec[tag]
    for j, (name, (lo, hi)) in enumerate(WINDOWS.items()):
        sel = (rest > lo) & (rest < hi)
        ax, axr = axes[2 * k, j], axes[2 * k + 1, j]
        d, e = flam(flux, good)[sel], flam(unc, good)[sel]
        ax.fill_between(rest[sel], d - e, d + e, color="0.8", step="mid", lw=0)
        ax.step(rest[sel], d, "k", where="mid", lw=0.7, label="DESI, degraded")
        ax.plot(rest[sel], flam(s["sp"], good)[sel], "C3", lw=1, label="max-L")
        ax.plot(rest[sel], flam(s["sp_narrow"], good)[sel], "C0--", lw=0.8, label="max-L, f_b = 0")
        axr.step(rest[sel], s["chi"][sel], "C3", where="mid", lw=0.8)
        axr.axhline(0, color="0.3", lw=0.5)
        for h in (-3, 3):
            axr.axhline(h, color="0.6", lw=0.5, ls=":")
        ax.set_title(f"{tag} {tid % 100000}: {name}", fontsize=9)
        ax.set_xlim(lo, hi)
        axr.set_xlim(lo, hi)
        c = s["chi"][sel]
        c = c[np.isfinite(c)]
        summary.append(
            f"{tag} {name:13s} n={len(c):3d} rms chi {np.sqrt(np.mean(c**2)):6.2f}  "
            f"mean chi {np.mean(c):+6.2f}  min {c.min():+7.2f}  max {c.max():+7.2f}"
        )
        if k == 1:
            axr.set_xlabel(r"rest wavelength [$\AA$]")
axes[0, 0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "line_windows.png", dpi=140)
plt.close(fig)
print("\n".join(summary))

# %% full spectra with chi residuals
for tag, tid in RUNS.items():
    z, flux, unc, good = data[tag]
    s = spec[tag]
    fig, ax = spectrum_figure(
        WAVE_OBS,
        z=z,
        figsize=(11, 6),
        data=flam(flux, good),
        unc=flam(unc, good),
        band_kw={},
        data_kw={"label": "DESI spectrum, degraded"},
        models=[
            {
                "flux": flam(s["sp"], good),
                "color": "C3",
                "lw": 0.8,
                "label": rf"max-L, $\chi^2_\nu$ = {s['chi2_red']:.2f}",
            }
        ],
    )
    plot_residual(ax[1], WAVE_OBS, z=z, chi=s["chi"], lw=0.5, color="C3", alpha=0.8)
    ax[1].set_ylim(-8, 8)
    fig.savefig(OUT / f"spectrum_{tid}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

(OUT / "analysis_summary.txt").write_text(
    "\n".join(
        [
            f"{t}: N_eff {post[t]['n_eff']:.0f}, log Z {post[t]['log_z']:.2f}, max lnL {post[t]['max_lnl']:.2f}, "
            f"chi2_nu {spec[t]['chi2_red']:.3f}"
            for t in RUNS
        ]
        + [table, ""]
        + summary
    )
    + "\n"
)
