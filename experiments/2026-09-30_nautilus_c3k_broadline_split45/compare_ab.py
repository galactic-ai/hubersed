"""QUESTION: for 37084, does freeing the forbidden-line broad fraction (run B, forbidden_broad="free")
change the fit relative to run A (forbidden_broad="shared", f_forb tied to the Balmer f_b)?
HYPOTHESIS (recorded 2026-10-01, before B converged; plan_T1c_low_high_ion_2026-10-01.md section 2):
f_forb is dominated by [OIII] and lands near 1.2 x f_b, about 0.2 (A has f_b = 0.181). If the [OIII]
and Hbeta core residuals in A (-/+/- pattern, design note 6.8 item 5) come from the shared profile,
the [OIII] core residual shrinks in B; Hbeta's does not, since its profile is unchanged.
INPUTS: results/2026-09-30_nautilus_c3k_broadline_split45/39627770174637084_broad_{shared,free}_split45_unmask_rest9000_seed0.{h5,json}
SEED: none needed (no random draws).
COMMAND: uv run python experiments/2026-09-30_nautilus_c3k_broadline_split45/compare_ab.py
[--b-dir DIR --tag TAG] (a copy of an unconverged B checkpoint can be read from DIR; outputs then go
to compare_ab_TAG/)
RESULT: compare_ab/compare_ab.txt next to the figures.
FIGURES: compare_ab/{corner_ab.png, line_windows_ab.png, fforb_ratio.png}
NOTE: nautilus 1.0.6 gives no log Z uncertainty (sampler.py, log_z is a logsumexp over shells). The
importance-sampling part is about 1/sqrt(N_eff); shell-volume and seed-to-seed scatter are unknown.
"""

# %%
import argparse
import importlib.util
import json
import warnings

import corner
import matplotlib

matplotlib.use("Agg")
import astropy.units as u  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from nautilus import Sampler  # noqa: E402

from hubersed.conversion import to_flambda  # noqa: E402
from hubersed.fitting.chi2 import WAVE_OBS  # noqa: E402
from hubersed.fitting.map_fits import chi2_parts, get_sps  # noqa: E402
from hubersed.paths import PATHS  # noqa: E402
from hubersed.plotting.spectra import residual_chi  # noqa: E402

HERE = PATHS["ROOT"] / "experiments/2026-09-30_nautilus_c3k_broadline_split45"
RES = PATHS["RESULTS"] / "2026-09-30_nautilus_c3k_broadline_split45"
TID = 39627770174637084
MODES = {"A": "shared", "B": "free"}
KEY = [
    "eline_sigma",
    "eline_sigma_forb",
    "eline_fbroad",
    "eline_fbroad_forb",
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
CORNER = [
    "eline_sigma",
    "eline_sigma_forb",
    "eline_fbroad",
    "eline_fbroad_forb",
    "eline_sigma_broad",
    "eline_vbroad",
    "gas_logu",
    "gas_lognH",
    "gas_logco",
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
    """Posterior samples of one parameter."""
    return p["points"][:, p["labels"].index(lab)]


def flam(maggies, good):
    """Convert maggies to f_lambda, NaN outside ``good``."""
    f = to_flambda(WAVE_OBS * u.AA, np.asarray(maggies, float) * u.mgy).value
    return np.where(good, f, np.nan)


parser = argparse.ArgumentParser()
parser.add_argument("--b-dir", default=None, help="directory holding a copy of B's checkpoint")
parser.add_argument("--tag", default=None, help="suffix of the output directory")
args = parser.parse_args()
OUT = RES / ("compare_ab" + (f"_{args.tag}" if args.tag else ""))
OUT.mkdir(exist_ok=True)
fit = load_module(HERE / "fit.py", "fit0930")

# %% posteriors
post, models, data = {}, {}, {}
for tag, mode in MODES.items():
    stem = f"{TID}_broad_{mode}_split45_unmask_rest9000_seed0"
    src = RES if (tag == "A" or args.b_dir is None) else PATHS["ROOT"] / args.b_dir
    cfg = json.loads((src / f"{stem}.json").read_text())
    z, flux, unc, good, _ = fit.load_data(TID, cfg["rest_max"], unmask=cfg["unmask_sky"])
    m = fit.make_model(z, cfg["forbidden_broad"], cfg["sigma_split"])
    assert list(m.theta_labels()) == cfg["labels"]
    models[tag], data[tag] = m, (z, flux, unc, good)
    post[tag] = read_checkpoint(src / f"{stem}.h5", m)

lines = []
for tag in MODES:
    p = post[tag]
    lines.append(
        f"{tag} ({MODES[tag]}): N_eff {p['n_eff']:.0f}, calls {p['n_like']}, "
        f"log Z {p['log_z']:.2f}, max lnL {p['max_lnl']:.2f}, ndim {len(p['labels'])}"
    )
dlz = post["B"]["log_z"] - post["A"]["log_z"]
lines.append(
    f"log Z(B) - log Z(A) = {dlz:+.2f}  (importance-sampling error ~ "
    f"{np.hypot(*[1 / np.sqrt(post[t]['n_eff']) for t in MODES]):.2f}; other errors unknown)"
)
lines.append(
    f"max lnL(B) - max lnL(A) = {post['B']['max_lnl'] - post['A']['max_lnl']:+.2f}  "
    "(B nests A, so this should be >= 0 if both found the peak)"
)

# %% parameter table and the forbidden/Balmer broad-fraction ratio
lines.append(f"\n{'parameter':18s}{'A shared':>36s}{'B free':>36s}")
for lab in KEY:
    row = f"{lab:18s}"
    for tag in MODES:
        p = post[tag]
        if lab not in p["labels"]:
            row += f"{'-':>36s}"
            continue
        a, mid, b = wquantile(column(p, lab), p["w"], [0.16, 0.5, 0.84])
        lo, hi = p["prior"][lab][:2]
        rail = (
            " FLOOR"
            if (mid - lo) < 0.02 * (hi - lo)
            else " CEIL"
            if (hi - mid) < 0.02 * (hi - lo)
            else ""
        )
        row += f"{mid:10.3f} [{a:8.3f},{b:8.3f}]{rail:6s}".rjust(36)
    lines.append(row)
pb = post["B"]
ratio = column(pb, "eline_fbroad_forb") / column(pb, "eline_fbroad")
q = wquantile(ratio, pb["w"], [0.16, 0.5, 0.84])
p_gt = float(np.sum(pb["w"][ratio > 1]))
lines.append(
    f"\nB: f_forb / f_b = {q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}], P(f_forb > f_b) = {p_gt:.3f}"
)
lines.append("Prediction (10-01): f_forb ~ 0.2, f_forb / f_b ~ 1.2.")

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.hist(ratio, bins=60, weights=pb["w"], histtype="step", color="C3", density=True)
ax.axvline(1, color="0.4", ls="--", lw=0.8, label="shared (run A)")
ax.axvline(1.2, color="C0", ls=":", lw=0.8, label="prediction 1.2")
ax.set_xlabel("f_forb / f_b in run B")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fforb_ratio.png", dpi=120)
plt.close(fig)

# %% corner of the line and gas parameters, A over B
labs = [lab for lab in CORNER if lab in post["B"]["labels"]]
rng = []
for lab in labs:
    ts = [t for t in MODES if lab in post[t]["labels"]]
    a = min(wquantile(column(post[t], lab), post[t]["w"], 0.001) for t in ts)
    b = max(wquantile(column(post[t], lab), post[t]["w"], 0.999) for t in ts)
    d = (b - a) * 0.05 or 1e-3
    rng.append((a - d, b + d))
# in A the forbidden broad fraction is tied to the Balmer one, so its column is f_b
xa = np.column_stack(
    [
        column(post["A"], lab) if lab in post["A"]["labels"] else column(post["A"], "eline_fbroad")
        for lab in labs
    ]
)
xb = np.column_stack([column(post["B"], lab) for lab in labs])
fig = corner.corner(
    xa,
    weights=post["A"]["w"],
    color="0.5",
    labels=labs,
    range=rng,
    plot_datapoints=False,
    plot_density=False,
    smooth=1.0,
    bins=30,
    levels=(0.393, 0.865),
    hist_kwargs={"density": True},
)
corner.corner(
    xb,
    weights=post["B"]["w"],
    fig=fig,
    color="C3",
    range=rng,
    plot_datapoints=False,
    plot_density=False,
    smooth=1.0,
    bins=30,
    levels=(0.393, 0.865),
    hist_kwargs={"density": True},
)
fig.legend(
    handles=[
        plt.Line2D([], [], color="0.5", label="A shared (f_forb = f_b)"),
        plt.Line2D([], [], color="C3", label="B free"),
    ],
    loc="upper right",
    frameon=False,
)
fig.savefig(OUT / "corner_ab.png", dpi=100)
plt.close(fig)

# %% max-L spectra and line-window residuals
cue = get_sps(zero_library_resolution=False)["cue"]
spec = {}
for tag in MODES:
    z, flux, unc, good = data[tag]
    m, p = models[tag], post[tag]
    lnl = fit.loglike(p["best"], TID, MODES[tag], 45.0, True)
    obs = fit.make_obs(flux, unc, good)
    sp, stats = chi2_parts(m, p["best"], obs, cue, np.zeros_like(good))
    nb = p["best"].copy()
    for lab in ("eline_fbroad", "eline_fbroad_forb"):
        if lab in p["labels"]:
            nb[p["labels"].index(lab)] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp0, _ = chi2_parts(m, nb, obs, cue, np.zeros_like(good))
    spec[tag] = dict(
        sp=sp, sp_narrow=sp0, chi=residual_chi(flux, sp, unc, good), chi2_red=stats["chi2_red"]
    )
    lines.append(
        f"{tag}: max-L lnL recomputed {lnl:.2f} vs checkpoint {p['max_lnl']:.2f}; "
        f"chi2_nu {stats['chi2_red']:.3f}"
    )

z, flux, unc, good = data["A"]
rest = WAVE_OBS / (1 + z)
fig, axes = plt.subplots(
    3, len(WINDOWS), figsize=(16, 7), gridspec_kw=dict(height_ratios=[3, 1.2, 1.2])
)
lines.append("\nline windows (chi = (data - max-L model) / sigma)")
for j, (name, (lo, hi)) in enumerate(WINDOWS.items()):
    sel = (rest > lo) & (rest < hi)
    ax = axes[0, j]
    d, e = flam(flux, good)[sel], flam(unc, good)[sel]
    ax.fill_between(rest[sel], d - e, d + e, color="0.8", step="mid", lw=0)
    ax.step(rest[sel], d, "k", where="mid", lw=0.7, label="DESI, degraded")
    for k, (tag, color) in enumerate([("A", "0.4"), ("B", "C3")]):
        s = spec[tag]
        ax.plot(
            rest[sel],
            flam(s["sp"], good)[sel],
            color=color,
            lw=1,
            label=f"{tag} max-L, chi2_nu {s['chi2_red']:.2f}",
        )
        axr = axes[1 + k, j]
        axr.step(rest[sel], s["chi"][sel], color=color, where="mid", lw=0.8)
        axr.axhline(0, color="0.3", lw=0.5)
        for h in (-3, 3):
            axr.axhline(h, color="0.6", lw=0.5, ls=":")
        axr.set_xlim(lo, hi)
        axr.set_ylabel(f"chi {tag}", fontsize=8)
        c = s["chi"][sel]
        c = c[np.isfinite(c)]
        lines.append(
            f"{tag} {name:13s} n={len(c):3d} rms chi {np.sqrt(np.mean(c**2)):6.2f}  "
            f"mean {np.mean(c):+6.2f}  min {c.min():+7.2f}  max {c.max():+7.2f}"
        )
    ax.set_title(name, fontsize=9)
    ax.set_xlim(lo, hi)
    axes[2, j].set_xlabel(r"rest wavelength [$\AA$]")
axes[0, 0].legend(fontsize=7, frameon=False)
fig.suptitle(
    f"37084: run A (shared) vs run B (free), max-L models{' ' + args.tag if args.tag else ''}"
)
fig.tight_layout()
fig.savefig(OUT / "line_windows_ab.png", dpi=130)
plt.close(fig)

text = "\n".join(lines) + "\n"
print(text)
(OUT / "compare_ab.txt").write_text(text)
