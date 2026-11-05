"""Same-ion null test: do two lines of one ion get the same second-component fraction?

QUESTION: [O III] 4959 and 5007, and [N II] 6548 and 6584, each come from one upper level of one
ion, so their second-component fractions must be equal. With a free fraction per line, how far
apart are they, and is that larger than the difference between the low-ionization group and the
Balmer lines (0.21 vs 0.18) found in 2026-10-13_oiii_second_component?
HYPOTHESIS: the pair differences measure the floor of the fit. If they are as large as the group
difference, the group difference is not a measurement.
INPUTS: results/2026-10-13_same_ion_null/fit_*.npz and profile_*.npz from fit.py.
SEED: none.
COMMAND: ``PYTHONPATH=experiments/2026-10-13_same_ion_null:experiments/2026-10-13_oiii_second_component:experiments/2026-10-13_nautilus_resolution_matrix
uv run --no-sync python experiments/2026-10-13_same_ion_null/analysis.py``
RESULT: see same_ion_null.txt.
FIGURES: same_ion_profiles.png in results/2026-10-13_same_ion_null/.

Each profile fixes one per-line fraction and re-optimizes the line profile, the other per-line
fractions and the gas parameters. A parabola through the points within 30 of the minimum gives the
best value and the Delta chi2 = 1 half-width sigma (statistical, unscaled). The slope of the partner
line's fraction along the profile is b = rho sigma_partner / sigma_self, so the covariance of the
pair is b sigma_self^2 and the error of the difference is sqrt(s1^2 + s2^2 - 2 cov). Both profiles
of a pair give an estimate of the covariance; their mean is used.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from same_ion import OUT, PAIRS, PER_LINE

from hubersed.plotting.style import use_apj_style

fits = {p.stem: np.load(p) for p in sorted(OUT.glob("fit_*.npz"))}
best = min(fits, key=lambda k: float(fits[k]["fun"]))
labels = [str(x) for x in fits[best]["labels"]]
theta = dict(zip(labels, fits[best]["theta"], strict=True))
chi2_best = float(fits[best]["chi2"])


def load_profile(name):
    """Return the grid values, chi2 and the fitted values of every parameter along a profile."""
    files = sorted(OUT.glob(f"profile_{name}_*.npz"))
    x = np.array([float(p.stem.rsplit("_", 1)[1]) for p in files])
    rows = [np.load(p) for p in files]
    chi2 = np.array([float(r["chi2"]) for r in rows])
    th = np.array([r["theta"] for r in rows])
    o = np.argsort(x)
    return x[o], chi2[o], th[o]


def parabola(x, c, cut=30.0):
    """Return the minimum and the Delta chi2 = 1 half-width of a parabola through the low points."""
    m = c - c.min() < cut
    if m.sum() < 3:
        return np.nan, np.nan
    a, b, _ = np.polyfit(x[m], c[m], 2)
    return -b / (2 * a), 1 / np.sqrt(a) if a > 0 else np.nan


# %% per-line profiles
prof = {}
for name in PER_LINE:
    x, c, th = load_profile(name)
    if x.size:
        xmin, sig = parabola(x, c)
        prof[name] = dict(x=x, c=c, th=th, xmin=xmin, sig=sig)

lines = [
    f"Same-ion null test of 39627770174637084, best fit {best}: chi2 {chi2_best:.2f}",
    "",
    f"{'per-line fraction':>24} {'MAP':>7} {'profile min':>12} {'sigma':>7}",
]
for name in PER_LINE:
    p = prof.get(name, {})
    lines.append(
        f"{name:>24} {theta[name]:7.3f} {p.get('xmin', np.nan):12.3f} {p.get('sig', np.nan):7.3f}"
    )
lines.append("")
lines.append(f"{'pair':>20} {'difference':>10} {'sigma_diff':>10} {'diff/sigma':>10} {'rho':>6}")
for pair, (a, b) in PAIRS.items():
    d = theta[a] - theta[b]
    if a in prof and b in prof:
        sa, sb = prof[a]["sig"], prof[b]["sig"]
        ia, ib = labels.index(a), labels.index(b)
        # slope of the partner along each profile
        slope_ab = np.polyfit(prof[a]["x"], prof[a]["th"][:, ib], 1)[0]
        slope_ba = np.polyfit(prof[b]["x"], prof[b]["th"][:, ia], 1)[0]
        cov = 0.5 * (slope_ab * sa**2 + slope_ba * sb**2)
        sd = np.sqrt(max(sa**2 + sb**2 - 2 * cov, 0.0))
        rho = cov / (sa * sb)
        lines.append(f"{pair:>20} {d:10.3f} {sd:10.3f} {d / sd:10.2f} {rho:6.2f}")
    else:
        lines.append(f"{pair:>20} {d:10.3f} {'n/a':>10}")
lines.append("")
lines.append(
    "Group values at the best fit: Balmer eline_fbroad "
    f"{theta['eline_fbroad']:.3f}, low-ion eline_fbroad_forb {theta['eline_fbroad_forb']:.3f}, "
    f"high-ion eline_fbroad_forb_hi {theta['eline_fbroad_forb_hi']:.3f} (now only the high-ion "
    "lines other than [O III] 4959 and 5007)."
)
lines.append("Window chi2 at the best fit:")
for w, v in zip(fits[best]["windows"], fits[best]["window_chi2"], strict=True):
    lines.append(f"  {str(w):>12} {float(v):8.2f}")
text = "\n".join(lines)
print(text)
(OUT / "same_ion_null.txt").write_text(text + "\n")

# %% figure
use_apj_style()
fig, axes = plt.subplots(1, len(prof), figsize=(3.2 * max(len(prof), 1), 3), squeeze=False)
for ax, (name, p) in zip(axes[0], prof.items(), strict=True):
    ax.plot(p["x"], p["c"] - chi2_best, "o-", color="C0")
    ax.axhline(1, color="0.5", ls="--", lw=0.6)
    ax.axvline(theta[name], color="C3", lw=0.8)
    ax.set_title(PER_LINE[name][0])
    ax.set_xlabel("second-component fraction")
axes[0, 0].set_ylabel(r"$\Delta\chi^2$")
fig.tight_layout()
fig.savefig(OUT / "same_ion_profiles.png", dpi=200)
