"""Does the spectrum's colour disagree with the imaging colour for the gallery92 artifacts?

QUESTION: does Delta(g-r), spectrum-synthesized minus Legacy Survey fibre (g-r), separate the
gallery92 artifacts from the physical classes, and how do candidates compare with a control?
HYPOTHESIS: none stated before the run.
INPUTS: FLUX_SYNTH_{G,R,Z} and FIBERFLUX_{G,R,Z} of data/fastspec-iron-sv3-bright.fits,
results/2026-11-10_data_quality_screen/targets.npz, gallery92 labels.
SEED: none.
COMMAND: ``uv run python experiments/2026-11-10_data_quality_screen/colour.py``.
RESULT: D(g-r) outside the control central 95% [-0.12, +0.18] in bad_sky 80%, bad_data 69%,
physical 40%, control 5%. D(g-r) < -0.25 flags bad_sky 7/10, emission_line 0/17, control 0%.
Assumptions: FIBERFLUX and spectrum see the same light. AGN variability, strong lines shift colours.
FIGURES: none (colour.txt).
"""

# %%
import glob
import os
import re

import numpy as np
from astropy.io import fits

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-11-10_data_quality_screen"
ARTIFACT = ("bad_sky", "bad_data", "star", "fibre_2675")
PHYSICAL = ("AGN", "emission_line", "quiescent", "composite")
lines = []

# %% colours
h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)
fs, meta = h["FASTSPEC"].data, h["METADATA"].data
tid = np.asarray(fs["TARGETID"], np.int64)
assert np.array_equal(tid, np.asarray(meta["TARGETID"], np.int64))


def mag(f):
    """Return -2.5 log10 of positive fluxes, NaN otherwise."""
    f = np.asarray(f, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, -2.5 * np.log10(f), np.nan)


d_gr = (mag(fs["FLUX_SYNTH_G"]) - mag(fs["FLUX_SYNTH_R"])) - (
    mag(meta["FIBERFLUX_G"]) - mag(meta["FIBERFLUX_R"])
)
d_rz = (mag(fs["FLUX_SYNTH_R"]) - mag(fs["FLUX_SYNTH_Z"])) - (
    mag(meta["FIBERFLUX_R"]) - mag(meta["FIBERFLUX_Z"])
)
d_r = mag(fs["FLUX_SYNTH_R"]) - mag(meta["FIBERFLUX_R"])
row = {int(t): i for i, t in enumerate(tid)}

# %% groups
tg = np.load(OUT / "targets.npz")
lab = {}
for d in glob.glob(str(RES / "2026-11-01_gallery92") + "/*/"):
    c = os.path.basename(d.rstrip("/"))
    for f in glob.glob(d + "*.png"):
        m = re.search(r"(\d{15,})", os.path.basename(f))
        if m and c != "cutouts":
            lab[int(m[1])] = c
groups = {
    "bad_sky": {t for t, c in lab.items() if c == "bad_sky"},
    "bad_data": {t for t, c in lab.items() if c == "bad_data"},
    "artifacts": {t for t, c in lab.items() if c in ARTIFACT},
    "physical": {t for t, c in lab.items() if c in PHYSICAL},
    "old cand": set(tg["old"].tolist()),
    "new cand": set(tg["new"].tolist()),
    "control": set(tg["control"].tolist()),
}


def get(x, group):
    """Return the finite values of x for a group."""
    v = np.array([x[row[t]] for t in group if t in row])
    return v[np.isfinite(v)]


# %% medians, spread, and the share beyond the control's central 95% (|value| tails)
for name, x in (("D(g-r)", d_gr), ("D(r-z)", d_rz), ("D(r) spec-fibre", d_r)):
    c = get(x, groups["control"])
    lo, hi = np.percentile(c, [2.5, 97.5])
    lines.append(f"{name}: control central 95% [{lo:+.3f}, {hi:+.3f}]")
    for g, v in groups.items():
        y = get(x, v)
        lines.append(
            f"   {g:>9} N {y.size:5d} median {np.median(y):+.3f} p16 {np.percentile(y, 16):+.3f} "
            f"p84 {np.percentile(y, 84):+.3f}  outside control 95% {np.mean((y < lo) | (y > hi)):.2f}"
        )

# %% candidate rules: per class caught, and the share of each candidate list and the control
x_gr, x_rz, x_r = (dict(zip(tid.tolist(), x, strict=True)) for x in (d_gr, d_rz, d_r))
c_rz = np.percentile(get(d_rz, groups["control"]), [2.5, 97.5])
c_r = np.percentile(get(d_r, groups["control"]), [2.5, 97.5])
rules = {
    "D(g-r) < -0.25": lambda t: x_gr.get(t, 0) < -0.25,
    "D(g-r) < -0.30": lambda t: x_gr.get(t, 0) < -0.30,
    "D(r-z) or D(r) outside control 95%": lambda t: (
        not (c_rz[0] <= x_rz.get(t, 0) <= c_rz[1]) or not (c_r[0] <= x_r.get(t, -0.4) <= c_r[1])
    ),
    "D(g-r) < -0.25 or D(r) outside 95%": lambda t: (
        x_gr.get(t, 0) < -0.25 or not (c_r[0] <= x_r.get(t, -0.4) <= c_r[1])
    ),
}
classes = (
    "bad_sky",
    "bad_data",
    "star",
    "fibre_2675",
    "AGN",
    "emission_line",
    "quiescent",
    "composite",
)
for name, rule in rules.items():
    per = "  ".join(
        f"{c} {sum(rule(t) for t, v in lab.items() if v == c)}/{sum(v == c for v in lab.values())}"
        for c in classes
    )
    share = "  ".join(
        f"{g} {np.mean([rule(t) for t in groups[g]]):.2f}"
        for g in ("old cand", "new cand", "control")
    )
    lines.append(f"\nrule {name}:\n   {per}\n   {share}")

text = "\n".join(lines)
print(text)
(OUT / "colour.txt").write_text(text + "\n")
