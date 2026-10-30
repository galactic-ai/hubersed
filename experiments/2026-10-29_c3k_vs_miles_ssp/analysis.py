"""C3K ER against MILES dust-free SSP continua at the same age and metallicity.

QUESTION: is the C3K continuum bowed against MILES at fixed age and metallicity, by the amount the
fits of 39633433328091196 imply (C3K best model about 6% fainter at rest 3400 A and 4% at 7000 A,
relative to V; 2026-10-29_continuum_39633433328091196_map/reddening.py)?
HYPOTHESIS: none registered.
INPUTS: ssp_c3k_er.npz and ssp_miles.npz of ssp.py (stampede3 job in ssp_s3.slurm), copied to
the Mac.
SEED: none.
COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-29_c3k_vs_miles_ssp/analysis.py``.
RESULT: 2026-10-29, ratio.txt. Same MIST, IMF and SPS_HOME: the continuum agrees within ~2-3% at
4500-7000 A; old ages at logzsol 0 differ in the Ca H&K bin (0.85), G band and Fe4383 (0.91-0.93)
and C2 4668 (1.04). Above the MILES cap (+0.3) the H&K bin falls to 0.76. Assumptions: scaled-solar
SSPs, 100 A bin means across different library resolutions, normalised at 5500 A.
FIGURES: results/2026-10-29_c3k_vs_miles_ssp/ratio.png and ratio.txt.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

OUT = PATHS["RESULTS"] / "2026-10-29_c3k_vs_miles_ssp"
MILES_RANGE = (3540.0, 7410.0)  # rest A, where FSPS uses MILES spectra
NORM = (5450.0, 5550.0)  # rest A window the ratio is normalised in
WINDOWS = (3800.0, 4400.0, 5500.0, 6200.0, 7000.0)  # rest A, +-50 A, as in reddening.py
HALF = 50.0
BIN = 100.0  # A, bin width of the plotted ratio

# %% load and compare on the C3K wavelength grid
c3k, miles = (np.load(OUT / f"ssp_{k}.npz") for k in ("c3k_er", "miles"))
assert np.allclose(c3k["ages"], miles["ages"]) and np.allclose(c3k["logzsol"], miles["logzsol"])
w = c3k["wave"]
keep = (w >= MILES_RANGE[0]) & (w <= MILES_RANGE[1])
edges = np.arange(MILES_RANGE[0], MILES_RANGE[1] + 1, BIN)
mid = 0.5 * (edges[1:] + edges[:-1])


def window_mean(x, lo, hi):
    """Return the mean of ``x`` over rest ``lo`` to ``hi`` on the C3K grid."""
    s = (w >= lo) & (w <= hi)
    return x[..., s].mean(axis=-1)


lines = [
    f"C3K ER / MILES dust-free SSP flux, each normalised by its mean over {NORM[0]:.0f}-{NORM[1]:.0f} A; "
    f"window means over +-{HALF:.0f} A",
    f"libraries {list(c3k['libraries'])} and {list(miles['libraries'])}",
    f"{'logzsol':>7} {'age Gyr':>7} " + " ".join(f"{x:>7.0f}" for x in WINDOWS),
]
ratio = {}
for i, z in enumerate(c3k["logzsol"]):
    for j, age in enumerate(c3k["ages"]):
        a = c3k["spec"][i, j]
        b = np.interp(w, miles["wave"], miles["spec"][i, j])
        r = (a / window_mean(a, *NORM)) / (b / window_mean(b, *NORM))
        ratio[(z, age)] = r
        vals = [window_mean(r, x - HALF, x + HALF) for x in WINDOWS]
        lines.append(f"{z:7.1f} {age:7.1f} " + " ".join(f"{v:7.3f}" for v in vals))
lines += [
    "",
    "for reference, the fits imply C3K/MILES intrinsic about 0.98 at 3800, 1.01 at 4400, 0.98 at "
    "6200 and 0.96 at 7000 A (10^(0.4 (A_C3K - A_MILES)) of reddening.py)",
]
text = "\n".join(lines) + "\n"
(OUT / "ratio.txt").write_text(text)
print(text)

# %% figure: binned ratio against rest wavelength, one panel per metallicity
use_apj_style()
fig, ax = plt.subplots(1, len(c3k["logzsol"]), figsize=(13, 4.5), sharey=True)
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(c3k["ages"])))
for a, z in zip(ax, c3k["logzsol"], strict=True):
    for age, color in zip(c3k["ages"], colors, strict=True):
        r = ratio[(z, age)]
        binned = [
            r[keep & (w >= lo) & (w < hi)].mean()
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
        a.plot(mid, binned, color=color, lw=1.2, label=f"{age:g} Gyr")
    a.axhline(1, color="0.3", lw=0.6)
    a.set_title(f"logzsol {z:+.1f}", fontsize=13)
    a.set_xlabel(r"rest wavelength [$\AA$]", fontsize=12)
ax[0].set_ylabel(f"C3K ER / MILES, {BIN:.0f} A bins", fontsize=12)
ax[-1].legend(frameon=False, fontsize=10)
fig.savefig(OUT / "ratio.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'ratio.png'}")
