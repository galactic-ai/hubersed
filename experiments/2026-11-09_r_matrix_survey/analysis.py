"""How much the DR1 coadd LSF width varies across parent targets.

QUESTION: Is the effective LSF sigma of the DR1 coadd resolution matrix stable enough across
targets that a ~10k random subset of real matrices covers the variation for the new mocks?
HYPOTHESIS: The 5-95% quantiles of sigma per wavelength settle by a few hundred targets.
INPUTS: results/2026-11-09_r_matrix_survey/survey.npz (survey.py run, 1,000 targets, seed 0).
SEED: 0 for the subsamples.
COMMAND: ``uv run python experiments/2026-11-09_r_matrix_survey/analysis.py``.
RESULT: 1,000 of 1,000 found (job 3698233, run with -p spr). sigma 55 km/s (3700 A) to 24 (9500 A);
(p95-p5)/p50 0.05-0.11. n=300 vs 1,000 within 1.5% (passes 2%), n=100 2.6% (fails).
Assumptions: second moment summarises R; tails beyond p5/p95 not checked.
FIGURES: none (table only).
"""

# %%
import numpy as np

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-09_r_matrix_survey"
THRESH = 0.02  # max relative difference of the 5-95% quantiles, 300 vs 1,000 (Nikhil 11-09)
LAMS = (3700, 4300, 4861, 5007, 5500, 6563, 7500, 8500, 9500)

d = np.load(OUT / "survey.npz")
wave, sig, rsum, fiber = d["wave"], d["sigma_kms"], d["row_sum"], d["fiber"]
found = d["found"]
sig, rsum, fiber = sig[found], rsum[found], fiber[found]
# rows with little weight sit at arm edges or masked pixels
sig = np.where(np.abs(rsum - 1) < 0.05, sig, np.nan)
lines = [f"targets found {found.sum()} of {found.size}"]

# %% quantiles per wavelength and their spread
q = np.nanpercentile(sig, [5, 50, 95], axis=0)
lines.append("lambda  p5  p50  p95  (p95-p5)/p50   sigma in km/s")
for lam in LAMS:
    i = np.argmin(abs(wave - lam))
    lines.append(
        f"{lam:6d} {q[0, i]:5.1f} {q[1, i]:5.1f} {q[2, i]:5.1f} {(q[2, i] - q[0, i]) / q[1, i]:6.3f}"
    )

# %% convergence: 100 and 300 random targets against all
rng = np.random.default_rng(0)
for n in (100, 300):
    sub = sig[rng.choice(len(sig), n, replace=False)]
    qs = np.nanpercentile(sub, [5, 95], axis=0)
    rel = np.nanmax(np.abs(qs - q[[0, 2]]) / q[[0, 2]], axis=1)
    lines.append(f"n={n}: max rel diff p5 {rel[0]:.3f}, p95 {rel[1]:.3f} (threshold {THRESH})")

# %% spread across petals (fiber // 500) at the median wavelength
petal = np.where(fiber >= 0, fiber // 500, -1)
i = np.argmin(abs(wave - 6000))
med = [np.nanmedian(sig[petal == p, i]) for p in range(10) if np.any(petal == p)]
lines.append(
    f"fiber known for {np.sum(fiber >= 0)}; petal medians at 6000 A: {np.round(med, 1).tolist()}"
)

text = "\n".join(lines)
print(text)
(OUT / "analysis.txt").write_text(text + "\n")
