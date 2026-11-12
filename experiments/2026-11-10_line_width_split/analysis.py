"""Narrow and second-component line widths of the parent sample, to set the broad-width prior.

QUESTION: What range of second-component width, as a ratio r = sigma_second / sigma_narrow and in
km/s, does EmFit find in our 104,990 parent galaxies? This sets the broad-width prior of the
line columns (sigma_broad = r * eline_sigma) instead of the fixed 250 km/s split.
HYPOTHESIS: Second components sit at r ~ 1.5-5 and sigma ~ 60-300 km/s (CLASSY, green peas); the
250 km/s lower bound excludes most of them.
INPUTS: emfit_widths_sv3_bright.fits (fetch.py, Data Lab desi_dr1.emfit) and
results/prescreened_sample_v1.npz (the parent). EmFit fits a second component only when a
Delta chi2 test passes at 5 sigma and defines it as the wider one (Pucha+25, arXiv:2411.00091,
pucha2024.tex l.432 and l.931); Halpha takes its widths from the [SII] fit (l.434), [OIII] is
fitted on its own (l.438). Sigmas are intrinsic only where the flag is 0.
SEED: none.
COMMAND: ``uv run python experiments/2026-11-10_line_width_split/analysis.py``.
RESULT: 87,278 of 104,990 parent in EmFit. Halpha narrow sigma p50 90, p99 408 km/s; second
component in 1.2%, sigma p50 82, p99 259 km/s, r p5 1.1, p50 2.4, p99 13.5; [OIII] second r p5 1.3,
p50 3.6, p99 38. So the 250 km/s split excludes ~99% of second components; prior set to narrow
TopHat(10, 450), r TopHat(1.2, 40). Assumption: EmFit 5-sigma detections stand for all wings.
FIGURES: none (table only).
"""

# %%
import numpy as np
from astropy.table import Table

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-10_line_width_split"
Q = (1, 5, 25, 50, 75, 95, 99)

tab = Table.read(OUT / "emfit_widths_sv3_bright.fits")
keep = np.load(PATHS["RESULTS"] / "prescreened_sample_v1.npz")["keep"]
t = tab[np.isin(np.asarray(tab["targetid"]), keep)]
lines = [f"EmFit rows {len(tab)}; parent galaxies found {len(t)} of {keep.size}"]


def col(name):
    """Return a column as a float array with masked entries as NaN."""
    c = t[name]
    if hasattr(c, "filled"):
        return np.asarray(c.astype(float).filled(np.nan), float)
    return np.asarray(c, float)


def qline(label, x):
    """Return one table row: label, N and the quantiles of finite x."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return f"{label:34s} N 0"
    q = np.percentile(x, Q)
    return f"{label:34s} N {x.size:6d}  " + "  ".join(
        f"p{p} {v:7.1f}" for p, v in zip(Q, q, strict=True)
    )


# %% narrow widths and second components, Halpha (via [SII]) and [OIII]5007
for narrow, second, name in (
    ("ha_n", "ha_out", "Halpha"),
    ("oiii5007", "oiii5007_out", "[OIII]5007"),
):
    s_n, f_n = col(f"{narrow}_sigma"), col(f"{narrow}_sigma_flag")
    s_2, f_2 = col(f"{second}_sigma"), col(f"{second}_sigma_flag")
    has2 = (col(f"{second}_flux") > 0) & (f_2 >= 0)
    both = has2 & (f_n == 0) & (f_2 == 0)
    lines.append(
        f"{name}: narrow resolved (flag 0) {np.mean(f_n == 0):.3f}, unresolved {np.mean(f_n == 1):.3f}; "
        f"second component in {has2.sum()} ({has2.mean():.4f}), both resolved {both.sum()}"
    )
    lines.append(qline("  narrow sigma, flag 0 [km/s]", np.where(f_n == 0, s_n, np.nan)))
    lines.append(qline("  second sigma, both flag 0 [km/s]", np.where(both, s_2, np.nan)))
    lines.append(qline("  r = second / narrow, both flag 0", np.where(both, s_2 / s_n, np.nan)))

# %% broad Balmer (AGN-like) components
pb, s_b, s_n = col("prob_broad"), col("ha_b_sigma"), col("ha_n_sigma")
bl = (pb >= 0.8) & (col("ha_b_sigma_flag") == 0) & (col("ha_n_sigma_flag") == 0)
lines.append(f"broad Halpha with prob_broad >= 0.8 and both flag 0: {bl.sum()}")
lines.append(qline("  broad sigma [km/s]", np.where(bl, s_b, np.nan)))
lines.append(qline("  r = broad / narrow", np.where(bl, s_b / s_n, np.nan)))

text = "\n".join(lines)
print(text)
(OUT / "analysis.txt").write_text(text + "\n")
