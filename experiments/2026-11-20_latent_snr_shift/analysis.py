"""How far the cont10 latents and outlier-flow log p move when only the S/N of a mock changes.

QUESTION: for the same mock, how much do the latent and the log p change between median S/N 3, 5,
10 and 20, compared with the mock population spread and the outlier threshold?
HYPOTHESIS: none stated before the run. Negligible if the median latent shift between S/N 3 and
20 is below 0.1 sigma and the median log p change is below 10% of the median-to-threshold gap.
INPUTS: results/2026-11-20_latent_snr_shift/shift.npz (shift.py).
SEED: 0.
COMMAND: ``uv run python experiments/2026-11-20_latent_snr_shift/analysis.py``.
RESULT: not negligible. From S/N 20 to 3 the median latent moves 0.50 sigma (draws scatter 0.05-0.10)
and log p rises by 3.0, 20% of the median-to-threshold gap. Lower S/N moves mocks toward the dense
region: 48% fall below the threshold at S/N 3, 64% at S/N 20.
Assumptions: v2 mocks (C3K-ER) on the baseline encoder and flow, so the fraction below threshold is
high for all levels; one S/N curve rescaled per mock stands in for real noise.
FIGURES: none (analysis.txt).
"""

import numpy as np

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-20_latent_snr_shift"


def main():
    """Print latent shifts, log p changes and threshold crossings per S/N level."""
    d = np.load(OUT / "shift.npz")
    levels = d["levels"]
    z = (d["latents"] - d["scaler_mean"]) / d["scaler_scale"]  # (level, draw, mock, dim)
    logp = d["logp"]  # (level, draw, mock)
    ref = d["logp_ref_mocks"]
    thr, med = np.quantile(ref, 0.001), np.median(ref)
    gap = med - thr
    hi = len(levels) - 1

    mean_z = z.mean(axis=1)  # mean over draws, (level, mock, dim)
    draw_rms = np.sqrt(z.var(axis=1).mean(axis=-1))  # scatter between draws, per level and mock
    lines = [
        f"mocks {z.shape[2]}, draws {z.shape[1]}, levels {levels.tolist()}",
        f"reference mocks {ref.size}: median log p {med:.2f}, 0.1% threshold {thr:.2f}, gap {gap:.2f}",
        "",
        "level  measured S/N  draw-to-draw rms (sigma)  shift vs S/N 20 (sigma)  median dlogp vs S/N 20"
        "  below threshold",
    ]
    for i, level in enumerate(levels):
        shift = np.sqrt(((mean_z[i] - mean_z[hi]) ** 2).mean(axis=-1))
        dlogp = logp[i].mean(axis=0) - logp[hi].mean(axis=0)
        frac = (logp[i] <= thr).mean()
        lines.append(
            f"{level:5g}  {np.median(d['snr_measured'][i]):12.2f}  {np.median(draw_rms[i]):24.3f}"
            f"  {np.median(shift):23.3f}  {np.median(dlogp):22.2f}  {frac:15.4f}"
        )
    shift_3 = np.median(np.sqrt(((mean_z[0] - mean_z[hi]) ** 2).mean(axis=-1)))
    dlogp_3 = np.median(logp[0].mean(axis=0) - logp[hi].mean(axis=0))
    lines += [
        "",
        f"S/N 3 vs 20: median latent shift {shift_3:.3f} sigma (threshold 0.1) -> "
        + ("negligible" if shift_3 < 0.1 else "NOT negligible"),
        f"S/N 3 vs 20: median |dlogp| {abs(dlogp_3):.2f} = {abs(dlogp_3) / gap:.1%} of gap (threshold 10%) -> "
        + ("negligible" if abs(dlogp_3) < 0.1 * gap else "NOT negligible"),
    ]
    text = "\n".join(lines)
    (OUT / "analysis.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
