"""Do the metal-rich galaxies move to high O/H with the auroral lines, the Nicholls prior or Z* 0.0?

QUESTION: residuals.py shows Cue fits the strong lines of the six metal-rich galaxies (N2 > -0.8)
with a low-Z solution that over-predicts the held-out [OII]7325 (median 1.63x). Which change moves
them to high O/H, and what does it cost in fit to the data?

Variants, all from fit.py: strong flat (fits/..._strong_emfit_mu0.4_zs-0.41.npz), all flat,
strong Nicholls prior, all Nicholls prior, strong with --zstar 0.0. For each: weighted median
12+log(O/H) = gas_logz + 8.71 with 16-84, N/O, U, [OII]7325 model/data, and the best data log
likelihood (log_l_data where saved, else log_l, which equals it for flat fits). Likelihoods compare
only fits of the same lines. Points more than e^-30 below the heaviest are dropped. SEED: none.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/rich_compare.py``
from the repository root.

OUTPUT: rich_compare.txt in results/2026-10-14_cue_lineflux_calibration/.
"""

import numpy as np

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
CUE_OH_ZERO = 8.71
IDS = (
    1078898594938887,
    39627805926884333,
    39627860859685607,
    39627824604123900,
    39628384589841385,
    39633456296102164,
)
VARIANTS = {
    "strong flat": "strong_emfit_mu0.4_zs-0.41",
    "strong Z*0.0": "strong_emfit_mu0.4_zs+0.00",
    "strong prior": "strong_emfit_mu0.4_zs-0.41_pnicholls",
    "all flat": "all_emfit_mu0.4_zs-0.41",
    "all prior": "all_emfit_mu0.4_zs-0.41_pnicholls",
}


def summarise(path):
    """Return O/H quantiles, N/O and U medians, 7325 model/data and the best data log likelihood."""
    d = np.load(path)
    lab = [str(x) for x in d["labels"]]
    lw = d["log_w"]
    keep = lw > lw.max() - 30
    w = np.exp(lw[keep] - lw.max())
    p = d["points"][keep]

    def q(name, qs=(0.16, 0.5, 0.84)):
        x = p[:, lab.index(name)]
        i = np.argsort(x)
        c = np.cumsum(w[i])
        return np.interp(qs, c / c[-1], x[i])

    lines = [str(x) for x in d["lines"]]
    j = lines.index("OII7325")
    r7325 = float(np.median(d["draw_flux"][:, j]) / d["obs"][j])
    lnl = d["log_l_data"] if "log_l_data" in d.files else d["log_l"]
    return q("gas_logz") + CUE_OH_ZERO, q("gas_logno")[1], q("gas_logu")[1], r7325, float(lnl.max())


out = [
    "12+log(O/H) [16, 84], N/O, U, [OII]7325 model/data, best data lnL; dlnL vs strong flat or all flat"
]
for tid in IDS:
    out.append(f"\n{tid}")
    ref = {}
    for name, tag in VARIANTS.items():
        f = OUT / "fits" / f"{tid}_{tag}.npz"
        if not f.exists():
            out.append(f"  {name:>13}: missing")
            continue
        oh, no, u, r, lnl = summarise(f)
        base = "strong" if name.startswith("strong") else "all"
        ref.setdefault(base, lnl)
        out.append(
            f"  {name:>13}: O/H {oh[1]:5.2f} [{oh[0]:5.2f}, {oh[2]:5.2f}]  N/O {no:+5.2f}  U {u:+5.2f}  "
            f"7325 m/d {r:5.2f}  lnL {lnl:8.2f}  dlnL {lnl - ref[base]:+6.2f}"
        )
text = "\n".join(out)
print(text)
(OUT / "rich_compare.txt").write_text(text + "\n")
