"""Force Cue onto the upper metallicity branch for metal-rich galaxies: which lines fail, and is it dust?

QUESTION: over 78 galaxies Cue never takes the upper branch (posterior weight at 12+log(O/H) > 8.41
median 0.00 for N2 > -0.8), and A_V rises with N2 and mass (Spearman +0.73, +0.71). Forced to
12+log(O/H) >= ZMIN_OH, how much worse do metal-rich galaxies fit, in which lines, and does it
differ between low and high A_V?

``pick`` writes forced_ids.txt: among galaxies of subset_20.txt and subset_ew60.txt at z > 0.01 with
observed N2 > N2_MIN, the N_EACH highest-N2 galaxies with free-fit A_V < AV_LOW and the N_EACH with
A_V > AV_HIGH (A_V is the weighted median of the flat strong-line fit). ``compare`` then reads the
free fits and fit.py --zmin fits (gas_logz >= ZMIN_OH - 8.71) and writes forced_compare.txt: O/H,
the change of the best data log likelihood, and model/data per line (median over posterior draws)
for both. Points more than e^-30 below the heaviest are dropped. SEED: none.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/forced.py pick``;
then for each TARGETID in forced_ids.txt, ``fit.py TARGETID --lines strong --zmin -0.31``; then
``forced.py compare``, all from the repository root.

OUTPUT: forced_ids.txt and forced_compare.txt in results/2026-10-14_cue_lineflux_calibration/.
"""

import sys

import numpy as np
from astropy.table import Table
from fit import LINES

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
CUE_OH_ZERO = 8.71
ZMIN_OH = 8.4
N2_MIN = -0.8
AV_LOW, AV_HIGH = 1.0, 1.5
N_EACH = 5
Z_MIN = 0.01
FREE = "strong_emfit_mu0.4_zs-0.41"
FORCED = f"{FREE}_zmin{ZMIN_OH - CUE_OH_ZERO:+.2f}"


def load(tid, tag):
    """Return weighted medians, O/H 16-84, best data lnL and model/data per line of one fit."""
    d = np.load(OUT / "fits" / f"{tid}_{tag}.npz")
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
    md = {}
    for j, k in enumerate(lines):
        obs, err = float(d["obs"][j]), float(d["err"][j])
        if np.isfinite(obs) and err > 0 and obs / err >= 3:
            md[k] = float(np.median(d["draw_flux"][:, j]) / obs)
    lnl = d["log_l_data"] if "log_l_data" in d.files else d["log_l"]
    return {
        "oh": q("gas_logz") + CUE_OH_ZERO,
        "av": q("av")[1],
        "no": q("gas_logno")[1],
        "u": q("gas_logu")[1],
        "lnl": float(lnl.max()),
        "md": md,
    }


def pick():
    """Write forced_ids.txt with the low- and high-A_V metal-rich galaxies."""
    t = Table.read(OUT / "lineflux_table.fits")
    ids = np.concatenate(
        [np.loadtxt(OUT / f, dtype=np.int64) for f in ("subset_20.txt", "subset_ew60.txt")]
    )
    rows = []
    for tid in ids:
        r = t[t["TARGETID"] == tid][0]
        n2 = float(np.log10(r["F_NII6584"] / r["F_HA"])) if r["F_NII6584"] > 0 else np.nan
        if r["Z"] <= Z_MIN or not n2 > N2_MIN:
            continue
        rows.append((n2, int(tid), load(tid, FREE)["av"]))
    rows.sort(reverse=True)
    low = [r for r in rows if r[2] < AV_LOW][:N_EACH]
    high = [r for r in rows if r[2] > AV_HIGH][:N_EACH]
    for name, group in (("low A_V", low), ("high A_V", high)):
        for n2, tid, av in group:
            print(f"{name}: {tid} N2 {n2:+.2f} A_V {av:.2f}")
    np.savetxt(OUT / "forced_ids.txt", [r[1] for r in low + high], fmt="%d")


def compare():
    """Write forced_compare.txt: free against forced fits for each picked galaxy."""
    ids = np.loadtxt(OUT / "forced_ids.txt", dtype=np.int64)
    order = [k for k in LINES]
    out = [
        f"free vs forced (12+log(O/H) >= {ZMIN_OH}) strong-line fits; m/d = model/data, S/N >= 3",
        f"{'TARGETID':>18} {'fit':>6} {'O/H':>5} {'A_V':>5} {'N/O':>6} {'U':>6} {'dlnL':>7}  "
        + " ".join(f"{k:>9}" for k in order),
    ]
    for tid in ids:
        free, forced = load(tid, FREE), load(tid, FORCED)
        for name, f in (("free", free), ("forced", forced)):
            cells = [f"{f['md'][k]:9.2f}" if k in f["md"] else f"{'-':>9}" for k in order]
            out.append(
                f"{tid:>18} {name:>6} {f['oh'][1]:5.2f} {f['av']:5.2f} {f['no']:+6.2f} {f['u']:+6.2f} "
                f"{f['lnl'] - free['lnl']:+7.2f}  " + " ".join(cells)
            )
    text = "\n".join(out)
    print(text)
    (OUT / "forced_compare.txt").write_text(text + "\n")


if __name__ == "__main__":
    {"pick": pick, "compare": compare}[sys.argv[1]]()
