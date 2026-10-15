"""Compare Cue strong-line fits with the direct method and the held-out auroral lines.

QUESTION: is Cue's gas too hot in general, and does that bias its gas-phase O/H against the direct
(T_e) method, as a function of stellar mass?

HYPOTHESIS: from 39627770174637084, Cue over-predicts the held-out auroral lines ([OIII]4363 x2.6,
[SIII]6312 x1.4, [OII]7325 x1.18) and its O/H is 0.31 dex below the direct value. If general, the
gap should be positive across mass; if it grows with mass or EW(Hbeta), it changes the MZR shape.

INPUTS: lineflux_table.fits (data.py), fits/<TARGETID>_strong_emfit_mu0.4_zs-0.41.npz (fit.py
--lines strong) and direct_wild0.4_campbell_r2_subset_20.fits (direct.py --ids subset_20.txt) in
results/2026-10-14_cue_lineflux_calibration/. Cue gas-phase 12+log(O/H) = gas_logz + 8.71
(before depletion, handoff 2026-10-13e). Points more than e^-30 below the heaviest are dropped.
SEED: none.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/analysis.py``
from the repository root.

RESULT: preliminary, 20 galaxies (subset.py), zstar -0.41, mu 0.4. Held out, Cue over-predicts
[OII]7325 by a median 1.13 (20 galaxies, 0.41-2.76) and [SIII]6312 by 1.19 (7). Direct - Cue gap
median +0.08 dex, 16-84 [-0.41, +0.49]; it tracks the [OII]7325 residual, since both use the [OII]
thermometer, and galaxies with n_e at the 10 cm^-3 floor get high T(O+) and low direct O/H. Not as
expected: Cue strong-line gas_logz shows no mass trend; at log M* 9.8 and 11.2 it gives Z ~ -0.77
with N/O ~ +0.2 and U -3.4 to -3.9 (direct 8.61, 9.00), and 39627824604123900 is bimodal (Z -0.76
and +0.08). The Z, N/O and U degeneracy under a flat N/O prior looks larger than the temperature
effect for an MZR.

FIGURES: compare_subset.txt and compare_subset.png in results/2026-10-14_cue_lineflux_calibration/.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
CUE_OH_ZERO = 8.71
AURORAL = ("OII7325", "SIII6312", "OIII4363")
SNR_SHOW = 3.0


def weighted_q(x, w, q=(0.16, 0.5, 0.84)):
    """Return weighted quantiles of x."""
    i = np.argsort(x)
    c = np.cumsum(w[i])
    return np.interp(q, c / c[-1], x[i])


# %%
lines = Table.read(OUT / "lineflux_table.fits")
direct = Table.read(OUT / "direct_wild0.4_campbell_r2_subset_20.fits")
rows = []
for f in sorted((OUT / "fits").glob("*_strong_emfit_mu0.4_zs-0.41.npz")):
    d = np.load(f)
    tid = int(f.name.split("_")[0])
    labels = [str(x) for x in d["labels"]]
    lw = d["log_w"]
    keep = lw > lw.max() - 30
    w = np.exp(lw[keep] - lw.max())
    z = weighted_q(d["points"][keep, labels.index("gas_logz")], w) + CUE_OH_ZERO
    lr = [str(x) for x in d["lines"]]
    ratio = {}
    for k in AURORAL:
        j = lr.index(k)
        obs, err = float(d["obs"][j]), float(d["err"][j])
        ok = np.isfinite(obs) and err > 0 and obs / err >= SNR_SHOW
        ratio[k] = np.median(d["draw_flux"][:, j]) / obs if ok else np.nan
    li = lines[lines["TARGETID"] == tid][0]
    di = direct[direct["TARGETID"] == tid][0]
    rows.append(
        {
            "TARGETID": tid,
            "LOGMSTAR": float(li["LOGMSTAR"]),
            "Z": float(li["Z"]),
            "EW_HB": float(li["EW_HB"]),
            "CUE_OH": z[1],
            "CUE_OH_ERR": (z[2] - z[0]) / 2,
            "DIRECT_OH": float(di["OH_P50"]),
            "DIRECT_OH_ERR": float(di["OH_P84"] - di["OH_P16"]) / 2,
            "DIRECT_KEEP": bool(di["KEEP"]),
            "T_FROM_4363": bool(di["T_FROM_4363"]),
            "T_LOW": float(di["T_LOW_P50"]),
            "NE_BOUND": float(di["NE_BOUND_FRAC"]),
            **{f"R_{k}": ratio[k] for k in AURORAL},
        }
    )
t = Table(rows=rows)
t.sort("LOGMSTAR")
t["GAP"] = t["DIRECT_OH"] - t["CUE_OH"]

# %%
head = (
    f"{'TARGETID':>18} {'logM*':>6} {'z':>6} {'EW_Hb':>6} {'Cue O/H':>8} {'direct':>7} {'gap':>6} "
    f"{'T(O+)':>6} {'ne@lim':>6} {'4363':>4} {'keep':>4} "
    + " ".join(f"{'m/d ' + k:>12}" for k in AURORAL)
)
out = [
    "Cue strong-line fits vs direct method; m/d = Cue model / data for held-out lines "
    f"(shown where the line has S/N >= {SNR_SHOW})",
    head,
]
for r in t:
    out.append(
        f"{r['TARGETID']:>18} {r['LOGMSTAR']:6.2f} {r['Z']:6.3f} {r['EW_HB']:6.1f} "
        f"{r['CUE_OH']:8.2f} {r['DIRECT_OH']:7.2f} {r['GAP']:+6.2f} {r['T_LOW']:6.0f} {r['NE_BOUND']:6.2f} "
        f"{'yes' if r['T_FROM_4363'] else 'no':>4} {'yes' if r['DIRECT_KEEP'] else 'no':>4} "
        + " ".join(f"{r[f'R_{k}']:12.2f}" for k in AURORAL)
    )
k = np.asarray(t["DIRECT_KEEP"])
gap = np.asarray(t["GAP"])[k]
out += [
    "",
    f"kept {k.sum()} of {len(t)}; gap median {np.median(gap):+.2f}, 16-84 "
    f"[{np.percentile(gap, 16):+.2f}, {np.percentile(gap, 84):+.2f}] dex",
]
for key in AURORAL:
    v = np.asarray(t[f"R_{key}"])
    v = v[np.isfinite(v)]
    if v.size:
        out.append(f"{key}: model/data median {np.median(v):.2f} over {v.size} galaxies")
text = "\n".join(out)
print(text)
(OUT / "compare_subset.txt").write_text(text + "\n")

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
m = np.asarray(t["LOGMSTAR"])
ax[0].errorbar(m[k], gap, yerr=np.asarray(t["DIRECT_OH_ERR"])[k], fmt="o", ms=4)
ax[0].axhline(0, color="0.5", lw=0.8)
ax[0].set_xlabel(r"log $M_*$ (FastSpecFit)")
ax[0].set_ylabel("direct - Cue 12+log(O/H)")
for key, mk in zip(AURORAL, ("o", "s", "^"), strict=True):
    ax[1].plot(m, np.asarray(t[f"R_{key}"]), mk, ms=5, label=key)
ax[1].axhline(1, color="0.5", lw=0.8)
ax[1].set_yscale("log")
ax[1].set_xlabel(r"log $M_*$ (FastSpecFit)")
ax[1].set_ylabel("Cue model / data, held out")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "compare_subset.png", dpi=150)
