"""Recovery of gas O/H from the line-flux mocks with flat and Nicholls N/O and C/O priors.

QUESTION: with strong lines only, can Cue recover gas O/H when N/O and C/O follow the Nicholls+17
relations, with flat priors and with the relations as priors? Does the prior dominate when the true
N/O is 0.3 dex off the relation?

HYPOTHESIS: flat priors trade Z against N/O, so recovered Z is biased low at high Z; the prior
removes most of that bias. If the prior dominates, the control mocks shift Z by about
offset / (d gas_logno / d gas_logz), about +-0.34 dex near solar.

INPUTS: mocks/mock_table.fits (mocks.py) and mocks/fits/<TARGETID>_strong_emfit_mu0.4_zs-0.41
[_pnicholls].npz (fit.py --table --prior) in results/2026-10-14_cue_lineflux_calibration/. Recovered
12+log(O/H) = gas_logz + 8.71, the same scale as the truth. Points more than e^-30 below the
heaviest are dropped. The mocks and the fits share Cue and the ionizing spectrum, so this tests
identifiability, not whether Cue fits real galaxies. SEED: none.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/analysis_mocks.py``
from the repository root.

RESULT: 54 mocks (6 Z x 2 U x 3 donors, plus 18 N/O-offset controls). Not as expected: flat
priors recover O/H well, mean bias -0.004 dex, rms 0.040, 68% coverage 0.75 (prior: -0.009, 0.036,
0.78), including at 8.8-9.0, and N/O is recovered to about +-0.03 dex. Control Z shifts are -0.015
to +0.025 dex against +-0.33-0.39 for full prior dominance. So within Cue the strong lines separate
Z from N/O, and the flat Cue MZR of the real subset is not an identifiability problem; it points to
real metal-rich spectra that Cue does not reproduce at their true Z.

FIGURES: mocks/compare_mocks.txt and mocks/compare_mocks.png.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration" / "mocks"
CUE_OH_ZERO = 8.71
PRIORS = ("flat", "nicholls")


def weighted_q(x, w, q=(0.16, 0.5, 0.84)):
    """Return weighted quantiles of x."""
    i = np.argsort(x)
    c = np.cumsum(w[i])
    return np.interp(q, c / c[-1], x[i])


def load(tid, prior):
    """Return quantiles of the fitted parameters and the best data log likelihood, or None."""
    tag = "" if prior == "flat" else f"_p{prior}"
    f = OUT / "fits" / f"{tid}_strong_emfit_mu0.4_zs-0.41{tag}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    lab = [str(x) for x in d["labels"]]
    lw = d["log_w"]
    keep = lw > lw.max() - 30
    w = np.exp(lw[keep] - lw.max())
    p = d["points"][keep]
    out = {
        k: weighted_q(p[:, lab.index(k)], w)
        for k in ("gas_logz", "gas_logno", "gas_logco", "gas_logu")
    }
    out["best_lnl"] = float(np.max(d["log_l_data"]))
    return out


# %%
t = Table.read(OUT / "mock_table.fits")
rows = []
for r in t:
    tid = int(r["TARGETID"])
    fits = {p: load(tid, p) for p in PRIORS}
    if any(v is None for v in fits.values()):
        continue
    row = {
        "TARGETID": tid,
        "TRUE_OH": float(r["TRUE_OH"]),
        "TRUE_LOGU": float(r["TRUE_LOGU"]),
        "NO_OFFSET": float(r["NO_OFFSET"]),
        "TRUE_LOGNO": float(r["TRUE_LOGNO"]),
    }
    for p, v in fits.items():
        oh = v["gas_logz"] + CUE_OH_ZERO
        row[f"BIAS_{p}"] = oh[1] - row["TRUE_OH"]
        row[f"COVER_{p}"] = bool(oh[0] <= row["TRUE_OH"] <= oh[2])
        row[f"WIDTH_{p}"] = (oh[2] - oh[0]) / 2
        row[f"NO_{p}"] = v["gas_logno"][1]
        row[f"CO_{p}"] = v["gas_logco"][1]
        row[f"LNL_{p}"] = v["best_lnl"]
    row["DLNL"] = row["LNL_nicholls"] - row["LNL_flat"]
    rows.append(row)
m = Table(rows=rows)

# %%
out = [f"{len(m)} of {len(t)} mocks with both fits; bias = recovered - true 12+log(O/H), dex", ""]
out.append(
    f"{'OH':>4} {'U':>5} {'off':>5} {'n':>2} | {'bias flat':>9} {'cover':>5} {'width':>5} | "
    f"{'bias prior':>10} {'cover':>5} {'width':>5} | {'dlnL':>6} | {'N/O true':>8} {'flat':>6} {'prior':>6}"
)
keys = sorted(
    {(r["TRUE_OH"], r["TRUE_LOGU"], r["NO_OFFSET"]) for r in m}, key=lambda k: (k[2], k[1], k[0])
)
for oh, u, off in keys:
    s = m[(m["TRUE_OH"] == oh) & (m["TRUE_LOGU"] == u) & (m["NO_OFFSET"] == off)]
    out.append(
        f"{oh:4.1f} {u:5.1f} {off:+5.1f} {len(s):2d} | "
        f"{np.mean(s['BIAS_flat']):+9.3f} {np.mean(s['COVER_flat']):5.2f} {np.mean(s['WIDTH_flat']):5.3f} | "
        f"{np.mean(s['BIAS_nicholls']):+10.3f} {np.mean(s['COVER_nicholls']):5.2f} "
        f"{np.mean(s['WIDTH_nicholls']):5.3f} | {np.mean(s['DLNL']):+6.2f} | "
        f"{s['TRUE_LOGNO'][0]:8.3f} {np.mean(s['NO_flat']):6.3f} {np.mean(s['NO_nicholls']):6.3f}"
    )
on = m["NO_OFFSET"] == 0
out += ["", "on-relation mocks, all U and Z:"]
for p in PRIORS:
    b = np.asarray(m[f"BIAS_{p}"][on])
    out.append(
        f"  {p:>8}: mean bias {b.mean():+.3f}, rms {np.sqrt(np.mean(b**2)):.3f}, "
        f"68% coverage {np.mean(m[f'COVER_{p}'][on]):.2f}"
    )
slope = {
    oh: 10 ** (oh - CUE_OH_ZERO) / (10**-0.852 + 10 ** (oh - CUE_OH_ZERO)) for oh in (8.4, 8.6, 8.8)
}
out += [
    "",
    "control: shift of recovered Z with the N/O offset, prior fits, against full prior dominance",
]
for oh in (8.4, 8.6, 8.8):
    for off in (-0.3, 0.3):
        s = m[(m["TRUE_OH"] == oh) & (m["NO_OFFSET"] == off)]
        r0 = m[(m["TRUE_OH"] == oh) & (m["NO_OFFSET"] == 0) & (m["TRUE_LOGU"] == -2.5)]
        if len(s) and len(r0):
            shift = np.mean(s["BIAS_nicholls"]) - np.mean(r0["BIAS_nicholls"])
            out.append(
                f"  OH {oh} offset {off:+.1f}: Z shift {shift:+.3f} dex "
                f"(prior-dominated about {off / slope[oh]:+.2f})"
            )
text = "\n".join(out)
print(text)
(OUT / "compare_mocks.txt").write_text(text + "\n")

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, p in zip(ax, PRIORS, strict=True):
    for u, mk in ((-3.0, "o"), (-2.5, "s")):
        s = m[on & (m["TRUE_LOGU"] == u)]
        a.plot(s["TRUE_OH"], s["TRUE_OH"] + s[f"BIAS_{p}"], mk, ms=5, alpha=0.7, label=f"U {u}")
    a.plot([7.9, 9.1], [7.9, 9.1], color="0.5", lw=0.8)
    a.set_xlabel("true 12+log(O/H)")
    a.set_title(f"prior: {p}")
    a.legend(fontsize=8)
ax[0].set_ylabel("recovered 12+log(O/H)")
fig.tight_layout()
fig.savefig(OUT / "compare_mocks.png", dpi=150)
