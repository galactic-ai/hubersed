"""Bias of the Cue gas and dust parameters when the true line profiles differ from our model.

QUESTION: our line model takes each line's total flux from Cue and only splits the profile, with
the same broad fraction, width and shift for every line (forbidden broad shared). If the true
spectrum has a different profile (S1, EmFit's three components), per-line broad fractions (S2)
or a broad component with different line ratios (S3), how biased are gas_logz, gas_logu,
gas_lognH, gas_logno, gas_logco and the dust parameters? Does the broad component reduce or add
bias compared with no broad component at all (F-none)?

HYPOTHESIS: S1 and S2 keep every Cue flux, so the gas parameters should move little and most of
the mismatch should go into the line widths and f_b. S3 changes total forbidden fluxes ([O III]
-7%, [N II] +7%, [O I] +39%), which the model can only absorb through the gas parameters, so S3
should bias gas_logu (from [O III]/[O II]) and gas_logno (from [N II]) at the level of the 10-09
posterior widths or more. F-none should leave larger residuals at the lines and, through the
line windows, bias sigma_smooth and the line widths more than F-shared.

INPUTS: fits/*.npz from fit.py, mocks.npz and validation.txt from mocks.py, the truth point
(map_rmatrix_mask.npz of 2026-10-13_nautilus_resolution_matrix) and the 10-09 nautilus posterior
(the _pred.npz of 2026-10-09_nautilus_c3k_er_broadline, weighted by exp(log_w)) for the
posterior standard deviations.

SEED: noise seeds 1000 + s, s = 0..4 (and the noiseless mock, s = -1); jitter starts 7000 + s.
The same noise seed is used in every scenario, so scenario shifts can be compared seed by seed.

COMMAND: ``uv run --no-sync python experiments/2026-10-13_broadline_injection/analysis.py`` from
the repository root (needs PYTHONPATH with experiments/2026-10-13_nautilus_resolution_matrix, as
fit.py, for the paths in injection.py).

RESULT: partial, 2026-10-13. Only 24 of 216 fits ran (S0, noise seeds -1, 0, 1; seed 1 without
jitter starts) before the run was stopped, so S1 to S3 are not done. In S0 the jittered starts
(5% of each prior range) ended in worse local optima (-ln p 25 to 624 above the best start, with
gas_lognH at its floor), while the truth and max-L starts agreed to within 1.6 in -ln p. F-none on
the noiseless S0 mock gives chi2 +2016 and gas_logz +0.8, gas_logno -0.7, gas_logco -5.4 posterior
sd. See bias_table.txt.

FIGURES: bias_gas.png (gas bias in posterior sd), bias_other.png (dust, logzsol, sigma_smooth,
line parameters), starts.png (-ln p of each start minus the best start).
"""

# %%
import glob

import matplotlib.pyplot as plt
import numpy as np
from injection import FIT_MODELS, OUT, POST_NPZ, SCENARIOS, load_truth

from hubersed.plotting.style import use_apj_style

GAS = ("gas_logz", "gas_logu", "gas_lognH", "gas_logno", "gas_logco")
OTHER = ("dust2", "dust_ratio", "dust_index", "logzsol", "sigma_smooth", "logmass")
LINE = (
    "eline_sigma",
    "eline_fbroad",
    "eline_sigma_broad",
    "eline_vbroad",
    "eline_sigma_forb",
    "eline_fbroad_forb",
)
PARAMS = GAS + OTHER + LINE
STARTS = ("truth", "maxl", "jitter")


def load_fits():
    """Return every fit result as a dict keyed by (scenario, seed, fit model, start)."""
    fits = {}
    for f in sorted(glob.glob(str(OUT / "fits" / "*.npz"))):
        d = np.load(f)
        key = (str(d["scenario"]), int(d["seed"]), str(d["fit_model"]), str(d["start"]))
        fits[key] = {k: d[k] for k in d.files}
        fits[key]["values"] = dict(zip([str(x) for x in d["labels"]], d["theta"], strict=True))
    return fits


def posterior_sd():
    """Return the weighted standard deviation of each parameter of the 10-09 posterior."""
    d = np.load(POST_NPZ, allow_pickle=True)
    w = np.exp(d["log_w"] - d["log_w"].max())
    w /= w.sum()
    pts = d["points"]
    mean = w @ pts
    sd = np.sqrt(w @ (pts - mean) ** 2)
    return dict(zip([str(x) for x in d["labels"]], sd, strict=True))


def best_of_starts(fits, sc, seed, fm):
    """Return the lowest -ln p fit over the starts, and the per-start results."""
    per = {st: fits[(sc, seed, fm, st)] for st in STARTS if (sc, seed, fm, st) in fits}
    if not per:
        return None, per
    return min(per.values(), key=lambda r: float(r["fun"])), per


fits = load_fits()
truth, _ = load_truth()
truth["eline_fbroad_forb"] = truth["eline_fbroad"]
psd = posterior_sd()
seeds = sorted({k[1] for k in fits if k[1] >= 0})
print(f"{len(fits)} fits loaded; noise seeds {seeds}")

# %% bias, scatter, chi2, edges and start agreement per scenario and fit model
rows, summary = [], {}
for sc in SCENARIOS:
    for fm in FIT_MODELS:
        best = {s: best_of_starts(fits, sc, s, fm) for s in [-1, *seeds]}
        noisy = [best[s][0] for s in seeds if best[s][0] is not None]
        if not noisy:
            continue
        nn = best[-1][0]
        rec = {"n": len(noisy)}
        for p in PARAMS:
            if p not in noisy[0]["values"]:
                continue
            d = np.array([r["values"][p] - truth[p] for r in noisy])
            rec[p] = (
                np.median(d),
                np.std(d, ddof=1) if d.size > 1 else np.nan,
                nn["values"][p] - truth[p] if nn is not None else np.nan,
            )
        # chi2 of the fit minus chi2 of the truth model (shared model at the truth) on the same mock
        ref = {s: fits.get((sc, s, "shared", "truth")) for s in [-1, *seeds]}
        dchi = [
            float(best[s][0]["chi2"]) - float(ref[s]["chi2_truth"])
            for s in seeds
            if best[s][0] is not None and ref[s] is not None
        ]
        rec["chi2"] = np.median([float(r["chi2"]) for r in noisy])
        rec["dchi2"] = np.median(dchi) if dchi else np.nan
        rec["dchi2_nonoise"] = (
            float(nn["chi2"]) - float(ref[-1]["chi2_truth"])
            if nn is not None and ref[-1]
            else np.nan
        )
        labels = [str(x) for x in noisy[0]["labels"]]
        edges = {}
        for r in noisy:
            for lab, e in zip(labels, r["edge"], strict=True):
                if e:
                    edges[lab] = edges.get(lab, 0) + 1
        rec["edges"] = edges
        # start agreement: gap in -ln p between the best and worst start, and which start won
        gaps, wins, spread = [], {st: 0 for st in STARTS}, []
        for s in [-1, *seeds]:
            b, per = best[s]
            if b is None or len(per) < 2:
                continue
            funs = {st: float(r["fun"]) for st, r in per.items()}
            wins[min(funs, key=funs.get)] += 1
            gaps.append(max(funs.values()) - min(funs.values()))
            spread.append(
                max(
                    abs(r["values"][p] - b["values"][p]) / psd[p] for r in per.values() for p in GAS
                )
            )
        rec["gap_max"] = max(gaps) if gaps else np.nan
        rec["gap_median"] = np.median(gaps) if gaps else np.nan
        rec["wins"] = wins
        rec["start_spread_gas_sd"] = max(spread) if spread else np.nan
        summary[(sc, fm)] = rec

# %% paired shift: the same seed in scenario Sk minus S0, same fit model
paired = {}
for sc in SCENARIOS[1:]:
    for fm in FIT_MODELS:
        d = {p: [] for p in GAS + OTHER}
        for s in seeds:
            a, b = best_of_starts(fits, sc, s, fm)[0], best_of_starts(fits, "S0", s, fm)[0]
            if a is None or b is None:
                continue
            for p in d:
                d[p].append(a["values"][p] - b["values"][p])
        if d["gas_logz"]:
            paired[(sc, fm)] = {p: np.median(v) for p, v in d.items()}

# %% line fluxes: fitted Cue fluxes against the injected total fluxes
mk = np.load(OUT / "mocks.npz")
mult = dict(zip([str(x) for x in mk["scenarios"]], mk["multipliers"], strict=True))
names = [str(x) for x in mk["line_names"]]
LINE_MEMBERS = {
    "Halpha": ("Ba-alpha 6563",),
    "Hbeta": ("Ba-beta 4861",),
    "[OIII]5007": ("[O III] 5007",),
    "[NII]6584": ("[N II] 6584",),
    "[SII]6716+6731": ("[S II] 6716", "[S II] 6731"),
    "[OII]3726+3729": ("[O II] 3726", "[O II] 3729"),
    "[OI]6300": ("[O I] 6300",),
    "[NeIII]3869": ("[Ne III] 3869",),
}
flux_bias = {}
for sc in SCENARIOS:
    for fm in FIT_MODELS:
        noisy = [best_of_starts(fits, sc, s, fm)[0] for s in seeds]
        noisy = [r for r in noisy if r is not None]
        if not noisy:
            continue
        ln = [str(x) for x in noisy[0]["line_names"]]
        ratio = []
        for r in noisy:
            # true total flux = Cue flux at the truth times the scenario's multiplier
            truth_tot = []
            for j, key in enumerate(ln):
                members = LINE_MEMBERS[key]
                m = np.mean([mult[sc][names.index(n)] for n in members])
                truth_tot.append(float(r["lines_truth"][j]) * m)
            ratio.append(np.log10(np.asarray(r["lines_fit"], float) / np.array(truth_tot)))
        flux_bias[(sc, fm)] = dict(zip(ln, np.median(ratio, axis=0), strict=True))

# %% write the table
out = [
    "Broad-line injection test of 39627770174637084: bias = fitted minus truth.",
    f"Best of the starts {STARTS} per seed (lowest -ln p). Noise seeds {seeds} (n per cell below).",
    "Columns per parameter: median bias over noise seeds, scatter (std over seeds), noiseless-mock bias.",
    "Gas rows also give median bias / 10-09 posterior sd and noiseless bias / sd.",
    "",
]
out.append("10-09 posterior sd: " + ", ".join(f"{p} {psd[p]:.4f}" for p in GAS + OTHER if p in psd))
out.append("truth: " + ", ".join(f"{p} {truth[p]:.4f}" for p in PARAMS))
for (sc, fm), rec in summary.items():
    out += ["", f"== {sc}  F-{fm}  (n = {rec['n']})"]
    for p in PARAMS:
        if p not in rec:
            continue
        med, sd, nn = rec[p]
        extra = f"  [{med / psd[p]:+6.2f} sd, noiseless {nn / psd[p]:+6.2f} sd]" if p in GAS else ""
        out.append(f"   {p:18s} {med:+9.4f} +- {sd:7.4f}   noiseless {nn:+9.4f}{extra}")
    out.append(
        f"   chi2 median {rec['chi2']:.1f}; chi2(fit) - chi2(truth model) median {rec['dchi2']:+.1f},"
        f" noiseless {rec['dchi2_nonoise']:+.2f}"
    )
    out.append(f"   prior-edge hits (fits of {rec['n']}): {rec['edges'] or 'none'}")
    out.append(
        f"   starts: wins {rec['wins']}; -ln p gap best-worst median {rec['gap_median']:.3f}, max"
        f" {rec['gap_max']:.3f}; max gas spread between starts {rec['start_spread_gas_sd']:.3f} sd"
    )
    if (sc, fm) in flux_bias:
        out.append(
            "   log10(fitted Cue flux / injected total flux), median: "
            + ", ".join(f"{k} {v:+.3f}" for k, v in flux_bias[(sc, fm)].items())
        )
    if (sc, fm) in paired:
        out.append(
            "   paired shift vs S0 (same seed, same fit model), in posterior sd: "
            + ", ".join(f"{p} {paired[(sc, fm)][p] / psd[p]:+.2f}" for p in GAS + OTHER)
        )

# compact gas table
out += ["", "Compact: median gas bias / 10-09 posterior sd (noiseless in brackets)"]
out.append(f"{'':14s}" + "".join(f"{p:>20s}" for p in GAS))
for (sc, fm), rec in summary.items():
    out.append(
        f"{sc + ' F-' + fm:14s}"
        + "".join(f"{rec[p][0] / psd[p]:+9.2f} [{rec[p][2] / psd[p]:+7.2f}]" for p in GAS)
    )
text = "\n".join(out) + "\n"
print(text)
(OUT / "bias_table.txt").write_text(text)

# %% figures
use_apj_style()
colors = {"shared": "#b2182b", "free": "#e08214", "none": "#2166ac"}
scs = [sc for sc in SCENARIOS if any((sc, fm) in summary for fm in FIT_MODELS)]


def bias_panel(params, path, in_sd):
    """Plot the median bias with the seed scatter, one panel per parameter."""
    fig, axes = plt.subplots(1, len(params), figsize=(2.6 * len(params), 3.0), sharey=False)
    for ax, p in zip(np.atleast_1d(axes), params, strict=True):
        for j, fm in enumerate(FIT_MODELS):
            for i, sc in enumerate(scs):
                rec = summary.get((sc, fm))
                if rec is None or p not in rec:
                    continue
                med, sd, nn = rec[p]
                scale = psd[p] if in_sd else 1.0
                x = i + (j - 1) * 0.22
                ax.errorbar(
                    x,
                    med / scale,
                    yerr=sd / scale,
                    fmt="o",
                    color=colors[fm],
                    ms=4,
                    label=f"F-{fm}" if i == 0 else None,
                )
                ax.plot(x, nn / scale, "x", color=colors[fm], ms=5)
        ax.axhline(0, color="0.5", lw=0.6)
        if in_sd:
            for h in (-1, 1):
                ax.axhline(h, color="0.7", lw=0.5, ls="--")
        ax.set_xticks(range(len(scs)), scs)
        ax.set_title(p, fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("bias / posterior sd" if in_sd else "bias")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


bias_panel(GAS, OUT / "bias_gas.png", in_sd=True)
bias_panel(OTHER, OUT / "bias_other.png", in_sd=True)
bias_panel(tuple(p for p in LINE if p != "eline_fbroad_forb"), OUT / "bias_lines.png", in_sd=False)

fig, ax = plt.subplots(figsize=(7, 3))
k = 0
ticks = []
for sc in scs:
    for fm in FIT_MODELS:
        for s in [-1, *seeds]:
            b, per = best_of_starts(fits, sc, s, fm)
            if b is None:
                continue
            for st, r in per.items():
                ax.plot(
                    k,
                    max(float(r["fun"]) - float(b["fun"]), 1e-3),
                    "o^s"[STARTS.index(st)],
                    color=colors[fm],
                    ms=3,
                )
        ticks.append((k, f"{sc}\n{fm}"))
        k += 1
ax.set_yscale("log")
ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=5)
ax.set_ylabel(r"$-\ln p$ minus best start (floor 1e-3)")
ax.set_title("circle truth, triangle max-L, square jitter", fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "starts.png", dpi=150)
plt.close(fig)
