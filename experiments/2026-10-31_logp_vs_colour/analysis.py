"""Flow log p against spectrum-minus-photometry colour for all DESI S/N > 3 spectra (#17).

QUESTION: does seed 0 continuum-flow log p fall as the spectrum's colour departs from its fiber
photometry, across the whole sample?
HYPOTHESIS: none registered.
INPUTS: stored log p of results/wide_flow_corrected/, data/fastspec-iron-sv3-bright.fits.
dgr = -2.5 log10[(FLUX_SYNTH_G / FIBERTOTFLUX_G) / (FLUX_SYNTH_R / FIBERTOTFLUX_R)], drz likewise.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_logp_vs_colour/analysis.py``.
RESULT: 2026-10-31. Flag rate rises steeply for dgr < 0 only (cont10 0.8% at 0 to 0.05, 15% at
-0.15 to -0.1, 25% at -0.25 to -0.2), in every z x SNR_B cell. dgr < -0.1 is 2.5% of all spectra,
24% of cont10 flags and 39% of cont15 flags.
FIGURES: results/2026-10-31_logp_vs_colour/logp_colour.txt and logp_colour.png.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from scipy.stats import spearmanr

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

OUT = PATHS["RESULTS"] / "2026-10-31_logp_vs_colour"
OUT.mkdir(parents=True, exist_ok=True)
TAGS = ("cont10", "cont15")
EDGES = np.concatenate([[-np.inf], np.round(np.arange(-1.0, 0.5001, 0.05), 2), [np.inf]])

# %% stored log p and the VAC colours
lp, thr = {}, {}
for tag in TAGS:
    o = torch.load(
        PATHS["RESULTS"] / "wide_flow_corrected" / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt",
        weights_only=False,
        map_location="cpu",
    )
    tid = np.asarray(o["desi_target_ids"], np.int64)
    order = np.argsort(tid)
    lp[tag] = dict(tid=tid[order], lp=np.asarray(o["log_p_desi"], float)[order])
    thr[tag] = float(o["threshold"])
assert np.array_equal(lp["cont10"]["tid"], lp["cont15"]["tid"]), "flows score different galaxies"
tid = lp["cont10"]["tid"]

h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)
M, S = h["METADATA"].data, h["FASTSPEC"].data
vt = np.asarray(M["TARGETID"], np.int64)
vo = np.argsort(vt)
pos = np.clip(np.searchsorted(vt[vo], tid), 0, len(vt) - 1)
invac = vt[vo][pos] == tid
row = vo[pos[invac]]
tid = tid[invac]
for tag in TAGS:
    lp[tag] = lp[tag]["lp"][invac]


def colour(b1, b2):
    """Return the spectrum-minus-photometry colour b1 - b2, NaN without positive fluxes."""
    f = {}
    for b in (b1, b2):
        fs = np.asarray(S[f"FLUX_SYNTH_{b}"][row], float)
        ft = np.asarray(M[f"FIBERTOTFLUX_{b}"][row], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            f[b] = np.where((fs > 0) & (ft > 0), fs / ft, np.nan)
    return -2.5 * np.log10(f[b1] / f[b2])


dgr, drz = colour("G", "R"), colour("R", "Z")
z = np.asarray(M["Z"][row], float)
snr = np.asarray(S["SNR_B"][row], float)
ok = np.isfinite(dgr) & np.isfinite(drz)
lines = [
    f"{len(invac)} latent spectra, {int(invac.sum())} in the VAC, {int(ok.sum())} with both colours "
    f"({int((~ok).sum())} dropped: no positive flux).",
    "",
]

# %% per bin: N, median log p, flagged fraction
binned = {}
for tag in TAGS:
    idx = np.digitize(dgr[ok], EDGES) - 1
    rows = []
    for k in range(len(EDGES) - 1):
        sel = idx == k
        n = int(sel.sum())
        f = float(np.mean(lp[tag][ok][sel] <= thr[tag])) if n else np.nan
        med = float(np.median(lp[tag][ok][sel])) if n else np.nan
        rows.append((EDGES[k], EDGES[k + 1], n, med, f, np.sqrt(f * (1 - f) / n) if n else np.nan))
    binned[tag] = rows
lines.append("dgr bin       N   " + "   ".join(f"{t}: med log p, flagged %" for t in TAGS))
for k in range(len(EDGES) - 1):
    lo, hi, n = binned["cont10"][k][:3]
    cells = "   ".join(f"{binned[t][k][3]:8.2f} {100 * binned[t][k][4]:7.2f}" for t in TAGS)
    lines.append(f"[{lo:5.2f},{hi:5.2f}) {n:6d}   {cells}")
lines.append("")

# %% Spearman of log p with |colour|, all and within z x SNR_B quartile cells
for tag in TAGS:
    for name, c in (("|dgr|", np.abs(dgr)), ("|drz|", np.abs(drz))):
        rho, p = spearmanr(lp[tag][ok], c[ok])
        lines.append(f"{tag} log p vs {name}: rho {rho:+.3f} (p {p:.1g}), N {int(ok.sum())}")
lines.append("")
zq = np.quantile(z[ok], [0.25, 0.5, 0.75])
sq = np.quantile(snr[ok], [0.25, 0.5, 0.75])
zi, si = np.digitize(z, zq), np.digitize(snr, sq)
lines.append("rho(log p, |dgr|) within z (rows) x SNR_B (columns) quartile cells")
for tag in TAGS:
    lines.append(f"{tag}:")
    for a in range(4):
        cells = []
        for b in range(4):
            sel = ok & (zi == a) & (si == b)
            cells.append(f"{spearmanr(lp[tag][sel], np.abs(dgr[sel]))[0]:+.3f} ({int(sel.sum())})")
        lines.append("   " + "  ".join(cells))
lines.append("")
for tag in TAGS:
    fl = ok & (lp[tag] <= thr[tag])
    for cut in (-0.1, -0.2):
        lines.append(
            f"{tag}: {int((fl & (dgr < cut)).sum())} of {int(fl.sum())} flagged have dgr < {cut} "
            f"({100 * np.mean(dgr[fl] < cut):.1f}%; {100 * np.mean(dgr[ok] < cut):.1f}% of all)"
        )
lines.append("")
for tag in TAGS:
    rho, p = spearmanr(lp[tag][ok], dgr[ok])
    lines.append(f"{tag} log p vs signed dgr: rho {rho:+.3f} (p {p:.1g})")
lines.append("")
lines.append(
    "flagged % for dgr < -0.1 against |dgr| < 0.05, within z (rows) x SNR_B (columns) cells"
)
for tag in TAGS:
    lines.append(f"{tag}:")
    for a in range(4):
        cells = []
        for b in range(4):
            cell = ok & (zi == a) & (si == b)
            blue, mid = cell & (dgr < -0.1), cell & (np.abs(dgr) < 0.05)
            fb = 100 * np.mean(lp[tag][blue] <= thr[tag]) if blue.any() else np.nan
            fm = 100 * np.mean(lp[tag][mid] <= thr[tag]) if mid.any() else np.nan
            cells.append(f"{fb:5.1f} ({int(blue.sum()):4d}) / {fm:4.2f}")
        lines.append("   " + "  ".join(cells))
text = "\n".join(lines)
(OUT / "logp_colour.txt").write_text(text + "\n")
np.savez(
    OUT / "binned.npz",
    edges=EDGES,
    **{f"{t}_{k}": np.array([r[i] for r in binned[t]]) for t in TAGS
       for i, k in ((2, "n"), (3, "median"), (4, "flagged"), (5, "err"))},
)  # fmt: skip
print(text)

# %% figure
use_apj_style()
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
x = np.clip(dgr[ok], -1.0, 0.5)  # edge bins collect values beyond the range
for j, tag in enumerate(TAGS):
    ax = axes[0, j]
    ax.hexbin(x, lp[tag][ok], gridsize=80, bins="log", cmap="Greys", mincnt=1)
    mid = [0.5 * (max(r[0], -1.0) + min(r[1], 0.5)) for r in binned[tag]]
    ax.plot(mid, [r[3] for r in binned[tag]], "o-", color="#b2182b", ms=4, label="bin median")
    ax.axhline(thr[tag], color="k", ls="--", lw=1.2, label=r"threshold (0.1\% mock)")
    ax.set_ylabel(rf"$\log p$, {tag}latent flow (seed 0)", fontsize=13)
    ax.legend(frameon=False, fontsize=11, loc="lower left")
    ax = axes[1, j]
    f = np.array([r[4] for r in binned[tag]])
    e = np.array([r[5] for r in binned[tag]])
    ax.errorbar(mid, 100 * f, yerr=100 * e, fmt="o-", color="k", ms=4)
    ax.set_ylabel("flagged [\\%]", fontsize=13)
    ax.set_xlabel(r"$\Delta(g-r)$ spectrum $-$ fiber total [mag]", fontsize=13)
fig.savefig(OUT / "logp_colour.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'logp_colour.png'}")
