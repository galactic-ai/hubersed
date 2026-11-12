"""Strong-line N/O-O/H relation of our parent sample, as a model-free reference.

QUESTION: Do DESI SV3 bright star-forming galaxies follow the N/O-O/H relation, with N/O scatter
tied to M*, as González-Koda+26 find for SDSS (arXiv:2611.09330)? This is the reference that
Cue posteriors under a flat, independent gas_logno prior must reproduce (plan
~/.claude/plans/no_oh_population_test_plan.md, route A).
HYPOTHESIS: none stated before the run.
INPUTS: data/fastspec-iron-sv3-bright.fits (fluxes, LOGMSTAR); prescreen_steps.npz from
2026-11-10_prescreen_replay (step 0 parent, step 1 removed by latent S/N <= 3).
METHOD, as González-Koda+26: S/N > 3 (flux * sqrt(ivar)) for [OII]3726+29, Hbeta, [OIII]5007,
Halpha, [NII]6584; star-forming below Kauffmann+03 eq. 1 (astro-ph/0304239 l.334); Balmer
decrement, case B 2.86, Cardelli R_V = 3.1 (extinction.ccm89), drop A_V < 0. N/O from N2O2 with
Florido+22 eq. fit1 (arXiv:2203.17021 l.265, valid -1.74 < N2O2 < 0.62). O/H from R23 and O32
with M91 as given in Mallery+07 (arXiv:0706.0238 l.456-490, branch rule included). Residual
xi_N/O from a cubic spline through medians in 12 equal O/H bins, above the 1st O/H percentile;
Spearman rho of xi_N/O with the same residual of log M*.
SEED: none.
COMMAND: ``uv run python experiments/2026-11-10_no_oh_reference/analysis.py``.
RESULT: parent 24,125 of 104,990 pass all cuts (32,641 have the five lines at S/N > 3); S/N <= 3
set 34,064 of 140,230. Both follow a plateau (log N/O ~ -1.4 below 12+log(O/H) ~ 8.3) then a
rise to -0.65 at 9.0; xi_N/O std 0.057 (parent), 0.072 dex (S/N <= 3); Spearman(xi_N/O, dlogM*)
0.52 and 0.50 (González-Koda+26: ~0.42). Assumptions: N/O and O/H share [OII] and the branch
choice uses N2O2, so part of the tightness is by construction; fastspecfit fluxes are
Galactic-extinction corrected (Nikhil 11-10); 1.5" fibres at z up to 0.6 cover far more of each
galaxy than the SDSS 3" fibres at z < 0.05.
FIGURES: no_oh_plane.png (N/O vs O/H density, binned medians, parent and S/N <= 3 panels).
"""

# %%
import extinction
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.interpolate import CubicSpline
from scipy.stats import spearmanr

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-10_no_oh_reference"
OUT.mkdir(parents=True, exist_ok=True)
SNR = 3.0
# Air wavelengths [A]; [OII] uses the doublet mean.
WAVE = {
    "oii": 3727.4,
    "hb": 4861.33,
    "o4959": 4958.91,
    "o5007": 5006.84,
    "ha": 6562.80,
    "nii": 6583.45,
}
COLS = {
    "oii": ("OII_3726", "OII_3729"),
    "hb": ("HBETA",),
    "o4959": ("OIII_4959",),
    "o5007": ("OIII_5007",),
    "ha": ("HALPHA",),
    "nii": ("NII_6584",),
}
NBIN = 12

h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)
fs = h["FASTSPEC"].data
tid = np.asarray(fs["TARGETID"], np.int64)
flux, snr = {}, {}
for k, names in COLS.items():
    f = sum(np.asarray(fs[f"{n}_FLUX"], float) for n in names)
    var = sum(
        np.where(
            np.asarray(fs[f"{n}_FLUX_IVAR"], float) > 0,
            1 / np.asarray(fs[f"{n}_FLUX_IVAR"], float),
            np.inf,
        )
        for n in names
    )
    flux[k], snr[k] = f, f / np.sqrt(var)
logm = np.asarray(fs["LOGMSTAR"], float)

st = np.load(PATHS["RESULTS"] / "2026-11-10_prescreen_replay" / "prescreen_steps.npz")
step = dict(zip(st["tid"].tolist(), st["step"].tolist(), strict=True))
samp = np.array([step.get(int(t), -1) for t in tid])
lines = [f"fastspecfit rows {tid.size}; parent {np.sum(samp == 0)}; S/N<=3 {np.sum(samp == 1)}"]

# %% cuts
det = np.all([snr[k] > SNR for k in ("oii", "hb", "o5007", "ha", "nii")], axis=0)
with np.errstate(divide="ignore", invalid="ignore"):
    x_bpt = np.log10(flux["nii"] / flux["ha"])
    y_bpt = np.log10(flux["o5007"] / flux["hb"])
    sf = det & (x_bpt < 0.05) & (y_bpt < 0.61 / (x_bpt - 0.05) + 1.3)
    k = {n: extinction.ccm89(np.array([w]), 1.0, 3.1)[0] for n, w in WAVE.items()}
    av = 2.5 / (k["hb"] - k["ha"]) * np.log10(flux["ha"] / flux["hb"] / 2.86)
    good = sf & np.isfinite(av) & (av >= 0)
    c = {n: flux[n] * 10 ** (0.4 * av * k[n]) for n in WAVE}

    # %% abundances
    n2o2 = np.log10(c["nii"] / c["oii"])
    logno = -0.102 * n2o2**2 + 0.528 * n2o2 - 0.634
    x = np.log10((c["oii"] + c["o4959"] + c["o5007"]) / c["hb"])
    y = np.log10((c["o4959"] + c["o5007"]) / c["oii"])
low = 7.065 + 0.767 * x + 0.602 * x**2 - y * (0.29 + 0.332 * x - 0.3318 * x**2)
up = (
    9.061
    - 0.2 * x
    - 0.237 * x**2
    - 0.305 * x**3
    - 0.0283 * x**4
    - y * (0.0047 - 0.0221 * x - 0.102 * x**2 - 0.0817 * x**3 - 0.00717 * x**4)
)
lnh, lno = x_bpt, n2o2
# Mallery+07: [NII]/[OII] decides outside (-1.5, -0.8), [NII]/Halpha inside; average on conflict.
by_nh = np.where(lnh < -1, low, up)
oh = np.where(
    lno < -1.5,
    np.where(lnh < -1, low, 0.5 * (low + up)),
    np.where(lno > -0.8, np.where(lnh > -1, up, 0.5 * (low + up)), by_nh),
)
valid = good & (n2o2 > -1.74) & (n2o2 < 0.62) & np.isfinite(oh) & np.isfinite(logm)

# %% per sample: counts, binned medians, spline residuals, Spearman with M*
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
for ax, (s, name) in zip(axes, ((0, "parent"), (1, "S/N <= 3")), strict=True):
    m = samp == s
    lines.append(
        f"{name}: N {m.sum()}, lines S/N>3 {np.sum(m & det)}, SF {np.sum(m & sf)}, "
        f"A_V>=0 {np.sum(m & good)}, N2O2 in range {np.sum(m & valid)}"
    )
    v = m & valid
    o, n, lm = oh[v], logno[v], logm[v]
    lo = np.percentile(o, 1)
    edges = np.linspace(lo, o.max(), NBIN + 1)
    idx = np.digitize(o, edges[1:-1])
    med = np.array(
        [
            [np.median(o[idx == i]), np.median(n[idx == i]), np.median(lm[idx == i])]
            for i in range(NBIN)
            if np.sum(idx == i) >= 20
        ]
    )
    sp_n, sp_m = CubicSpline(med[:, 0], med[:, 1]), CubicSpline(med[:, 0], med[:, 2])
    inr = (o >= med[0, 0]) & (o <= med[-1, 0])
    xi, dm = n[inr] - sp_n(o[inr]), lm[inr] - sp_m(o[inr])
    rho = spearmanr(xi, dm).statistic
    lines.append(
        f"  12+log(O/H) p1 {lo:.2f}, p50 {np.median(o):.2f}, p99 {np.percentile(o, 99):.2f}"
    )
    lines.append(
        "  binned medians (O/H, log N/O): "
        + ", ".join(f"({a:.2f}, {b:.2f})" for a, b in med[:, :2])
    )
    lines.append(
        f"  xi_N/O std {np.std(xi):.3f} dex; Spearman(xi_N/O, dlogM*) {rho:.2f} on {inr.sum()}"
    )
    ax.hexbin(o, n, gridsize=60, bins="log", mincnt=1, cmap="Greys")
    ax.plot(med[:, 0], med[:, 1], "o-", color="C3", ms=4)
    ax.set_title(f"{name} (N = {v.sum()})")
    ax.set_xlabel("12 + log(O/H)  [M91, R23-O32]")
axes[0].set_ylabel("log(N/O)  [F22, N2O2]")
fig.tight_layout()
fig.savefig(OUT / "no_oh_plane.png", dpi=150)

text = "\n".join(lines)
print(text)
(OUT / "analysis.txt").write_text(text + "\n")
