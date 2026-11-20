"""Where do the model L fits of the five emission-line outliers miss: line kinematics, continuum or noise?

QUESTION: why do gallery105 28, 34 and 80 fit worse under model L (chi2/N 1.42-1.49) than 25 and 64,
and does the refit on the SV3 recoadd (L_recoadd) remove the misfit?
HYPOTHESIS: a line is offset if |dv| > 3 err, its width is off if |sigma ratio - 1| > 3 err; continuum
pixel-scale chi2/N > 1.1 points to ivar or absorption lines, broad excess > 0.1 to continuum shape.
INPUTS: results/2026-11-14_poco_psb_sfh/<TID>/<MODEL>/maxl.npz (data, uncertainty, max-likelihood model).
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-16_emline_L_residuals/analysis.py [L_recoadd]``
(MODEL, default L).
RESULT: 2026-11-20. L_recoadd lowers continuum pixel chi2/N of 28, 34, 80 from 1.20-1.30 to 1.07-1.09,
but the three highest-S/N targets still rank highest, and line cores stay offset (28 Hbeta, [OIII],
Halpha -4.4 to -3.0 km/s; 34 [OIII] +2.1; 80 Halpha, [NII], [SII] +1.6 to +4.0; 25 [OIII] +1.9).
Assumptions: core window from the model line; one line velocity; per-exposure shifts not modelled.
FIGURES: results/2026-11-16_emline_L_residuals/broad_residual.png; tables lines.txt, continuum.txt,
snr.txt. For L_recoadd the same names with the suffix _L_recoadd.
"""

# %%
import sys

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

from hubersed.conversion import to_flambda
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

SRC = PATHS["RESULTS"] / "2026-11-14_poco_psb_sfh"
OUT = PATHS["RESULTS"] / "2026-11-16_emline_L_residuals"
OUT.mkdir(parents=True, exist_ok=True)
# model from the command line; a notebook kernel passes "-f ...", which keeps L
MODEL = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "L"
SUFFIX = "" if MODEL == "L" else f"_{MODEL}"
TARGETS = {  # gallery105 number, from results/2026-11-13_gallery105/gallery105.tsv
    "39627818010675704": "28",
    "39627854832472157": "34",
    "39633318999755849": "80",
    "39627806493117457": "25",
    "39633127609469239": "64",
}
TARGETS = {t: nn for t, nn in TARGETS.items() if (SRC / t / MODEL / "maxl.npz").is_file()}
print(f"model {MODEL}: {', '.join(TARGETS.values())}")
C_KMS = 299792.458
# rest vacuum wavelengths and continuum side bands (A), as the EMLINES of 2026-11-14_poco_psb_sfh
LINES = {
    "Hbeta": (4862.68, ((4830, 4845), (4880, 4895))),
    "[OIII]5007": (5008.24, ((4980, 4995), (5020, 5035))),
    "Halpha": (6564.61, ((6520, 6540), (6600, 6615))),
    "[NII]6585": (6585.27, ((6520, 6540), (6600, 6615))),
    "[SII]6718": (6718.29, ((6690, 6705), (6745, 6760))),
    "[SII]6733": (6732.67, ((6690, 6705), (6745, 6760))),
}
# every line marked in the 2026-11-14 fit figures, masked +-15 A for the continuum test
MASK_LINES = [3727.09, 3729.88, 3869.86, 4102.89, 4341.69, 4364.44, 4862.68, 4960.30, 5008.24,
              5877.25, 6302.05, 6549.86, 6564.61, 6585.27, 6718.29, 6732.67, 7137.77, 7321.94,
              7332.21]  # fmt: skip
MASK_HALF = 15.0
CONT_RANGE = (3700.0, 9000.0)
SNR_RANGE = (4000.0, 4400.0)
W_MEDIAN = 101
CORE = 0.1  # line core = model line above this fraction of its peak, so absorption wings stay out
WINDOWS = {"Halpha+[NII]": (6540, 6595), "[OIII]5007": (4997, 5020)}  # as 2026-11-14 analysis.py


def load(tid):
    """Return rest wavelength, data, model and uncertainty in flux density, and the good mask."""
    d = np.load(SRC / tid / MODEL / "maxl.npz")
    wave = np.asarray(d["wave"], float)
    good = np.asarray(d["good"], bool)

    def fl(x):
        return to_flambda(wave * u.AA, np.asarray(x, float) * u.mgy).value

    unc = np.where(good, fl(np.where(good, d["unc"], 0.0)), np.inf)
    return wave / (1 + float(d["z"])), fl(d["flux"]), fl(d["sp_best"]), unc, good


def side_continuum(rest, f, good, bands):
    """Return a straight line fitted to ``f`` in the two side bands, evaluated at ``rest``."""
    s = good & np.any([(rest >= a) & (rest < b) for a, b in bands], axis=0)
    p = np.polyfit(rest[s], f[s], 1)
    return np.polyval(p, rest)


def moments(lam, w):
    """Return the flux-weighted centroid and second-moment width."""
    mu = np.sum(lam * w) / np.sum(w)
    return mu, np.sqrt(np.sum((lam - mu) ** 2 * w) / np.sum(w))


def core_window(rest, lm, lam0):
    """Return the contiguous pixels around the model line peak where the model line is > CORE of it."""
    near = np.flatnonzero((rest > lam0 - 6) & (rest < lam0 + 6))
    k = near[np.argmax(lm[near])]
    if lm[k] <= 0:
        return None
    lo = hi = k
    while lo - 1 >= near[0] and lm[lo - 1] > CORE * lm[k]:
        lo -= 1
    while hi + 1 <= near[-1] and lm[hi + 1] > CORE * lm[k]:
        hi += 1
    win = np.zeros(rest.size, bool)
    win[lo : hi + 1] = True
    return win


def line_test(rest, data, model, unc, good, lam0, bands):
    """Return dv, its error, the width ratio, its error, the odd chi2 share and pixel count.

    Returns a string with the reason when the line cannot be measured.
    """
    ld = data - side_continuum(rest, data, good, bands)
    lm = model - side_continuum(rest, model, good, bands)
    win = core_window(rest, lm, lam0)
    if win is None:
        return "no model line above the side-band continuum"
    if not good[win].all():
        return "masked pixels in the line core"
    if win.sum() < 3:
        return "line core under 3 pixels"
    mu_m, sig_m = moments(rest[win], lm[win])
    lam, w, e = rest[win], ld[win], unc[win]
    wsum = w.sum()
    mu_d, sig_d = moments(lam, w)
    err_mu = np.sqrt(np.sum(((lam - mu_d) / wsum) ** 2 * e**2))
    err_var = np.sqrt(np.sum((((lam - mu_d) ** 2 - sig_d**2) / wsum) ** 2 * e**2))
    chi = (data[win] - model[win]) / e
    mirror = np.interp(2 * mu_m - lam, lam, chi)
    odd, even = 0.5 * (chi - mirror), 0.5 * (chi + mirror)
    share = np.sum(odd**2) / (np.sum(odd**2) + np.sum(even**2))
    return (C_KMS * (mu_d - mu_m) / mu_m, C_KMS * err_mu / mu_m, sig_d / sig_m,
            err_var / (2 * sig_d) / sig_m, share, int(win.sum()))  # fmt: skip


def continuum_mask(rest, good):
    """Return good pixels in CONT_RANGE outside +-MASK_HALF of every masked line."""
    m = good & (rest >= CONT_RANGE[0]) & (rest < CONT_RANGE[1])
    for lam in MASK_LINES:
        m &= np.abs(rest - lam) > MASK_HALF
    return m


# %% tests 1-3
use_apj_style()
lines_out = [f"Line kinematics at the model {MODEL} max-likelihood point. dv = c (centroid data - model) / lambda,",
             "ratio = data / model second-moment width, both in the line core (model line > 10% of peak);",
             "odd = share of core chi2 in the part of chi odd about the model centroid. Flags beyond 3 err.",
             f"{'nn':>3} {'line':>11} {'dv km/s':>16} {'width ratio':>16} {'odd':>5} {'px':>3} flag"]  # fmt: skip
cont_out = [f"Continuum chi at rest {CONT_RANGE[0]:.0f}-{CONT_RANGE[1]:.0f} A, lines masked +-{MASK_HALF:.0f} A;"
            f" broad = running median over {W_MEDIAN} px.",
            f"{'nn':>3} {'N':>5} {'chi2/N':>7} {'pixel chi2/N':>13} {'broad excess':>13} {'broad rms':>10}"]  # fmt: skip
snr_out = [f"Integrated line S/N in the moment window, median continuum S/N at rest {SNR_RANGE[0]:.0f}-"
           f"{SNR_RANGE[1]:.0f} A, window chi2/N (2026-11-14 windows) and continuum chi2/N.",
           f"{'nn':>3} {'S/N Halpha':>11} {'S/N [OIII]':>11} {'S/N cont':>9} {'chi2/N Ha+[NII]':>16} "
           f"{'chi2/N [OIII]':>14} {'chi2/N cont':>12}"]  # fmt: skip
broad_series = {}
for tid, nn in TARGETS.items():
    rest, data, model, unc, good = load(tid)
    sn = {}
    for name, (lam0, bands) in LINES.items():
        r = line_test(rest, data, model, unc, good, lam0, bands)
        if isinstance(r, str):
            lines_out.append(f"{nn:>3} {name:>11}  {r}")
            continue
        dv, edv, ratio, eratio, share, npx = r
        flag = ",".join(f for f, bad in (("offset", abs(dv) > 3 * edv),
                                           ("width", abs(ratio - 1) > 3 * eratio)) if bad)  # fmt: skip
        lines_out.append(f"{nn:>3} {name:>11} {dv:+8.1f} +- {edv:5.1f} {ratio:7.3f} +- {eratio:5.3f} "
                         f"{share:5.2f} {npx:3d} {flag or '-'}")  # fmt: skip
        if name in ("Halpha", "[OIII]5007"):
            ld = data - side_continuum(rest, data, good, bands)
            win = core_window(rest, model - side_continuum(rest, model, good, bands), lam0)
            sn[name] = ld[win].sum() / np.sqrt(np.sum(unc[win] ** 2))
    m = continuum_mask(rest, good)
    chi = ((data - model) / unc)[m]
    broad = median_filter(chi, size=W_MEDIAN, mode="nearest")
    pix = chi - broad
    tot, pxs = np.mean(chi**2), np.mean(pix**2)
    cont_out.append(f"{nn:>3} {m.sum():5d} {tot:7.3f} {pxs:13.3f} {tot - pxs:13.3f} "
                    f"{np.sqrt(np.mean(broad**2)):10.3f}")  # fmt: skip
    broad_series[tid] = (rest[m], chi, broad)
    s = good & (rest >= SNR_RANGE[0]) & (rest < SNR_RANGE[1])
    for lam in MASK_LINES:
        s &= np.abs(rest - lam) > MASK_HALF
    wchi = {}
    for k, (a, b) in WINDOWS.items():
        w = good & (rest >= a) & (rest < b)
        wchi[k] = np.mean(((data - model) / unc)[w] ** 2)
    snr_out.append(f"{nn:>3} {sn.get('Halpha', np.nan):11.1f} {sn.get('[OIII]5007', np.nan):11.1f} "
                   f"{np.median((data / unc)[s]):9.1f} {wchi['Halpha+[NII]']:16.2f} "
                   f"{wchi['[OIII]5007']:14.2f} {tot:12.3f}")  # fmt: skip
for name, text in (("lines", lines_out), ("continuum", cont_out), ("snr", snr_out)):
    (OUT / f"{name}{SUFFIX}.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text), "\n")


# %% figure: continuum chi per pixel (grey) and its running median (red), lines masked
fig, axes = plt.subplots(len(TARGETS), 1, figsize=(14, 2.2 * len(TARGETS)), sharex=True,
                         squeeze=False, gridspec_kw={"hspace": 0.35})  # fmt: skip
axes = axes[:, 0]
for ax, (tid, nn) in zip(axes, TARGETS.items(), strict=True):
    rest, chi, broad = broad_series[tid]
    ax.plot(rest, chi, ".", color="0.7", ms=1.5)
    ax.plot(rest, broad, color="#b2182b", lw=1.5)
    for h in (-2, 2):
        ax.axhline(h, color="0.3", ls="--", lw=1)
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_ylim(-5, 5)
    ax.set_ylabel(r"$\chi$", fontsize=13)
    ax.set_title(f"{nn}  {tid}", fontsize=12)
axes[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=13)
fig.savefig(OUT / f"broad_residual{SUFFIX}.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / f'broad_residual{SUFFIX}.png'}")
