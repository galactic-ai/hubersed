"""Are selected gallery105 figures AGN or PSB? Their DR1 coadds, line windows, MgII, Halpha and photometry.

Targets (TARGETS): 16, 18, 22, 41, 42, 67, 74, 77, 78, 81 (coadd comparison only), 82, 86, 89, 96 and 99 (coadd comparison), 102 (no SPARCL coadds, screened out) and 104.

QUESTION: 16 (39627788365337938) has a featureless continuum with no break, Hbeta emission and no
[OII]; 18 (39627788382109772) has a possibly broad MgII (FastSpecFit sigma 1322 km/s, S/N 4.0), narrow
Hbeta and [OIII] and almost no [OII]. Do all their DR1 coadds agree in continuum and lines, is the
MgII of 18 broad in each coadd, and what do the optical and WISE colours say?
HYPOTHESIS: none registered. Descriptive.
INPUTS: results/2026-11-10_data_quality_screen/sparcl_spectra.npz (all DR1 coadds of each target,
SPARCL) and repeats.csv (number of coadds the flows flag; which coadd is not saved);
data/fastspec-iron-sv3-bright.fits (METADATA photometry, FASTSPEC fit values).
METHOD: good pixels have mask 0 and ivar > 0. Each coadd is put in its own rest frame and divided
by its median over rest 4200-4600 A. The ratio panel is each coadd over the deepest (largest
COADD_NUMEXP, then TSNR2_BGS), 51 px median filter. MgII fit on each coadd over rest 2700-2900 A:
linear continuum plus two Gaussians at 2796.35 and 2803.53 A (vacuum, from memory) with one shared
velocity and sigma and free amplitudes, weighted by ivar (scipy curve_fit). sigma is not corrected
for the instrument (about 50-70 km/s, small next to the widths tested). Fe II emission is not
modelled. Broad Halpha (figures 41, 77, 78 added 2026-11-13): on each coadd over rest 6480-6650 A, a
linear continuum with narrow Halpha and [NII] 6548/6584 (ratio 1/3, one velocity and sigma <= 400 km/s)
is fit with and without one broad Halpha Gaussian (sigma 400-5000 km/s); dBIC = dchi2 - 3 ln N, positive
favours the broad line. Line wavelengths and the 1/3 ratio are from memory. Colours are AB from the Legacy Surveys fluxes (nanomaggies). W1-W2 in Vega adds
3.339 - 2.699 = 0.640 (m_AB = m_Vega + dm, WISE All-Sky Explanatory Supplement IV.4.h Table 3, the
offsets DR9 uses), for the Stern et al. (2012, arXiv:1205.0811) AGN cut W1-W2 >= 0.8 (Vega), set at
W2 <~ 15.05 (Vega, 10 sigma) in COSMOS; it picks AGN-dominated sources, AGN fraction >~ 80% (their
Fig. 2 caption, wise_cosmos.tex l.262-264).
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-13_agn_check_16_18/analysis.py``.
RESULT: pending.
FIGURES: results/2026-11-13_agn_check_16_18/{coadds,zoom}_<nn>.png; table agn_check.txt.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import medfilt

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

RES = PATHS["RESULTS"]
OUT = RES / "2026-11-13_agn_check_16_18"
TARGETS = {
    16: 39627788365337938,
    18: 39627788382109772,
    22: 39627800440737892,
    42: 39628357138121066,
    67: 39633136522363591,
    74: 39633158257249026,
    41: 39628351614225514,
    77: 39633282689666443,
    78: 39633290038083787,
    81: 39633322472637297,
    82: 39633329376461212,
    86: 39633402046974648,
    89: 39633407818337381,
    96: 39633419155538001,
    99: 39633421919584782,
    102: 39633425279222058,
    104: 39633458753965639,
}
NORM = (4200, 4600)
WINDOWS = {
    "MgII": (2700, 2900),
    "break and [OII]": (3600, 4200),
    "Hdelta and Hgamma": (4050, 4400),
    "Hbeta and [OIII]": (4800, 5050),
    "Halpha and [NII]": (6480, 6650),
}
MGII = (2796.35, 2803.53)
HA, N2A, N2B = 6564.61, 6549.86, 6585.27  # vacuum rest A, from memory
W12_VEGA_MINUS_AB = 3.339 - 2.699
C_KMS = 299792.458
N_FLUX, N_RATIO = 9, 51
YLABEL = r"$f_\lambda$ / median(4200-4600 \AA)"


def coadds(sp, tid):
    """Return the coadds of one target, deepest first, normalised in their own rest frame."""
    out = []
    for k in np.flatnonzero(sp["targetid"] == tid):
        z = float(sp["redshift"][k])
        rest = sp["wavelength"] / (1 + z)
        good = (sp["mask"][k] == 0) & (sp["ivar"][k] > 0)
        sel = good & (rest >= NORM[0]) & (rest <= NORM[1])
        norm = float(np.median(sp["flux"][k][sel]))
        out.append(dict(
            label=f"{sp['survey'][k]} {sp['program'][k]}, {sp['coadd_numexp'][k]} exp, "
                  f"dchi2 {sp['deltachi2'][k]:.0f}",
            z=z, rest=rest, good=good, flux=sp["flux"][k], ivar=sp["ivar"][k], norm=norm,
            nexp=int(sp["coadd_numexp"][k]), tsnr=float(sp["tsnr2_bgs"][k]),
        ))  # fmt: skip
    return sorted(out, key=lambda c: (-c["nexp"], -c["tsnr"]))


def mgii_model(w, c0, c1, a1, a2, v, sig):
    """Linear continuum plus the MgII doublet with one velocity (km/s) and sigma (km/s)."""
    m = c0 + c1 * (w - 2800)
    for lam, a in zip(MGII, (a1, a2), strict=True):
        mu = lam * (1 + v / C_KMS)
        s = lam * sig / C_KMS
        m = m + a * np.exp(-0.5 * ((w - mu) / s) ** 2)
    return m


def fit_mgii(c):
    """Return sigma, its error and the doublet flux S/N of one coadd, or NaNs if the fit fails."""
    sel = c["good"] & (c["rest"] >= WINDOWS["MgII"][0]) & (c["rest"] <= WINDOWS["MgII"][1])
    if sel.sum() < 50:
        return np.nan, np.nan, np.nan
    w, f, e = c["rest"][sel], c["flux"][sel], 1 / np.sqrt(c["ivar"][sel])
    p0 = [np.median(f), 0.0, f.max() - np.median(f), f.max() - np.median(f), 0.0, 1000.0]
    lo = [-np.inf, -np.inf, 0, 0, -1500, 50]
    hi = [np.inf, np.inf, np.inf, np.inf, 1500, 8000]
    try:
        p, cov = curve_fit(mgii_model, w, f, p0=p0, sigma=e, absolute_sigma=True, bounds=(lo, hi))
    except RuntimeError:
        return np.nan, np.nan, np.nan
    s = np.sqrt(np.diag(cov))
    # Doublet flux in rest-frame units: sum of a_i * sqrt(2 pi) * sigma_lambda_i.
    sl = np.array(MGII) * p[5] / C_KMS
    flux = np.sqrt(2 * np.pi) * (p[2] * sl[0] + p[3] * sl[1])
    jac = np.sqrt(2 * np.pi) * np.array([sl[0], sl[1]])
    var = jac @ cov[2:4, 2:4] @ jac
    return float(p[5]), float(s[5]), float(flux / np.sqrt(var)) if var > 0 else np.nan


def ha_model(w, c0, c1, a_ha, a_n2, v, sig, a_b=0.0, v_b=0.0, sig_b=1000.0):
    """Linear continuum, narrow Halpha and [NII] (6548 = 6584 / 3) and an optional broad Halpha."""
    m = c0 + c1 * (w - HA)
    for lam, a in ((HA, a_ha), (N2B, a_n2), (N2A, a_n2 / 3)):
        mu, s = lam * (1 + v / C_KMS), lam * sig / C_KMS
        m = m + a * np.exp(-0.5 * ((w - mu) / s) ** 2)
    mu, s = HA * (1 + v_b / C_KMS), HA * sig_b / C_KMS
    return m + a_b * np.exp(-0.5 * ((w - mu) / s) ** 2)


def fit_halpha(c):
    """Return the broad Halpha sigma, its flux S/N, delta chi2 and delta BIC (narrow minus broad)."""
    lo_w, hi_w = WINDOWS["Halpha and [NII]"]
    sel = c["good"] & (c["rest"] >= lo_w) & (c["rest"] <= hi_w)
    if sel.sum() < 80:
        return (np.nan,) * 4
    w, f, e = c["rest"][sel], c["flux"][sel], 1 / np.sqrt(c["ivar"][sel])
    a0 = max(f.max() - np.median(f), 1e-3)
    p0 = [np.median(f), 0.0, a0, a0 / 3, 0.0, 100.0]
    lo = [-np.inf, -np.inf, 0, 0, -500, 20]
    hi = [np.inf, np.inf, np.inf, np.inf, 500, 400]
    try:
        pn, _ = curve_fit(ha_model, w, f, p0=p0, sigma=e, absolute_sigma=True, bounds=(lo, hi))
        pb, cov = curve_fit(ha_model, w, f, p0=list(pn) + [a0 / 5, 0.0, 1000.0], sigma=e,
                            absolute_sigma=True,
                            bounds=(lo + [0, -2000, 400], hi + [np.inf, 2000, 5000]))  # fmt: skip
    except RuntimeError:
        return (np.nan,) * 4
    chi_n = float(np.sum(((f - ha_model(w, *pn)) / e) ** 2))
    chi_b = float(np.sum(((f - ha_model(w, *pb)) / e) ** 2))
    sl = HA * pb[8] / C_KMS
    flux = np.sqrt(2 * np.pi) * pb[6] * sl
    # Error on the broad flux from the amplitude and width terms of the covariance.
    jac = np.array([np.sqrt(2 * np.pi) * sl, np.sqrt(2 * np.pi) * pb[6] * HA / C_KMS])
    var = jac @ cov[np.ix_([6, 8], [6, 8])] @ jac
    sn = float(flux / np.sqrt(var)) if var > 0 else np.nan
    dbic = chi_n - chi_b - 3 * np.log(w.size)
    return float(pb[8]), sn, chi_n - chi_b, float(dbic)


def mag(f):
    """Return the AB magnitude of a flux in nanomaggies, NaN when not positive."""
    return 22.5 - 2.5 * np.log10(f) if f > 0 else np.nan


def main():
    """Write the table and the figures."""
    OUT.mkdir(parents=True, exist_ok=True)
    sp = np.load(RES / "2026-11-10_data_quality_screen" / "sparcl_spectra.npz")
    rep = {}
    with open(RES / "2026-11-10_data_quality_screen" / "repeats.csv") as f:
        for r in csv.DictReader(f):
            if r["flows"] == "new":
                rep[int(r["target_id"])] = r
    with fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits") as h:
        fs, md = h["FASTSPEC"].data, h["METADATA"].data
        row = {int(t): i for i, t in enumerate(md["TARGETID"])}
        lines = []
        use_apj_style()
        for nn, tid in TARGETS.items():
            cs = coadds(sp, tid)
            i = row[tid]
            # Targets removed by the data-quality screen have no repeats row and no SPARCL coadds.
            r = rep.get(tid, {"n_flagged": "-", "n_usable": "-", "state": "not in the repeat test"})
            g, rr, zz, w1, w2 = (
                mag(float(md[f"FLUX_{b}"][i])) for b in ("G", "R", "Z", "W1", "W2")
            )
            lines += [
                f"figure {nn}, {tid}: {len(cs)} DR1 coadds; flows flag {r['n_flagged']} of "
                f"{r['n_usable']} usable repeats (state {r['state']}).",
                f"  AB colours: g-r {g - rr:.2f}, r-z {rr - zz:.2f}, z-W1 {zz - w1:.2f}, "
                f"W1-W2 {w1 - w2:.2f} (W1 {w1:.2f}, W2 {w2:.2f}); Vega W1-W2 "
                f"{w1 - w2 + W12_VEGA_MINUS_AB:.2f}, W2 {w2 - 3.339:.2f} (Stern+12 AGN: >= 0.8).",
                "  FastSpecFit: "
                + ", ".join(
                    f"{k} {float(fs[k][i]):.3g}"
                    for k in ("AGE", "AV", "ZZSUN", "VDISP", "LOGMSTAR", "SFR", "RCHI2_CONT")
                ),
                "  MgII doublet fit per coadd: sigma [km/s] +- error, doublet flux S/N",
            ]
            for c in cs:
                s, es, sn = fit_mgii(c)
                lines.append(
                    f"    {c['label']:>40}  z {c['z']:.4f}  sigma {s:7.0f} +- {es:5.0f}  S/N {sn:5.1f}"
                )
            lines.append(
                "  broad Halpha fit per coadd: sigma_b [km/s], broad flux S/N, dchi2, dBIC"
            )
            for c in cs:
                sb, snb, dchi, dbic = fit_halpha(c)
                lines.append(
                    f"    {c['label']:>40}  sigma_b {sb:6.0f}  S/N {snb:5.1f}  dchi2 {dchi:7.1f}"
                    f"  dBIC {dbic:7.1f}"
                )
            lines.append("")
            if not cs:
                continue

            # Coadds overlaid and their ratio to the deepest.
            fig, ax = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})  # fmt: skip
            ref = cs[0]
            for j, c in enumerate(cs):
                col = "k" if j == 0 else plt.cm.tab10(j)
                ax[0].plot(c["rest"][c["good"]], medfilt(c["flux"][c["good"]] / c["norm"], N_FLUX),
                           color=col, lw=0.7, label=c["label"])  # fmt: skip
                if j:
                    both = c["good"] & ref["good"]
                    rat = (c["flux"] / c["norm"]) / np.interp(
                        c["rest"], ref["rest"], ref["flux"] / ref["norm"]
                    )
                    ax[1].plot(c["rest"][both], medfilt(rat[both], N_RATIO), color=col, lw=0.7)
            ax[1].axhline(1, color="0.3", lw=0.6)
            ax[0].set_ylabel(YLABEL, fontsize=13)
            ax[1].set_ylabel(f"over deepest, {N_RATIO} px", fontsize=12)
            ax[1].set_xlabel(r"rest wavelength [\AA]", fontsize=13)
            ax[0].legend(frameon=False, fontsize=9, loc="upper right")
            ax[0].set_title(f"figure {nn}   {tid}", fontsize=13)
            fig.savefig(OUT / f"coadds_{nn}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)

            # Line windows, unsmoothed.
            fig, axs = plt.subplots(1, len(WINDOWS), figsize=(18, 4.6))
            for ax1, (name, (lo, hi)) in zip(axs, WINDOWS.items(), strict=True):
                for j, c in enumerate(cs):
                    sel = c["good"] & (c["rest"] >= lo) & (c["rest"] <= hi)
                    ax1.plot(c["rest"][sel], c["flux"][sel] / c["norm"],
                             color="k" if j == 0 else plt.cm.tab10(j), lw=0.9 if j == 0 else 0.5)  # fmt: skip
                ax1.set_title(name, fontsize=12)
                ax1.set_xlabel(r"rest wavelength [\AA]", fontsize=11)
            axs[0].set_ylabel(YLABEL, fontsize=11)
            fig.savefig(OUT / f"zoom_{nn}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
    text = "\n".join(lines)
    (OUT / "agn_check.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
