"""Check where the continuum chi2 excess of the ER maximum likelihood spectrum comes from.

QUESTION
    The continuum pixels of the ER fit have chi2 6199 over 5259 pixels. Is the excess from a
    uniform noise underestimate, sky or telluric residuals, a few outlier pixels, correlated
    residuals, or model mismatch at stellar absorption features?

INPUTS
    results/2026-10-09_nautilus_c3k_er_broadline/
    39627770174637084_er_broad_shared_split45_rest9000_seed0_pred.npz from predict.py, and
    39627770174637084/jacobian.npz from jacobian.py, whose region array marks the continuum
    pixels. No model is evaluated here.

    The telluric band edges are approximate and were written from memory. The night sky line
    wavelengths are air values converted to vacuum. They match the vacuum values in spender's
    sky-lines.txt. spender already masks these sky lines in the data, so no continuum pixel
    falls near them.

    Some continuum pixels sit on emission lines that the model line list (eline_wave in
    jacobian.npz) does not have, such as Pa11 to Pa16, H11, H12 and He I 7281. Their air
    wavelengths were written from memory. The budget is given twice. Order A is the one asked
    for. Order B takes these lines out first and uses |chi| > 4 instead of the worst 1%, because
    the worst 1% of unit Gaussian noise already carries a large excess.

SEED
    None. Nothing is random.

COMMAND
    uv run python experiments/2026-10-09_nautilus_c3k_er_broadline/continuum_check.py

OUTPUT
    results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/continuum_check.txt with the
    tables and the excess budget, and continuum_check.png with chi against observed wavelength
    and the chi histogram against a unit Gaussian.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from hubersed.paths import PATHS

TID = 39627770174637084
OUT = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline"
PRED = OUT / f"{TID}_er_broad_shared_split45_rest9000_seed0_pred.npz"
JAC = OUT / str(TID) / "jacobian.npz"

# observed frame, Angstrom; approximate edges written from memory
TELLURIC = {
    "O2 B": (6860, 6960),
    "H2O 7160-7340": (7160, 7340),
    "O2 A": (7590, 7700),
    "H2O 8130-8350": (8130, 8350),
    "H2O 8950-9250": (8950, 9250),
}
# bright night sky lines, air wavelengths in Angstrom
SKY_AIR = [5577.339, 5889.950, 5895.924, 6300.304, 6363.776]
SKY_HALF = 3.0
OH_MIN = 7200.0
# rest frame features, air wavelengths in Angstrom, each widened by FEAT_HALF
FEAT_HALF = 10.0
FEATURES_AIR = {
    "Ca II K": [3933.66],
    "Ca II H": [3968.47],
    "G band": [4300.0],
    "Mg b": [5167.32, 5172.68, 5183.60],
    "Na D": [5889.95, 5895.92],
    "Ca T": [8498.02, 8542.09, 8662.14],
}
# rest frame vacuum Balmer wings, 10 A on each side of the jacobian.py windows
BALMER_WINGS = {
    "Hdelta wings": [(4085, 4095), (4110, 4120)],
    "Hgamma wings": [(4323, 4333), (4349, 4359)],
    "Hbeta wings": [(4835, 4845), (4880, 4890)],
}
BIN = 250.0
# rest frame emission lines that are absent from the model line list (model._eline_wave in
# jacobian.npz) and fall on continuum pixels; air wavelengths in Angstrom, from memory
UNMODELLED_AIR = {
    "H12": 3750.15,
    "H11": 3770.63,
    "HeI 7281": 7281.35,
    "Pa16": 8502.48,
    "Pa15": 8545.38,
    "Pa14": 8598.39,
    "Pa13": 8665.02,
    "Pa12": 8750.47,
    "Pa11": 8862.78,
}
LINE_HALF = 4.0


def air_to_vac(wave_air):
    """Convert air wavelengths in Angstrom to vacuum with the Morton (2000) formula."""
    wave_air = np.asarray(wave_air, dtype=float)
    s2 = (1e4 / wave_air) ** 2
    n = 1 + 8.336624212083e-5 + 2.408926869968e-2 / (130.1065924522 - s2)
    n += 1.599740894897e-4 / (38.92568793293 - s2)
    return wave_air * n


def gauss_tail_excess(frac):
    """Mean of chi^2 - 1 over the worst ``frac`` of |chi| for unit Gaussian noise."""
    lo, hi = 0.0, 10.0
    for _ in range(100):
        c = 0.5 * (lo + hi)
        lo, hi = (c, hi) if math.erfc(c / math.sqrt(2)) > frac else (lo, c)
    return 2 * c * math.exp(-0.5 * c**2) / math.sqrt(2 * math.pi) / frac


def in_ranges(x, ranges):
    """Return True where x falls in any of the (lo, hi) ranges."""
    sel = np.zeros(x.size, dtype=bool)
    for lo, hi in ranges:
        sel |= (x >= lo) & (x < hi)
    return sel


def stats(chi, sel):
    """Return pixel count, chi2, chi2 per pixel and excess chi2 of the selected pixels."""
    n = int(sel.sum())
    c2 = float((chi[sel] ** 2).sum())
    return n, c2, c2 / max(n, 1), c2 - n


def autocorr(chi_full, cont_full, lag):
    """Lag autocorrelation of chi over pixel pairs that are both continuum pixels."""
    a = cont_full[:-lag] & cont_full[lag:]
    x, y = chi_full[:-lag][a], chi_full[lag:][a]
    raw = np.sum(x * y) / np.sqrt(np.sum(x**2) * np.sum(y**2))
    return a.sum(), raw, np.corrcoef(x, y)[0, 1]


def main():
    p = np.load(PRED)
    j = np.load(JAC)
    wave, flux, unc, good, z = (
        p["wave"].astype(float),
        p["flux"],
        p["unc"],
        p["good"],
        float(p["z"]),
    )
    region = j["region"]
    assert np.allclose(wave[good], j["wave"])

    cont_full = np.zeros(wave.size, dtype=bool)
    cont_full[np.flatnonzero(good)[region == "continuum"]] = True
    chi_full = np.zeros(wave.size)
    chi_full[cont_full] = (flux - p["sp_best"])[cont_full] / unc[cont_full]

    w = wave[cont_full]
    rest = w / (1 + z)
    chi = chi_full[cont_full]
    snr = flux[cont_full] / unc[cont_full]
    n_tot, c2_tot, _, ex_tot = stats(chi, np.ones(chi.size, dtype=bool))
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out(f"TARGETID {TID}, z = {z:.6f}, ER fit maximum likelihood spectrum, continuum pixels")
    out(f"pixels {n_tot}, chi2 {c2_tot:.1f}, chi2/px {c2_tot / n_tot:.4f}, excess {ex_tot:+.1f}")
    out(f"pure noise sd of the excess sqrt(2N) = {np.sqrt(2 * n_tot):.1f}")
    out()

    # 1. observed frame bins
    out("1. chi2/px in observed frame bins of 250 A")
    out(
        f"{'obs lo':>7} {'obs hi':>7} {'rest lo':>7} {'N':>5} {'chi2/px':>8} {'excess':>8} {'mean chi':>9}"
    )
    edges = np.arange(3600.0, wave[good].max() + BIN, BIN)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (w >= lo) & (w < hi)
        if sel.sum() == 0:
            continue
        n, _, cpp, ex = stats(chi, sel)
        out(
            f"{lo:7.0f} {hi:7.0f} {lo / (1 + z):7.0f} {n:5d} {cpp:8.3f} {ex:+8.1f} {chi[sel].mean():+9.3f}"
        )
    out()

    # 2. outliers and robust scale
    out("2. outliers and robust scale")
    for t, gauss in [(3, 2.7e-3), (5, 5.7e-7)]:
        f = np.mean(np.abs(chi) > t)
        out(
            f"fraction |chi| > {t}: {f:.4f} ({int((np.abs(chi) > t).sum())} px), Gaussian {gauss:.1e}"
        )
    order = np.argsort(-np.abs(chi))
    for frac in (0.01, 0.05):
        k = int(round(frac * n_tot))
        ex = float((chi[order[:k]] ** 2 - 1).sum())
        out(f"worst {frac:.0%} ({k} px): excess {ex:+.1f} = {ex / ex_tot:.1%} of the total excess")
        out(f"    unit Gaussian noise gives {k * gauss_tail_excess(frac):+.1f} for the same pixels")
    mad = 1.4826 * np.median(np.abs(chi - np.median(chi)))
    out(
        f"median chi {np.median(chi):+.4f}, mean chi {chi.mean():+.4f}, rms chi {np.sqrt(np.mean(chi**2)):.4f}"
    )
    out(f"robust scale 1.4826 MAD(chi) = {mad:.4f}, so robust chi2/px = {mad**2:.4f}")
    sel5 = np.abs(chi) <= 5
    n, _, cpp, ex = stats(chi, sel5)
    out(f"after removing |chi| > 5: N {n}, chi2/px {cpp:.4f}, excess {ex:+.1f}")
    out()

    # 3. telluric
    out("3. telluric bands (observed frame, approximate edges from memory)")
    tell = np.zeros(chi.size, dtype=bool)
    for name, (lo, hi) in TELLURIC.items():
        sel = (w >= lo) & (w < hi)
        tell |= sel
        n, _, cpp, ex = stats(chi, sel)
        out(f"{name:>14}: N {n:4d}, chi2/px {cpp:.3f}, excess {ex:+7.1f}")
    n, _, cpp, ex = stats(chi, tell)
    out(f"{'all telluric':>14}: N {n:4d}, chi2/px {cpp:.3f}, excess {ex:+7.1f} = {ex / ex_tot:.1%}")
    out()

    # 4. sky lines
    sky_vac = air_to_vac(SKY_AIR)
    out(
        "4. night sky lines, air to vacuum: "
        + ", ".join(f"{a:.1f}->{v:.2f}" for a, v in zip(SKY_AIR, sky_vac, strict=True))
    )
    sky = np.zeros(chi.size, dtype=bool)
    fit_range = (wave / (1 + z) >= 3320) & (wave / (1 + z) < 9000)
    for v in sky_vac:
        sel = np.abs(w - v) <= SKY_HALF
        sky |= sel
        near = np.abs(wave - v) <= SKY_HALF
        n_masked = int((near & fit_range & ~good).sum())
        n, _, cpp, ex = stats(chi, sel)
        out(
            f"{v:8.2f} +/- {SKY_HALF:g} A: continuum N {n}, masked by spender {n_masked}, excess {ex:+.1f}"
        )
    n, _, cpp, ex = stats(chi, sky)
    out(f"all listed sky lines: N {n}, excess {ex:+.1f}")
    oh = w > OH_MIN
    n, _, cpp, ex = stats(chi, oh)
    out(
        f"OH region obs > {OH_MIN:g} A: N {n}, chi2/px {cpp:.3f}, excess {ex:+.1f} = {ex / ex_tot:.1%}"
    )
    n, _, cpp, ex = stats(chi, oh & ~tell)
    out(f"OH region outside telluric bands: N {n}, chi2/px {cpp:.3f}, excess {ex:+.1f}")
    n, _, cpp, ex = stats(chi, ~oh)
    out(f"obs < {OH_MIN:g} A: N {n}, chi2/px {cpp:.3f}, excess {ex:+.1f}")
    n_spender = int((fit_range & ~good).sum())
    out(f"pixels in the fitted rest range already dropped by the spender mask: {n_spender}")
    out("no DESI sky model per pixel was found in data/ or src/hubersed/io/, so no sky correlation")
    out()

    # 5. stellar features
    out(f"5. stellar absorption features, rest +/- {FEAT_HALF:g} A (vacuum), continuum pixels only")
    feats = {
        k: [(c - FEAT_HALF, c + FEAT_HALF) for c in air_to_vac(v)] for k, v in FEATURES_AIR.items()
    }
    feats["Mg b"] = [(air_to_vac(5167.32) - FEAT_HALF, air_to_vac(5183.60) + FEAT_HALF)]
    feats.update(BALMER_WINGS)
    absf = np.zeros(chi.size, dtype=bool)
    for name, ranges in feats.items():
        sel = in_ranges(rest, ranges)
        absf |= sel
        n, _, cpp, ex = stats(chi, sel)
        out(
            f"{name:>13}: N {n:3d}, chi2/px {cpp:.3f}, excess {ex:+6.1f}, mean chi {chi[sel].mean():+.3f}"
        )
    n, _, cpp, ex = stats(chi, absf)
    n2, _, cpp2, _ = stats(chi, ~absf)
    out(f"all features: N {n}, chi2/px {cpp:.3f}, excess {ex:+.1f}; elsewhere chi2/px {cpp2:.3f}")
    out(f"feature excess above the elsewhere level: {ex - (cpp2 - 1) * n:+.1f}")
    out()

    # 5b. emission lines missing from the model
    out(f"5b. emission lines absent from the model line list, rest +/- {LINE_HALF:g} A (vacuum)")
    ewave = j["eline_wave"]
    unmod = np.zeros(chi.size, dtype=bool)
    for name, v in zip(UNMODELLED_AIR, air_to_vac(list(UNMODELLED_AIR.values())), strict=True):
        sel = np.abs(rest - v) <= LINE_HALF
        unmod |= sel
        n, _, cpp, ex = stats(chi, sel)
        mean = chi[sel].mean() if n else np.nan
        near = np.min(np.abs(ewave - v))
        out(
            f"{name:>9} {v:8.2f}: nearest model line {near:6.1f} A away, N {n:2d}, "
            f"excess {ex:+6.1f}, mean chi {mean:+.2f}"
        )
    n, _, cpp, ex = stats(chi, unmod)
    out(f"all: N {n}, excess {ex:+.1f} = {ex / ex_tot:.1%}")
    n, _, _, ex = stats(chi, unmod & tell)
    out(f"of which inside the telluric bands: N {n}, excess {ex:+.1f}")
    n, _, _, ex = stats(chi, unmod & absf)
    out(f"of which inside the absorption feature windows: N {n}, excess {ex:+.1f}")
    out()

    # 6. correlation and posterior spread
    out("6. autocorrelation of chi on contiguous continuum pixels")
    for lag in range(1, 6):
        npair, raw, pear = autocorr(chi_full, cont_full, lag)
        out(
            f"lag {lag}: pairs {npair}, r (no mean removed) {raw:+.3f}, Pearson {pear:+.3f}, 1/sqrt(N) {1 / np.sqrt(npair):.3f}"
        )
    sd = p["draw_spec"].std(axis=0)[cont_full]
    ratio = sd / unc[cont_full]
    out(
        f"posterior draw sd / unc: median {np.median(ratio):.4f}, 95th pct {np.percentile(ratio, 95):.4f}, max {ratio.max():.4f}"
    )
    c2_inf = float(np.sum((flux - p["sp_best"])[cont_full] ** 2 / (unc[cont_full] ** 2 + sd**2)))
    out(f"chi2 with unc^2 + sd^2: {c2_inf:.1f} (change {c2_inf - c2_tot:+.1f})")
    out()

    # extra: chi2/px by signal to noise quartile
    out("extra: chi2/px in quartiles of flux/unc")
    q = np.percentile(snr, [0, 25, 50, 75, 100])
    for lo, hi in zip(q[:-1], q[1:], strict=True):
        sel = (snr >= lo) & (snr <= hi)
        n, _, cpp, ex = stats(chi, sel)
        out(f"S/N {lo:6.1f}-{hi:6.1f}: N {n}, chi2/px {cpp:.3f}, excess {ex:+.1f}")
    out()

    # 7. budget, each pixel counted once in the listed order
    worst = np.zeros(chi.size, dtype=bool)
    worst[order[: int(round(0.01 * n_tot))]] = True
    orders = {
        "A, the requested order": [
            ("telluric", tell),
            ("sky lines +/-3 A", sky),
            ("absorption features", absf),
            ("worst 1% |chi|", worst),
        ],
        "B, missing emission lines first, then |chi| > 4 instead of the worst 1%": [
            ("missing emission lines", unmod),
            ("telluric", tell),
            ("sky lines +/-3 A", sky),
            ("absorption features", absf),
            ("|chi| > 4", np.abs(chi) > 4),
        ],
    }
    for title, groups in orders.items():
        out(f"7{title[0]}. excess budget, order {title}; each pixel goes to its first group")
        used = np.zeros(chi.size, dtype=bool)
        for name, sel in groups:
            n, c2, cpp, ex = stats(chi, sel & ~used)
            used |= sel
            out(f"{name:>22}: N {n:4d}, chi2 {c2:8.1f}, excess {ex:+7.1f} = {ex / ex_tot:6.1%}")
        if title.startswith("A"):
            k = int((worst & ~(tell | sky | absf)).sum())
            out(
                f"    worst 1% group: unit Gaussian noise would give {k * gauss_tail_excess(0.01):+.1f}"
            )
        rem = ~used
        n, c2, cpp, ex = stats(chi, rem)
        out(f"{'remainder':>22}: N {n:4d}, chi2 {c2:8.1f}, excess {ex:+7.1f} = {ex / ex_tot:6.1%}")
        mad_rem = 1.4826 * np.median(np.abs(chi[rem] - np.median(chi[rem])))
        out(f"noise scale factor for the remainder to reach chi2/px = 1: {np.sqrt(cpp):.4f}")
        out(f"robust scale of the remainder 1.4826 MAD = {mad_rem:.4f}")
        out()
    out(f"noise scale factor for all continuum pixels: {np.sqrt(c2_tot / n_tot):.4f}")
    (OUT / str(TID) / "continuum_check.txt").write_text("\n".join(lines) + "\n")

    # figure
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})
    lo_fit, hi_fit = 3320 * (1 + z), 9000 * (1 + z)
    spm = ~good & (wave >= lo_fit) & (wave < hi_fit)
    for i in np.flatnonzero(spm):
        ax0.axvspan(wave[i] - 0.4, wave[i] + 0.4, color="0.75", lw=0)
    emis = np.zeros(wave.size, dtype=bool)
    emis[np.flatnonzero(good)[region != "continuum"]] = True
    for i in np.flatnonzero(emis):
        ax0.axvspan(wave[i] - 0.4, wave[i] + 0.4, color="tab:blue", alpha=0.3, lw=0)
    for lo, hi in TELLURIC.values():
        ax0.axvspan(lo, hi, color="tab:orange", alpha=0.15, lw=0)
    for v in sky_vac:
        ax0.axvline(v, color="tab:red", lw=0.6, alpha=0.6)
    for v in air_to_vac(list(UNMODELLED_AIR.values())):
        ax0.axvline(v * (1 + z), color="tab:green", lw=0.6, alpha=0.8)
    ax0.plot(w, chi, ".", ms=1.5, color="k")
    ax0.set_ylim(-11, 11)
    ax0.set_xlim(wave[good].min() - 20, wave[good].max() + 20)
    ax0.set_xlabel("observed vacuum wavelength [A]")
    ax0.set_ylabel("chi, continuum pixels")
    ax0.set_title(
        fontsize=9,
        label="grey: spender mask, blue: emission windows, orange: telluric (approximate), red: sky lines, "
        "green: emission lines missing from the model",
    )
    bins = np.linspace(-11, 11, 111)
    ax1.hist(chi, bins=bins, histtype="step", color="k", label="continuum chi")
    x = np.linspace(-11, 11, 600)
    dx = bins[1] - bins[0]
    ax1.plot(
        x,
        n_tot * dx * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi),
        color="tab:red",
        label="unit Gaussian",
    )
    ax1.plot(
        x,
        n_tot * dx * np.exp(-0.5 * (x / mad) ** 2) / (mad * np.sqrt(2 * np.pi)),
        color="tab:blue",
        ls="--",
        label=f"Gaussian, sigma = 1.4826 MAD = {mad:.3f}",
    )
    ax1.set_yscale("log")
    ax1.set_ylim(0.5, None)
    ax1.set_xlabel("chi")
    ax1.set_ylabel("pixels")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / str(TID) / "continuum_check.png", dpi=130)


if __name__ == "__main__":
    main()
