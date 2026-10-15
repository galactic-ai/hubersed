"""Plot data, model and residuals at the chi2 minimum of each n_H profile variant of map_unmask.py.

One full-spectrum figure and one line-window figure per variant in DROPS. Each uses the lowest chi2
profile point of that variant on its own pixels, scored with n_H put back as in map_unmask.py
profiles. [SII]6731 is fitted in all of them. Pixels a variant masks are shaded, and their
residuals are drawn in grey with the same model. A line-window figure of the rmatrix nautilus
max-L point (from its _pred.npz, [SII]6731 masked) is made the same way for comparison.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_nautilus_resolution_matrix/map_unmask_spectrum.py``.

OUTPUT: map_unmask_spectrum_<variant>.png, lines_<variant>.png and lines_nautilus.png in
results/2026-10-13_nautilus_resolution_matrix/.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from fit import stem
from map_unmask import DIR, DROPS, OUT, WINDOWS, build
from model_setup import MISSING_LINES, WAVE_OBS, load_data

from hubersed.conversion import to_flambda
from hubersed.plotting.spectra import plot_residual, residual_chi, spectrum_figure

COL = "#e08214"
COL_NAUT = "#1b7837"
# rest-frame windows in A for the line-window figures
LINE_WINDOWS = {
    "[OII]3726,3729": (3715, 3742),
    "[OIII]4363": (4352, 4374),
    "Hbeta": (4850, 4876),
    "[OI]6300 + [SIII]6312": (6292, 6320),
    "Halpha + [NII]": (6540, 6592),
    "[SII]6716,6731": (6706, 6742),
    "[OII]7320,7330": (7312, 7340),
}


def flam(maggies, keep):
    """Convert maggies to DESI f_lambda units, NaN outside ``keep``."""
    f = to_flambda(WAVE_OBS * u.AA, np.asarray(maggies, float) * u.mgy).value
    return np.where(keep, f, np.nan)


def best_point(drop, lab, chi2_of):
    """Return n_H, theta and number of free parameters of the lowest chi2 profile point."""
    pattern = "profile_nH*.npz" if drop == "none" else f"profile_{drop}_nH*_s*.npz"
    rows = []
    for f in DIR.glob(pattern):
        d = np.load(f)
        fit = dict(zip([str(x) for x in d["labels"]], d["theta"], strict=True))
        fit["gas_lognH"] = float(d["nh"])
        theta = np.array([fit[x] for x in lab])
        rows.append((chi2_of(theta), float(d["nh"]), theta, len(d["labels"])))
    return min(rows, key=lambda r: r[0])[1:]


def line_windows(rest, data, model, chi, chi_grey, fit, title, color, path):
    """Draw data and model (top) and residuals (bottom) in each of LINE_WINDOWS.

    ``chi`` is drawn on the fitted pixels in ``color``, ``chi_grey`` in grey, and pixels not in
    ``fit`` are shaded. Each panel gives the chi2 of its fitted and grey pixels.
    """
    n = len(LINE_WINDOWS)
    fig, ax = plt.subplots(
        2, n, figsize=(2.6 * n, 5), sharex="col", gridspec_kw={"height_ratios": [2, 1]}
    )
    dw = np.median(np.diff(rest))
    for j, (name, (a, b)) in enumerate(LINE_WINDOWS.items()):
        s = (rest >= a) & (rest <= b)
        top, bot = ax[0, j], ax[1, j]
        for r in rest[s & ~fit]:
            for x in (top, bot):
                x.axvspan(r - dw / 2, r + dw / 2, color="0.88", lw=0, zorder=0)
        top.step(rest[s], data[s], where="mid", color="0.2", lw=0.8)
        top.plot(rest[s], model[s], color=color, lw=1.0)
        bot.step(rest[s], chi[s], where="mid", color=color, lw=0.8)
        bot.step(rest[s], chi_grey[s], where="mid", color="0.45", lw=0.8)
        bot.axhline(0, color="0.5", lw=0.6)
        for lv in (-2, 2):
            bot.axhline(lv, color="0.5", lw=0.6, ls="--")
        bot.set_ylim(-8, 8)
        bot.set_xlim(a, b)
        c2, g2 = np.nansum(chi[s] ** 2), np.nansum(chi_grey[s] ** 2)
        label = f"$\\chi^2$ {c2:.0f}" + (f", grey {g2:.0f}" if g2 > 0 else "")
        bot.text(0.03, 0.05, label, transform=bot.transAxes, fontsize=7)
        top.set_title(name, fontsize=8)
        top.tick_params(labelsize=7)
        bot.tick_params(labelsize=7)
    ax[0, 0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=8)
    ax[1, 0].set_ylabel(r"$\chi$", fontsize=8)
    fig.supxlabel(r"rest wavelength [$\AA$]", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


model, obs_all, cue, _ = build(True)
lab = model.theta_labels()
o_all = obs_all[0]
zred = load_data(tuple(MISSING_LINES), ("[SII]6731",))[0]
rest = WAVE_OBS / (1 + zred)
for drop in DROPS:
    obs = build(True, drop=drop)[1]
    o = obs[0]

    def chi2_of(theta, obs=obs, o=o):
        """Return chi2 on this variant's pixels."""
        sp = model.predict(theta, observations=obs, sps=cue)[0][0]
        return float(np.sum(((o.flux - sp)[o.mask] / o.uncertainty[o.mask]) ** 2))

    nh, theta, n_free = best_point(drop, lab, chi2_of)
    sp = model.predict(theta, observations=obs, sps=cue)[0][0]
    fit = o.mask
    dropped = o_all.mask & ~fit
    chi = residual_chi(o.flux, sp, o.uncertainty, fit)
    chi_drop = residual_chi(o_all.flux, sp, o_all.uncertainty, dropped)
    n_pix = int(fit.sum())
    chi2_nu = float(np.nansum(chi**2) / (n_pix - n_free))
    masked = ", ".join(DROPS[drop]) if DROPS[drop] else "nothing extra"
    fig, ax = spectrum_figure(
        WAVE_OBS,
        z=zred,
        figsize=(11, 6),
        data=flam(o_all.flux, o_all.mask),
        unc=flam(o_all.uncertainty, o_all.mask),
        band_kw={},
        data_kw={"label": "DESI spectrum (row-weighted recoadd, corrected ivar), [SII]6731 fitted"},
        models=[
            {
                "flux": flam(sp, o_all.mask),
                "color": COL,
                "lw": 0.8,
                "label": rf"MAP, masked {masked}; log $n_H$ = {nh:.1f}, $\chi^2_\nu$ = {chi2_nu:.3f}",
            }
        ],
    )
    for w in DROPS[drop]:
        for a in ax[:2]:
            a.axvspan(*WINDOWS[w], color="0.85", lw=0, zorder=0)
    plot_residual(ax[1], WAVE_OBS, z=zred, chi=chi, lw=0.5, color=COL, alpha=0.8)
    ax[1].plot(rest, chi_drop, color="0.4", lw=0.7)
    ax[1].set_ylim(-8, 8)
    ax[1].set_xlim(rest[o_all.mask].min(), rest[o_all.mask].max())
    fig.savefig(OUT / f"map_unmask_spectrum_{drop}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    line_windows(
        rest,
        flam(o_all.flux, np.isfinite(o_all.flux)),
        flam(sp, np.ones(rest.size, bool)),
        chi,
        chi_drop,
        fit,
        rf"MAP with [SII]6731 fitted, masked {masked}; log $n_H$ = {nh:.1f}, "
        rf"$\chi^2_\nu$ = {chi2_nu:.3f} (grey: masked pixels, not fitted)",
        COL,
        OUT / f"lines_{drop}.png",
    )
    print(
        f"{drop}: n_H {nh:.1f}, chi2 {np.nansum(chi**2):.1f} over {n_pix} pixels, "
        f"chi2_nu {chi2_nu:.4f}, masked-window chi2 {np.nansum(chi_drop**2):.1f}"
    )

d = np.load(OUT / f"{stem('rmatrix', 'mask')}_pred.npz")
assert np.allclose(d["wave"], WAVE_OBS)
chi = residual_chi(d["flux"], d["sp_best"], d["unc"], d["good"])
n_pix = int(d["good"].sum())
chi2_nu = float(np.nansum(chi**2) / (n_pix - len(d["labels"])))
nh = float(d["best"][list(d["labels"]).index("gas_lognH")])
line_windows(
    WAVE_OBS / (1 + float(d["z"])),
    flam(d["flux"], np.isfinite(d["flux"])),
    flam(d["sp_best"], np.ones(WAVE_OBS.size, bool)),
    chi,
    np.full(WAVE_OBS.size, np.nan),
    d["good"],
    rf"rmatrix nautilus max-L, [SII]6731 masked; log $n_H$ = {nh:.2f}, $\chi^2_\nu$ = {chi2_nu:.3f}",
    COL_NAUT,
    OUT / "lines_nautilus.png",
)
print(f"nautilus: n_H {nh:.2f}, chi2 {np.nansum(chi**2):.1f} over {n_pix} pixels")
