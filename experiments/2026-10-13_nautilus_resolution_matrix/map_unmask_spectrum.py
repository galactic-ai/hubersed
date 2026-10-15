"""Plot data, model and residuals at the chi2 minimum of each n_H profile variant of map_unmask.py.

One figure per variant in DROPS. Each uses the lowest chi2 profile point of that variant on its own
pixels, scored with n_H put back as in map_unmask.py profiles. [SII]6731 is fitted in all of them.
Pixels a variant masks are shaded, and their residuals are drawn in grey with the same model.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_nautilus_resolution_matrix/map_unmask_spectrum.py``.

OUTPUT: map_unmask_spectrum_<variant>.png in results/2026-10-13_nautilus_resolution_matrix/.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from map_unmask import DIR, DROPS, OUT, WINDOWS, build
from model_setup import MISSING_LINES, WAVE_OBS, load_data

from hubersed.conversion import to_flambda
from hubersed.plotting.spectra import plot_residual, residual_chi, spectrum_figure

COL = "#e08214"


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


model, obs_all, cue, _ = build(True)
lab = model.theta_labels()
o_all = obs_all[0]
zred = load_data(tuple(MISSING_LINES), ("[SII]6731",))[0]
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
    ax[1].plot(WAVE_OBS / (1 + zred), chi_drop, color="0.4", lw=0.7)
    ax[1].set_ylim(-8, 8)
    rest = WAVE_OBS / (1 + zred)
    ax[1].set_xlim(rest[o_all.mask].min(), rest[o_all.mask].max())
    fig.savefig(OUT / f"map_unmask_spectrum_{drop}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"{drop}: n_H {nh:.1f}, chi2 {np.nansum(chi**2):.1f} over {n_pix} pixels, "
        f"chi2_nu {chi2_nu:.4f}, masked-window chi2 {np.nansum(chi_drop**2):.1f}"
    )
