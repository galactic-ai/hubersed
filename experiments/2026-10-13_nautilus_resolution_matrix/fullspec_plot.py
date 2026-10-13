"""Plot the full spectrum of 39627770174637084 with the 10-09 ER max-L and the masked rmatrix MAP.

The DESI data and both models on top, then the chi of the 10-09 ER nautilus max-L point (iron
coadd and ivar, median LSF), then the chi of the 10-13 MAP with this target's resolution matrix and the
missing-line mask (row-weighted recoadd, corrected ivar). Each chi uses its own data and errors.
Pixels that a fit does not use are left empty, and the masked lines are shaded.

INPUTS: the 10-09 ``_pred.npz`` (data, mask and max-L spectrum) and map_rmatrix_mask.npz of map.py.
SEED: none.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_nautilus_resolution_matrix/fullspec_plot.py``.

OUTPUT: fullspec_er_vs_rmatrix_mask.png in results/2026-10-13_nautilus_resolution_matrix/.
"""

import matplotlib.pyplot as plt
import numpy as np
from map import OUT, START, build
from model_setup import MISSING_LINES

from hubersed.plotting.style import use_apj_style


def main():
    use_apj_style()
    er = np.load(START, allow_pickle=True)
    wave, f_er, u_er, g_er = er["wave"], er["flux"], er["unc"], er["good"].astype(bool)
    sp_er = er["sp_best"]
    model, obs, cue, _ = build("rmatrix", "mask")
    theta = np.load(OUT / "map_rmatrix_mask.npz")["theta"]
    sp_r = model.predict(theta, observations=obs, sps=cue)[0][0]
    o = obs[0]
    f_r, u_r, g_r = o.flux, o.uncertainty, o.mask.astype(bool)
    z = float(er["z"])
    chi_er = np.where(g_er, (f_er - sp_er) / u_er, np.nan)
    chi_r = np.where(g_r, (f_r - sp_r) / u_r, np.nan)
    c2 = {
        "er": np.nansum(chi_er**2),
        "r": np.nansum(chi_r**2),
    }
    fig, (top, bot_er, bot_r) = plt.subplots(
        3,
        1,
        figsize=(16, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1], "hspace": 0.05},
    )
    top.plot(wave, np.where(g_r, f_r, np.nan), color="0.2", lw=0.6, label="DESI")
    top.plot(
        wave,
        np.where(g_er, sp_er, np.nan),
        color="#b2182b",
        lw=0.7,
        label=rf"10-09 ER max-L, median LSF, iron ivar ($\chi^2_r$ = {c2['er'] / g_er.sum():.2f})",
    )
    top.plot(
        wave,
        np.where(g_r, sp_r, np.nan),
        color="#e08214",
        lw=0.7,
        label=rf"10-13 MAP, target R, line mask, corrected ivar ($\chi^2_r$ = {c2['r'] / g_r.sum():.2f}",
    )
    bot_er.plot(wave, chi_er, color="#b2182b", lw=0.4)
    bot_r.plot(wave, chi_r, color="#e08214", lw=0.4)
    for lam in MISSING_LINES.values():
        for a in (top, bot_r):
            a.axvspan((lam - 4) * (1 + z), (lam + 4) * (1 + z), color="0.9", zorder=0)
    for a in (bot_er, bot_r):
        a.axhline(0, color="0.3", lw=0.5)
        for h in (-2, 2):
            a.axhline(h, color="0.5", ls="--", lw=0.5)
        a.set_ylim(-7, 7)
    top.set_ylabel("flux [maggies]")
    top.legend(frameon=False, fontsize=8, loc="upper right")
    bot_er.set_ylabel(r"$\chi$ 10-09 ER")
    bot_r.set_ylabel(r"$\chi$ 10-13 R")
    bot_r.set_xlabel(r"observed wavelength [$\AA$]")
    bot_r.set_xlim(wave[g_er | g_r].min() - 10, wave[g_er | g_r].max() + 10)
    fig.savefig(OUT / "fullspec_er_vs_rmatrix_mask.png", dpi=150, bbox_inches="tight")
    print(
        f"wrote {OUT / 'fullspec_er_vs_rmatrix_mask.png'}; chi2 ER {c2['er']:.1f}, R {c2['r']:.1f}"
    )


if __name__ == "__main__":
    main()
