"""Plot the full spectrum of 39627770174637084 with the model A and free-line MAP fits.

The data and both MAP spectra in DESI units, then the chi of each. The legend gives the reduced chi2
over the fitted pixels. Unused pixels are empty and the lines missing from Cue are shaded.

INPUTS: map_A.npz and map_free.npz of map.py. SEED: none.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-15_broadline_free_amplitude/fullspec.py``.

OUTPUT: fullspec_A_free.png in results/2026-10-15_broadline_free_amplitude/.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from map import OUT, build
from model_setup import MISSING_LINES

from hubersed.conversion import to_flambda
from hubersed.plotting.style import use_apj_style

COLORS = {"A": "#b2182b", "free": "#2166ac"}
LABELS = {"A": "model A", "free": "free-line"}


def desi(wave, maggies):
    """Return maggies as DESI flux density, 1e-17 erg/s/cm^2/A."""
    return to_flambda(wave * u.AA, maggies * u.mgy).value


def main():
    """Write the full-spectrum figure."""
    use_apj_style()
    spec = {}
    for v in COLORS:
        model, obs, cue, _ = build(v)
        theta = np.load(OUT / f"map_{v}.npz")["theta"]
        spec[v] = model.predict(theta, observations=obs, sps=cue)[0][0]
    o = obs[0]
    wave, good = o.wavelength, o.mask.astype(bool)
    z = float(np.atleast_1d(model.params["zred"])[0])
    flux = desi(wave, o.flux)

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1], "hspace": 0.05}
    )
    top = axes[0]
    top.plot(wave, np.where(good, flux, np.nan), color="0.2", lw=0.6, label="DESI")
    for ax, v in zip(axes[1:], COLORS, strict=True):
        m = desi(wave, spec[v])
        chi2 = float(np.sum(((o.flux - spec[v])[good] / o.uncertainty[good]) ** 2))
        top.plot(
            wave,
            np.where(good, m, np.nan),
            color=COLORS[v],
            lw=0.7,
            label=rf"{LABELS[v]} MAP, $\chi^2_r$ = {chi2 / good.sum():.3f}",
        )
        chi = np.where(good, (o.flux - spec[v]) / o.uncertainty, np.nan)
        ax.plot(wave, chi, color=COLORS[v], lw=0.4)
        ax.axhline(0, color="0.3", lw=0.5)
        for h in (-2, 2):
            ax.axhline(h, color="0.5", ls="--", lw=0.5)
        ax.set_ylabel(rf"$\chi$ {LABELS[v]}")
        ax.text(
            0.01,
            0.9,
            rf"{LABELS[v]}: $\chi^2_r$ = {chi2 / good.sum():.3f}",
            transform=ax.transAxes,
            fontsize=12,
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
        )
        ax.set_ylim(-7, 7)
    for lam in MISSING_LINES.values():
        for ax in axes:
            ax.axvspan((lam - 4) * (1 + z), (lam + 4) * (1 + z), color="0.93", zorder=0)
    top.set_ylabel(r"flux [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    top.legend(frameon=False, fontsize=11, loc="upper right")
    axes[-1].set_xlabel(r"observed wavelength [$\AA$]")
    axes[-1].set_xlim(wave[good].min() - 10, wave[good].max() + 10)
    path = OUT / "fullspec_A_free.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
