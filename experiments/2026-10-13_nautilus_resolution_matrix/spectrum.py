"""Plot the rmatrix fit's data, max-L model and residuals from its _pred.npz.

Writes spectrum_rmatrix.png in results/2026-10-13_nautilus_resolution_matrix/. Needs no SPS build.

COMMAND: ``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_nautilus_resolution_matrix/spectrum.py`` from the repository root.
"""

import astropy.units as u
import numpy as np
from fit import OUT, stem

from hubersed.conversion import to_flambda
from hubersed.plotting.spectra import plot_residual, residual_chi, spectrum_figure

COL = "#1b7837"

d = np.load(OUT / f"{stem('rmatrix', 'mask')}_pred.npz")
z = float(d["z"])
rest = d["wave"] / (1 + z)


def flam(maggies):
    """Convert maggies to DESI f_lambda units, NaN outside the fitted pixels."""
    f = to_flambda(d["wave"] * u.AA, np.asarray(maggies, float) * u.mgy).value
    return np.where(d["good"], f, np.nan)


chi = residual_chi(d["flux"], d["sp_best"], d["unc"], d["good"])
n_pix = int(np.isfinite(chi).sum())
chi2_nu = float(np.nansum(chi**2) / (n_pix - len(d["labels"])))
lo, hi = np.percentile(d["draw_spec"], [16, 84], axis=0)
fig, ax = spectrum_figure(
    d["wave"],
    z=z,
    figsize=(11, 6),
    data=flam(d["flux"]),
    unc=flam(d["unc"]),
    band_kw={},
    data_kw={"label": "DESI spectrum (row-weighted recoadd, corrected ivar)"},
    models=[
        {
            "flux": flam(d["sp_best"]),
            "color": COL,
            "lw": 0.8,
            "label": rf"rmatrix max-L, N_eff {float(d['n_eff']):.0f}, $\chi^2_\nu$ = {chi2_nu:.2f}",
        }
    ],
)
ax[0].fill_between(rest, flam(lo), flam(hi), color=COL, alpha=0.3, lw=0)
plot_residual(ax[1], d["wave"], z=z, chi=chi, lw=0.5, color=COL, alpha=0.8)
ax[1].set_ylim(-8, 8)
ax[1].set_xlim(rest[d["good"]].min(), rest[d["good"]].max())
fig.savefig(OUT / "spectrum_rmatrix.png", dpi=150, bbox_inches="tight")
print(f"chi2 {np.nansum(chi**2):.1f} over {n_pix} pixels, chi2_nu {chi2_nu:.3f}")
