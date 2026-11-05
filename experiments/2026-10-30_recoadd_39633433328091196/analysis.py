"""Single exposures and PR 2377 recoadd of the SV3 bright spectrum of 39633433328091196 (#17).

QUESTION: which of the three SV3 bright exposures carries the blue excess of the DR1 coadd, and
does the recoadd (row-weighted R, fixed flux-calibration ivar) remove it?
HYPOTHESIS: none registered.
INPUTS: sv3/exposures.npz (exposures.py) and sv3/resolution_matrix_rows_ivar.npz (ivar_recompute.py)
of stampede3 job 3662886 (recoadd.slurm), copied to the Mac. DR1 SV3 bright and dark through spender
as in 2026-10-30_sv3_bright_dark_39633433328091196. Sky g (mag/arcsec2, sky_mag_g_spec) and
efftime_spec from desi_dr1.exposure (Astro Data Lab, 2026-10-30).
SEED: none.
COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-30_recoadd_39633433328091196/analysis.py``.
RESULT: 2026-10-30, exposures.txt. Median flux at observed 3600-4000 A grows with sky brightness:
0.65 (87837, sky g 22.05), 3.22 (87838, 20.67), 7.64 (87839, 19.09); dark 0.34. Redward of 5500 A the
exposures agree within ~10%. The recoadd is within 2% of the DR1 coadd in every bin, so it does not
remove the excess. Assumptions: 51 px median filter; sky values from desi_dr1.exposure.
FIGURES: results/2026-10-30_recoadd_39633433328091196/exposures.png and exposures.txt.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from spender.data.desi import DESI

from hubersed.io.desi import load_spectrum
from hubersed.paths import get_paths
from hubersed.plotting.style import use_apj_style

TID = 39633433328091196
PATHS = get_paths()
OUT = PATHS["RESULTS"] / "2026-10-30_recoadd_39633433328091196"
KERNEL = 51  # pixels of the median filter
BINS = [(3600, 4000), (4000, 4620), (4620, 5500), (5500, 6500), (6500, 7600), (7600, 9824)]
SKY = {87837: (22.05, 192.3), 87838: (20.67, 180.5), 87839: (19.09, 109.9)}  # sky g, efftime s
TOL = 1e-4  # A


def stitch(d, i, grid):
    """Return flux and ivar of exposure row ``i`` with b, r, z combined by ivar on ``grid``.

    As spender DESI.prepare_spectra: masked pixels get no weight, overlaps are ivar-weighted.
    """
    num, den = np.zeros(grid.size), np.zeros(grid.size)
    for arm in "BRZ":
        w = d[f"{arm}_wave"]
        j = np.searchsorted(grid, w[0] - TOL)
        assert np.allclose(grid[j : j + w.size], w, atol=TOL), arm
        iv = d[f"{arm}_ivar"][i] * (d[f"{arm}_mask"][i] == 0)
        num[j : j + w.size] += iv * d[f"{arm}_flux"][i]
        den[j : j + w.size] += iv
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0.0), den


def smooth(flux, good):
    """Return the median-filtered flux on good pixels, NaN elsewhere."""
    out = np.full(flux.size, np.nan)
    out[good] = medfilt(flux[good], KERNEL)
    return out


# %% spectra on the spender grid
s = load_spectrum(TID)
grid = np.asarray(s.spectral_axis.value, float)
spectra = {"DR1 bright coadd": (np.asarray(s.flux.value, float), ~np.asarray(s.mask, bool))}
dark_file = PATHS["DATA"] / "sv3_dark_coadds" / "coadd-sv3-dark-10231.fits"
sp, w, _, tid, norm, _ = DESI.prepare_spectra(str(dark_file))
k = int(np.flatnonzero(tid.numpy() == TID)[0])
spectra["DR1 dark coadd"] = (sp[k].numpy() * float(norm[k]), w[k].numpy() > 0)
rec = np.load(OUT / "sv3" / "resolution_matrix_rows_ivar.npz")
assert np.allclose(rec["wave"], grid, atol=TOL)
spectra["bright recoadd"] = (rec["flux"], rec["ivar"] > 0)
ex = np.load(OUT / "sv3" / "exposures.npz")
for i, e in enumerate(ex["EXPID"]):
    f, iv = stitch(ex, i, grid)
    g, eff = SKY[int(e)]
    spectra[f"exposure {e}, sky g {g:.2f}, efftime {eff:.0f} s"] = (f, iv > 0)

# %% table: median flux in observed bins
names = list(spectra)
lines = ["median f_lambda (1e-17 erg/s/cm2/A) on good pixels, observed A"]
lines.append(f"{'obs A':>11} " + " ".join(f"{n.split(',')[0]:>18}" for n in names))
for lo, hi in BINS:
    sel = (grid >= lo) & (grid < hi)
    vals = [np.median(f[sel & g]) for f, g in spectra.values()]
    lines.append(f"{lo:5d}-{hi:5d} " + " ".join(f"{v:18.3f}" for v in vals))
text = "\n".join(lines)
(OUT / "exposures.txt").write_text(text + "\n")
print(text)

# %% figure: median-filtered spectra in one panel
use_apj_style()
colors = ["#b2182b", "#2166ac", "k", "#fdb863", "#e66101", "#5e3c99"]
fig, ax = plt.subplots(figsize=(16, 6))
for (name, (f, g)), c in zip(spectra.items(), colors, strict=True):
    ax.plot(grid, smooth(f, g), color=c, lw=1.2, label=f"{name}, median filter {KERNEL} px")
ax.set_xlabel(r"observed wavelength [$\AA$]", fontsize=14)
ax.set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
ax.legend(frameon=False, fontsize=11, loc="upper right")
fig.savefig(OUT / "exposures.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'exposures.png'}")
