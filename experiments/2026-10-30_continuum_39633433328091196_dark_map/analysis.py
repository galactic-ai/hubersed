"""C3K ER MAP of the SV3 dark spectrum against the SV3 bright MAP of 39633433328091196 (#17).

QUESTION: do the dust, metallicity and afe of the C3K ER MAP change when the SV3 bright spectrum
(sky residual in the blue) is replaced by the SV3 dark one?
HYPOTHESIS: none registered.
INPUTS: <TID>_map.npz of fit.py collect (spr job 3662995, 4 starts) in this experiment's results
folder, copied from stampede3, and the SV3 bright ER MAP of
2026-10-29_continuum_39633433328091196_map (icx job 3657895, 7 starts).
SEED: none.
COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-30_continuum_39633433328091196_dark_map/analysis.py``.
RESULT: 2026-10-30, dark_vs_bright.txt. Dark chi2/(N-16) 1.060 (bright 1.117), lower in every rest
bin. afe -0.125 -> +0.164, dust_index 0.400 (ceiling) -> 0.067, dust2 0.413 -> 0.390, logzsol 0.194 ->
0.148, log M 11.461 -> 11.477, sigma 316 -> 287 km/s. Assumptions: MAP from 4 starts (spread
delta ln p 0-12.7), SFH hyperparameters fixed at the bright C3K_HR max-L, 1 dark exposure.
FIGURES: results/2026-10-30_continuum_39633433328091196_dark_map/dark_vs_bright.png and .txt.
"""

# %%
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

from hubersed.conversion import to_flambda
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39633433328091196
OUT = PATHS["RESULTS"] / "2026-10-30_continuum_39633433328091196_dark_map"
FITS = {
    "SV3 bright": PATHS["RESULTS"]
    / "2026-10-29_continuum_39633433328091196_map"
    / f"{TID}_map.npz",
    "SV3 dark": OUT / f"{TID}_map.npz",
}
COLORS = {"SV3 bright": "#b2182b", "SV3 dark": "#2166ac"}
BINS = [(lo, lo + 500) for lo in range(3300, 7300, 500)]  # rest A
KERNEL = 51  # pixels of the median filter of the data


def flam(d, x):
    """Return maggies on the fit's wavelength grid in 1e-17 erg/s/cm^2/A."""
    return to_flambda(np.asarray(d["wave"], float) * u.AA, np.asarray(x, float) * u.mgy).value


def chi(d):
    """Return the chi of a fit on its own data, NaN on unfitted pixels."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d["good"], (d["flux"] - d["sp_best"]) / d["unc"], np.nan)


# %% table: parameters and chi2 by rest-frame bin
fits = {k: np.load(p) for k, p in FITS.items()}
pars = {
    k: dict(zip([str(x) for x in d["labels"]], d["best"], strict=True)) for k, d in fits.items()
}
lines = [f"{'parameter':>16} {'SV3 bright':>11} {'SV3 dark':>11} {'dark-bright':>12}"]
for p in pars["SV3 bright"]:
    b, dk = pars["SV3 bright"][p], pars["SV3 dark"][p]
    lines.append(f"{p:>16} {b:11.4f} {dk:11.4f} {dk - b:+12.4f}")
lines.append("")
for k, d in fits.items():
    g = d["good"]
    x2 = float(np.nansum(chi(d)[g] ** 2))
    starts = np.asarray(d["fun_all"], float)
    lines.append(
        f"{k}: chi2 {x2:.1f} on {g.sum()} px, chi2/(N-{len(d['labels'])}) "
        f"{x2 / (g.sum() - len(d['labels'])):.3f}; -lnp of each start minus best "
        + " ".join(f"{s - starts.min():.2f}" for s in starts)
    )
lines.append(f"{'rest A':>11} " + " ".join(f"{k + ' chi2/N':>16}" for k in fits))
for lo, hi in BINS:
    vals = []
    for d in fits.values():
        s = (
            d["good"]
            & (d["wave"] / (1 + float(d["z"])) >= lo)
            & (d["wave"] / (1 + float(d["z"])) < hi)
        )
        vals.append(f"{np.mean(chi(d)[s] ** 2):9.3f} ({s.sum():4d})" if s.any() else f"{'-':>16}")
    lines.append(f"{lo:5d}-{hi:5d} " + " ".join(f"{v:>16}" for v in vals))
text = "\n".join(lines)
(OUT / "dark_vs_bright.txt").write_text(text + "\n")
print(text)

# %% figure: each spectrum with its MAP, then the chi of each
use_apj_style()
fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.06})  # fmt: skip
for (k, d), a_chi in zip(fits.items(), ax[1:], strict=True):
    rest = d["wave"] / (1 + float(d["z"]))
    g = d["good"]
    data = flam(d, d["flux"])
    med = np.full(data.size, np.nan)
    med[g] = medfilt(data[g], KERNEL)
    ax[0].plot(rest, med, color=COLORS[k], lw=1.0, alpha=0.6,
               label=f"{k} DR1 coadd, median filter {KERNEL} px")  # fmt: skip
    ax[0].plot(rest, np.where(g, flam(d, d["sp_best"]), np.nan), color=COLORS[k], lw=0.8,
               ls="--", label=f"{k} C3K ER MAP")  # fmt: skip
    a_chi.plot(rest, chi(d), color=COLORS[k], lw=0.5)
    for h in (-2, 2):
        a_chi.axhline(h, color="0.3", ls="--", lw=1)
    a_chi.axhline(0, color="0.3", lw=0.6)
    a_chi.set_ylim(-5, 5)
    a_chi.set_ylabel(r"$\chi$", fontsize=14)
ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
ax[0].legend(frameon=False, fontsize=12, loc="lower right")
ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
fig.savefig(OUT / "dark_vs_bright.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'dark_vs_bright.png'}")
