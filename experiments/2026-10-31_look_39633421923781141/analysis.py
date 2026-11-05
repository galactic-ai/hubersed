"""Emission lines, redshift and neighbours of the screened continuum outlier 39633421923781141 (#17).

QUESTION: what is unusual about 39633421923781141 (z 0.232, a strong line at observed 8090 A, a blue
neighbour in the cutout, flagged in 18 of 20 seeds)?
HYPOTHESIS: none registered.
INPUTS: FastSpecFit VAC (redshift, line fits), fracflux.csv of 2026-10-31_screen_all, the spender
spectrum, the cutout of 2026-10-31_logp_vs_colour/cutouts. BPT line: Kewley et al. 2001, log
[OIII]/Hb = 0.61 / (log [NII]/Ha - 0.47) + 1.19 [from memory, unchecked].
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_look_39633421923781141/analysis.py``.
RESULT: 2026-10-31. Broad-line AGN: FastSpecFit broad Halpha S/N 94, sigma 533 km/s, EW 58 A, broad
Hbeta S/N 27. Narrow lines sit just below the Kewley line (log [NII]/Ha -0.34, log [OIII]/Hb
+0.38), so the narrow-line BPT screen kept it. Narrow Ha/Hb 4.4. fracflux_g 0.07.
FIGURES: results/2026-10-31_look_39633421923781141/look.txt and look.png.
"""

# %%
import csv

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import medfilt
from spender.data.desi import DESI

from hubersed.io.desi import load_by_index, tids_to_indices
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39633421923781141
RES = PATHS["RESULTS"]
OUT = RES / "2026-10-31_look_39633421923781141"
OUT.mkdir(parents=True, exist_ok=True)
LINES = ("OII_3726", "OII_3729", "HBETA", "OIII_5007", "HALPHA", "NII_6584", "SII_6716", "SII_6731",
         "HALPHA_BROAD", "HBETA_BROAD")  # fmt: skip
WINDOWS = {
    "[OII]": (3700, 3760),
    r"H$\beta$+[OIII]": (4820, 5040),
    r"H$\alpha$+[NII]+[SII]": (6520, 6760),
}

h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)
M, S = h["METADATA"].data, h["FASTSPEC"].data
i = int(np.flatnonzero(M["TARGETID"] == TID)[0])
z = float(M["Z"][i])
out = [f"{TID}: Z {z:.5f}, Z_RR {float(M['Z_RR'][i]):.5f}, DELTACHI2 {float(M['DELTACHI2'][i]):.1f}, "
       f"ZWARN {int(M['ZWARN'][i])}, SPECTYPE {M['SPECTYPE'][i]}",  # fmt: skip
       f"LOGMSTAR {float(S['LOGMSTAR'][i]):.2f}, DN4000 {float(S['DN4000'][i]):.3f}", "",
       f"{'line':>13} {'flux':>9} {'S/N':>6} {'EW':>7} {'sigma':>6} {'vshift':>7}"]  # fmt: skip
flux = {}
for ln in LINES:
    f, iv = float(S[f"{ln}_FLUX"][i]), float(S[f"{ln}_FLUX_IVAR"][i])
    flux[ln] = (f, f * np.sqrt(iv) if iv > 0 else 0.0)
    out.append(f"{ln:>13} {f:9.2f} {flux[ln][1]:6.1f} {float(S[f'{ln}_EW'][i]):7.1f} "
               f"{float(S[f'{ln}_SIGMA'][i]):6.1f} {float(S[f'{ln}_VSHIFT'][i]):7.1f}")  # fmt: skip


def ratio(a, b):
    """Return log10(a / b) of two line fluxes, NaN unless both have S/N > 3."""
    ok = flux[a][1] > 3 and flux[b][1] > 3
    return np.log10(flux[a][0] / flux[b][0]) if ok else np.nan


n2, o3 = ratio("NII_6584", "HALPHA"), ratio("OIII_5007", "HBETA")
kew = 0.61 / (n2 - 0.47) + 1.19 if np.isfinite(n2) and n2 < 0.47 else np.nan
ha_hb = flux["HALPHA"][0] / flux["HBETA"][0] if flux["HBETA"][1] > 3 else np.nan
out += ["", f"Ha/Hb {ha_hb:.2f} (Case B about 2.86 [from memory]); log [NII]/Ha {n2:+.2f}; "
        f"log [OIII]/Hb {o3:+.2f}; Kewley 2001 line at this [NII]/Ha: {kew:+.2f}"]  # fmt: skip
with open(RES / "2026-10-31_screen_all" / "fracflux.csv", newline="") as fh:
    tr = next(r for r in csv.DictReader(fh) if int(r["targetid"]) == TID)
out.append(f"Tractor: type {tr['type']}, fracflux_g {tr['fracflux_g']}, fracflux_r {tr['fracflux_r']}, "
           f"maskbits {tr['maskbits']}")  # fmt: skip
text = "\n".join(out)
(OUT / "look.txt").write_text(text + "\n")
print(text)

# %% figure
wave = DESI._wave_obs.numpy()
fl, iv, _, _ = load_by_index(int(tids_to_indices(np.array([TID]))[0]))
fl, iv = np.asarray(fl, float), np.asarray(iv, float)
good = iv > 0
rest = wave / (1 + z)
use_apj_style()
fig = plt.figure(figsize=(18, 9))
ax0 = fig.add_subplot(2, 1, 1)
ax0.plot(rest[good], fl[good], color="0.7", lw=0.4)
ax0.plot(rest[good], medfilt(fl[good], 21), color="k", lw=0.8)
ax0.set_xlabel(r"rest wavelength [\AA]", fontsize=12)
ax0.set_ylabel(r"$f_\lambda$ [$10^{-17}$ cgs]", fontsize=12)
ax0.set_title(f"{TID}  z {z:.4f}", fontsize=12)
ins = ax0.inset_axes([0.8, 0.45, 0.19, 0.53])
ins.imshow(plt.imread(RES / "2026-10-31_logp_vs_colour" / "cutouts" / f"{TID}.jpg"), origin="upper")
ins.add_patch(plt.Circle((47.5, 47.5), 0.75 / 0.262, fill=False, color="w", lw=1))
ins.set_xticks([])
ins.set_yticks([])
for k, (name, (lo, hi)) in enumerate(WINDOWS.items()):
    ax = fig.add_subplot(2, 3, 4 + k)
    s = good & (rest > lo) & (rest < hi)
    ax.step(rest[s], fl[s], where="mid", color="k", lw=0.8)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel(r"rest wavelength [\AA]", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "look.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'look.png'}")
