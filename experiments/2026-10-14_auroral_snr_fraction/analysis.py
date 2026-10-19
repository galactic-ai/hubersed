"""Fraction of BGS SV3 star-forming galaxies with the auroral lines detected in FastSpecFit.

QUESTION: in how many galaxies of the sample can [OII]7320,7330, [SIII]6312 and [OIII]4363 pull on
the gas temperature and n_H of a Cue fit, as they do in 39627770174637084?

INPUTS: data/all_target_ids.npy (the spender sample) and the iron SV3 bright FastSpecFit VAC in
data/. Star-forming means Halpha, Hbeta, [OIII]5007 and [NII]6584 at S/N >= 3 and below the
Kauffmann+03 line. [OII]7320,7330 is the sum of both lines. A line with FLUX_IVAR 0 counts as not
detected. SEED: none.

COMMAND: ``uv run python experiments/2026-10-14_auroral_snr_fraction/analysis.py`` from the repository root.

RESULT: see results/2026-10-14_auroral_snr_fraction.txt.
"""

import numpy as np
from astropy.io import fits

from hubersed.paths import PATHS

TID = 39627770174637084
FSF = PATHS["DATA"] / "fastspec-iron-sv3-bright.fits"
OUT = PATHS["RESULTS"] / "2026-10-14_auroral_snr_fraction.txt"
CUTS = (3, 5, 10)
MASS_BINS = (8.0, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0)


def snr(t, *lines):
    """Return the S/N of the summed flux of ``lines``, 0 where any line has no ivar."""
    flux = sum(t[f"{x}_FLUX"] for x in lines)
    ivar = [t[f"{x}_FLUX_IVAR"] for x in lines]
    ok = np.all([iv > 0 for iv in ivar], axis=0)
    var = sum(np.where(iv > 0, 1.0 / np.where(iv > 0, iv, 1.0), 0.0) for iv in ivar)
    return np.where(ok, flux / np.sqrt(np.where(ok, var, 1.0)), 0.0)


t = fits.getdata(FSF, "FASTSPEC")
t = t[np.isin(t["TARGETID"], np.load(PATHS["DATA"] / "all_target_ids.npy").astype(np.int64))]
s = {x: snr(t, x) for x in ("HALPHA", "HBETA", "OIII_5007", "NII_6584")}
with np.errstate(divide="ignore", invalid="ignore"):
    x = np.log10(t["NII_6584_FLUX"] / t["HALPHA_FLUX"])
    y = np.log10(t["OIII_5007_FLUX"] / t["HBETA_FLUX"])
    sf = np.all([v >= 3 for v in s.values()], axis=0) & (x < 0.05) & (y < 0.61 / (x - 0.05) + 1.3)
aur = {
    "[OII]7320,7330": snr(t, "OII_7320", "OII_7330"),
    "[SIII]6312": snr(t, "SIII_6312"),
    "[OIII]4363": snr(t, "OIII_4363"),
}
aur_any = np.max(list(aur.values()), axis=0)
mass = t["LOGMSTAR"]

out = [f"{len(t)} sample galaxies in FastSpecFit, {sf.sum()} star-forming ({sf.mean():.1%})", ""]
out.append("fraction of star-forming galaxies with S/N >= cut")
out.append(f"{'line':>16} " + " ".join(f"{'S/N>=' + str(c):>9}" for c in CUTS))
for name, v in {**aur, "any of the three": aur_any}.items():
    out.append(f"{name:>16} " + " ".join(f"{(v[sf] >= c).mean():9.1%}" for c in CUTS))
out += ["", "by log M* (FastSpecFit LOGMSTAR), star-forming only, any auroral line"]
out.append(f"{'log M*':>11} {'N':>7} " + " ".join(f"{'S/N>=' + str(c):>9}" for c in CUTS))
for lo, hi in zip(MASS_BINS[:-1], MASS_BINS[1:], strict=True):
    m = sf & (mass >= lo) & (mass < hi)
    out.append(
        f"{lo:5.1f}-{hi:4.1f} {m.sum():7d} "
        + " ".join(f"{(aur_any[m] >= c).mean():9.1%}" for c in CUTS)
    )
i = np.flatnonzero(t["TARGETID"] == TID)[0]
out += ["", f"{TID}: log M* {mass[i]:.2f}, star-forming {bool(sf[i])}"]
for name, v in aur.items():
    out.append(
        f"{name:>16} S/N {v[i]:6.1f}, percentile among star-forming {(v[sf] < v[i]).mean():.1%}"
    )
text = "\n".join(out)
print(text)
OUT.write_text(text + "\n")
