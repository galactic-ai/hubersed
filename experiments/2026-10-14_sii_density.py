"""Electron density of 39627770174637084 from the [SII] 6716/6731 ratio of EmFit and FastSpecFit.

QUESTION: what density does the measured [SII] doublet imply, compared with the Cue log n_H of our
fits (HR 2.03, ER 1.39, rmatrix 1.12)?

INPUTS: the EmFit row in UberSED knowledge/_evidence/2026-09-30_kin_vs_ion/emfit_rows_2obj.csv
(narrow plus outflow fluxes) and the iron SV3 bright FastSpecFit VAC in data/. SEED: 0 (error draws).

COMMAND: ``uv run python experiments/2026-10-14_sii_density.py`` from the repository root.

RESULT: see results/2026-10-14_sii_density.txt.
"""

import csv
from pathlib import Path

import numpy as np
import pyneb as pn
from astropy.io import fits

from hubersed.paths import PATHS

TID = 39627770174637084
EMFIT = (
    Path.home() / "Documents/UberSED/knowledge/_evidence/2026-09-30_kin_vs_ion/emfit_rows_2obj.csv"
)
FSF = PATHS["DATA"] / "fastspec-iron-sv3-bright.fits"
TEMPS = (8000.0, 10000.0, 12000.0)
N_DRAW = 2000

row = next(r for r in csv.DictReader(EMFIT.open()) if r["targetid"] == str(TID))
f16 = float(row["sii6716_flux"]) + float(row["sii6716_out_flux"])
f31 = float(row["sii6731_flux"]) + float(row["sii6731_out_flux"])
e16 = np.hypot(float(row["sii6716_flux_err"]), float(row["sii6716_out_flux_err"]))
e31 = np.hypot(float(row["sii6731_flux_err"]), float(row["sii6731_out_flux_err"]))
d = fits.getdata(FSF, "FASTSPEC")
k = np.flatnonzero(d["TARGETID"] == TID)[0]
fsf = [
    d[c][k] for c in ("SII_6716_FLUX", "SII_6716_FLUX_IVAR", "SII_6731_FLUX", "SII_6731_FLUX_IVAR")
]
fluxes = {
    "EmFit (narrow + outflow)": (f16, e16, f31, e31),
    "FastSpecFit": (fsf[0], fsf[1] ** -0.5, fsf[2], fsf[3] ** -0.5),
}

s2 = pn.Atom("S", 2)
rng = np.random.default_rng(0)
out = [f"{TID} [SII] 6716/6731 density (PyNeb {pn.__version__}, default S II atomic data)"]
for name, (a, ea, b, eb) in fluxes.items():
    ratio = a / b
    draws = rng.normal(a, ea, N_DRAW) / rng.normal(b, eb, N_DRAW)
    out.append(f"{name}: ratio {ratio:.3f} +- {draws.std():.3f}")
    for t in TEMPS:
        ne = s2.getTemDen(draws, tem=t, wave1=6716, wave2=6731)
        lo, med, hi = np.nanpercentile(np.log10(ne), [16, 50, 84])
        out.append(f"  T {t:.0f} K: log n_e {med:.2f} [{lo:.2f}, {hi:.2f}]")
lim = s2.getEmissivity(1e4, 1e0, wave=6716) / s2.getEmissivity(1e4, 1e0, wave=6731)
out.append(f"low-density limit ratio at 1e4 K: {lim:.3f}")
text = "\n".join(out)
print(text)
(PATHS["RESULTS"] / "2026-10-14_sii_density.txt").write_text(text + "\n")
