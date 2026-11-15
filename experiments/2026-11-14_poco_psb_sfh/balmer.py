"""Balmer decrement of gallery105 99 from FastSpecFit and from our Cue fits.

QUESTION: Is 99's Halpha/Hbeta near the Case B value (no dust, consistent with dust2 ~ 0 in the fits)?
HYPOTHESIS: none registered. Descriptive; compared with Case B 2.86 (from memory, not checked here).
INPUTS: data/fastspec-iron-sv3-bright.fits (HALPHA_FLUX, HBETA_FLUX and their IVAR, SV3 coadd);
results/2026-11-14_poco_psb_sfh/39633421919584782/<model>/maxl.npz (predict.py) for the Cue models:
flux, unc, good, sp_best and sp_cont (stellar continuum, nebular emission off).
METHOD: (1) FastSpecFit fluxes. (2) For each Cue model with sp_cont: line flux = sum over good pixels of
(data - sp_cont) * dlambda in rest +-6 A windows around Halpha 6564.61 and Hbeta 4862.68 A (vacuum,
from memory), f_lambda; errors from unc. The model's own ratio uses sp_best - sp_cont in the same
windows. The +-6 A windows exclude [NII] 6549.9 and 6585.3 A.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-14_poco_psb_sfh/balmer.py``.
RESULT: 2026-11-15, balmer.txt. FastSpecFit 4.09 +- 0.86 (+1.4 sigma from Case B). Data minus our
stellar continuum: 2.51 +- 0.53 (AC), 2.60 +- 0.57 (BC, AC0), 2.36-2.46 +- 1.3 (main coadd), all within
1 sigma of Case B; the gap to FastSpecFit is Hbeta (about 10 vs 5.2), set by how deep the stellar Hbeta
absorption is. Model ratios 2.94 (AC) to 3.27 (AC0, dust2 0.47). The decrement does not separate the
low- and high-dust solutions. Assumptions: +-6 A windows; nebular continuum left in the residual.
FIGURES: none (balmer.txt).
"""

import astropy.units as u
import numpy as np
from astropy.io import fits

from hubersed.conversion import to_flambda
from hubersed.paths import PATHS

TID = 39633421919584782
OUT = PATHS["RESULTS"] / "2026-11-14_poco_psb_sfh"
LINES = {"Halpha": 6564.61, "Hbeta": 4862.68}
HALF = 6.0
CASE_B = 2.86


def ratio(a, ea, b, eb):
    """Return a/b and its error for independent errors."""
    r = a / b
    return r, abs(r) * np.hypot(ea / a, eb / b)


lines = [f"99 ({TID}) Balmer decrement; Case B {CASE_B}"]
with fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True) as h:
    f, m = h["FASTSPEC"].data, h["METADATA"].data
    k = int(np.flatnonzero(np.asarray(m["TARGETID"], np.int64) == TID)[0])
    ha, hb = float(f["HALPHA_FLUX"][k]), float(f["HBETA_FLUX"][k])
    eha, ehb = (1 / np.sqrt(float(f[c][k])) for c in ("HALPHA_FLUX_IVAR", "HBETA_FLUX_IVAR"))
r, er = ratio(ha, eha, hb, ehb)
lines.append(f"FastSpecFit (SV3 coadd): Halpha {ha:.2f} +- {eha:.2f}, Hbeta {hb:.2f} +- {ehb:.2f} "
             f"(1e-17 erg/s/cm2); ratio {r:.2f} +- {er:.2f}, {(r - CASE_B) / er:+.1f} sigma from Case B")  # fmt: skip

for name in ("AC", "BC", "AC0", "AC_main", "BC_main"):
    path = OUT / str(TID) / name / "maxl.npz"
    if not path.is_file():
        continue
    d = np.load(path)
    if "sp_cont" not in d.files:
        lines.append(f"{name}: no sp_cont (rerun predict.py)")
        continue
    wave = np.asarray(d["wave"], float)
    rest = wave / (1 + float(d["z"]))
    good = d["good"]
    dlam = np.gradient(wave)

    def flam(x, wave=wave):
        return to_flambda(wave * u.AA, np.asarray(x, float) * u.mgy).value

    data, cont, best = flam(d["flux"]), flam(d["sp_cont"]), flam(d["sp_best"])
    err = flam(np.where(good, d["unc"], 0.0))
    vals = {}
    for ln, lam in LINES.items():
        s = good & (np.abs(rest - lam) <= HALF)
        vals[ln] = (
            float(np.sum((data - cont)[s] * dlam[s])),
            float(np.sqrt(np.sum((err[s] * dlam[s]) ** 2))),
            float(np.sum((best - cont)[s] * dlam[s])),
            int(s.sum()),
        )
    (a, ea, ma, na), (b, eb, mb, nb) = vals["Halpha"], vals["Hbeta"]
    r, er = ratio(a, ea, b, eb)
    lines.append(f"{name}: data - continuum Halpha {a:.2f} +- {ea:.2f} ({na} px), Hbeta {b:.2f} +- "
                 f"{eb:.2f} ({nb} px); ratio {r:.2f} +- {er:.2f}, {(r - CASE_B) / er:+.1f} sigma from "
                 f"Case B; model ratio {ma / mb:.2f}")  # fmt: skip
text = "\n".join(lines)
(OUT / "balmer.txt").write_text(text + "\n")
print(text)
