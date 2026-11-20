"""Score the model L max-likelihood spectra of the emission-line outliers against the SV3 recoadd.

QUESTION: before refitting, does the recoadd ivar or flux lower the misfit of 28, 34 and 80?
HYPOTHESIS: as 2026-11-16_emline_L_residuals. The recoadd explains the continuum excess if pixel-scale
chi2/N of 28, 34, 80 falls below 1.1 and no longer ranks with continuum S/N.
INPUTS: <TID>/L/maxl.npz (DR1 data, model) and <TID>/recoadd/resolution_matrix_rows_ivar.npz.
SEED: none.
COMMAND: ``PYTHONPATH=experiments/2026-10-15_broadline_free_amplitude uv run --no-sync python
experiments/2026-11-14_poco_psb_sfh/recoadd_check.py``.
RESULT: 2026-11-20. The recoadd ivar on the DR1 flux lowers continuum pixel chi2/N of 28, 34, 80 from
1.241, 1.298, 1.200 to 1.108, 1.097, 1.108; the recoadd flux adds nothing (1.112, 1.097, 1.103).
Assumptions: the model was fitted to the DR1 coadd with the DR1 R, so the recoadd
columns are indicative only.
FIGURES: results/2026-11-14_poco_psb_sfh/recoadd_check.txt.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fitpsb", HERE / "fit.py")
fit = importlib.util.module_from_spec(_spec)
sys.modules["fitpsb"] = fit
_spec.loader.exec_module(fit)
OUT = fit.OUT
TARGETS = {"39627818010675704": "28", "39627854832472157": "34", "39633318999755849": "80",
           "39627806493117457": "25", "39633127609469239": "64"}  # fmt: skip
# as 2026-11-16_emline_L_residuals: lines masked +-15 A, rest 3700-9000 A, running median 101 px
MASK_LINES = [3727.09, 3729.88, 3869.86, 4102.89, 4341.69, 4364.44, 4862.68, 4960.30, 5008.24,
              5877.25, 6302.05, 6549.86, 6564.61, 6585.27, 6718.29, 6732.67, 7137.77, 7321.94,
              7332.21]  # fmt: skip
WINDOWS = {"Ha+[NII]": (6540, 6595), "[OIII]5007": (4997, 5020)}
W_MEDIAN = 101


def scores(rest, flux, unc, model, good):
    """Return chi2/N overall, in the two line windows, and continuum total and pixel-scale chi2/N."""
    chi = np.where(good, (flux - model) / unc, np.nan)
    out = [np.nanmean(chi**2)]
    for a, b in WINDOWS.values():
        out.append(np.nanmean(chi[(rest >= a) & (rest < b)] ** 2))
    m = good & (rest >= 3700) & (rest < 9000)
    for lam in MASK_LINES:
        m &= np.abs(rest - lam) > 15
    c = chi[m]
    pix = c - median_filter(c, size=W_MEDIAN, mode="nearest")
    return out + [np.mean(c**2), np.mean(pix**2)]


lines = ["chi2/N of the model L max-likelihood spectrum (fitted to the DR1 coadd) on common good pixels",
         "dr1 = DR1 flux and ivar; ivar = DR1 flux with recoadd ivar; recoadd = recoadd flux and ivar",
         f"{'nn':>3} {'data':>8} {'N':>5} {'all':>6} {'Ha+[NII]':>9} {'[OIII]':>7} {'cont':>6} {'cont pix':>9}"]  # fmt: skip
for tid, nn in TARGETS.items():
    if not fit.matrix_path(int(tid), "recoadd").is_file():
        lines.append(f"{nn:>3} no recoadd yet")
        continue
    d = np.load(OUT / tid / "L" / "maxl.npz")
    z, rflux, runc, rgood, _ = fit.load_data(int(tid), "recoadd")
    assert np.isclose(z, float(d["z"])), "recoadd z differs from the DR1 fit"
    good = np.asarray(d["good"], bool) & rgood
    rest = np.asarray(d["wave"], float) / (1 + z)
    model = np.asarray(d["sp_best"], float)
    for label, flux, unc in (("dr1", d["flux"], d["unc"]), ("ivar", d["flux"], runc),
                             ("recoadd", rflux, runc)):  # fmt: skip
        s = scores(rest, np.asarray(flux, float), np.asarray(unc, float), model, good)
        lines.append(f"{nn:>3} {label:>8} {good.sum():5d} " + " ".join(
            f"{v:{w}.3f}" for v, w in zip(s, (6, 9, 7, 6, 9), strict=True)))  # fmt: skip
(OUT / "recoadd_check.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
