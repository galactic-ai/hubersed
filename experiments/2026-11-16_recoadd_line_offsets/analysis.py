"""Do the SV3 recoadds move the emission-line cores relative to the DR1 coadd?

QUESTION: are the larger line offsets of 28 under L_recoadd already in the recoadd data, with no model?
HYPOTHESIS: the recoadd shifts a line if |dv(recoadd - DR1)| > 3 err, with err from both uncertainties
in quadrature (an overestimate, since both coadds share exposures).
INPUTS: results/2026-11-14_poco_psb_sfh/<TID>/L/maxl.npz (DR1 data, line window) and
<TID>/recoadd/resolution_matrix_rows_ivar.npz; for 39627770174637084 (issue 14) the
2026-11-07_emline_ladder steps 7 (DR1, line window) and 8 (recoadd) predict.npz.
SEED: none.
COMMAND: ``PYTHONPATH=experiments/2026-10-15_broadline_free_amplitude uv run --no-sync python
experiments/2026-11-16_recoadd_line_offsets/analysis.py``.
RESULT: 2026-11-16, offsets.txt. The recoadd moves Halpha of 28 by -1.96 +- 0.21 km/s and [OIII]5007
of 25 by +2.49 +- 0.63, about the change in their L to L_recoadd line offsets. 34 (one exposure) is
unchanged, as it must be; 64, 80 and #14 move less than 3 err. Assumptions: window from the DR1 model
line; err ignores the shared exposures.
FIGURES: results/2026-11-16_recoadd_line_offsets/offsets.txt.
"""

# %%
import importlib.util
import sys

import astropy.units as u
import numpy as np

from hubersed.conversion import to_flambda
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.sps.parameter_file import WAVE_OBS

_spec = importlib.util.spec_from_file_location(
    "fitpsb", PATHS["ROOT"] / "experiments" / "2026-11-14_poco_psb_sfh" / "fit.py"
)
fit = importlib.util.module_from_spec(_spec)
sys.modules["fitpsb"] = fit
_spec.loader.exec_module(fit)

SRC = PATHS["RESULTS"] / "2026-11-14_poco_psb_sfh"
LADDER = PATHS["RESULTS"] / "2026-11-07_emline_ladder_39627770174637084"
OUT = PATHS["RESULTS"] / "2026-11-16_recoadd_line_offsets"
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = {  # gallery105 number, from results/2026-11-13_gallery105/gallery105.tsv
    "39627818010675704": "28",
    "39627854832472157": "34",
    "39633318999755849": "80",
    "39627806493117457": "25",
    "39633127609469239": "64",
}
TID14 = "39627770174637084"
C_KMS = 299792.458
# rest vacuum wavelengths and side bands (A), as 2026-11-16_emline_L_residuals
LINES = {
    "Hbeta": (4862.68, ((4830, 4845), (4880, 4895))),
    "[OIII]5007": (5008.24, ((4980, 4995), (5020, 5035))),
    "Halpha": (6564.61, ((6520, 6540), (6600, 6615))),
}
CORE = (
    0.1  # line core = model line above this fraction of its peak, as 2026-11-16_emline_L_residuals
)


def side_continuum(rest, f, good, bands):
    """Return a straight line fitted to ``f`` in the two side bands, evaluated at ``rest``."""
    s = good & np.any([(rest >= a) & (rest < b) for a, b in bands], axis=0)
    return np.polyval(np.polyfit(rest[s], f[s], 1), rest)


def core_window(rest, lm, lam0):
    """Return the contiguous pixels around the model line peak where the model line is > CORE of it."""
    near = np.flatnonzero((rest > lam0 - 6) & (rest < lam0 + 6))
    k = near[np.argmax(lm[near])]
    if lm[k] <= 0:
        return None
    lo = hi = k
    while lo - 1 >= near[0] and lm[lo - 1] > CORE * lm[k]:
        lo -= 1
    while hi + 1 <= near[-1] and lm[hi + 1] > CORE * lm[k]:
        hi += 1
    win = np.zeros(rest.size, bool)
    win[lo : hi + 1] = True
    return win


def centroid(rest, f, unc, good, win, bands):
    """Return the continuum-subtracted flux-weighted centroid in ``win`` and its error."""
    lf = f - side_continuum(rest, f, good, bands)
    lam, w, e = rest[win], lf[win], unc[win]
    mu = np.sum(lam * w) / w.sum()
    return mu, np.sqrt(np.sum(((lam - mu) / w.sum()) ** 2 * e**2))


def flam(x):
    """Return maggies on WAVE_OBS in flux density."""
    return to_flambda(WAVE_OBS * u.AA, np.asarray(x, float) * u.mgy).value


def load(tid):
    """Return z, the model, and flux, uncertainty and good mask of the DR1 coadd and the recoadd."""
    if tid == TID14:
        p7 = np.load(LADDER / "step7_dr1_rmatrix_cols_ratio_neff1024" / "predict.npz")
        p8 = np.load(LADDER / "step8_recoadd_rmatrix_cols_ratio_neff1024" / "predict.npz")
        for p in (p7, p8):
            assert np.allclose(p["wave"], WAVE_OBS), "predict.npz grid differs from WAVE_OBS"
        z = float(load_spectrum(int(tid)).redshift.value)
        dr1 = (p7["flux"], p7["unc"], np.asarray(p7["mask"], bool))
        rec = (p8["flux"], p8["unc"], np.asarray(p8["mask"], bool))
        return z, p7["spec_ml"], dr1, rec
    d = np.load(SRC / tid / "L" / "maxl.npz")
    z = float(d["z"])
    zr, rflux, runc, rgood, _ = fit.load_data(int(tid), "recoadd")
    assert np.isclose(zr, z), "recoadd z differs from the DR1 fit"
    return z, d["sp_best"], (d["flux"], d["unc"], np.asarray(d["good"], bool)), (rflux, runc, rgood)


# %%
rows = [
    "Line-core centroid of the recoadd minus the DR1 coadd, data only: continuum from the side bands,",
    "core window where the DR1 max-likelihood model line (L; ladder step 7 for #14) is > 10% of its peak,",
    "pixels good in both coadds. err = both centroid errors in quadrature. Flag beyond 3 err.",
    f"{'nn':>3} {'TARGETID':>18} {'line':>11} {'dv rec-DR1 km/s':>18} {'px':>3} flag",
]
for tid, nn in (TARGETS | {TID14: "#14"}).items():
    z, model, (fd, ud, gd), (fr, ur, gr) = load(tid)
    rest = WAVE_OBS / (1 + z)
    good = gd & gr
    for name, (lam0, bands) in LINES.items():
        lm = flam(model) - side_continuum(rest, flam(model), good, bands)
        win = core_window(rest, lm, lam0)
        if win is None or not good[win].all() or win.sum() < 3:
            rows.append(f"{nn:>3} {tid:>18} {name:>11}  no clean core window")
            continue
        mu_d, e_d = centroid(rest, flam(fd), flam(ud), good, win, bands)
        mu_r, e_r = centroid(rest, flam(fr), flam(ur), good, win, bands)
        dv, err = C_KMS * (mu_r - mu_d) / mu_d, C_KMS * np.hypot(e_r, e_d) / mu_d
        flag = "shift" if abs(dv) > 3 * err else "-"
        rows.append(f"{nn:>3} {tid:>18} {name:>11} {dv:+9.2f} +- {err:5.2f} {win.sum():3d} {flag}")
text = "\n".join(rows)
(OUT / "offsets.txt").write_text(text + "\n")
print(text)
