"""Do the exposures of one target agree in emission-line centroid, per arm?

QUESTION: do the per-exposure line centroids of gallery105 28, 25, 80, 64 (and the #14 target)
disagree beyond their noise, do b-arm and r-arm lines disagree per exposure (34 too), and do the
iron and recoadd masks and weights then give the recoadd - DR1 shifts of 2026-11-16_recoadd_line_offsets?
HYPOTHESIS: exposures disagree if chi2 about their weighted mean exceeds the 99.7% point for n - 1 dof;
an arm offset is significant beyond 3 err; a rebuilt coadd shift matches test A within 3 of its err.
INPUTS: results/2026-11-16_exposure_line_centroids/rows_<TID>.npz (extract.py), the DR1 model line of
2026-11-14_poco_psb_sfh <TID>/L/maxl.npz (ladder step 7 predict.npz for #14), and offsets.txt of
2026-11-16_recoadd_line_offsets.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-16_exposure_line_centroids/analysis.py``.
RESULT: exposures disagree in 28 (Hbeta, [OIII], Halpha; excess 2.2-3.6 km/s), 80 (Halpha, [NII];
1.1-1.4), 25 ([OIII]; 2.0) and the #14 control (Halpha; 0.6); b - r offsets of 2-3 km/s in 34, 80 and
#14; the rebuilt coadds give all 15 test A shifts.
FIGURES: results/2026-11-16_exposure_line_centroids/centroids.txt.
"""

# %%
import astropy.units as u
import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2 as chi2_dist

from hubersed.conversion import to_flambda
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.sps.parameter_file import WAVE_OBS

OUT = PATHS["RESULTS"] / "2026-11-16_exposure_line_centroids"
SRC = PATHS["RESULTS"] / "2026-11-14_poco_psb_sfh"
LADDER = PATHS["RESULTS"] / "2026-11-07_emline_ladder_39627770174637084"
TEST_A = PATHS["RESULTS"] / "2026-11-16_recoadd_line_offsets" / "offsets.txt"
TID14 = "39627770174637084"
TARGETS = {"39627818010675704": "28", "39627854832472157": "34", "39633318999755849": "80",
           "39627806493117457": "25", "39633127609469239": "64", TID14: "#14"}  # fmt: skip
C_KMS = 299792.458
# rest vacuum wavelengths and side bands (A), as 2026-11-16_emline_L_residuals
LINES = {
    "Hbeta": (4862.68, ((4830, 4845), (4880, 4895))),
    "[OIII]5007": (5008.24, ((4980, 4995), (5020, 5035))),
    "Halpha": (6564.61, ((6520, 6540), (6600, 6615))),
    "[NII]6585": (6585.27, ((6520, 6540), (6600, 6615))),
}
ARM_PAIRS = (("Hbeta", "Halpha"), ("[OIII]5007", "Halpha"))  # b-arm line minus r-arm line
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


def velocity(rest, f, ivar, good, win, lam0, bands):
    """Return the core centroid as a velocity (km/s) about lam0 at the DR1 z, and its error."""
    lf = f - side_continuum(rest, f, good, bands)
    lam, w, e = rest[win], lf[win], 1 / np.sqrt(ivar[win])
    mu = np.sum(lam * w) / w.sum()
    err = np.sqrt(np.sum(((lam - mu) / w.sum()) ** 2 * e**2))
    return C_KMS * (mu / lam0 - 1), C_KMS * err / lam0


def model_of(tid):
    """Return the DR1 redshift and max-likelihood model (maggies on WAVE_OBS) that set the windows."""
    if tid == TID14:
        p = np.load(LADDER / "step7_dr1_rmatrix_cols_ratio_neff1024" / "predict.npz")
        return float(load_spectrum(int(tid)).redshift.value), p["spec_ml"]
    d = np.load(SRC / tid / "L" / "maxl.npz")
    return float(d["z"]), d["sp_best"]


def coadd(flux, ivar, mask, cosmic):
    """Return the ivar-weighted coadd of one arm with masks, as iron coaddition (unmasked if all bad)."""
    w = ivar * (mask == 0) * ~cosmic
    bad = w.sum(0) == 0
    w[:, bad] = ivar[:, bad]
    tiv = w.sum(0)
    return (w * flux).sum(0) / np.where(tiv > 0, tiv, 1), tiv


def excess(v, e):
    """Return the extra scatter (km/s) that brings chi2/dof about the weighted mean to 1."""

    def f(s):
        w = 1 / (e**2 + s**2)
        m = np.sum(w * v) / w.sum()
        return np.sum(w * (v - m) ** 2) - (v.size - 1)

    return 0.0 if f(0.0) <= 0 else brentq(f, 0.0, 100.0)


# %%
per_exp = ["Per exposure: core centroid velocity about the rest wavelength at the DR1 z (km/s), iron ivar,",
           "pixels unmasked in that exposure; cosmic = iron / PR 2377-new-ivar cosmic pixels in the core.",
           f"{'nn':>3} {'expid':>6} {'night':>8} {'tile':>5} {'fiber':>5} {'heliocor':>10} {'line':>11} "
           f"{'v km/s':>16} {'px':>3} {'cosmic':>7}"]  # fmt: skip
consist = ["Consistency of the exposures per line: chi2 about the weighted mean, n - 1 dof, 99.7% point,",
           "and the excess scatter that brings chi2/dof to 1 (Guy+23: 0.8 km/s between exposures).",
           f"{'nn':>3} {'line':>11} {'n':>2} {'chi2':>8} {'99.7%':>7} {'excess km/s':>12} flag"]  # fmt: skip
arms = ["Arms per exposure: b-arm line minus r-arm line velocity (km/s); chi2 of that difference over",
        "exposures about its weighted mean. The mean includes any real velocity difference of the lines.",
        f"{'nn':>3} {'pair':>18} {'expid':>6} {'b - r km/s':>16} flag"]  # fmt: skip
rebuild = ["Coadds rebuilt from the exposure rows: iron (iron ivar and cosmics) and recoadd (new ivar,",
           "PR 2377 cosmics on it); recoadd - iron core centroid vs test A (offsets.txt).",
           f"{'nn':>3} {'line':>11} {'rebuilt km/s':>16} {'test A km/s':>16} match"]  # fmt: skip
shift_a = {}
for row in TEST_A.read_text().splitlines():
    p = row.split()
    if len(p) >= 6 and p[4] == "+-":
        shift_a[(p[0], p[2])] = (float(p[3]), float(p[5]))
for tid, nn in TARGETS.items():
    rows = np.load(OUT / f"rows_{tid}.npz")
    z, sp = model_of(tid)
    vel = {}
    for name, (lam0, bands) in LINES.items():
        lo, hi = (bands[0][0] - 5) * (1 + z), (bands[1][1] + 5) * (1 + z)
        arm = next(
            (a for a in "br" if rows[f"{a}_wave"][0] < lo and rows[f"{a}_wave"][-1] > hi), None
        )
        if arm is None:
            per_exp.append(
                f"{nn:>3} {'':>6} {'':>8} {'':>5} {'':>5} {'':>10} {name:>11}  not inside one arm"
            )
            continue
        wave = rows[f"{arm}_wave"]
        i0 = int(np.argmin(np.abs(WAVE_OBS - wave[0])))
        assert np.allclose(WAVE_OBS[i0 : i0 + wave.size], wave, atol=1e-4), "arm grid off WAVE_OBS"
        rest = wave / (1 + z)
        model = to_flambda(wave * u.AA, sp[i0 : i0 + wave.size] * u.mgy).value
        allgood = np.ones(wave.size, bool)
        win = core_window(rest, model - side_continuum(rest, model, allgood, bands), lam0)
        flux, iv, mask = rows[f"{arm}_flux"], rows[f"{arm}_ivar_iron"], rows[f"{arm}_mask"]
        ci, cn = rows[f"{arm}_cosmic_iron"], rows[f"{arm}_cosmic_new"]
        meta, keys = rows[f"{arm}_meta"], [str(k) for k in rows["meta_keys"]]
        v, e = [], []
        for j in range(flux.shape[0]):
            good = (iv[j] > 0) & (mask[j] == 0)
            m = dict(zip(keys, meta[j], strict=True))
            head = (f"{nn:>3} {int(m['EXPID']):>6} {int(m['NIGHT']):>8} {int(m['TILEID']):>5} "
                    f"{int(rows[f'{arm}_fiber'][j]):>5} {m['HELIOCOR']:>10.7f} {name:>11}")  # fmt: skip
            if win is None or not good[win].all() or win.sum() < 3:
                per_exp.append(f"{head}  no clean core window")
                continue
            vj, ej = velocity(rest, flux[j], iv[j], good, win, lam0, bands)
            v.append(vj)
            e.append(ej)
            vel[(name, int(m["EXPID"]))] = (vj, ej)
            per_exp.append(f"{head} {vj:+8.2f} +- {ej:5.2f} {win.sum():3d} "
                           f"{ci[j, win].sum():>3d}/{cn[j, win].sum():<3d}")  # fmt: skip
        if len(v) >= 2:
            v, e = np.array(v), np.array(e)
            w = 1 / e**2
            c2 = float(np.sum(w * (v - np.sum(w * v) / w.sum()) ** 2))
            thr = float(chi2_dist.ppf(0.997, v.size - 1))
            consist.append(f"{nn:>3} {name:>11} {v.size:2d} {c2:8.2f} {thr:7.2f} {excess(v, e):12.2f} "
                           f"{'disagree' if c2 > thr else '-'}")  # fmt: skip
        if flux.shape[0] >= 2 and win is not None:
            fd, td = coadd(flux, iv, mask, ci)
            fr, tr = coadd(flux, rows[f"{arm}_ivar_new"], mask, cn)
            gd, gr = td > 0, tr > 0
            if gd[win].all() and gr[win].all():
                vd, ed = velocity(rest, fd, td, gd, win, lam0, bands)
                vr, er = velocity(rest, fr, tr, gr, win, lam0, bands)
                dv, err = vr - vd, np.hypot(er, ed)
                a = shift_a.get((nn, name))
                ok = "-" if a is None else ("yes" if abs(dv - a[0]) < 3 * a[1] else "no")
                atext = "" if a is None else f"{a[0]:+8.2f} +- {a[1]:5.2f}"
                rebuild.append(f"{nn:>3} {name:>11} {dv:+8.2f} +- {err:5.2f} {atext:>16} {ok}")
    for b, r in ARM_PAIRS:
        expids = sorted({k[1] for k in vel if k[0] == b} & {k[1] for k in vel if k[0] == r})
        d = [
            (vel[(b, x)][0] - vel[(r, x)][0], np.hypot(vel[(b, x)][1], vel[(r, x)][1]))
            for x in expids
        ]
        for x, (dd, ee) in zip(expids, d, strict=True):
            arms.append(f"{nn:>3} {b + ' - ' + r:>18} {x:>6} {dd:+8.2f} +- {ee:5.2f} "
                        f"{'offset' if abs(dd) > 3 * ee else '-'}")  # fmt: skip
        if len(d) >= 2:
            dv, de = np.array([x[0] for x in d]), np.array([x[1] for x in d])
            w = 1 / de**2
            c2 = float(np.sum(w * (dv - np.sum(w * dv) / w.sum()) ** 2))
            thr = float(chi2_dist.ppf(0.997, dv.size - 1))
            arms.append(f"{nn:>3} {b + ' - ' + r:>18} {'chi2':>6} {c2:8.2f} (99.7% {thr:.2f}) "
                        f"{'varies' if c2 > thr else '-'}")  # fmt: skip
text = "\n\n".join("\n".join(t) for t in (per_exp, consist, arms, rebuild))
(OUT / "centroids.txt").write_text(text + "\n")
print(text)
