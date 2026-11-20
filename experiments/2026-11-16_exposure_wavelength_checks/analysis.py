"""Do the sky lines in the target fibers shift between exposures like the galaxy lines?

QUESTION: is the per-exposure line shift of 2026-11-16_exposure_line_centroids from the wavelength
calibration of that fiber and arm (sky lines shift too) or from the galaxy light (they do not)?
HYPOTHESIS: calibration explains a line if v_gal - v_sky passes the 99.7% chi2 point for n - 1 dof where
v_gal failed; it is excluded if the 99.7% upper limit on the sky scatter is below the galaxy excess.
INPUTS: results/2026-11-16_exposure_wavelength_checks/sky_<EXPID>_<CAMERA>.npz (extract.py) and
results/2026-11-16_exposure_line_centroids/centroids.txt.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-16_exposure_wavelength_checks/analysis.py``.
RESULT: pending.
FIGURES: results/2026-11-16_exposure_wavelength_checks/skylines.txt.
"""

# %%
import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2 as chi2_dist

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-16_exposure_wavelength_checks"
CENT = PATHS["RESULTS"] / "2026-11-16_exposure_line_centroids" / "centroids.txt"
C_KMS = 299792.458
# desispec 0.56.5 sky.py get_sky_lines up to the r-arm edge: vacuum, earth frame (A)
SKY_LINES = np.array([
    4359.55, 5199.27, 5462.38, 5578.85, 5891.47, 5897.51, 5917.04, 5934.63, 5955.11, 5978.80, 6172.39,
    6204.48, 6223.56, 6237.36, 6259.62, 6266.96, 6289.21, 6302.06, 6308.70, 6323.09, 6331.63, 6350.26,
    6358.10, 6365.57, 6388.31, 6467.83, 6472.63, 6500.48, 6506.85, 6524.31, 6534.88, 6545.95, 6555.44,
    6564.59, 6570.69, 6579.14, 6606.02, 6831.19, 6836.16, 6843.66, 6865.78, 6873.06, 6883.14, 6891.22,
    6902.79, 6914.52, 6925.11, 6941.43, 6950.99, 6971.84, 6980.36, 7005.76, 7013.36, 7050.01, 7242.08,
    7247.18, 7255.13, 7278.23, 7286.57, 7298.03, 7305.76, 7318.31, 7331.26, 7342.93, 7360.71, 7371.44,
    7394.23, 7403.92, 7431.82, 7440.56, 7468.66, 7473.57, 7475.80, 7481.74, 7485.52, 7495.74, 7526.02,
    7532.80, 7559.59, 7573.90, 7588.21, 7600.49, 7620.20, 7630.86,
])  # fmt: skip
MATCH = 1.0  # largest distance (A) from a peak to its list line
ARM = {"Hbeta": "b", "[OIII]5007": "b", "Halpha": "r", "[NII]6585": "r"}
NAMES = {39627818010675704: "28", 39633318999755849: "80", 39627806493117457: "25",
         39633127609469239: "64", 39627770174637084: "#14"}  # fmt: skip
P6_FIBERS = (3159, 3160, 3173, 3189, 3488)  # target fibers on petal 6 (28 and 25)


def fit_a(f, iv, s, ds):
    """Return the desispec 0.56.5 sky-line fit of one window for every fiber.

    The model is a + b S + c dS/dlambda with ivar weights and S the camera mean sky. A fiber passes if
    0 < err(c) < 0.1 A; the desispec chi2/dof < 2 cut is not applied, since S does not carry each
    fiber's LSF. Returns the line offset -c/b (A), its error, the pass flag and chi2/dof.
    """
    M = np.array([np.ones_like(s), s, ds])
    AA = np.einsum("jk,ik,lk->ijl", M, iv, M)
    BB = np.einsum("jk,ik->ij", M, iv * f)
    off, err, ok = np.full(len(f), np.nan), np.full(len(f), np.nan), np.zeros(len(f), bool)
    c2 = np.full(len(f), np.nan)
    for i in range(len(f)):
        try:
            Ai = np.linalg.inv(AA[i])
        except np.linalg.LinAlgError:
            continue
        X = Ai @ BB[i]
        E = np.sqrt(np.diag(Ai) * (np.diag(Ai) > 0))
        if X[1] <= 0:
            continue
        res = f[i] - X @ M
        var = 1 / (iv[i] + (iv[i] == 0)) + (0.05 * s) ** 2
        c2[i] = np.sum((iv[i] > 0) / var * res**2) / (s.size - 3)
        ok[i] = (E[2] > 0) & (E[2] < 0.1)
        off[i], err[i] = -X[2] / X[1], E[2] / X[1]
    return off, err, ok, c2


def fit_b(lam, f, iv):
    """Return the centroid (A) and error of one window, after a constant from its two end pixels."""
    if np.any(iv <= 0):
        return np.nan, np.nan
    cont = np.sum(iv[[0, -1]] * f[[0, -1]]) / iv[[0, -1]].sum()
    w = f - cont
    if w.sum() <= 0:
        return np.nan, np.nan
    mu = np.sum(lam * w) / w.sum()
    return mu, np.sqrt(np.sum(((lam - mu) / w.sum()) ** 2 / iv))


def wmean(v, e):
    """Return the weighted mean and its error, or nan if empty."""
    v, e = np.asarray(v, float), np.asarray(e, float)
    if v.size == 0:
        return np.nan, np.nan
    w = 1 / e**2
    return np.sum(w * v) / w.sum(), 1 / np.sqrt(w.sum())


def chi2(v, e):
    """Return chi2 about the weighted mean."""
    m, _ = wmean(v, e)
    return float(np.sum((v - m) ** 2 / e**2))


def excess(v, e):
    """Return the extra scatter (km/s) that brings chi2/dof about the weighted mean to 1."""

    def f(s):
        w = 1 / (e**2 + s**2)
        return np.sum(w * (v - np.sum(w * v) / w.sum()) ** 2) - (v.size - 1)

    return 0.0 if f(0.0) <= 0 else brentq(f, 0.0, 1e4)


def upper(v, e, q=0.003):
    """Return the 99.7% upper limit (km/s) on extra scatter about the weighted mean."""

    def f(s):
        w = 1 / (e**2 + s**2)
        return np.sum(w * (v - np.sum(w * v) / w.sum()) ** 2) - chi2_dist.ppf(q, v.size - 1)

    return 0.0 if f(0.0) <= 0 else brentq(f, 0.0, 1e4)


# %%
# per exposure and camera: Fit A and Fit B for every fiber and peak, keyed by the matched list line
fits = {}
for path in sorted(OUT.glob("sky_*.npz")):
    d = np.load(path)
    expid, cam = int(path.stem.split("_")[1]), path.stem.split("_")[2]
    meta = dict(zip([str(k) for k in d["meta_keys"]], d["meta"], strict=True))
    helio = meta["HELIOCOR"]
    wave, meansky = d["wave"], d["meansky"]
    dpix = int(3 // np.mean(np.gradient(wave))) + 1  # as sky.py
    c0 = d["win_wave"].shape[1] // 2
    sl = slice(c0 - dpix, c0 + dpix + 1)
    dsky = np.gradient(meansky) / np.gradient(wave)
    good = (d["win_mask"] == 0) & (d["fiberstatus"][:, None, None] == 0)
    per = {}
    for j, p in enumerate(d["peaks"]):
        lam = d["win_wave"][j, sl]
        near = np.abs(SKY_LINES - wave[p] / helio)
        if near.min() > MATCH:
            continue
        lvac = float(SKY_LINES[np.argmin(near)])
        f = d["win_flux"][:, j, sl].astype(float)
        iv = d["win_ivar"][:, j, sl].astype(float) * good[:, j, sl]
        off, err, ok, c2 = fit_a(
            f, iv, meansky[p - dpix : p + dpix + 1], dsky[p - dpix : p + dpix + 1]
        )
        mu_c, _ = fit_b(lam, meansky[p - dpix : p + dpix + 1], np.ones(lam.size))
        v_cam = C_KMS * (mu_c / (lvac * helio) - 1)
        vb = np.full(len(f), np.nan)
        eb = np.full(len(f), np.nan)
        for i in range(len(f)):
            mu, e = fit_b(lam, f[i], iv[i])
            vb[i], eb[i] = C_KMS * (mu / (lvac * helio) - 1), C_KMS * e / (lvac * helio)
        per[lvac] = dict(
            va=C_KMS * off / wave[p], ea=C_KMS * err / wave[p], oka=ok, c2=c2, vb=vb, eb=eb,
            okb=np.isfinite(vb), v_cam=v_cam,
        )  # fmt: skip
    fits[(expid, cam)] = dict(d=d, meta=meta, per=per)

# %%
peaks_txt = ["Sky peaks per exposure and camera (desispec rule) matched to the desispec list (vacuum,",
             "earth frame); v_cam = Fit B centroid of the camera mean sky about lambda_vac * HELIOCOR (km/s);",
             "n_A, n_B = fibers passing Fit A and Fit B; chi2 = median Fit A chi2/dof of those fibers.",
             f"{'expid':>6} {'cam':>3} {'lambda_vac':>10} {'v_cam':>7} {'n_A':>4} {'n_B':>4} {'chi2':>6}"]  # fmt: skip
cams = ["Per exposure and camera: Fit A offsets (fiber minus camera mean sky, km/s) over all fibers and",
        "sky fibers (median, NMAD), and the target fiber: A, A + v_cam, B, and its A percentile.",
        f"{'expid':>6} {'cam':>3} {'all med':>7} {'nmad':>5} {'sky med':>7} {'nmad':>5} {'nn':>3} "
        f"{'fiber':>5} {'A':>14} {'A+cam':>8} {'B':>14} {'pct':>4}"]  # fmt: skip


def combine(per, rows, keys=None, absolute=False):
    """Return Fit A and Fit B weighted means over the given peaks for the given fiber rows."""
    keys = per.keys() if keys is None else keys
    out = []
    for r in rows:
        a = [(per[k]["va"][r] + (per[k]["v_cam"] if absolute else 0.0), per[k]["ea"][r]) for k in keys
             if per[k]["oka"][r] and (np.isfinite(per[k]["v_cam"]) or not absolute)]  # fmt: skip
        b = [(per[k]["vb"][r], per[k]["eb"][r]) for k in keys if per[k]["okb"][r]]
        out.append((wmean(*zip(*a, strict=True)) if a else (np.nan, np.nan),
                    wmean(*zip(*b, strict=True)) if b else (np.nan, np.nan)))  # fmt: skip
    return out


def nmad(x):
    """Return the normalised median absolute deviation."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))


for (expid, cam), fx in sorted(fits.items()):
    d, per = fx["d"], fx["per"]
    for k, p in per.items():
        mc2 = np.median(p["c2"][p["oka"]]) if p["oka"].any() else np.nan
        peaks_txt.append(f"{expid:>6} {cam:>3} {k:10.2f} {p['v_cam']:+7.2f} {p['oka'].sum():4d} "
                         f"{p['okb'].sum():4d} {mc2:6.2f}")  # fmt: skip
    allf = combine(per, range(len(d["fiber"])))
    va = np.array([x[0][0] for x in allf])
    fin = np.isfinite(va)
    sky = fin & (d["objtype"] == "SKY")
    for tid, row in zip(d["tids"], d["target_rows"], strict=True):
        (a, ea), (b, eb) = allf[row]
        (aa, _), _ = combine(per, [row], absolute=True)[0]
        pct = 100 * np.mean(va[fin] < a) if np.isfinite(a) else np.nan
        cams.append(f"{expid:>6} {cam:>3} {np.median(va[fin]):+7.2f} {nmad(va[fin]):5.2f} "
                    f"{np.median(va[sky]):+7.2f} {nmad(va[sky]):5.2f} {NAMES[int(tid)]:>3} "
                    f"{int(d['fiber'][row]):>5} {a:+7.2f} +-{ea:5.2f} {aa:+8.2f} {b:+7.2f} +-{eb:5.2f} "
                    f"{pct:4.0f}")  # fmt: skip

# %%
p6 = ["Petal 6 target fibers over the 9 exposures of 28 and 25: Fit A offset per exposure (km/s), chi2",
      "about the weighted mean (99.7% point for n - 1 dof) and the mean (offset if beyond 3 err).",
      f"{'cam':>3} {'fiber':>5} {'n':>2} {'mean':>14} {'chi2':>7} {'99.7%':>6} flags"]  # fmt: skip
for arm in "br":
    keys6 = sorted(k for k in fits if k[1] == f"{arm}6")
    for fib in P6_FIBERS:
        v, e = [], []
        for k in keys6:
            d = fits[k]["d"]
            row = int(np.where(d["fiber"] == fib)[0][0])
            (a, ea), _ = combine(fits[k]["per"], [row])[0]
            if np.isfinite(a):
                v.append(a)
                e.append(ea)
        if len(v) < 2:
            continue
        v, e = np.array(v), np.array(e)
        m, em = wmean(v, e)
        c2, thr = chi2(v, e), float(chi2_dist.ppf(0.997, v.size - 1))
        flags = ("varies " if c2 > thr else "") + ("offset" if abs(m) > 3 * em else "")
        p6.append(f"{arm}6 {fib:>5} {v.size:2d} {m:+7.2f} +-{em:5.2f} {c2:7.2f} {thr:6.2f} "
                  f"{flags or '-'}")  # fmt: skip

# %%
# galaxy per-exposure velocities from the 11-16 table
gal = {}
for row in CENT.read_text().split("\n\n")[0].splitlines():
    p = row.split()
    if len(p) >= 10 and p[8] == "+-" and p[1].isdigit():
        gal.setdefault((p[0], p[6]), {})[int(p[1])] = (float(p[7]), float(p[9]))


def sky_series(tid, arm, absolute):
    """Return {expid: ((A, err), (B, err))} of a target fiber over the peaks common to its exposures."""
    keys = [k for k in fits if k[1][0] == arm and tid in fits[k]["d"]["tids"]]
    if not keys:
        return {}, []
    common = None
    for k in keys:
        d = fits[k]["d"]
        row = int(d["target_rows"][list(d["tids"]).index(tid)])
        valid = {lv for lv, p in fits[k]["per"].items() if p["oka"][row] and p["okb"][row]}
        common = valid if common is None else common & valid
    out = {}
    for k in keys:
        d = fits[k]["d"]
        row = int(d["target_rows"][list(d["tids"]).index(tid)])
        out[k[0]] = combine(fits[k]["per"], [row], sorted(common), absolute)[0]
    return out, sorted(common)


test = ["Galaxy minus sky per line: chi2 over exposures of v_gal, v_gal - v_sky(A + cam), v_gal - v_sky(B),",
        "99.7% point; galaxy excess; sky error (median) and the 99.7% upper limit on the sky scatter.",
        f"{'nn':>3} {'line':>11} {'n':>2} {'gal':>7} {'-A':>7} {'-B':>7} {'99.7%':>6} {'excess':>6} "
        f"{'sky err':>7} {'upA':>6} {'upB':>6} {'peaks':>18} verdict"]  # fmt: skip
armtxt = ["b - r per exposure: sky (A + cam, B) and galaxy (11-16) velocity differences (km/s); galaxy",
          "minus sky A flagged beyond 3 err.",
          f"{'nn':>3} {'expid':>6} {'sky A':>14} {'sky B':>14} {'pair':>18} {'gal':>14} {'gal - A':>14}"]  # fmt: skip
for tid, nn in NAMES.items():
    series = {arm: sky_series(tid, arm, absolute=True) for arm in "br"}
    for line, arm in ARM.items():
        g = gal.get((nn, line), {})
        s, common = series[arm]
        x = sorted(set(g) & set(s))
        if len(x) < 2:
            continue
        vg, eg = np.array([g[k][0] for k in x]), np.array([g[k][1] for k in x])
        va, ea = np.array([s[k][0][0] for k in x]), np.array([s[k][0][1] for k in x])
        vb, eb = np.array([s[k][1][0] for k in x]), np.array([s[k][1][1] for k in x])
        thr = float(chi2_dist.ppf(0.997, len(x) - 1))
        cg, ca, cb = chi2(vg, eg), chi2(vg - va, np.hypot(eg, ea)), chi2(vg - vb, np.hypot(eg, eb))
        exc, err = excess(vg, eg), float(np.median(ea))
        ua, ub = upper(va, ea), upper(vb, eb)
        if exc > 0 and err >= exc / 3:
            verdict = "sky too noisy"
        elif cg > thr and ca <= thr:
            verdict = "calibration explains (A)"
        elif exc > 0 and ua < exc:
            verdict = "calibration excluded (A)"
        else:
            verdict = "-"
        pk = ",".join(f"{c:.0f}" for c in common)
        test.append(f"{nn:>3} {line:>11} {len(x):2d} {cg:7.2f} {ca:7.2f} {cb:7.2f} {thr:6.2f} "
                    f"{exc:6.2f} {err:7.2f} {ua:6.2f} {ub:6.2f} {pk:>18} {verdict}")  # fmt: skip
    sb, sr = series["b"][0], series["r"][0]
    for x in sorted(set(sb) & set(sr)):
        da = sb[x][0][0] - sr[x][0][0], np.hypot(sb[x][0][1], sr[x][0][1])
        db = sb[x][1][0] - sr[x][1][0], np.hypot(sb[x][1][1], sr[x][1][1])
        for bl in ("Hbeta", "[OIII]5007"):
            gb, gr = gal.get((nn, bl), {}).get(x), gal.get((nn, "Halpha"), {}).get(x)
            if gb is None or gr is None:
                continue
            dg = gb[0] - gr[0], np.hypot(gb[1], gr[1])
            r = dg[0] - da[0], np.hypot(dg[1], da[1])
            armtxt.append(f"{nn:>3} {x:>6} {da[0]:+7.2f} +-{da[1]:5.2f} {db[0]:+7.2f} +-{db[1]:5.2f} "
                          f"{bl + ' - Halpha':>18} {dg[0]:+7.2f} +-{dg[1]:5.2f} {r[0]:+7.2f} +-{r[1]:5.2f}"
                          f"{' offset' if abs(r[0]) > 3 * r[1] else ''}")  # fmt: skip
text = "\n\n".join("\n".join(t) for t in (peaks_txt, cams, p6, test, armtxt))
(OUT / "skylines.txt").write_text(text + "\n")
print(text)
