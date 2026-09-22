"""
Old-population AGE scan for 42580 (ChangHoon's idea), done with genuine single-age
SSPs and the full physics: free logzsol, KC13 dust (free A_V + slope delta),
velocity smoothing (~galaxy sigma_v). No bins / step basis / mass-redistribution.

Model:  smooth_v[ dust_KC13(A_V, delta) * (f_old*SSP(A_old,Z) + f_int*SSP(A_int,Z)) ]
  - SSPs: FSPS sfh=0 (true single-age), precomputed on a Z grid.
  - f_old,f_int: NNLS (>=0).  Z, A_V, delta: grid-searched (best per old age).
  - KC13 in optical = Calzetti * (lambda_rest/5500)^delta (2175A bump is out of range).
Question: does any old SSP age produce enough 6500-7200 variation to fit the bump?

Run from hubersed root (needs fsps):
    python tmp/ssp_age_scan_42580.py
"""

import pickle
import numpy as np
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fsps

PKL, KEY = "results/mapfit_cont_line_examples.pkl", "continuum-only"
OUT = "results/ssp_age_scan_42580.pkl"
A_INT = 3.5
A_OLD = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 12.5, 13.0, 13.5])
ZGRID = np.array([-0.4, -0.2, 0.0, 0.1, 0.19])
AVS = np.linspace(0.0, 2.0, 9)
DELTAS = np.array([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4])
SIGMA_V = 207.0  # km/s (VAC VDISP); instrumental LSF is subdominant
c_kms = 2.998e5
c_AA = 2.998e18

d = pickle.load(open(PKL, "rb"))
wave = np.asarray(d["wave"], float)
r = d["results"][KEY]
z = r["z"]
flux = np.asarray(r["flux"], float)
unc = np.asarray(r["unc"], float)
mask = np.asarray(r["mask"], bool)
lam_rest = wave / (1 + z)
print(
    f"TID {r['id']}  z={z:.4f}  free logzsol(grid {ZGRID}), KC13 dust (A_V,delta), sigma_v={SIGMA_V} km/s"
)

sp = fsps.StellarPopulation(
    zcontinuous=1, sfh=0, imf_type=2, add_neb_emission=False, dust_type=0, dust2=0.0
)


def ssp_obs(age, logz):
    sp.params["logzsol"] = logz
    wr, fl = sp.get_spectrum(tage=age, peraa=True)
    wo = wr * (1 + z)
    fnu = (fl / (1 + z)) * wo**2 / c_AA
    s = np.interp(wave, wo, fnu)
    # velocity smoothing: convolve in ln-lambda (observed grid ~ uniform in lambda; approx sigma in pix)
    dln = np.gradient(np.log(wave))  # per-pixel d ln lambda
    sig_pix = (SIGMA_V / c_kms) / np.median(dln)
    return gaussian_filter1d(s, sig_pix)


# precompute SSPs on (age, Z) grid
ages = np.concatenate([[A_INT], A_OLD])
SSP = {(round(a, 3), round(zz, 3)): ssp_obs(a, zz) for a in ages for zz in ZGRID}


def calzetti_k(lr_aa):
    x = 1e4 / lr_aa
    lum = lr_aa / 1e4
    Rv = 4.05
    k = np.zeros_like(lum)
    b = lum < 0.63
    k[b] = 2.659 * (-2.156 + 1.509 * x[b] - 0.198 * x[b] ** 2 + 0.011 * x[b] ** 3) + Rv
    k[~b] = 2.659 * (-1.857 + 1.040 * x[~b]) + Rv
    return np.clip(k, 0, None)


KCAL = calzetti_k(lam_rest)
Rv = 4.05
good = mask & (unc > 0) & np.isfinite(flux)
wr = lam_rest


def chi2win(model, lo, hi):
    s = good & (wr >= lo) & (wr < hi)
    return float(np.nansum(((flux[s] - model[s]) / unc[s]) ** 2)) / s.sum()


def fit_age(A):
    best = None
    for logz in ZGRID:
        s_old = SSP[(round(A, 3), round(logz, 3))]
        s_int = SSP[(round(A_INT, 3), round(logz, 3))]
        for av in AVS:
            for dlt in DELTAS:
                att = 10 ** (-0.4 * av * (KCAL / Rv) * (lam_rest / 5500.0) ** dlt)
                M = np.vstack([att * s_old, att * s_int]).T
                Aw = M[good] / unc[good, None]
                bw = flux[good] / unc[good]
                coef, _ = nnls(Aw, bw)
                model = M @ coef
                chi2 = float(np.nansum(((flux[good] - model[good]) / unc[good]) ** 2))
                if best is None or chi2 < best["chi2"]:
                    best = dict(
                        A=A,
                        logz=logz,
                        av=av,
                        delta=dlt,
                        coef=coef,
                        model=model,
                        chi2=chi2,
                    )
    best["chi2_red"] = best["chi2"] / (good.sum() - 5)
    best["bump"] = chi2win(best["model"], 6500, 7200)
    best["control"] = chi2win(best["model"], 4500, 5500)
    best["fold"] = (
        best["coef"][0] / best["coef"].sum() if best["coef"].sum() > 0 else np.nan
    )
    return best


res = []
print(
    f"\n{'oldage':>7} {'chi2_red':>9} {'bump':>7} {'control':>8} {'logZ':>5} {'A_V':>5} {'delta':>6} {'f_old':>6}"
)
for A in A_OLD:
    x = fit_age(A)
    res.append(x)
    print(
        f"{A:>7.1f} {x['chi2_red']:>9.3f} {x['bump']:>7.2f} {x['control']:>8.2f} {x['logz']:>5.2f} {x['av']:>5.2f} {x['delta']:>6.2f} {x['fold']:>6.2f}"
    )
pickle.dump(
    dict(
        wave=wave,
        z=z,
        flux=flux,
        unc=unc,
        mask=mask,
        A_old=A_OLD,
        A_int=A_INT,
        results=res,
    ),
    open(OUT, "wb"),
)
print(f"saved {OUT}")

fl = flux.copy()
fl[~mask] = np.nan
flmed = medfilt(np.nan_to_num(flux, nan=0.0), 9)
flmed[~mask] = np.nan
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.5))
a1.plot(A_OLD, [x["bump"] for x in res], "o-", color="C3", label="6500-7200 (bump)")
a1.plot(
    A_OLD, [x["control"] for x in res], "s-", color="C0", label="4500-5500 (control)"
)
a1.set_xlabel("old-SSP age [Gyr]")
a1.set_ylabel("χ²/pixel")
a1.legend(fontsize=8)
a1.set_title("bump vs old-SSP age")
s = (wr >= 6000) & (wr <= 7300)
a2.plot(wr[s], flmed[s], color="k", lw=1.0, label="DESI (medfilt9)")
for x, cc in zip([res[0], res[len(res) // 2], res[-1]], ["C0", "C2", "C3"]):
    mm = x["model"].copy()
    mm[~mask] = np.nan
    a2.plot(wr[s], mm[s], color=cc, lw=0.9, label=f"old {x['A']:.1f} Gyr")
a2.set_xlim(6000, 7300)
a2.set_xlabel("rest λ (Å)")
a2.set_ylabel("flux (maggies)")
a2.legend(fontsize=8)
a2.set_title("best fits")

# --- DIRECT check: intrinsic SSP continua (no fitting), normalized in 6000-7300 ---
sref = (wr >= 6000) & (wr <= 7300)
for age, cc in zip([8.0, 10.0, 12.0, 13.5], ["C0", "C2", "C1", "C3"]):
    ss = SSP[(round(age, 3), 0.1)].copy()  # fixed Z=0.1, no dust, just the SSP
    ss = ss / np.nanmedian(ss[sref])  # normalize in the window
    ss[~mask] = np.nan
    a3.plot(wr[sref], ss[sref], color=cc, lw=1.0, label=f"SSP {age:.0f} Gyr")
dref = flmed / np.nanmedian(flmed[sref & np.isfinite(flmed)])
a3.plot(wr[sref], dref[sref], color="k", lw=1.2, label="DESI (norm)")
a3.set_xlim(6000, 7300)
a3.set_xlabel("rest λ (Å)")
a3.set_ylabel("normalized flux")
a3.legend(fontsize=8)
a3.set_title("intrinsic SSP continua (no fit)")
fig.tight_layout()
fig.savefig("tmp/ssp_age_scan_42580.png", dpi=140, bbox_inches="tight")
print("saved tmp/ssp_age_scan_42580.png")

# quantify: max fractional difference between SSP(8) and SSP(13.5) in the bump window
s67 = (wr >= 6500) & (wr <= 7200) & mask
a8 = SSP[(8.0, 0.1)]
a13 = SSP[(13.5, 0.1)]
a8n = a8 / np.nanmedian(a8[s67])
a13n = a13 / np.nanmedian(a13[s67])
print(
    f"max |SSP(13.5)-SSP(8)|/SSP in 6500-7200 (Z=0.1, normalized): {np.nanmax(np.abs(a13n[s67] - a8n[s67])) * 100:.2f}%  (bump to fit is ~2%)"
)
