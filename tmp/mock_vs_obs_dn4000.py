"""Do the TRAINING mocks reach the observed Dn4000 of massive DESI galaxies?

MODE=clean : Dn4000 from the h5 'fluxes' (NOISELESS model spectra)
MODE=noisy : Dn4000 from DESIcueprospector1024_*.pkl (DESI IVAR noise pasted by the spender
             decoder -- bin/spender/noise/make_prospector_noisy_sed.py)
MODE=both  : run both and print the difference.

WHY THIS MATTERS: the original 44.5%-uncovered number compared CLEAN mocks to NOISY observed
DN4000. That is biased: noise BROADENS the mock distribution -> raises the mock 99.5th pct ->
LOWERS the uncovered fraction. The apples-to-apples number is noisy-mock vs noisy-obs.

Safe to measure Dn4000 off the normalized f_noisy: normalize_spectra divides by a SCALAR
(median of rest-frame 5300-5850, quantities.py:19), so it cancels in the band ratio.
f_noisy is on wavelength[:-1] (make_prospector_noisy_sed.py:81 drops the last point).

Run from hubersed root (venv):  python tmp/mock_vs_obs_dn4000.py [clean|noisy|both]
"""

import sys, glob, pickle, re
import numpy as np, h5py
from astropy.io import fits
from hubersed.paths import PATHS

MODE = sys.argv[1] if len(sys.argv) > 1 else "both"
NPKL = int(sys.argv[2]) if len(sys.argv) > 2 else 60  # 60 x 1024 = ~61k mocks
DP = PATHS["DATA"] / "prospector_model"
H5 = DP / "prospector_stochastic_model_seds_cue_500000.h5"
BINS = [
    (11.0, 11.5, 0.05, 0.15, "massive"),
    (10.0, 10.5, 0.05, 0.15, "intermediate"),
    (9.25, 9.75, 0.05, 0.15, "low-mass"),
]

h = h5py.File(H5, "r")
wave_full = h["wavelength"][:]
mm_all = h["priors/stellar_masses"][:]
zz_all = h["priors/redshifts"][:]


def dn4000(fl, w, z, is_flambda=False):
    """Balogh+99 Dn4000 is a ratio of average F_NU. h5 'fluxes' are MAGGIES (~f_nu) -> use as-is.
    The noised pkls hold f_LAMBDA (maggies_to_flambda) -> must x lambda^2 to get back to f_nu.
    Skipping that biases Dn4000 by (3900/4050)^2 = 0.9273 -- verified numerically: the same 527
    massive mocks give med 1.247 as f_nu vs 1.157 as f_lambda, ratio 0.9275."""
    r = w / (1 + z)
    f = fl * (w**2) if is_flambda else fl
    b = np.nanmean(f[(r >= 3850) & (r <= 3950)])
    rr = np.nanmean(f[(r >= 4000) & (r <= 4100)])
    return rr / b if b > 0 else np.nan


# ---------- observed ----------
hv = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)


def col(n):
    for hd in hv:
        c = getattr(getattr(hd, "columns", None), "names", None)
        if c and n in c:
            return np.asarray(hd.data[n])


olm = col("LOGMSTAR").astype(float)
oz = col("Z").astype(float)
odn = col("DN4000").astype(float)


# ---------- clean mocks ----------
def get_clean():
    NBLOCK = 30000
    i0 = int(np.searchsorted(wave_full, 4000.0))
    i1 = int(np.searchsorted(wave_full, 4750.0))
    w = wave_full[i0:i1]
    F = h["fluxes"][0:NBLOCK, i0:i1]  # one contiguous read (mocks are iid)
    out = {}
    for m0, m1, z0, z1, lab in BINS:
        s = np.where(
            (mm_all[:NBLOCK] >= m0)
            & (mm_all[:NBLOCK] < m1)
            & (zz_all[:NBLOCK] >= z0)
            & (zz_all[:NBLOCK] < z1)
        )[0]
        D = np.array([dn4000(F[j], w, zz_all[j]) for j in s])
        out[lab] = D[np.isfinite(D) & (D > 0.5) & (D < 3.5)]
    return out


# ---------- noisy mocks (YOUR pipeline's output) ----------
def get_noisy():
    files = sorted(
        glob.glob(str(DP / "DESIcueprospector1024_*.pkl")),
        key=lambda p: int(re.search(r"_(\d+)\.pkl$", p).group(1)),
    )[:NPKL]
    assert files, "no DESIcueprospector1024_*.pkl found"
    w = wave_full[:-1]  # f_noisy drops the last point
    acc = {lab: [] for (_, _, _, _, lab) in BINS}
    for k, f in enumerate(files):
        with open(f, "rb") as fh:
            f_noisy, ivar, zb, tid, norms, zerr = pickle.load(fh)
        fn = np.asarray(f_noisy)
        zb = np.asarray(zb)
        tid = np.asarray(tid).astype(int)
        mass = mm_all[tid]
        for m0, m1, z0, z1, lab in BINS:
            s = np.where((mass >= m0) & (mass < m1) & (zb >= z0) & (zb < z1))[0]
            for j in s:
                acc[lab].append(dn4000(fn[j], w, zb[j], is_flambda=True))
        if (k + 1) % 10 == 0:
            print(
                f"  {k + 1}/{len(files)} pkls  "
                + "  ".join(f"{lab}:{len(v)}" for lab, v in acc.items())
            )
    return {
        lab: np.array(v)[np.isfinite(v) & (np.array(v) > 0.5) & (np.array(v) < 3.5)]
        for lab, v in acc.items()
    }


res = {}
if MODE in ("clean", "both"):
    print("reading CLEAN h5 ...")
    res["clean"] = get_clean()
if MODE in ("noisy", "both"):
    print("reading NOISY pkls ...")
    res["noisy"] = get_noisy()

print(
    f"\n{'bin':14s}{'set':7s}{'N':>7}{'med':>8}{'99.5th':>9}{'max':>8}{'% obs uncovered':>17}"
)
for m0, m1, z0, z1, lab in BINS:
    o = (
        (olm >= m0)
        & (olm < m1)
        & (oz >= z0)
        & (oz < z1)
        & np.isfinite(odn)
        & (odn > 0.5)
        & (odn < 3)
    )
    O = odn[o]
    for tag in ("clean", "noisy"):
        if tag not in res:
            continue
        D = res[tag][lab]
        if len(D) < 20:
            print(f"{lab:14s}{tag:7s}{len(D):7d}   too few")
            continue
        p995 = np.nanpercentile(D, 99.5)
        print(
            f"{lab:14s}{tag:7s}{len(D):7d}{np.median(D):8.3f}{p995:9.3f}{D.max():8.3f}"
            f"{100 * np.mean(O > p995):17.1f}"
        )
    print(
        f"{'':14s}{'OBS':7s}{len(O):7d}{np.median(O):8.3f}{np.percentile(O, 95):9.3f}{O.max():8.3f}"
    )
if MODE == "both":
    print(
        "\nCAVEAT -- the percentile metric is only trustworthy in the MASSIVE (high-S/N) bin."
    )
    print(
        "At low S/N, Dn4000 is a ratio of two noisy band means: the blue band (3850-3950)"
    )
    print(
        "fluctuates toward zero -> the ratio EXPLODES -> a fake high tail that READS as coverage"
    )
    print(
        "but gives the flow NO support (theta is unchanged; the noise is incoherent pixel scatter,"
    )
    print(
        "not a coherent 4000-A break). A faint mock reading Dn4000=3 is garbage, not an old galaxy."
    )
