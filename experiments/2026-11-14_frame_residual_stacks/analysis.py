"""Is the DESI minus best-fit continuum residual coherent in observed or in rest wavelength?

QUESTION: For the MAP fits of the 310 old outliers, does the residual F / (s M) agree between the
low-z and high-z halves when stacked in observed wavelength (instrument or calibration shared by
all DR1 spectra) or in rest wavelength (model physics or prior coverage)? Plan in
~/.claude/plans/calib_injection_2026-11-13.md.
HYPOTHESIS: the frame with the higher split-half correlation wins when the difference exceeds 3
bootstrap sigma; otherwise undecided. A residual broader than the frame shift cannot decide.
INPUTS: results/map_fits_outliers310_v2/<tid>.pkl (flux, unc, model on the DESI grid, z; prospect
Spectrum with response 1, so no calibration polynomial in the model).
METHOD: good pixels have finite flux, unc > 0, model > 0, are outside the spender sky-line mask and
the telluric A (7590-7700 A) and B (6860-6950 A) bands, and more than 15 A rest from the strong
lines in LINES. r = F / (s M) with s the median of F / M over good pixels at rest 5300-5850 A
(galaxies with under 10 such pixels dropped). Each r is averaged in 10 A bins in observed
(3600-9800 A) and rest (2800-9400 A) wavelength. Stacks are the median over galaxies per bin, kept
where 10 or more galaxies contribute. The sample is split at its median z; per frame the Pearson
correlation of the two half stacks (minus 1) is taken over bins both cover. The bootstrap takes
1000 resamples of galaxies within each half, on the difference corr_obs - corr_rest.
SEED: 0 (bootstrap).
COMMAND: ``uv run --no-sync python experiments/2026-11-14_frame_residual_stacks/analysis.py``.
RESULT: 2026-11-14, frame_stacks.txt. 310 fits, z median 0.075 (10-90% 0.030-0.374). Split-half
correlation 0.69 in observed and 0.79 in rest frame (491 and 528 bins); difference -0.10 +- 0.08
(-1.2 sigma): undecided. Stack rms(r - 1) 0.046 observed, 0.056 rest. Both frames correlate
strongly, as a residual broader than the frame shift (~770 A at 5800 A between the half medians)
does. Assumptions: the 310 are old outliers at low z, not the high-S/N cont outliers; MAP fits of
the 10-2025 model; no scale separation of broad and narrow residuals.
FIGURES: none (frame_stacks.txt, frame_stacks.npz).
"""

import glob
import pickle

import numpy as np
from spender.data.desi import DESI
from spender.instrument import get_skyline_mask

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-11-14_frame_residual_stacks"
FITS = RES / "map_fits_outliers310_v2"
TELLURIC = ((6860.0, 6950.0), (7590.0, 7700.0))
# vacuum rest wavelengths (A) of the strong lines masked at +-15 A
LINES = (3727.1, 3729.9, 3869.9, 3890.2, 3971.2, 4102.9, 4341.7, 4364.4, 4862.7, 4960.3, 5008.2,
         5877.3, 6302.0, 6365.5, 6549.9, 6564.6, 6585.3, 6718.3, 6732.7)  # fmt: skip
OBS_EDGES = np.arange(3600.0, 9810.0, 10.0)
REST_EDGES = np.arange(2800.0, 9410.0, 10.0)
NMIN = 10
NBOOT = 1000


def residuals():
    """Return z and the binned residuals of every usable fit in both frames."""
    wave = DESI._wave_obs.numpy().astype(float)
    sky = get_skyline_mask(DESI._wave_obs).numpy()
    tell = np.zeros(wave.size, bool)
    for lo, hi in TELLURIC:
        tell |= (wave >= lo) & (wave <= hi)
    zs, obs, rest = [], [], []
    for f in sorted(glob.glob(str(FITS / "*.pkl"))):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        if not isinstance(d, dict) or d.get("status") != "ok":
            continue
        assert np.allclose(np.asarray(d["wave"], float), wave), f
        fl, un, md = (np.asarray(d[k], float) for k in ("flux", "unc", "model"))
        z = float(d["z"])
        rw = wave / (1 + z)
        good = (
            np.isfinite(fl) & np.isfinite(un) & (un > 0) & np.isfinite(md) & (md > 0) & ~sky & ~tell
        )
        for lam in LINES:
            good &= np.abs(rw - lam) > 15
        sel = good & (rw > 5300) & (rw < 5850)
        if sel.sum() < 10:
            continue
        r = fl / (np.median(fl[sel] / md[sel]) * md)
        zs.append(z)
        obs.append(binned(wave[good], r[good], OBS_EDGES))
        rest.append(binned(rw[good], r[good], REST_EDGES))
    return np.array(zs), np.array(obs), np.array(rest)


def binned(x, y, edges):
    """Return the mean of y in each bin of x, NaN where the bin is empty."""
    k = np.digitize(x, edges) - 1
    ok = (k >= 0) & (k < edges.size - 1)
    s = np.bincount(k[ok], y[ok], minlength=edges.size - 1)
    n = np.bincount(k[ok], minlength=edges.size - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan)


def stack(r):
    """Return the median over galaxies per bin, NaN where fewer than NMIN contribute."""
    n = np.sum(np.isfinite(r), axis=0)
    with np.errstate(all="ignore"):
        m = np.nanmedian(r, axis=0) if r.shape[0] else np.full(r.shape[1], np.nan)
    return np.where(n >= NMIN, m, np.nan)


def corr(a, b):
    """Return the Pearson correlation of a - 1 and b - 1 over bins both cover, and the bin count."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(a[ok] - 1, b[ok] - 1)[0, 1]), int(ok.sum())


def main():
    """Write the split-half correlations, the bootstrap and the stacks."""
    OUT.mkdir(parents=True, exist_ok=True)
    z, obs, rest = residuals()
    lo = z <= np.median(z)
    hi = ~lo
    lines = [f"fits used {z.size}; z 10/50/90 {' '.join(f'{v:.3f}' for v in np.percentile(z, [10, 50, 90]))}; "
             f"low half z <= {np.median(z):.3f} (N {lo.sum()}), high half (N {hi.sum()})"]  # fmt: skip
    res = {}
    for name, r, edges in (("observed", obs, OBS_EDGES), ("rest", rest, REST_EDGES)):
        s_all, s_lo, s_hi = stack(r), stack(r[lo]), stack(r[hi])
        c, nb = corr(s_lo, s_hi)
        rms = np.sqrt(np.nanmean((s_all - 1) ** 2))
        res[name] = (s_lo, s_hi)
        lines.append(f"{name}: split-half correlation {c:.3f} over {nb} bins; stack rms(r - 1) {rms:.4f} "
                     f"over {np.isfinite(s_all).sum()} bins")  # fmt: skip
        np.savez(
            OUT / f"stack_{name}.npz",
            centres=(edges[:-1] + edges[1:]) / 2,
            all=s_all,
            low=s_lo,
            high=s_hi,
        )
    rng = np.random.default_rng(0)
    il, ih = np.flatnonzero(lo), np.flatnonzero(hi)
    diff = []
    for _ in range(NBOOT):
        bl, bh = rng.choice(il, il.size), rng.choice(ih, ih.size)
        co = corr(stack(obs[bl]), stack(obs[bh]))[0]
        cr = corr(stack(rest[bl]), stack(rest[bh]))[0]
        diff.append(co - cr)
    diff = np.array(diff)
    d0 = corr(*res["observed"])[0] - corr(*res["rest"])[0]
    sig = np.nanstd(diff)
    verdict = ("observed frame (instrument)" if d0 > 3 * sig else
               "rest frame (model)" if d0 < -3 * sig else "undecided")  # fmt: skip
    lines.append(f"corr_obs - corr_rest {d0:.3f}, bootstrap sigma {sig:.3f} ({d0 / sig:.1f} sigma); "
                 f"bootstrap 16/50/84 {' '.join(f'{v:.3f}' for v in np.nanpercentile(diff, [16, 50, 84]))}")  # fmt: skip
    lines.append(f"rule: {verdict}")
    text = "\n".join(lines)
    (OUT / "frame_stacks.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
