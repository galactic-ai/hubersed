"""
tmp/basel_masking.py  --  how much does the BaSeL red (rest > 7200 A) actually matter?

The MILES-resolution fix drops everything outside the MILES window (rest 3750-7200 A).
Geometrically that's a big chunk of the red at low z -- but the question Hahn raised is
whether it carries any chi^2 weight, given (a) BaSeL is only R~200 so there are no
absorption features to fit there anyway, and (b) the red is where DESI sky residuals
live, so it's often masked / low-ivar.

Fast: just loads the data, no fitting. Run in your MILES env:
    python tmp/basel_masking.py            # all 240
    python tmp/basel_masking.py --n 50     # match the oldnew_diff sample

Per bin it prints medians of:
    red pixels          -- geometric fraction of pixels with rest > 7200 A (the BaSeL red)
    usable & red        -- of the good (ivar>0) pixels, the fraction that are red
    chi2-weight in red  -- fraction of total sum(ivar) sitting in the red   <-- the key number
    red already masked  -- of the red pixels, the fraction that are already bad/masked (ivar<=0)

If "chi2-weight in red" is small (few %), Hahn is right: dropping BaSeL costs almost
nothing even though it's a big fraction of the pixels.
"""
import sys
import numpy as np

from hubersed.paths import PATHS
from hubersed.fitting.chi2 import load_by_index, tids_to_indices, WAVE_OBS
from hubersed.conversion import flambda_to_maggies, ivar_flambda_to_ivar_maggies

MRMIN, MRMAX = 3750.0, 7200.0   # MILES rest-frame window


def main():
    n = None
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    s = np.load(PATHS["RESULTS"] / "oldnew_sample.npz")
    bins = {"SF_lowM": s["sf_tids"], "QU_highM": s["qu_tids"]}

    print(f"{'bin':>9} | {'N':>3} | {'red pix':>8} | {'usable&red':>10} | "
          f"{'chi2-wt red':>11} | {'red masked':>10}")
    for label, tids in bins.items():
        tids = tids[:n] if n else tids
        idxs = tids_to_indices(np.asarray(tids, np.int64))
        rp, rg, rw, mk = [], [], [], []
        for idx in idxs:
            spec, ivar, z, tid = load_by_index(int(idx))
            iv = ivar_flambda_to_ivar_maggies(WAVE_OBS, ivar)
            fm = flambda_to_maggies(WAVE_OBS, spec)
            good = (iv > 0) & np.isfinite(fm) & np.isfinite(iv)
            red = (WAVE_OBS / (1.0 + z)) > MRMAX          # BaSeL red, dropped by the fix
            rp.append(float(red.mean()))
            if good.sum():
                rg.append(float((good & red).sum()) / float(good.sum()))
                rw.append(float(iv[good & red].sum()) / float(iv[good].sum()))
            if red.sum():
                mk.append(1.0 - float((good & red).sum()) / float(red.sum()))
        med = lambda a: (np.median(a) if len(a) else np.nan)
        print(f"{label:>9} | {len(idxs):>3} | {med(rp):>7.0%} | {med(rg):>9.0%} | "
              f"{med(rw):>10.0%} | {med(mk):>9.0%}")
    print("\nchi2-wt red = fraction of total sum(ivar) redward of the MILES window.")
    print("If that's small, the BaSeL red barely enters chi^2 -> safe to drop.")


if __name__ == "__main__":
    main()
