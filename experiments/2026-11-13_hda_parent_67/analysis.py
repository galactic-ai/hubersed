"""Is the Balmer absorption of gallery105 figures 67, 71, 88 and 99 unusual for the 4000 A break in the parent?

QUESTION: 67 (39633136522363591, z 0.478) has HdA 7.0 +- 1.1 and HgA 6.6 +- 1.1 at Dn4000 1.53 with no
emission. Where do these sit among parent galaxies with the same break strength and redshift?
HYPOTHESIS: none registered. If 67 is an older post-starburst its HdA should sit in the upper tail.
INPUTS: results/prescreened_sample_v1.npz (keep, the 104,990 parent); data/fastspec-iron-sv3-bright.fits
(DN4000, Z, HALPHA_EW and its IVAR, OII EWs); DR1 SV3 bright coadds through load_by_index.
Figure 71 (39633149579232417, z 0.467, HdA 7.8 +- 1.0, HgA 2.1, Dn4000 1.37) added on 2026-11-13.
METHOD: control = parent galaxies with FastSpecFit DN4000 in a 0.15-wide window at the target's break
(67: [1.45, 1.60]; 71: [1.30, 1.45]; 88: [1.25, 1.40]; 99: [1.12, 1.27], added 2026-11-13) and Z within 0.05 of the target, the target removed. HdA and HgA measured as gallery105 seed.py (lick_index copied unchanged:
Worthey & Ottaviani 1997 bands taken as air and moved to vacuum, mean form over good pixels, NaN below
25% good pixels, std over 200 noise draws). Reported for all controls and for those with no lines
(FastSpecFit EW Halpha < 3 A where Halpha is on the spectrum, and EW [OII] 3726+3729 < 2.5 A).
SEED: 0 (noise draws).
COMMAND: ``uv run --no-sync python experiments/2026-11-13_hda_parent_67/analysis.py``.
RESULT: 2026-11-13, hda_parent.txt. 182 controls (47 with no lines). 67's HdA 6.97 sits at the 90th
percentile (87th of no-line controls; 19 controls >= 67, p95 7.50); its HgA 6.63 at the 99th (98th; 2
controls >= 67, p95 4.39). The control spread (HdA p5-p95 -3.4 to 7.5) is mostly noise at this S/N.
71 (298 controls, 23 with no lines): HdA 7.81 at the 91st percentile (23 controls >= 71), HgA 2.08 at
the 71st (86 >= 71), so only HdA sits in the tail.
88 (255 controls, 14 with no lines): HdA 6.30 at the 86th percentile of all, 50th of the no-line
controls; HgA 5.22 at the 94th and 71st. Line-free galaxies at Dn4000 1.25-1.40 already have high HdA.
Assumptions: FastSpecFit DN4000 matches the break the indices see; one SV3 coadd per galaxy.
FIGURES: none (hda_parent.txt).
"""

import numpy as np
from astropy.io import fits
from spender.data.desi import DESI

from hubersed.io.desi import load_by_index, tids_to_indices
from hubersed.paths import PATHS
from hubersed.sps.utils import _airtovac

OUT = PATHS["RESULTS"] / "2026-11-13_hda_parent_67"
TARGETS = {
    "67": (39633136522363591, 0.4781, (1.45, 1.60)),
    "71": (39633149579232417, 0.4666, (1.30, 1.45)),
    "88": (39633407818335390, 0.4818, (1.25, 1.40)),
    "99": (39633421919584782, 0.0151, (1.12, 1.27)),
}
DZ = 0.05
BANDS = {
    "hda": ((4083.50, 4122.25), (4041.60, 4079.75), (4128.50, 4161.00)),
    "hga": ((4319.75, 4363.50), (4283.50, 4319.75), (4367.25, 4419.75)),
}
MIN_GOOD, N_MC, SEED = 0.25, 200, 0


def lick_index(rest, flux, ivar, bands, rng):
    """Return a Lick index, its noise-draw std and the smallest good-pixel fraction of its bands."""
    vac = [(_airtovac(lo), _airtovac(hi)) for lo, hi in bands]
    sel = [(rest >= lo) & (rest <= hi) for lo, hi in vac]
    good = min((ivar[s] > 0).mean() if s.any() else 0.0 for s in sel)
    if good < MIN_GOOD:
        return np.nan, np.nan, good
    sel = [s & (ivar > 0) for s in sel]
    (f_lo, f_hi), (b_lo, b_hi), (r_lo, r_hi) = vac
    wb, wr = (b_lo + b_hi) / 2, (r_lo + r_hi) / 2
    sig = 1 / np.sqrt(np.where(ivar > 0, ivar, np.inf))

    def index(fl):
        cb, cr = (fl[s].mean() for s in sel[1:])
        cont = cb + (cr - cb) * (rest[sel[0]] - wb) / (wr - wb)
        return float((f_hi - f_lo) * (1 - np.mean(fl[sel[0]] / cont)))

    draws = [index(flux + rng.standard_normal(flux.size) * sig) for _ in range(N_MC)]
    return index(flux), float(np.std(draws)), good


def compare(nn, t0, z0, dn_range, keep, f, m, wave):
    """Return the table lines of one target against its controls."""
    tid, z, dn = np.asarray(m["TARGETID"]), np.asarray(m["Z"]), np.asarray(f["DN4000"])
    sel = np.isin(tid, keep) & (dn >= dn_range[0]) & (dn <= dn_range[1])
    sel &= (np.abs(z - z0) <= DZ) & (tid != t0)
    ewha = np.where(np.asarray(f["HALPHA_EW_IVAR"]) > 0, np.asarray(f["HALPHA_EW"]), np.nan)
    ewoii = np.asarray(f["OII_3726_EW"]) + np.asarray(f["OII_3729_EW"])
    idx = np.flatnonzero(sel)
    quiet = np.array([(np.isnan(ewha[i]) or ewha[i] < 3) and ewoii[i] < 2.5 for i in idx])
    rng = np.random.default_rng(SEED)
    vals = {k: [] for k in BANDS}
    for t in tid[idx]:
        flux, ivar, zz, _ = load_by_index(int(tids_to_indices(np.array([t]))[0]))
        rest = wave / (1 + float(zz))
        for k, b in BANDS.items():
            vals[k].append(
                lick_index(rest, np.asarray(flux, float), np.asarray(ivar, float), b, rng)[0]
            )
    flux, ivar, zz, _ = load_by_index(int(tids_to_indices(np.array([t0]))[0]))
    own = {k: lick_index(wave / (1 + float(zz)), np.asarray(flux, float), np.asarray(ivar, float), b, rng)
           for k, b in BANDS.items()}  # fmt: skip
    lines = [f"figure {nn} ({t0}) control: {idx.size} parent galaxies, Dn4000 {dn_range}, "
             f"z {z0} +- {DZ}; {quiet.sum()} with no lines"]  # fmt: skip
    for k in BANDS:
        v = np.array(vals[k])
        for name, mk in (("all", np.ones(v.size, bool)), ("no lines", quiet)):
            x = v[mk & np.isfinite(v)]
            p = np.percentile(x, [5, 16, 50, 84, 95])
            lines.append(
                f"{k} {name:8} N {x.size:3d}  p5/16/50/84/95 "
                + " ".join(f"{q:5.2f}" for q in p)
                + f"  | {nn}: {own[k][0]:.2f} +- {own[k][1]:.2f}, percentile "
                f"{100 * np.mean(x < own[k][0]):.1f}, N >= {nn}: {int(np.sum(x >= own[k][0]))}"
            )
    return lines + [""]


def main():
    """Measure the control indices of each target and place the target among them."""
    OUT.mkdir(parents=True, exist_ok=True)
    keep = np.load(PATHS["RESULTS"] / "prescreened_sample_v1.npz")["keep"]
    wave = DESI._wave_obs.numpy()
    lines = []
    with fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits") as h:
        f, m = h["FASTSPEC"].data, h["METADATA"].data
        for nn, (t0, z0, dn_range) in TARGETS.items():
            lines += compare(nn, t0, z0, dn_range, keep, f, m, wave)
    text = "\n".join(lines)
    (OUT / "hda_parent.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
