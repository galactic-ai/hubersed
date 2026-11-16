"""At fixed S/N, does the DESI flow outlier rate depend on flux calibration quality?

QUESTION: Within each S/N bin, does the outlier rate of each flow rise with the predicted
calibration error f of the petals a target was observed on, and fall with the number of tiles?
Plan in ~/.claude/plans/calib_injection_2026-11-13.md.
HYPOTHESIS: if missing calibration distortions drive the high-S/N excess, the rate in the top
quartile of f exceeds the bottom quartile by more than 3 sigma in the S/N 8-15 and > 15 bins of
cont10 and cont15. If the excess is a model offset, the rate is flat in f.
INPUTS: data/iron_tileqa/<tile>/tile-qa-*.fits (latest file per tile; FIBERQA TARGETID, PETAL_LOC,
EFFTIME_SPEC, QAFIBERSTATUS; PETALQA NSTDSTAR, STARRMS) for the 214 SV3 bright tiles of
data/iron_tiles/tiles-iron.csv; results/2026-11-09_mocks_v2_outliers/flows/desi_outliers_flow_nsf_
<tag>latent_snr3.pt; data/dataset_z_snr.npz (the latent-cut S/N).
METHOD: f per tile-petal from the r-camera fit of results/2026-10-13_fluxcal_sample/all/
summarise_f.txt l.24, f = -0.0159 + 0.1209 / sqrt(NSTDSTAR) + 0.0582 STARRMS (NSTDSTAR 0 set to 1).
Per target, over the tiles it was observed on (QAFIBERSTATUS 0 rows first, all rows when none),
f_coadd = sqrt(mean(f^2) / n_tiles). Quartiles of f_coadd over the scored parent. Errors
sqrt(r (1 - r) / N). In the matched check, within each S/N bin, the 3+ tile rate is reweighted to
the 1-tile mix over cells of z (0.1 bins), log M* and Dn4000 (FastSpecFit LOGMSTAR and Dn4000 of
2026-11-13_psb_fraction_like_for_like indices_desi.npz) and S/N quartile; cells without both
groups are dropped (coverage given).
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-13_calib_outlier_dependence/analysis.py``.
RESULT: 2026-11-13, dependence.txt. Q4 minus Q1 of f: cont10 S/N 8-15 +1.9% (3.5 sigma), > 15
+4.3% (2.9 sigma); cont15 8-15 +0.2% (0.6 sigma), > 15 +2.7% (2.7 sigma). The 3 sigma rule holds
in one of four cells, so not met. Tile count matters more: above S/N 15, 1 tile vs 3+ tiles is
cont10 15.7 vs 9.3%, cont15 8.0 vs 2.8%, at equal median S/N; at S/N 3-5 there is no difference.
3+ tile targets still sit far above the mocks' 0.1%. Matched check: 1-tile targets are at lower z,
higher log M* and Dn4000 (S/N > 15 medians z 0.084 vs 0.116, Dn4000 1.90 vs 1.82). Reweighted, the
1-tile excess shrinks to cont10 16.0 vs 11.8 +- 1.6% (S/N > 15) and 7.2 vs 5.4 +- 0.6% (8-15),
cont15 8.1 vs 6.0 +- 1.2% and 2.2 vs 1.2 +- 0.4%: 1.7-2.7 sigma each, same sign; the full-spectrum
flows show no consistent sign. Assumptions: f from a fit to the r camera, tile-qa NSTDSTAR per tile
not per exposure, tiles independent, cells of z, log M*, Dn4000 and S/N remove the population
difference.
FIGURES: none (dependence.txt).
"""

import glob

import numpy as np
import pandas as pd
import torch
from astropy.io import fits

from hubersed.paths import PATHS

RES, DATA = PATHS["RESULTS"], PATHS["DATA"]
OUT = RES / "2026-11-13_calib_outlier_dependence"
FLOWS = RES / "2026-11-09_mocks_v2_outliers" / "flows"
TAGS = ("6", "10", "15", "cont10", "cont15")
SNR_BINS = (3, 5, 8, 15, np.inf)
F_FIT = (-0.0159, 0.1209, 0.0582)  # summarise_f.txt l.24, r camera
ZCELL = (0.1, 0.2, 0.3, 0.4)
MCELL = (9.5, 10.0, 10.5, 11.0)
DCELL = (1.2, 1.4, 1.6, 1.8)


def tile_rows():
    """Return one row per target and SV3 bright tile with the petal f and EFFTIME."""
    tiles = pd.read_csv(DATA / "iron_tiles" / "tiles-iron.csv")
    tiles = tiles[(tiles.SURVEY == "sv3") & (tiles.PROGRAM == "bright")].TILEID.astype(int)
    rows = []
    for tile in tiles:
        path = sorted(glob.glob(str(DATA / "iron_tileqa" / str(tile) / "tile-qa-*.fits")))[-1]
        with fits.open(path) as h:
            fq, pq = h["FIBERQA"].data, h["PETALQA"].data
            nstd = np.clip(np.asarray(pq["NSTDSTAR"], float), 1, None)
            f = F_FIT[0] + F_FIT[1] / np.sqrt(nstd) + F_FIT[2] * np.asarray(pq["STARRMS"], float)
            fpet = dict(zip(np.asarray(pq["PETAL_LOC"], int).tolist(), f.tolist(), strict=True))
            npet = dict(zip(np.asarray(pq["PETAL_LOC"], int).tolist(), nstd.tolist(), strict=True))
            for t, p, st in zip(np.asarray(fq["TARGETID"], np.int64), np.asarray(fq["PETAL_LOC"], int),
                                np.asarray(fq["QAFIBERSTATUS"], int), strict=True):  # fmt: skip
                rows.append((int(t), tile, fpet.get(p, np.nan), npet.get(p, np.nan), st == 0))
    return pd.DataFrame(rows, columns=["tid", "tile", "f", "nstd", "good"])


def per_target(rows, tids):
    """Return f_coadd, n_tiles and the minimum NSTDSTAR of each scored target."""
    rows = rows[rows.tid.isin(set(tids.tolist()))]
    has_good = rows.groupby("tid").good.transform("any")
    rows = rows[rows.good | ~has_good]
    g = rows.groupby("tid")
    tab = pd.DataFrame({"n_tiles": g.size(), "mean_f2": g.f.apply(lambda x: np.nanmean(x**2)),
                        "min_nstd": g.nstd.min()})  # fmt: skip
    tab["f_coadd"] = np.sqrt(tab.mean_f2 / tab.n_tiles)
    return tab.reindex(tids)


def rate(hit):
    """Return the rate, its error and N as text."""
    n = hit.size
    if n == 0:
        return "   -              (N     0)"
    r = hit.mean()
    return f"{100 * r:5.2f} +- {100 * np.sqrt(r * (1 - r) / n):4.2f}% (N {n:5d})"


def covariates(tids):
    """Return z, log M* and Dn4000 of each target from the like-for-like DESI indices."""
    d = np.load(RES / "2026-11-13_psb_fraction_like_for_like" / "indices_desi.npz")
    pos = {int(t): k for k, t in enumerate(d["id"])}
    k = np.array([pos.get(int(t), -1) for t in tids])
    get = lambda a: np.where(k >= 0, a[np.clip(k, 0, None)], np.nan)  # noqa: E731
    return get(d["z"]), get(d["logmass"]), get(d["dn4000"])


def cells(z, lm, dn, snr, s):
    """Return a cell label per target from bins of z, log M*, Dn4000 and S/N quartile within s."""
    b = lambda x, e: np.where(np.isfinite(x), np.digitize(x, e), -1)  # noqa: E731
    sq = np.nanpercentile(snr[s], [25, 50, 75]) if s.any() else [0, 0, 0]
    return (b(z, ZCELL) * 100 + b(lm, MCELL)) * 10000 + b(dn, DCELL) * 10 + b(snr, sq)


def standardised(out, ref, other, lab):
    """Return the rate of ``other`` reweighted to the cell mix of ``ref``, with the ref rate."""
    shared = np.intersect1d(np.unique(lab[ref]), np.unique(lab[other]))
    r_ref = ref & np.isin(lab, shared)
    w = np.array([np.sum(r_ref & (lab == c)) for c in shared], float)
    w /= w.sum()
    rc = np.array([out[other & (lab == c)].mean() for c in shared])
    nc = np.array([np.sum(other & (lab == c)) for c in shared])
    r = np.sum(w * rc)
    err = np.sqrt(np.sum(w**2 * rc * (1 - rc) / nc))
    return out[r_ref].mean(), r, err, r_ref.sum() / ref.sum()


def main():
    """Write the outlier rate per S/N bin and calibration bin for every flow."""
    OUT.mkdir(parents=True, exist_ok=True)
    zs = np.load(DATA / "dataset_z_snr.npz")
    snr_of = dict(zip(zs["tid"].tolist(), zs["snr"].tolist(), strict=True))
    rows = tile_rows()
    lines = []
    conf = ["matched check: 3+ tile rate reweighted to the 1-tile mix of z, log M*, Dn4000 and S/N"]
    for tag in TAGS:
        res = torch.load(FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt", weights_only=False)
        lp, tids, thr = (
            np.asarray(res["log_p_desi"]),
            np.asarray(res["desi_target_ids"]),
            res["threshold"],
        )
        tab = per_target(rows, tids)
        snr = np.array([snr_of.get(int(t), np.nan) for t in tids])
        z, lm, dn = covariates(tids)
        out = lp <= thr
        fc, nt = tab.f_coadd.to_numpy(), tab.n_tiles.to_numpy()
        q = np.nanpercentile(fc, [25, 50, 75])
        fq = np.digitize(fc, q)
        lines.append(f"{tag}latent: N {tids.size}, no tile-qa match {np.isnan(fc).sum()}, "
                     f"f_coadd quartile edges {' '.join(f'{x:.4f}' for x in q)}")  # fmt: skip
        for lo, hi in zip(SNR_BINS[:-1], SNR_BINS[1:], strict=True):
            s = (snr > lo) & (snr <= hi) & np.isfinite(fc)
            lines.append(f"  S/N {lo}-{hi}: all {rate(out[s])}")
            for k in range(4):
                c = s & (fq == k)
                lines.append(f"    f quartile {k + 1}: {rate(out[c])}, median S/N "
                             f"{np.median(snr[c]) if c.any() else np.nan:5.2f}")  # fmt: skip
            for lab, c in (("1 tile", nt == 1), ("2 tiles", nt == 2), ("3+ tiles", nt >= 3)):
                c = s & c
                lines.append(f"    {lab:8}: {rate(out[c])}, median S/N "
                             f"{np.median(snr[c]) if c.any() else np.nan:5.2f}")  # fmt: skip
            one, many = s & (nt == 1), s & (nt >= 3)
            r1, r3w, e3w, cov = standardised(out, one, many, cells(z, lm, dn, snr, s))
            e1 = np.sqrt(r1 * (1 - r1) / one.sum())
            med = lambda x, c: np.nanmedian(x[c])  # noqa: E731
            conf.append(
                f"  {tag} S/N {lo}-{hi}: 1 tile {100 * r1:5.2f} +- {100 * e1:4.2f}% | 3+ tiles "
                f"reweighted {100 * r3w:5.2f} +- {100 * e3w:4.2f}% (1-tile coverage {cov:.3f}); "
                f"medians z {med(z, one):.3f}/{med(z, many):.3f}, log M* {med(lm, one):.2f}/"
                f"{med(lm, many):.2f}, Dn4000 {med(dn, one):.2f}/{med(dn, many):.2f}"
            )
        lines.append("")
    lines += conf
    text = "\n".join(lines)
    (OUT / "dependence.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
