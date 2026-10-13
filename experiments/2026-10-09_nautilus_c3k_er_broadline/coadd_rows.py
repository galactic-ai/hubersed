"""Recoadd 39627770174637084 with the row-weighted resolution matrix of desispec PR 2377.

QUESTION: how much does the DR1 coadd resolution matrix of this two-exposure target change when
the exposures are combined with the corrected weights, and do the per-exposure matrices and ivars
behave as the resolution checks assume?

DR1 (iron) coadds weight each exposure's banded matrix by the ivar of the column pixel
(desispec 0.56 coaddition.coadd). The DR1 paper lists this as a known issue (arXiv:2503.14745,
A5_knownissues.tex:29) and desispec issue 2372 derives the correct form, the ivar of the row
pixel, R_hat = (sum_e C_e^-1)^-1 sum_e C_e^-1 R_e. PR 2377 implements it as _resolution_coadd,
where data[k, j] is weighted by the normalised ivar at pixel j + k - ndiag//2. The arms are then
merged the same way (coadd_cameras in PR 2377). Cosmic masking needs three or more exposures, so
it does not apply here.

The script does five things from the iron spectra file of healpix 26004.

- Lists this target's exposures, and every (night, tile) in the file, for the model variance
  check of the tile-night logs.
- Rebuilds the coadd with the old column weights and checks it against resolution_matrix.npz (the
  DR1 coadd), as a test of the reading.
- Builds the row-weighted coadd per arm and merged over the arms, and compares it with the old one
  at the strong lines.
- Tests R = S Q on each exposure, where it holds exactly (Guy et al. 2023, extraction.tex:43-61):
  R_ij sqrt(ivar_i) should be symmetric.
- Compares the two exposures pixel by pixel, (f1 - a f2) / sqrt(1/ivar1 + a^2/ivar2) with a scalar a
  per arm, in bins of S/N, as in desispec issue 2420. Near strong lines the two matrices differ,
  which adds to the scatter there.

INPUTS: spectra-sv3-bright-26004.fits.gz (iron), resolution_matrix.npz of resolution_check.py.
SEED: none.

COMMAND: on an ls6 compute node from the repository root,
``.venv/bin/python experiments/2026-10-09_nautilus_c3k_er_broadline/coadd_rows.py
/corral/utexas/AST25023/nikhilgaruda/hubersed/data/desi_iron_spectra/spectra-sv3-bright-26004.fits.gz``.

OUTPUT: coadd_rows.txt and resolution_matrix_rows.npz (same keys as resolution_matrix.npz, with the
row-weighted matrices) in results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/.
"""

import argparse

import numpy as np
from astropy.io import fits

from hubersed.paths import PATHS
from hubersed.sps.lsf import DESI_WAV

TID = 39627770174637084
Z = 0.025319
OUT = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline" / str(TID)
LINES = {
    "[OIII]5007": 5008.240,
    "[OIII]4959": 4960.295,
    "Hbeta": 4862.683,
    "Halpha": 6564.614,
    "[NII]6584": 6585.268,
    "[SII]6718": 6718.294,
}


def column_coadd(res, ivar, mask):
    """Return the DR1 coadd of one arm, matrix weighted by the ivar of the column pixel."""
    good = ivar * (mask == 0)
    tivar = good.sum(0)
    flux_w = good
    rdata = (ivar[:, None, :] * res).sum(0)
    un = ivar.sum(0)
    rdata[:, un > 0] /= un[un > 0]
    return tivar, flux_w, rdata


def row_weights(res, weights):
    """Return the per-element row weights of PR 2377, weights[e, j + k - ndiag//2]."""
    nd, n = res.shape[1], res.shape[2]
    ww = nd // 2
    idx = (np.arange(n)[None, :] + np.arange(-ww, ww + 1)[:, None]) % n
    return np.array([w[idx] for w in weights])


def row_coadd(res, ivar, mask):
    """Return the PR 2377 coadd of one arm: flux weights and the row-weighted matrix."""
    masked = ivar * (mask == 0)
    tivar = masked.sum(0)
    weights = masked.copy()
    bad = tivar == 0
    weights[:, bad] = ivar[:, bad]
    tivar[bad] = weights[:, bad].sum(0)
    weights = weights / (tivar + (tivar == 0))
    rdata = (row_weights(res, weights) * res).sum(0)
    return tivar, weights, rdata


def merge_arms(arms, key):
    """Merge per-arm coadds onto DESI_WAV. Rows weighted as coadd_cameras in PR 2377 if key is rows."""
    wave = DESI_WAV
    nd = max(a["res"].shape[0] for a in arms.values())
    flux, ivar, ivar_un = np.zeros(wave.size), np.zeros(wave.size), np.zeros(wave.size)
    acc, norm = np.zeros((nd, wave.size)), np.zeros((nd, wave.size))
    for a in arms.values():
        imin = np.argmin(abs(a["wave"][0] - wave))
        w = np.arange(imin, imin + a["wave"].size)
        assert np.allclose(a["wave"], wave[w], atol=1e-4)
        good = a["ivar"] * (a["mask"] == 0)
        ivar[w] += good
        ivar_un[w] += a["ivar"]
        flux[w] += good * a["flux"]
        off = (nd - a["res"].shape[0]) // 2
        sl = slice(off, nd - off)
        if key == "rows":
            wt = row_weights(a["res"][None], good[None])[0]
            acc[sl, w] += wt * a["res"]
            norm[sl, w] += wt
        else:
            acc[sl, w] += a["ivar"] * a["res"]
            norm[sl, w] += a["ivar"]
    flux[ivar > 0] /= ivar[ivar > 0]
    res = acc / np.where(norm > 0, norm, 1.0)
    return wave, flux, ivar, res


def dense(data):
    """Return the dense matrix of a banded one, R[j - offset_k, j] = data[k, j]."""
    nd, n = data.shape
    c = nd // 2
    R = np.zeros((n, n))
    j = np.arange(n)
    for k in range(nd):
        i = j - (c - k)
        ok = (i >= 0) & (i < n)
        R[i[ok], j[ok]] = data[k, ok]
    return R


def asym(R, i0, half=8, band=3):
    """Return sum |B - B^T| / sum |B + B^T| on the band of R around pixel i0."""
    s = slice(i0 - half, i0 + half + 1)
    B = R[s, s]
    m = np.abs(np.subtract.outer(np.arange(B.shape[0]), np.arange(B.shape[0]))) <= band
    return np.abs(B - B.T)[m].sum() / np.abs(B + B.T)[m].sum()


def main(path):
    out = [f"Recoadd of {TID} from {path}"]
    with fits.open(path) as h:
        fm = h["FIBERMAP"].data
        rows = np.where(fm["TARGETID"] == TID)[0]
        out.append(f"exposures of this target: {rows.size}")
        for r in rows:
            out.append(
                f"  NIGHT {fm['NIGHT'][r]} EXPID {fm['EXPID'][r]} TILEID {fm['TILEID'][r]} "
                f"FIBER {fm['FIBER'][r]} FIBERSTATUS {fm['FIBERSTATUS'][r]}"
            )
        pairs = sorted({(int(n), int(t)) for n, t in zip(fm["NIGHT"], fm["TILEID"], strict=True)})
        out.append(f"(night, tile) pairs in the file: {len(pairs)}: {pairs}")
        exps = {}
        for b in ("b", "r", "z"):
            B = b.upper()
            exps[b] = dict(
                wave=np.asarray(h[f"{B}_WAVELENGTH"].data, float),
                flux=np.asarray(h[f"{B}_FLUX"].data[rows], float),
                ivar=np.asarray(h[f"{B}_IVAR"].data[rows], float),
                mask=np.asarray(h[f"{B}_MASK"].data[rows]),
                res=np.asarray(h[f"{B}_RESOLUTION"].data[rows], float),
            )
    dr1 = np.load(OUT / "resolution_matrix.npz")

    old, new = {}, {}
    out.append("")
    out.append("Check: column-weighted rebuild against the DR1 coadd (max abs difference)")
    for b, e in exps.items():
        tivar, fw, r_old = column_coadd(e["res"], e["ivar"], e["mask"])
        f_old = (fw * e["flux"]).sum(0) / np.where(tivar > 0, tivar, 1.0)
        out.append(
            f"  {b}: R {np.max(abs(r_old - dr1[f'{b}_res'])):.2e}, flux rel "
            f"{np.nanmax(abs(f_old - dr1[f'{b}_flux']) / np.maximum(abs(dr1[f'{b}_flux']), 1e-3)):.2e}, "
            f"ivar rel {np.nanmax(abs(tivar - dr1[f'{b}_ivar']) / np.maximum(dr1[f'{b}_ivar'], 1e-12)):.2e}"
        )
        tivar_n, w, r_new = row_coadd(e["res"], e["ivar"], e["mask"])
        f_new = (w * e["flux"]).sum(0)
        mask_new = np.bitwise_and.reduce(e["mask"], axis=0) * (tivar_n == 0)
        old[b] = dict(wave=e["wave"], flux=f_old, ivar=tivar, mask=dr1[f"{b}_mask"], res=r_old)
        new[b] = dict(wave=e["wave"], flux=f_new, ivar=tivar_n, mask=mask_new, res=r_new)

    out.append("")
    out.append("Row-weighted against column-weighted coadd, per arm")
    for b in exps:
        d = abs(new[b]["res"] - old[b]["res"])
        out.append(
            f"  {b}: max |dR| {d.max():.3f}, pixels with max |dR| > 0.01: "
            f"{(d.max(0) > 0.01).sum()} of {d.shape[1]}, > 0.1: {(d.max(0) > 0.1).sum()}"
        )
    out.append("")
    out.append(
        "At the lines: column sum at the line centre and the max row-sum deviation within +-3 px,"
        " old (DR1) and new (rows); asym of R and of R sqrt(ivar_row) per exposure."
    )
    for name, rest in LINES.items():
        lam = rest * (1 + Z)
        for b, e in exps.items():
            if not (e["wave"][10] < lam < e["wave"][-10]):
                continue
            i0 = int(np.argmin(abs(e["wave"] - lam)))
            line = f"  {name:>11} {b}:"
            for lab, cc in (("old", old[b]), ("new", new[b])):
                Rd = dense(cc["res"])
                rs = Rd.sum(1)[i0 - 3 : i0 + 4]
                line += f" {lab} colsum {Rd[:, i0].sum():.3f} rowdev {np.max(abs(rs - 1)):.4f};"
            for x in range(e["res"].shape[0]):
                Rd = dense(e["res"][x])
                M = Rd * np.sqrt(e["ivar"][x])[:, None]
                line += f" exp{x} asym R {asym(Rd, i0):.4f} R*sqrt(iv) {asym(M, i0):.4f}"
            out.append(line)

    out.append("")
    out.append(
        "Repeat test: std of (f1 - a f2) / sqrt(1/iv1 + a^2/iv2) in bins of S/N of the coadd "
        "(a = median f1/f2 per arm on S/N > 5). 1 means the ivars are right."
    )
    for b, e in exps.items():
        if e["flux"].shape[0] < 2:
            continue
        f1, f2 = e["flux"][0], e["flux"][1]
        i1, i2 = e["ivar"][0], e["ivar"][1]
        ok = (i1 > 0) & (i2 > 0) & (e["mask"][0] == 0) & (e["mask"][1] == 0)
        sn = new[b]["flux"] * np.sqrt(new[b]["ivar"])
        hi = ok & (sn > 5)
        a = np.median(f1[hi] / f2[hi]) if hi.sum() > 10 else 1.0
        d = (f1 - a * f2) / np.sqrt(1 / np.where(ok, i1, 1) + a**2 / np.where(ok, i2, 1))
        row = f"  {b}: a {a:.3f};"
        for lo, up in ((-np.inf, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, np.inf)):
            m = ok & (sn >= lo) & (sn < up)
            if m.sum() >= 3:
                row += f" S/N {lo:g}-{up:g}: {np.std(d[m]):.2f} (n {m.sum()});"
        out.append(row)

    wave, flux, ivar, res = merge_arms(new, "rows")
    _, _, _, res_oldmerge = merge_arms(old, "cols")
    out.append("")
    out.append(
        f"Merged: max |R_rows - R_DR1-merged-as-resolution_check| {np.max(abs(res - dr1['res'])):.3f}; "
        f"old merge reproduces resolution_matrix.npz to {np.max(abs(res_oldmerge - dr1['res'])):.2e}"
    )
    save = {
        "targetid": TID,
        "spectra": str(path),
        "wave": wave,
        "res": res,
        "flux": flux,
        "ivar": ivar,
    }
    for b in exps:
        for k in ("wave", "flux", "ivar", "mask", "res"):
            save[f"{b}_{k}"] = new[b][k]
        save[f"{b}_exp_res"] = exps[b]["res"]
        save[f"{b}_exp_ivar"] = exps[b]["ivar"]
        save[f"{b}_exp_flux"] = exps[b]["flux"]
    np.savez(OUT / "resolution_matrix_rows.npz", **save)
    text = "\n".join(out) + "\n"
    print(text)
    (OUT / "coadd_rows.txt").write_text(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spectra", help="iron spectra file of healpix 26004")
    main(ap.parse_args().spectra)
