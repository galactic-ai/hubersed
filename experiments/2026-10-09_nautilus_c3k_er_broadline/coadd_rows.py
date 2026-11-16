"""Recoadd a target with the row-weighted resolution matrix of desispec PR 2377.

Also checks the iron column-weighted rebuild, R = S Q per exposure and the exposure pair scatter.
Run from the repository root as
``.venv/bin/python experiments/2026-10-09_nautilus_c3k_er_broadline/coadd_rows.py
<iron spectra folder>/spectra-sv3-bright-26004.fits.gz``,
and with ``--tid <targetid>`` and that target's spectra file for the others.
"""

import argparse

import numpy as np
import scipy.stats
from astropy.io import fits
from targets import REFERENCE, target

from hubersed.sps.lsf import DESI_WAV

LINES = {
    "[OIII]5007": 5008.240,
    "[OIII]4959": 4960.295,
    "Hbeta": 4862.683,
    "Halpha": 6564.614,
    "[NII]6584": 6585.268,
    "[SII]6718": 6718.294,
}


COSMICS_NSIG = 4.0  # desispec coadd default, 0.56.2 and PR 2377


def iron_cosmics(wave, flux, ivar, mask, nsig=COSMICS_NSIG):
    """Return the desispec 0.56.2 cosmic mask of one arm, exposures by pixels (coaddition.py:377-431)."""
    cosmic = np.zeros(ivar.shape, dtype=bool)
    if ivar.shape[0] <= 2:
        return cosmic
    masked = ivar * (mask == 0)
    grad, gradvar = [], []
    for j in range(ivar.shape[0]):
        good = masked[j] > 0
        bad = ~good
        if good.sum() == 0:
            continue
        ttflux = flux[j].copy()
        ttivar = ivar[j].copy()
        if bad.any():
            ttflux[bad] = np.interp(wave[bad], wave[good], ttflux[good])
            ttivar[bad] = np.interp(wave[bad], wave[good], ttivar[good])
        ttvar = 1.0 / (ttivar + (ttivar == 0))
        ttflux[1:] = ttflux[1:] - ttflux[:-1]
        ttvar[1:] = ttvar[1:] + ttvar[:-1]
        ttflux[0] = 0
        grad.append(ttflux)
        gradvar.append(ttvar)
    grad, gradvar = np.array(grad), np.array(gradvar)
    gradivar = (gradvar > 0) / np.array(gradvar + (gradvar == 0))
    nspec = grad.shape[0]
    sgradivar = np.sum(gradivar)
    if sgradivar > 0:
        meangrad = np.sum(gradivar * grad, axis=0) / sgradivar
        deltagrad = grad - meangrad
        chi2 = np.sum(gradivar * deltagrad**2, axis=0) / (nspec - 1)
        for bi in np.where(chi2 > nsig**2)[0]:
            k = np.argmax(gradivar[:, bi] * deltagrad[:, bi] ** 2)
            cosmic[k, bi] = True
    return cosmic


def chi2_threshold(nspec, nsig):
    """Return the chi2 of nspec - 1 dof whose tail holds the nsig one-sided normal tail (PR 2377)."""
    return scipy.stats.chi2(nspec - 1).isf(scipy.stats.norm.cdf(-nsig))


def iterative_masker(vec, ivar, nsig, min_for_cosmics, threshold=None):
    """Return the values PR 2377 masks at one pixel, dropping the worst until chi2 passes."""
    good = np.ones(len(vec), dtype=bool)
    while True:
        mean = (ivar[good] * vec[good]).sum() / ivar[good].sum()
        metric = (vec - mean) ** 2 * ivar
        chi2 = metric[good].sum()
        cur = chi2_threshold(good.sum(), nsig) if threshold is None else threshold
        if chi2 > cur:
            good[np.argmax(metric * good)] = False
        else:
            break
        if good.sum() < min_for_cosmics:
            break
    return ~good


def pr2377_cosmics(wave, flux, ivar, nsig=COSMICS_NSIG):
    """Return the PR 2377 cosmic mask of one arm, exposures by pixels (coaddition.py:512-641)."""
    min_for_cosmics = 3
    cosmic = np.zeros(ivar.shape, dtype=bool)
    if ivar.shape[0] < min_for_cosmics:
        return cosmic
    grad, gradvar, spec_pos = [], [], []
    for j in range(ivar.shape[0]):
        good = ivar[j] > 0
        bad = ~good
        if good.sum() == 0:
            continue
        spec_pos.append(j)
        ttflux = flux[j].copy()
        ttivar = ivar[j].copy()
        if bad.any():
            ttflux[bad] = np.interp(wave[bad], wave[good], ttflux[good])
            ttivar[bad] = np.interp(wave[bad], wave[good], ttivar[good])
        ttvar = 1.0 / (ttivar + (ttivar == 0))
        grad.append(ttflux[1:] - ttflux[:-1])
        gradvar.append(ttvar[1:] + ttvar[:-1])
    spec_pos, grad, gradvar = (np.array(x) for x in (spec_pos, grad, gradvar))
    gradivar = (gradvar > 0) / np.array(gradvar + (gradvar == 0))
    nspec = grad.shape[0]
    if nspec < min_for_cosmics:
        return cosmic
    sgradivar = np.sum(gradivar, axis=0)
    bad = sgradivar == 0
    if (~bad).any():
        meangrad = np.sum(gradivar * grad, axis=0) / (sgradivar + bad.astype(int))
        deltagrad = grad - meangrad
        chi2 = np.sum(gradivar * deltagrad**2, axis=0)
        med_chi2 = np.median(chi2)
        if med_chi2 > 2 * scipy.stats.chi2(nspec - 1).ppf(0.5):
            threshold = med_chi2 + (scipy.stats.scoreatpercentile(chi2, 84) - med_chi2) * nsig
            fix_threshold = threshold
        else:
            threshold = chi2_threshold(nspec, nsig)
            fix_threshold = None
        for bi in np.where((chi2 > threshold) & (~bad))[0]:
            cur = iterative_masker(
                deltagrad[:, bi], gradivar[:, bi], nsig, min_for_cosmics, threshold=fix_threshold
            )
            cosmic[spec_pos[cur], bi] = True
            cosmic[spec_pos[cur], bi + 1] = True
    return cosmic


def column_coadd(res, ivar, mask, cosmic=None):
    """Return the DR1 coadd of one arm, matrix weighted by the ivar of the column pixel."""
    good = ivar * (mask == 0)
    if cosmic is not None:
        good[cosmic] = 0
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


def row_coadd(res, ivar, mask, cosmic=None):
    """Return the PR 2377 coadd of one arm, with its flux weights and row-weighted matrix."""
    masked = ivar * (mask == 0)
    if cosmic is not None:
        masked[cosmic] = 0
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


def main(path, tid=REFERENCE):
    TID, Z, OUT = tid, target(tid)["z"], target(tid)["out"]
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
    cosmics = []
    for b, e in exps.items():
        c_iron = iron_cosmics(e["wave"], e["flux"], e["ivar"], e["mask"])
        c_new = pr2377_cosmics(e["wave"], e["flux"], e["ivar"] * (e["mask"] == 0))
        near = np.zeros(e["wave"].size, dtype=bool)
        for rest in LINES.values():
            near |= np.abs(e["wave"] - rest * (1 + Z)) < 10 * np.median(np.diff(e["wave"]))
        cosmics.append(
            f"  {b}: iron {c_iron.sum(1).tolist()} ({c_iron[:, near].sum()} near lines), "
            f"PR 2377 {c_new.sum(1).tolist()} ({c_new[:, near].sum()} near lines)"
        )
        tivar, fw, r_old = column_coadd(e["res"], e["ivar"], e["mask"], c_iron)
        f_old = (fw * e["flux"]).sum(0) / np.where(tivar > 0, tivar, 1.0)
        out.append(
            f"  {b}: R {np.max(abs(r_old - dr1[f'{b}_res'])):.2e}, flux rel "
            f"{np.nanmax(abs(f_old - dr1[f'{b}_flux']) / np.maximum(abs(dr1[f'{b}_flux']), 1e-3)):.2e}, "
            f"ivar rel {np.nanmax(abs(tivar - dr1[f'{b}_ivar']) / np.maximum(dr1[f'{b}_ivar'], 1e-12)):.2e}"
        )
        tivar_n, w, r_new = row_coadd(e["res"], e["ivar"], e["mask"], c_new)
        f_new = (w * e["flux"]).sum(0)
        mask_new = np.bitwise_and.reduce(e["mask"], axis=0) * (tivar_n == 0)
        old[b] = dict(wave=e["wave"], flux=f_old, ivar=tivar, mask=dr1[f"{b}_mask"], res=r_old)
        new[b] = dict(wave=e["wave"], flux=f_new, ivar=tivar_n, mask=mask_new, res=r_new)
    if rows.size > 2:
        out.append("")
        out.append(
            f"Cosmic-masked pixels per exposure ({COSMICS_NSIG:g} sigma), iron 0.56.2 and PR 2377, "
            "and how many lie within 10 px of the lines"
        )
        out += cosmics

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
    ap.add_argument("spectra", help="iron spectra file of the target's healpix")
    ap.add_argument("--tid", type=int, default=REFERENCE, help="target of targets.py")
    args = ap.parse_args()
    main(args.spectra, args.tid)
