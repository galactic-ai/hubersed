"""Recompute the calibrated ivar of 39627770174637084 with the desispec fixes and recoadd it.

QUESTION: with the corrected flux-calibration variance, do the two exposures of this target agree
within their errors at high S/N, and what are the coadded flux and ivar to fit?

For each exposure and camera (petal 1) the calibrated variance is recomputed with the formula of
desispec PR 2422, Var(F') = 1/(ivar_F C^2) + F^2/(ivar_C C^4), with F the sky-subtracted,
flat-fielded flux before calibration (iron sframe), C the iron calibration vector and ivar_C from
fluxcal_rerun.py with PRs 2496, 2827 and 2452 (patched). The iron formula, with the calibrated flux
in place of F, is checked against the iron cframe ivar first. Flux, mask and resolution matrix are
iron's (cframe). The exposures are then coadded per arm with the row weights of desispec PR 2377
(coadd_rows.py) and merged over the arms. The repeat test of coadd_rows.py is rerun with the new
ivars.

INPUTS: iron sframe, cframe and fluxcalib files of exposures 87113 and 88269 (petal 1) under
data/desi_iron_exposures/<night>/<expid>/, the patched fluxcalib files of fluxcal_rerun.py, and
resolution_matrix_rows.npz of coadd_rows.py for the comparison. SEED: none.

COMMAND: on an ls6 compute node from the repository root,
``.venv/bin/python experiments/2026-10-09_nautilus_c3k_er_broadline/ivar_recompute.py``.

OUTPUT: ivar_recompute.txt and resolution_matrix_rows_ivar.npz (keys as resolution_matrix_rows.npz)
in results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/.
"""

import numpy as np
from astropy.io import fits
from coadd_rows import merge_arms, row_coadd

from hubersed.paths import PATHS

TID = 39627770174637084
EXPOSURES = {87113: "20210503", 88269: "20210513"}
DATA = PATHS["DATA"] / "desi_iron_exposures"
OUT = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline" / str(TID)
PATCHED = OUT / "fluxcalib_rerun" / "patched"


def exposure(night, expid, cam):
    """Return the target's iron cframe data and the recomputed ivar for one camera."""
    d = DATA / night / f"{expid:08d}"
    name = f"{cam}-{expid:08d}.fits.gz"
    with (
        fits.open(d / f"cframe-{name}") as cf,
        fits.open(d / f"sframe-{name}") as sf,
        fits.open(d / f"fluxcalib-{name}") as fc,
        fits.open(PATCHED / f"fluxcalib-{name}") as fp,
    ):
        row = int(np.where(cf["FIBERMAP"].data["TARGETID"] == TID)[0][0])
        Fp = cf["FLUX"].data[row].astype(float)
        iv = cf["IVAR"].data[row].astype(float)
        F = sf["FLUX"].data[row].astype(float)
        ivF = sf["IVAR"].data[row].astype(float)
        C = fc["FLUXCALIB"].data[row].astype(float)
        ivC_old = fc["IVAR"].data[row].astype(float)
        ivC_new = fp["IVAR"].data[row].astype(float)
        assert np.array_equal(fp["FLUXCALIB"].data[row], fc["FLUXCALIB"].data[row])
        out = dict(
            wave=cf["WAVELENGTH"].data.astype(float),
            flux=Fp,
            mask=cf["MASK"].data[row],
            res=cf["RESOLUTION"].data[row].astype(float),
            ivar_iron=iv,
        )
    ok = (iv > 0) & (ivF > 0) & (ivC_new > 0) & (C > 0)
    var_old = 1 / (ivF * C**2) + (F / C) ** 2 / (ivC_old * C**4)
    var_new = 1 / (ivF * C**2) + F**2 / (ivC_new * C**4)
    ivar = np.zeros_like(iv)
    ivar[ok] = 1 / var_new[ok]
    out["ivar"] = ivar
    out["check_old"] = np.percentile(np.abs(var_old[ok] * iv[ok] - 1), 99)
    out["ok"] = ok
    return out


def repeat_test(e1, e2, sn):
    """Return the std of the normalised two-exposure difference in bins of coadd S/N."""
    f1, f2, i1, i2 = e1["flux"], e2["flux"], e1["ivar"], e2["ivar"]
    ok = (i1 > 0) & (i2 > 0) & (e1["mask"] == 0) & (e2["mask"] == 0)
    hi = ok & (sn > 5)
    a = np.median(f1[hi] / f2[hi])
    d = (f1 - a * f2) / np.sqrt(1 / np.where(ok, i1, 1) + a**2 / np.where(ok, i2, 1))
    row = f"a {a:.3f};"
    for lo, up in ((-np.inf, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, np.inf)):
        m = ok & (sn >= lo) & (sn < up)
        if m.sum() >= 3:
            row += f" S/N {lo:g}-{up:g}: {np.std(d[m]):.2f} (n {m.sum()});"
    return row


def main():
    lines = [f"Recomputed calibrated ivar of {TID} (PR 2422 formula, patched ivar_C)"]
    exps = {b: [] for b in "brz"}
    for expid, night in EXPOSURES.items():
        for cam in ("b1", "r1", "z1"):
            e = exposure(night, expid, cam)
            exps[cam[0]].append(e)
            sn = e["flux"] * np.sqrt(e["ivar_iron"])
            r = np.sqrt(e["ivar_iron"][e["ok"]] / e["ivar"][e["ok"]])
            row = f"{expid} {cam}: iron formula vs cframe 99% {e['check_old']:.1e}; sigma_new/sigma_iron"
            for lo, up in ((5, 20), (20, 40), (40, 80), (80, np.inf)):
                m = (sn[e["ok"]] >= lo) & (sn[e["ok"]] < up)
                if m.sum():
                    row += f" S/N {lo:g}-{up:g}: {np.median(r[m]):.2f};"
            lines.append(row)
    arms = {}
    lines.append("")
    lines.append("Repeat test with the new ivars (std of the normalised difference; 1 is right)")
    for b, (e1, e2) in exps.items():
        stack = {k: np.array([e1[k], e2[k]]) for k in ("flux", "ivar", "mask", "res")}
        tivar, w, rdata = row_coadd(stack["res"], stack["ivar"], stack["mask"])
        flux = (w * stack["flux"]).sum(0)
        mask = np.bitwise_and.reduce(stack["mask"], axis=0) * (tivar == 0)
        arms[b] = dict(wave=e1["wave"], flux=flux, ivar=tivar, mask=mask, res=rdata)
        lines.append(f"  {b}: " + repeat_test(e1, e2, flux * np.sqrt(tivar)))
    old = np.load(OUT / "resolution_matrix_rows.npz")
    wave, flux, ivar, res = merge_arms(arms, "rows")
    ok = (ivar > 0) & (old["ivar"] > 0)
    lines.append("")
    lines.append(
        f"Merged coadd: max |R - R_rows| {np.max(abs(res - old['res'])):.3f}; flux rel change 99% "
        f"{np.percentile(abs(flux[ok] / old['flux'][ok] - 1), 99):.2e}; median sigma_new/sigma_rows "
        f"{np.median(np.sqrt(old['ivar'][ok] / ivar[ok])):.3f}"
    )
    save = {"targetid": TID, "wave": wave, "res": res, "flux": flux, "ivar": ivar}
    for b, a in arms.items():
        for k in ("wave", "flux", "ivar", "mask", "res"):
            save[f"{b}_{k}"] = a[k]
    np.savez(OUT / "resolution_matrix_rows_ivar.npz", **save)
    text = "\n".join(lines) + "\n"
    print(text)
    (OUT / "ivar_recompute.txt").write_text(text)


if __name__ == "__main__":
    main()
