"""Check how well a fractional error floor on the iron ivar reproduces the corrected ivar.

QUESTION: can the corrected calibrated variance of DESI DR1 spectra be approximated for a sample by
adding a fractional floor, Var = 1/ivar_iron + (f F')^2, without rerunning the flux calibration of
every exposure?

The corrected variance of one exposure (desispec PR 2422 formula with the ivar_C of PRs 2496, 2827
and 2452) is Var_new = 1/(ivar_F C^2) + F'^2 / (ivar_C,new C^2), with F' the calibrated flux. The
iron variance is 1/(ivar_F C^2) + F'^2 / (ivar_C,old C^4). So Var_new = 1/ivar_iron - F'^2 /
(ivar_C,old C^4) + (F' sigma_C / C)^2, a fractional floor with f = sigma_C / C, which depends on
exposure, camera and wavelength. This script builds Var_new exactly for both exposures of this
target and compares it with floor models: a constant f per exposure and camera (the median of
sigma_C / C), and one global f for everything (chosen to fit best at coadd S/N above 20). The
comparison is sigma_floor / sigma_exact of the row-weighted coadd in bins of S/N.

INPUTS: resolution_matrix_rows.npz (iron per-exposure flux and ivar, from the iron spectra file),
resolution_matrix_rows_ivar.npz (the exact recoadd, as a check) and the unpatched and patched
fluxcalib files of fluxcal_rerun.py, all in results/2026-10-09_nautilus_c3k_er_broadline/
39627770174637084/. SEED: none.

COMMAND: from the repository root, ``uv run python
experiments/2026-10-13_nautilus_resolution_matrix/ivar_floor_check.py``.

OUTPUT: ivar_floor_check.txt in results/2026-10-13_nautilus_resolution_matrix/.
"""

import numpy as np
from astropy.io import fits

from hubersed.paths import PATHS

TID = 39627770174637084
ER = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline" / str(TID)
OUT = PATHS["RESULTS"] / "2026-10-13_nautilus_resolution_matrix"
EXPOSURES = (
    (87113, 224),
    (88269, 204),
)  # order of the exposures in the spectra file, fluxcalib row
SN_BINS = ((5, 20), (20, 40), (40, 80), (80, np.inf))


def calib(mode, cam, expid, row):
    """Return the calibration vector and its ivar of one fiber from a rerun fluxcalib file."""
    path = ER / "fluxcalib_rerun" / mode / f"fluxcalib-{cam}-{expid:08d}.fits.gz"
    with fits.open(path) as h:
        return h["FLUXCALIB"].data[row].astype(float), h["IVAR"].data[row].astype(float)


def summary(ratio, sn, label):
    """Return one line with the median and 5-95% of a sigma ratio in bins of S/N."""
    row = f"  {label:<28}"
    for lo, hi in SN_BINS:
        m = np.isfinite(ratio) & (sn >= lo) & (sn < hi)
        if m.sum() >= 3:
            q = np.percentile(ratio[m], [5, 50, 95])
            row += f" S/N {lo:g}-{hi:g}: {q[1]:.3f} [{q[0]:.3f},{q[2]:.3f}] n {m.sum()};"
    return row


def main():
    rows = np.load(ER / "resolution_matrix_rows.npz")
    exact_coadd = np.load(ER / "resolution_matrix_rows_ivar.npz")
    lines = [f"Fractional floor against the exact corrected ivar of {TID}"]
    per = {}
    for arm in "brz":
        cam = f"{arm}1"
        flux, ivar = rows[f"{arm}_exp_flux"], rows[f"{arm}_exp_ivar"]
        var_exact, frac = [], []
        for e, (expid, row) in enumerate(EXPOSURES):
            C, ivc_old = calib("unpatched", cam, expid, row)
            C2, ivc_new = calib("patched", cam, expid, row)
            assert np.array_equal(C, C2)
            Fp, iv = flux[e], ivar[e]
            ok = (iv > 0) & (C > 0) & (ivc_old > 0) & (ivc_new > 0)
            v = np.full(Fp.size, np.inf)
            v[ok] = (
                1 / iv[ok]
                - Fp[ok] ** 2 / (ivc_old[ok] * C[ok] ** 4)
                + Fp[ok] ** 2 / (ivc_new[ok] * C[ok] ** 2)
            )
            f = np.full(Fp.size, np.nan)
            f[ok] = 1 / (C[ok] * np.sqrt(ivc_new[ok]))
            var_exact.append(v)
            frac.append(f)
            q = np.nanpercentile(f, [5, 50, 95])
            lines.append(
                f"{expid} {cam}: sigma_C/C corrected median {q[1]:.4f}, 5-95% {q[0]:.4f}-{q[2]:.4f}"
            )
        per[arm] = dict(flux=flux, ivar=ivar, var=np.array(var_exact), f=np.array(frac))
        iv_exact = np.where(np.isfinite(per[arm]["var"]), 1 / per[arm]["var"], 0).sum(0)
        ok = (iv_exact > 0) & (exact_coadd[f"{arm}_ivar"] > 0)
        lines.append(
            f"  check {arm}: coadd of the exact ivar vs resolution_matrix_rows_ivar.npz, max rel "
            f"{np.max(abs(iv_exact[ok] / exact_coadd[f'{arm}_ivar'][ok] - 1)):.1e}"
        )

    def coadd_sigma(arm, var):
        iv = np.where(np.isfinite(var) & (var > 0), 1 / var, 0).sum(0)
        return np.where(iv > 0, 1 / np.sqrt(iv), np.nan)

    # global f: best median |ln ratio| at coadd S/N > 20
    grid = np.arange(0.005, 0.0401, 0.0005)
    sig_exact, sn = {}, {}
    for arm, p in per.items():
        sig_exact[arm] = coadd_sigma(arm, p["var"])
        fl = (np.where(p["ivar"] > 0, p["ivar"] * p["flux"], 0)).sum(0) / np.maximum(
            (p["ivar"] > 0) * p["ivar"], 0
        ).sum(0).clip(1e-300)
        sn[arm] = fl / sig_exact[arm]
    score = []
    for f in grid:
        r = []
        for arm, p in per.items():
            v = np.where(
                p["ivar"] > 0,
                1 / np.where(p["ivar"] > 0, p["ivar"], 1) + (f * p["flux"]) ** 2,
                np.inf,
            )
            rr = coadd_sigma(arm, v) / sig_exact[arm]
            m = np.isfinite(rr) & (sn[arm] > 20)
            r.append(np.abs(np.log(rr[m])))
        score.append(np.median(np.concatenate(r)))
    f_glob = float(grid[int(np.argmin(score))])
    lines.append("")
    lines.append(f"Global f fitted at coadd S/N > 20: {f_glob:.4f}")
    lines.append("")
    lines.append("sigma_model / sigma_exact of the coadd, median [5%, 95%], in bins of coadd S/N")
    for arm, p in per.items():
        lines.append(f"{arm} arm:")
        iron = np.where(p["ivar"] > 0, 1 / np.where(p["ivar"] > 0, p["ivar"], 1), np.inf)
        lines.append(summary(coadd_sigma(arm, iron) / sig_exact[arm], sn[arm], "iron, no floor"))
        fmed = np.nanmedian(p["f"], axis=1)[:, None]
        v1 = iron + (fmed * p["flux"]) ** 2
        lines.append(
            summary(coadd_sigma(arm, v1) / sig_exact[arm], sn[arm], "floor, f per exposure-camera")
        )
        v2 = iron + (f_glob * p["flux"]) ** 2
        lines.append(
            summary(coadd_sigma(arm, v2) / sig_exact[arm], sn[arm], f"floor, global f {f_glob:.4f}")
        )
    text = "\n".join(lines) + "\n"
    print(text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ivar_floor_check.txt").write_text(text)


if __name__ == "__main__":
    main()
