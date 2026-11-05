"""Build the per exposure-petal-camera calibration error table of the SV3 bright sample and check it.

QUESTION: what is the corrected calibration error f = sigma_C / C of every exposure-petal-camera that
holds our sample, did the reruns reproduce iron, and what predicts f?

Reads the merged tables of fluxcal_sample.py (the two halves of the run ``all``) and the unit list
units_all.csv of make_units.py. For each exposure, petal and camera it keeps the median over good
fibers (FIBERSTATUS 0) of the per-fiber median f, corrected and iron, the 5-95% spread of the
corrected f over fibers, the checks against iron, and the unit's QA values. Units that were skipped
or failed get f = NaN and a status. The text output gives the checks, the distribution of f per
camera, weighted by the number of sample targets, and how f depends on NSTDSTAR and STARRMS.

INPUTS: results/2026-10-13_fluxcal_sample/all/{cameras,fibers}_part{0,1}.fits, units_all.csv.
SEED: none.

COMMAND: from the repository root, ``uv run python experiments/2026-10-13_fluxcal_sample/summarise_f.py``.

OUTPUT: f_table.fits and summarise_f.txt in results/2026-10-13_fluxcal_sample/all/.
"""

import numpy as np
from astropy.table import Table, join, vstack

from hubersed.paths import PATHS

HERE = PATHS["ROOT"] / "experiments" / "2026-10-13_fluxcal_sample"
RUN = PATHS["RESULTS"] / "2026-10-13_fluxcal_sample" / "all"
KEYS = ["EXPID", "PETAL", "CAMERA"]


def main():
    parts = []
    for k in (0, 1):
        c = Table.read(RUN / f"cameras_part{k}.fits")
        for col in ("FAILED", "SKIPPED", "CAMERA"):
            v = np.asarray(c[col])
            c[col] = np.char.strip(v.astype(str)) if v.dtype.kind in "SU" else np.full(len(c), "")
        parts.append(c)
    cams = vstack(parts)
    cols = [
        "EXPID",
        "PETAL",
        "CAMERA",
        "FIBERSTATUS",
        "F_PATCH_P50",
        "F_IRON_P50",
        "FLAT_TO_PSF_FLUX",
        "IN_SAMPLE",
    ]
    fib = vstack([Table.read(RUN / f"fibers_part{k}.fits")[cols] for k in (0, 1)])
    units = Table.read(HERE / "units_all.csv")
    cams["CAMERA"] = np.char.strip(cams["CAMERA"].astype(str))
    fib["CAMERA"] = np.char.strip(fib["CAMERA"].astype(str))
    lines = [f"Calibration error table of the SV3 bright sample from {RUN}"]
    lines.append(f"camera rows {len(cams)} (expected {3 * len(units)}), fiber rows {len(fib)}")

    good = (fib["FIBERSTATUS"] == 0) & np.isfinite(fib["F_PATCH_P50"])
    rows = []
    key = np.char.add(
        np.char.add(fib["EXPID"].astype(str), "-"),
        np.char.add(fib["PETAL"].astype(str), fib["CAMERA"]),
    )
    order = np.argsort(key, kind="stable")
    ks, starts = np.unique(key[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    for _, a, b in zip(ks, starts, ends, strict=True):
        idx = order[a:b]
        g = idx[good[idx]]
        fp = np.asarray(fib["F_PATCH_P50"][g])
        rows.append(
            dict(
                EXPID=int(fib["EXPID"][idx[0]]),
                PETAL=int(fib["PETAL"][idx[0]]),
                CAMERA=str(fib["CAMERA"][idx[0]]),
                NGOOD=len(g),
                F=float(np.median(fp)) if fp.size else np.nan,
                F_FIB_P05=float(np.percentile(fp, 5)) if fp.size else np.nan,
                F_FIB_P95=float(np.percentile(fp, 95)) if fp.size else np.nan,
                F_IRON=float(np.median(fib["F_IRON_P50"][g])) if fp.size else np.nan,
                PSC_MED=float(np.median(fib["FLAT_TO_PSF_FLUX"][g])) if fp.size else np.nan,
            )
        )
    f = Table(rows=rows)
    t = join(cams, f, keys=KEYS, join_type="left")
    t = join(t, units, keys=["EXPID", "PETAL"], join_type="left", table_names=["", "unit"])
    status = np.array(["ok"] * len(t), dtype="<32U")
    skipped = np.char.strip(np.asarray(t["SKIPPED"]).astype(str))
    status[skipped != ""] = skipped[skipped != ""]
    nof = (status == "ok") & ~np.isfinite(np.asarray(t["F"], dtype=float))
    status[nof] = "FAILED"
    t["STATUS"] = status
    t.write(RUN / "f_table.fits", overwrite=True)

    ok = t["STATUS"] == "ok"
    lines.append(
        "status: "
        + ", ".join(f"{s} {int((t['STATUS'] == s).sum())}" for s in np.unique(t["STATUS"]))
    )
    d_un = np.asarray(t["UNPATCHED_VS_IRON_MAXABS"][ok], float)
    d_c = np.asarray(t["PATCHED_C_VS_IRON_MAXABS"][ok], float)
    lines.append(
        f"unpatched vs iron, max abs over cameras {np.nanmax(d_un):.2e}; patched C vs iron {np.nanmax(d_c):.2e}"
    )
    spread = np.asarray(t["F_FIB_P95"][ok] / t["F_FIB_P05"][ok] - 1, float)
    lines.append(
        f"f spread over fibers within a camera (p95/p05 - 1): median {np.nanmedian(spread):.2e}, max {np.nanmax(spread):.2e}"
    )
    failed = t[t["STATUS"] != "ok"]
    lines.append(
        "not ok: "
        + "; ".join(
            f"{r['EXPID']} p{r['PETAL']} {r['CAMERA']} {r['STATUS']} (NSAMPLE {r['NSAMPLE']}, NSTDSTAR {r['NSTDSTAR']})"
            for r in failed
            if r["STATUS"] == "FAILED"
        )
    )
    w_bad = int(np.sum(np.asarray(units["NSAMPLE"])[np.asarray(units["N_BADPETALSTDSTAR"]) > 0]))
    lines.append(
        f"sample target-exposure pairs on BADPETALSTDSTAR units {w_bad} of {int(np.sum(units['NSAMPLE']))}"
    )
    lines.append("")
    lines.append(
        "corrected f per camera over units (unweighted, and weighted by NSAMPLE): p05 p16 p50 p84 p95"
    )
    for c in "brz":
        m = ok & (t["CAMERA"] == c)
        v = np.asarray(t["F"][m], float)
        w = np.asarray(t["NSAMPLE"][m], float)
        o = np.argsort(v)
        cw = np.cumsum(w[o]) / w.sum()
        wq = [v[o][np.searchsorted(cw, q)] for q in (0.05, 0.16, 0.5, 0.84, 0.95)]
        lines.append(
            f"  {c}: n {m.sum()}  "
            + " ".join(f"{x:.4f}" for x in np.percentile(v, [5, 16, 50, 84, 95]))
            + "  | weighted "
            + " ".join(f"{x:.4f}" for x in wq)
        )
        vi = np.asarray(t["F_IRON"][m], float)
        lines.append(
            f"     iron f median {np.median(vi):.4f}; corrected/iron median {np.median(v / vi):.3f}, 5-95% {np.percentile(v / vi, 5):.3f}-{np.percentile(v / vi, 95):.3f}"
        )
    lines.append("")
    lines.append("median corrected f (r camera) in bins of NSTDSTAR")
    m = ok & (t["CAMERA"] == "r")
    ns = np.asarray(t["NSTDSTAR"][m])
    v = np.asarray(t["F"][m], float)
    for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 100)):
        s = (ns >= lo) & (ns < hi)
        if s.sum():
            lines.append(
                f"  NSTDSTAR {lo}-{hi - 1}: n {s.sum()}, f median {np.median(v[s]):.4f}, 16-84% {np.percentile(v[s], 16):.4f}-{np.percentile(v[s], 84):.4f}"
            )
    x = np.column_stack(
        [np.ones(m.sum()), 1 / np.sqrt(np.maximum(ns, 1)), np.asarray(t["STARRMS"][m], float)]
    )
    good_x = np.all(np.isfinite(x), axis=1) & np.isfinite(v)
    coef, *_ = np.linalg.lstsq(x[good_x], v[good_x], rcond=None)
    pred = x[good_x] @ coef
    lines.append(
        f"  fit f = a + b/sqrt(NSTDSTAR) + c STARRMS (r): a {coef[0]:.4f}, b {coef[1]:.4f}, c {coef[2]:.4f}; residual rms {np.std(v[good_x] - pred):.4f} vs f rms {np.std(v[good_x]):.4f}"
    )
    text = "\n".join(lines) + "\n"
    print(text)
    (RUN / "summarise_f.txt").write_text(text)


if __name__ == "__main__":
    main()
