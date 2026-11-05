"""Add [SIII]9069 and He I 5876 to the Cue fits: do semi-strong lines put galaxies on the right branch?

QUESTION: with DESI strong lines Cue does not recover the direct T_e ranking (nosii.py RESULT). In
Y_P galaxies, adding [SIII]9069,9531, [SII], He I 5876 and [NeIII] to the strong lines (line list 2)
removes the strong-line offset, rms 0.06 dex, and [SIII] matters most (Korhonen Cuestas+26
arXiv:2609.22488 Table 2, second_draft.tex:496, :506-508; tested only below ~8.3 with near-solar
S/O). Does fit.py --lines semi --drop OI6300 (their list 2 plus Hgamma, [OI] dropped as they do)
follow direct T_e O/H and N2 for BGS galaxies with [SIII]9069 in the DESI range (z < ~0.08)?

``ids`` writes semi/semi_ids.txt: galaxies of lineflux_table.fits at z > Z_MIN with [SIII]9069 and
He I 5876 at S/N >= SNR_MIN. ``jobs`` writes semi/jobs.txt: a semi fit per galaxy and a strong fit where none exists. ``compare``
writes semi/semi_compare.txt: per galaxy N2, log M*, direct T_e O/H, Cue O/H with strong lines and
with semi lines, the posterior weight above 8.4, and model/data of [SII]6716+6731, [SIII]9069, their
ratio and held-out [OI]; then Spearman rho and offsets against direct O/H (direct.py --ids
semi_ids.txt, KEEP) and against N2. If [SII] and [SIII] are off by the same factor, Cue's fixed S/O
is the likely cause; if their ratio is off, low-ionization emission from outside HII regions is
(von Raesfeld+26 arXiv:2609.27081 Fig. 3). SEED: none.

RESULT: 2026-10-15, 444 galaxies, 443 semi fits (ls6 idev; 1 not finished), direct T_e kept for 433
(rho with N2 +0.53, median 8.09 to 8.59 from N2 <= -1.2 to > -0.6). Cue semi O/H is flat at 8.1-8.2
in every N2 bin (rho with direct -0.09, with N2 +0.09; median offset -0.14 [-0.49, +0.20]); strong
on the 107 with both: rho -0.22. Metal-poorer galaxies (N2 <= -0.8, 309): model/data [SII] 0.95,
[SIII] 1.31, [SII]/[SIII] 0.70, held-out [OI] 0.53, the low-ionization excess of von Raesfeld+26;
N2 > -0.8 (134): 0.99, 1.04, 0.95, 0.72. Adding [SIII] and He I does not break the branch
degeneracy here; only 9% of SV3 star-forming galaxies (2% above log M* 10) have [SIII]9069 anyway.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/semi.py ids``;
``direct.py --ids results/2026-10-14_cue_lineflux_calibration/semi/semi_ids.txt``; ``semi.py jobs``;
each line of semi/jobs.txt from the repository root (ls6, one process per core); then ``semi.py
compare``.

OUTPUT: semi_ids.txt, jobs.txt and semi_compare.txt in results/2026-10-14_cue_lineflux_calibration/semi/.
"""

import sys

import forced
import numpy as np
from astropy.table import Table
from scipy.stats import spearmanr

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
SEM = OUT / "semi"
EXP = "experiments/2026-10-14_cue_lineflux_calibration"
Z_MIN = 0.01
SNR_MIN = 3.0
N2_MIN = -0.8
STRONG = forced.FREE
SEMI = "semi_noOI6300_emfit_mu0.4_zs-0.41"
DIRECT = "direct_wild0.4_campbell_r2_semi_ids.fits"


def ids():
    """Write semi/semi_ids.txt."""
    t = Table.read(OUT / "lineflux_table.fits")
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = t["Z"] > Z_MIN
        for k in ("SIII9069", "HEI5876"):
            ok &= np.nan_to_num(t[f"F_{k}"] / t[f"E_{k}"]) >= SNR_MIN
    SEM.mkdir(parents=True, exist_ok=True)
    np.savetxt(SEM / "semi_ids.txt", t["TARGETID"][ok], fmt="%d")
    print(f"{ok.sum()} galaxies")


def jobs():
    """Write semi/jobs.txt: semi fits for all, strong fits for galaxies that lack one."""
    run = f"python {EXP}/fit.py"
    tids = np.loadtxt(SEM / "semi_ids.txt", dtype=np.int64)
    lines = [f"{run} {t} --lines semi --drop OI6300" for t in tids]
    lines += [
        f"{run} {t} --lines strong"
        for t in tids
        if not (OUT / "fits" / f"{t}_{STRONG}.npz").exists()
    ]
    (SEM / "jobs.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs")


def ratios(tid):
    """Return model/data medians of [SII] summed, [SIII]9069, their ratio and [OI] for the semi fit."""
    d = np.load(OUT / "fits" / f"{tid}_{SEMI}.npz")
    lines = [str(x) for x in d["lines"]]
    obs, draws = d["obs"], d["draw_flux"]

    def col(k):
        return draws[:, lines.index(k)], obs[lines.index(k)]

    (m16, o16), (m31, o31) = col("SII6716"), col("SII6731")
    m9, o9 = col("SIII9069")
    m1, o1 = col("OI6300")
    s2 = np.median((m16 + m31) / (o16 + o31))
    s3 = np.median(m9 / o9)
    r = np.median(((m16 + m31) / m9) / ((o16 + o31) / o9))
    oi = np.median(m1 / o1) if np.isfinite(o1) else np.nan
    return s2, s3, r, oi


def compare():
    """Write semi/semi_compare.txt."""
    t = {int(r["TARGETID"]): r for r in Table.read(OUT / "lineflux_table.fits")}
    d = Table.read(OUT / DIRECT)
    direct = {int(r["TARGETID"]): r for r in d}
    forced.OUT = OUT
    out = [
        "Cue O/H strong (fit.py --lines strong) and semi (--lines semi --drop OI6300) vs direct T_e "
        "O/H; m/d = model/data of the semi fit, OI held out",
        f"{'TARGETID':>18} {'N2':>6} {'logM':>5} {'direct':>6} {'keep':>4} {'strong':>6} {'semi':>5} "
        f"{'w>8.4':>5} {'SII':>5} {'SIII':>5} {'S2/S3':>5} {'OI':>5}",
    ]
    rows = []
    missing = []
    for tid in np.loadtxt(SEM / "semi_ids.txt", dtype=np.int64):
        tid = int(tid)
        if not (OUT / "fits" / f"{tid}_{SEMI}.npz").exists():
            missing.append(tid)
            continue
        r = t[tid]
        n2 = float(np.log10(r["F_NII6584"] / r["F_HA"]))
        s = forced.load(tid, SEMI)
        dr = direct.get(tid)
        dir_oh = float(dr["OH_P50"]) if dr is not None else np.nan
        keep = bool(dr["KEEP"]) if dr is not None else False
        have = (OUT / "fits" / f"{tid}_{STRONG}.npz").exists()
        st = forced.load(tid, STRONG)["oh"][1] if have else np.nan
        dz = np.load(OUT / "fits" / f"{tid}_{SEMI}.npz")
        w = np.exp(dz["log_w"] - dz["log_w"].max())
        z = dz["points"][:, [str(x) for x in dz["labels"]].index("gas_logz")] + forced.CUE_OH_ZERO
        wa = float(w[z >= forced.ZMIN_OH].sum() / w.sum())
        s2, s3, rr, oi = ratios(tid)
        out.append(
            f"{tid:>18} {n2:+6.2f} {r['LOGMSTAR']:5.2f} {dir_oh:6.2f} {'yes' if keep else 'no':>4} "
            f"{st:6.2f} {s['oh'][1]:5.2f} {wa:5.2f} {s2:5.2f} {s3:5.2f} {rr:5.2f} {oi:5.2f}"
        )
        rows.append((n2, dir_oh if keep else np.nan, st, s["oh"][1], s2, s3, rr, oi))
    a = np.array(rows, float)
    out.append("")
    out.append(f"semi fits missing: {len(missing)} {missing}")
    out.append(
        f"N = {len(a)}; Cue semi O/H vs N2: rho {spearmanr(a[:, 3], a[:, 0]).statistic:+.2f}"
    )
    k = np.isfinite(a[:, 1])
    out.append(
        f"direct kept: N = {k.sum()}, direct vs N2 rho {spearmanr(a[k, 1], a[k, 0]).statistic:+.2f}"
    )
    for j, lab in ((2, "strong"), (3, "semi")):
        m = k & np.isfinite(a[:, j])
        diff = a[m, j] - a[m, 1]
        lo, mid, hi = np.percentile(diff, [16, 50, 84])
        rho = spearmanr(a[m, j], a[m, 1]).statistic
        out.append(
            f"  Cue {lab} - direct (N = {m.sum()}): median {mid:+.2f} [{lo:+.2f}, {hi:+.2f}], rho {rho:+.2f}"
        )
    for lab, m in (("N2 > -0.8", a[:, 0] > N2_MIN), ("N2 <= -0.8", a[:, 0] <= N2_MIN)):
        med = [np.nanmedian(a[m, j]) for j in (4, 5, 6, 7)]
        out.append(
            f"  {lab} ({m.sum()}): median m/d [SII] {med[0]:.2f}, [SIII] {med[1]:.2f}, "
            f"[SII]/[SIII] {med[2]:.2f}, [OI] {med[3]:.2f}"
        )
    text = "\n".join(out)
    print(text)
    (SEM / "semi_compare.txt").write_text(text + "\n")


if __name__ == "__main__":
    {"ids": ids, "jobs": jobs, "compare": compare}[sys.argv[1]]()
