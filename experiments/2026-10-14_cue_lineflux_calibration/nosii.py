"""Drop [SII] (and [OI]) from the Cue fits: do they hold metal-rich galaxies on the low branch?

QUESTION: forced onto the upper branch, Cue raises U, over-predicts [OIII] and [SII]6716 and
under-predicts [OII]3727. Two reasons [SII] could mislead a single Cue cloud: Cue fixes S/O at
solar (Li+24 arXiv:2405.04598 main.tex:91; a fixed S/O biases Cue O/H, Korhonen Cuestas+26
arXiv:2609.22488 second_draft.tex:406-407), and [SII] and [OI] can carry emission from gas outside
HII regions (Raesfeld+26 arXiv:2609.27081 p6-7 fit Cue without S, Ar and low-ionization lines and
find ~50% excess in both). With the lines left out of the likelihood (still predicted), does Cue
O/H move to the upper branch and follow N2 and mass? Held-out [SII] and [OI] under-predicted points
to a low-ionization excess; [SII] over-predicted with [OI] fine points to S/O.

An optional second argument names extra lines to drop with [SII], for example ``OI6300`` (as
Raesfeld+26 do); files then get the suffix _<lines>.

``jobs`` writes nosii/jobs.txt, one fit.py command per line: free fits without [SII] for the 80
galaxies of subset_20.txt and subset_ew60.txt and the 10 of control/control_ids.txt, and fits forced to
12+log(O/H) >= forced.ZMIN_OH without [SII] for forced_ids.txt and the controls. ``compare`` writes
nosii/nosii_compare.txt: per galaxy, O/H, U and N/O with and without [SII], the change of the best
data log likelihood forced minus free without [SII], and model/data of the held-out [SII] and of
[OIII]5007 and [OII]3727; then Spearman rho of Cue O/H with N2 and log M* over galaxies at z > Z_MIN,
and the median posterior weight above forced.ZMIN_OH for N2 > N2_MIN. SEED: none.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/nosii.py jobs
[OI6300]``; then each line of the job file from the repository root (ls6 idev); then ``nosii.py
compare [OI6300]``.

OUTPUT: jobs[_<lines>].txt and nosii_compare[_<lines>].txt in
results/2026-10-14_cue_lineflux_calibration/nosii/.
"""

import sys

import forced
import numpy as np
from astropy.table import Table, vstack
from scipy.stats import spearmanr

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
CTL = OUT / "control"
NOS = OUT / "nosii"
EXP = "experiments/2026-10-14_cue_lineflux_calibration"
EXTRA = sys.argv[2:]
SUFFIX = "".join(f"_{k}" for k in EXTRA)
DROP = "--drop " + " ".join(["SII6716", "SII6731", *EXTRA])
FREE = forced.FREE
FREE_NOS = "strong" + "".join(f"_no{k}" for k in ["SII6716", "SII6731", *EXTRA])
FREE_NOS += "_emfit_mu0.4_zs-0.41"
FORCED_NOS = f"{FREE_NOS}_zmin{forced.ZMIN_OH - forced.CUE_OH_ZERO:+.2f}"
Z_MIN = 0.01
N2_MIN = -0.8


def ids(name, where=OUT):
    """Return the TARGETIDs listed in a text file."""
    return [int(x) for x in np.loadtxt(where / name, dtype=np.int64)]


def jobs():
    """Write nosii/jobs.txt."""
    table = f"--table {CTL.relative_to(PATHS['RESULTS'].parent)}/control_table.fits"
    zmin = f"--zmin {forced.ZMIN_OH - forced.CUE_OH_ZERO:+.2f}"
    calib = ids("subset_20.txt") + ids("subset_ew60.txt")
    ctl = ids("control_ids.txt", CTL)
    run = f"python {EXP}/fit.py"
    lines = [f"{run} {t} --lines strong {DROP}" for t in calib]
    lines += [f"{run} {t} --lines strong {DROP} {table}" for t in ctl]
    lines += [f"{run} {t} --lines strong {DROP} {zmin}" for t in ids("forced_ids.txt")]
    lines += [f"{run} {t} --lines strong {DROP} {zmin} {table}" for t in ctl]
    NOS.mkdir(parents=True, exist_ok=True)
    (NOS / f"jobs{SUFFIX}.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs")


def weight_above(where, tid, tag):
    """Return the posterior weight at 12+log(O/H) >= forced.ZMIN_OH."""
    d = np.load(where / "fits" / f"{tid}_{tag}.npz")
    w = np.exp(d["log_w"] - d["log_w"].max())
    z = d["points"][:, [str(x) for x in d["labels"]].index("gas_logz")] + forced.CUE_OH_ZERO
    return float(w[z >= forced.ZMIN_OH].sum() / w.sum())


def load(where, tid, tag):
    """Return forced.load for a fit in the fits/ folder under where."""
    # forced.load reads its module OUT when called
    forced.OUT = where
    return forced.load(tid, tag)


def compare():
    """Write nosii/nosii_compare.txt."""
    t = vstack([Table.read(OUT / "lineflux_table.fits"), Table.read(CTL / "control_table.fits")])
    rows = {int(r["TARGETID"]): r for r in t}
    calib = ids("subset_20.txt") + ids("subset_ew60.txt")
    ctl = ids("control_ids.txt", CTL)
    forced_set = set(ids("forced_ids.txt")) | set(ctl)
    out = [
        f"free fits with and without [SII]; forced = 12+log(O/H) >= {forced.ZMIN_OH} without [SII]; "
        "m/d = model/data (S/N >= 3); dlnL = best data lnL forced minus free, both without [SII]",
        f"{'TARGETID':>18} {'grp':>4} {'N2':>6} {'logM':>5} {'O/H':>5} {'O/H-S':>6} {'U':>6} "
        f"{'U-S':>6} {'NO-S':>6} {'w>8.4':>6} {'SII16':>6} {'SII31':>6} {'OIII':>5} {'OII':>5} "
        f"{'fO/H':>5} {'fdlnL':>7} {'fSII16':>6} {'fOIII':>5} {'fOII':>5}",
    ]
    stats = []
    for grp, where, group in (("cal", OUT, calib), ("ctl", CTL, ctl)):
        for tid in group:
            r = rows[tid]
            n2 = float(np.log10(r["F_NII6584"] / r["F_HA"])) if r["F_NII6584"] > 0 else np.nan
            a, b = load(where, tid, FREE), load(where, tid, FREE_NOS)
            w = weight_above(where, tid, FREE_NOS)

            def md(f, k):
                return f"{f['md'][k]:.2f}" if k in f["md"] else "-"

            cells = [
                f"{tid:>18} {grp:>4} {n2:+6.2f} {r['LOGMSTAR']:5.2f} {a['oh'][1]:5.2f} "
                f"{b['oh'][1]:6.2f} {a['u'][1]:+6.2f} {b['u'][1]:+6.2f} {b['no']:+6.2f} {w:6.2f} "
                f"{md(b, 'SII6716'):>6} {md(b, 'SII6731'):>6} {md(b, 'OIII5007'):>5} "
                f"{md(b, 'OII3727'):>5}"
            ]
            if tid in forced_set:
                f = load(where, tid, FORCED_NOS)
                cells.append(
                    f" {f['oh'][1]:5.2f} {f['lnl'] - b['lnl']:+7.2f} {md(f, 'SII6716'):>6} "
                    f"{md(f, 'OIII5007'):>5} {md(f, 'OII3727'):>5}"
                )
            out.append("".join(cells))
            if r["Z"] > Z_MIN:
                stats.append((grp, n2, float(r["LOGMSTAR"]), a["oh"][1], b["oh"][1], w))
    s = np.array([x[1:] for x in stats], dtype=float)
    cal = np.array([x[0] == "cal" for x in stats])
    out.append("")
    for name, m in (("calibration (z > 0.01)", cal), ("calibration + controls", np.ones_like(cal))):
        out.append(f"{name}, N = {m.sum()}")
        for j, lab in ((2, "with [SII]"), (3, "without [SII]")):
            rn2 = spearmanr(s[m, j], s[m, 0]).statistic
            rm = spearmanr(s[m, j], s[m, 1]).statistic
            out.append(f"  Cue O/H {lab}: rho vs N2 {rn2:+.2f}, vs log M* {rm:+.2f}")
        rich = m & (s[:, 0] > N2_MIN)
        out.append(
            f"  N2 > {N2_MIN} ({rich.sum()}): median O/H with [SII] {np.median(s[rich, 2]):.2f}, "
            f"without {np.median(s[rich, 3]):.2f}; median weight above {forced.ZMIN_OH} without "
            f"[SII] {np.median(s[rich, 4]):.2f}"
        )
    text = "\n".join(out)
    print(text)
    (NOS / f"nosii_compare{SUFFIX}.txt").write_text(text + "\n")


if __name__ == "__main__":
    {"jobs": jobs, "compare": compare}[sys.argv[1]]()
