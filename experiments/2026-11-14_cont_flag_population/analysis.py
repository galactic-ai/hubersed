"""Are the cont flags of gallery105 67, 71, 88 and 99 part of a population-wide offset?

QUESTION: What share of DESI quiescent galaxies at the redshift of 67, 71, 88 (z 0.47-0.48) and 99
(z 0.015) is past both cont thresholds, against all DESI and against line emitters; and where do the
four sit in log p among their matched controls?
HYPOTHESIS: none registered. Plan ~/.claude/plans/cont_flag_population_2026-11-14.md. A quiescent
flagged share at z 0.45-0.50 at least 5x the all-DESI share means the four are the tail of a
population offset; within 2x, with the four below the control p5 of log p, means specific to them.
INPUTS: results/2026-11-09_mocks_v2_outliers/flows/desi_outliers_flow_nsf_cont{10,15}latent_snr3.pt
(log p, thresholds), data/fastspec-iron-sv3-bright.fits (Z, DN4000, HALPHA_EW and IVAR, OII EWs),
results/prescreened_sample_v1.npz (keep), data/dataset_z_snr.npz (median S/N).
METHOD: population = prescreen keep with cont log p (the S/N > 3 latent set). Flag = log p at or
below the saved threshold in both cont10 and cont15 (the cont branch of the flag rule). Quiescent =
no lines as 2026-11-13_hda_parent_67, EW Halpha < 3 A (or no Halpha measurement) and EW [OII]
3726+3729 < 2.5 A; the rest are line emitters. Flagged share with binomial error per z bin (0.05
wide), split at the median S/N. Controls as hda_parent_67 (same Dn4000 window and
|z - z_target| <= 0.05, target removed); target log p percentile among the quiescent controls.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-14_cont_flag_population/analysis.py``.
RESULT: 2026-11-14, cont_flags.txt. 0.11% of all DESI are cont-flagged (quiescent 0.15%, line
emitters 0.09%), but the share rises steeply with z for both groups: z 0.45-0.50 quiescent 2.2 +- 1.1%,
emitters 2.6 +- 0.7%; z 0.50-0.55 11 +- 4% and 4.0 +- 1.4%; z 0.55-0.60 33 +- 10% and 8.7 +- 3.4%
(about 0.1% or less below z 0.35). By the plan rule (>= 5x) 67, 71 and 88 are the tail of a
population offset, and it is a redshift offset, not a quiescent one. Among quiescent controls 71 and
88 are typical (cont log p percentile 4-36; 17 +- 8% and 29 +- 12% of those controls flagged); 67 is
below the control p5 in cont10 (percentile 4.3). 99 (z 0.015) is below every control in both flows:
specific to it. Assumptions: FastSpecFit EWs and Dn4000; flag rule as the 11-09 run; S/N > 3 set;
cause of the z trend not tested (mock z or S/N coverage at z > 0.45 is one candidate).
FIGURES: none (cont_flags.txt).
"""

import numpy as np
import torch
from astropy.io import fits

from hubersed.paths import PATHS

RES, DATA = PATHS["RESULTS"], PATHS["DATA"]
OUT = RES / "2026-11-14_cont_flag_population"
FLOWS = RES / "2026-11-09_mocks_v2_outliers" / "flows"
TARGETS = {  # TARGETID, z, Dn4000 window, as 2026-11-13_hda_parent_67
    "67": (39633136522363591, 0.4781, (1.45, 1.60)),
    "71": (39633149579232417, 0.4666, (1.30, 1.45)),
    "88": (39633407818335390, 0.4818, (1.25, 1.40)),
    "99": (39633421919584782, 0.0151, (1.12, 1.27)),
}
DZ = 0.05
Z_EDGES = np.round(np.arange(0.0, 0.65, 0.05), 2)


def share(mask):
    """Return the flagged share in percent, its binomial error and the count of ``mask``."""
    n = mask.sum()
    if n == 0:
        return np.nan, np.nan, 0
    p = FLAG[mask].mean()
    return 100 * p, 100 * np.sqrt(p * (1 - p) / n), n


def fmt(s):
    """Return a share tuple as text."""
    return f"{s[0]:6.2f} +- {s[1]:5.2f}% (N {s[2]:6d})"


# population and flags
lp, thr = {}, {}
for tag in ("cont10", "cont15"):
    r = torch.load(FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt", weights_only=False)
    lp[tag] = dict(zip(np.asarray(r["desi_target_ids"]).tolist(), np.asarray(r["log_p_desi"], float),
                       strict=True))  # fmt: skip
    thr[tag] = float(r["threshold"])
keep = set(np.load(RES / "prescreened_sample_v1.npz")["keep"].astype(np.int64).tolist())
zs = np.load(DATA / "dataset_z_snr.npz")
snr_of = dict(zip(zs["tid"].tolist(), zs["snr"].tolist(), strict=True))
with fits.open(DATA / "fastspec-iron-sv3-bright.fits") as h:
    f, m = h["FASTSPEC"].data, h["METADATA"].data
    tid_all = np.asarray(m["TARGETID"], np.int64)
    first = {}
    for i, t in enumerate(tid_all.tolist()):
        first.setdefault(t, i)
    tids = np.array([t for t in first if t in keep and t in lp["cont10"] and t in lp["cont15"]])
    rows = np.array([first[t] for t in tids])
    z = np.asarray(m["Z"], float)[rows]
    dn = np.asarray(f["DN4000"], float)[rows]
    ewha = np.where(
        np.asarray(f["HALPHA_EW_IVAR"])[rows] > 0, np.asarray(f["HALPHA_EW"])[rows], np.nan
    )
    ewoii = (np.asarray(f["OII_3726_EW"]) + np.asarray(f["OII_3729_EW"]))[rows]
L10 = np.array([lp["cont10"][t] for t in tids])
L15 = np.array([lp["cont15"][t] for t in tids])
FLAG = (L10 <= thr["cont10"]) & (L15 <= thr["cont15"])
QUIET = (np.isnan(ewha) | (ewha < 3)) & (ewoii < 2.5)
snr = np.array([snr_of.get(int(t), np.nan) for t in tids])
HIGH = snr >= np.nanmedian(snr)
ALL = np.ones(tids.size, bool)

lines = [
    f"{tids.size} galaxies (prescreen keep with cont log p; {len(tid_all) - len(first)} duplicate "
    f"fastspec rows dropped); thresholds cont10 {thr['cont10']:.2f}, cont15 {thr['cont15']:.2f}; "
    f"median S/N {np.nanmedian(snr):.1f}",
    f"all DESI flagged {fmt(share(ALL))}; quiescent {fmt(share(QUIET))}; "
    f"line emitters {fmt(share(~QUIET))}",
    "",
    "flagged share per z bin: quiescent | line emitters | quiescent S/N >= median | quiescent S/N < median",
]
for lo, hi in zip(Z_EDGES[:-1], Z_EDGES[1:], strict=True):
    zb = (z >= lo) & (z < hi)
    lines.append(f"  z {lo:.2f}-{hi:.2f}: {fmt(share(zb & QUIET))} | {fmt(share(zb & ~QUIET))} | "
                 f"{fmt(share(zb & QUIET & HIGH))} | {fmt(share(zb & QUIET & ~HIGH))}")  # fmt: skip
lines.append("")
pos = {int(t): i for i, t in enumerate(tids)}
for nn, (t0, z0, (d0, d1)) in TARGETS.items():
    i = pos.get(t0)
    if i is None:
        lines.append(f"{nn} ({t0}) not in the population")
        continue
    ctrl = (np.abs(z - z0) <= DZ) & (dn >= d0) & (dn <= d1) & (tids != t0)
    q = ctrl & QUIET
    lines.append(f"{nn} ({t0}) z {z0}, Dn4000 {d0}-{d1}: quiet {bool(QUIET[i])}, flagged "
                 f"{bool(FLAG[i])}; controls {fmt(share(ctrl))}, quiescent controls {fmt(share(q))}")  # fmt: skip
    for tag, L in (("cont10", L10), ("cont15", L15)):
        lines.append(f"    {tag}: log p {L[i]:.2f}, percentile among quiescent controls "
                     f"{100 * np.mean(L[q] < L[i]):.1f}, control p5 {np.percentile(L[q], 5):.2f}")  # fmt: skip
OUT.mkdir(parents=True, exist_ok=True)
text = "\n".join(lines)
(OUT / "cont_flags.txt").write_text(text + "\n")
print(text)
