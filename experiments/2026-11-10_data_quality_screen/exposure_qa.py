"""Do per-exposure QA columns separate the gallery92 artifacts from the physical outliers?

QUESTION: For each target, take the worst value over its SV3 bright exposures of the per-exposure
fibre QA (QAFIBERSTATUS, DELTA_X/Y, fibre EFFTIME_SPEC) and of its petal's QA (sky chi2 and sky
throughput rms per arm, STARRMS, NSTDSTAR, throughput fractions, WORSTREADNOISE). Which columns flag
the gallery92 artifacts (bad_sky above all) and not the physical classes?
HYPOTHESIS: none stated before the run.
INPUTS: results/2026-11-10_data_quality_screen/{targets.npz, exposure.fits} (fetch.py); the DR1
exposure-qa-<EXPID>.fits files (FIBERQA per fibre, PETALQA per petal), from the 2026-11-02 cache in
results/2026-11-02_exposure_checks/exposure_qa/ or downloaded from the public DR1 iron exposures
folder into results/2026-11-10_data_quality_screen/exposure_qa/. Targets are matched by TARGETID in
FIBERQA; a target's petal row is PETALQA at its PETAL_LOC in that exposure. Labels as analysis.py.
SEED: none.
COMMAND: ``uv run python experiments/2026-11-10_data_quality_screen/exposure_qa.py``.
RESULT: all 288 exposures, all 2,457 targets matched. No column separates: share above the control
p95 of the worst petal sky chi2 is bad_sky 2/10, artifacts 24%, physical 11%, control 5%; sky
throughput rms 20/13/9/5%; STARRMS 10/13/2/5%. Per-exposure QAFIBERSTATUS bits fire at similar
rates in every group (bits 9, 10 most often). bad_sky targets have a shallower worst exposure (fibre
EFFTIME median 108 s vs 168 s physical). DESI's QA does not flag the hand-labelled sky artifacts.
Assumption: the worst exposure per target summarises the per-exposure QA.
FIGURES: none (exposure_qa.txt).
"""

# %%
import glob
import os
import re
import shutil
import urllib.request

import numpy as np
from astropy.io import fits
from astropy.table import Table

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-11-10_data_quality_screen"
QA = OUT / "exposure_qa"
CACHE = RES / "2026-11-02_exposure_checks" / "exposure_qa"
URL = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/exposures/{n}/{e:08d}/exposure-qa-{e:08d}.fits"
ARTIFACT = ("bad_sky", "bad_data", "star", "fibre_2675")
PHYSICAL = ("AGN", "emission_line", "quiescent", "composite")
QA_BITS = (
    2,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
)
lines = []

# %% exposure-qa files
QA.mkdir(exist_ok=True)
ex = Table.read(OUT / "exposure.fits")
for r in ex:
    e, n = int(r["expid"]), int(r["night"])
    path = QA / f"exposure-qa-{e:08d}.fits"
    if path.exists():
        continue
    if (CACHE / path.name).exists():
        shutil.copy(CACHE / path.name, path)
    else:
        urllib.request.urlretrieve(URL.format(n=n, e=e), path)
lines.append(f"exposure-qa files: {len(ex)} exposures")

# %% per target: worst value over its exposures
tg = np.load(OUT / "targets.npz")
ours = set(np.concatenate([tg["old"], tg["new"], tg["control"]]).tolist())
worst = {}
for r in ex:
    h = fits.open(QA / f"exposure-qa-{int(r['expid']):08d}.fits")
    fq, pq = h["FIBERQA"].data, h["PETALQA"].data
    prow = {int(p): i for i, p in enumerate(pq["PETAL_LOC"])}
    for j in np.where(np.isin(np.asarray(fq["TARGETID"], np.int64), list(ours)))[0]:
        t = int(fq["TARGETID"][j])
        p = pq[prow[int(fq["PETAL_LOC"][j])]]
        v = {
            "qabits": int(fq["QAFIBERSTATUS"][j]),
            "delta": float(np.hypot(fq["DELTA_X"][j], fq["DELTA_Y"][j])),
            "fib_efftime": -float(fq["EFFTIME_SPEC"][j]),
            "skychi2": max(
                float(p["BSKYCHI2PDF"]), float(p["RSKYCHI2PDF"]), float(p["ZSKYCHI2PDF"])
            ),
            "skythrurms": max(
                float(p["BSKYTHRURMS"]), float(p["RSKYTHRURMS"]), float(p["ZSKYTHRURMS"])
            ),
            "starrms": float(p["STARRMS"]),
            "nstdstar": -float(p["NSTDSTAR"]),
            "thrufrac": -min(float(p["BTHRUFRAC"]), float(p["RTHRUFRAC"]), float(p["ZTHRUFRAC"])),
            "readnoise": float(p["WORSTREADNOISE"]),
        }
        w = worst.setdefault(t, {k: (0 if k == "qabits" else -np.inf) for k in v})
        for k, x in v.items():
            w[k] = w[k] | x if k == "qabits" else max(w[k], x)
lines.append(f"targets with exposure-qa rows: {len(worst)} of {len(ours)}")

# %% groups and labels
lab = {}
for d in glob.glob(str(RES / "2026-11-01_gallery92") + "/*/"):
    c = os.path.basename(d.rstrip("/"))
    for f in glob.glob(d + "*.png"):
        m = re.search(r"(\d{15,})", os.path.basename(f))
        if m and c != "cutouts":
            lab[int(m[1])] = c
groups = {
    "bad_sky": {t for t, c in lab.items() if c == "bad_sky"},
    "artifacts": {t for t, c in lab.items() if c in ARTIFACT},
    "physical": {t for t, c in lab.items() if c in PHYSICAL},
    "old cand": set(tg["old"].tolist()),
    "new cand": set(tg["new"].tolist()),
    "control": set(tg["control"].tolist()),
}


def vals(group, k):
    """Return the worst values of column k for the targets of a group."""
    return np.array([worst[t][k] for t in group if t in worst], float)


# %% per-exposure fibre bits
lines.append("\nper-exposure QAFIBERSTATUS bits, any exposure")
for b in QA_BITS:
    if not any(worst[t]["qabits"] >> b & 1 for t in worst):
        continue
    parts = []
    for k, v in groups.items():
        ids = [t for t in v if t in worst]
        hit = sum(worst[t]["qabits"] >> b & 1 for t in ids)
        parts.append(f"{k:>9} {hit}/{len(ids)}")
    lines.append(f"bit {b:2d}: " + "  ".join(parts))

# %% continuous columns: group quantiles, and the share above the control p95
lines.append(
    "\nworst value per target (signs flipped so that larger is worse): p50 / share above control p95"
)
for k in (
    "delta",
    "fib_efftime",
    "skychi2",
    "skythrurms",
    "starrms",
    "nstdstar",
    "thrufrac",
    "readnoise",
):
    c95 = np.percentile(vals(groups["control"], k), 95)
    parts = []
    for g, v in groups.items():
        x = vals(v, k)
        parts.append(f"{g:>9} {np.median(x):8.3f}/{np.mean(x > c95):.2f}")
    lines.append(f"{k:>11}: " + "  ".join(parts))

text = "\n".join(lines)
print(text)
(OUT / "exposure_qa.txt").write_text(text + "\n")
