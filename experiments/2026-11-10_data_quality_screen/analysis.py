"""Which data-quality checks catch the gallery92 artifacts, and how often each fires per group.

QUESTION: Do fibre QA bits, exposure conditions and Gaia neighbours within 5 arcsec separate the
hand-labelled gallery92 artifacts (bad_sky, bad_data, star, fibre_2675) from the physical classes
(AGN, emission_line, quiescent, composite)? How often does each check fire for the old and new
candidates and for a random control?
HYPOTHESIS: none stated before the run.
INPUTS: results/2026-11-10_data_quality_screen/{targets.npz, ztile.fits, exposure.fits,
gaia5.fits} (fetch.py); data/iron_tileqa/<tile>/tile-qa-*.fits (QAFIBERSTATUS per tile and
target); results/2026-11-01_gallery92/<class>/ (hand labels, the PNG names hold the TARGETID).
Fibre bits from desispec maskbits.py (main branch, read 2026-11-10); bits 0, 1, 3 are information.
SEED: none.
COMMAND: ``uv run python experiments/2026-11-10_data_quality_screen/analysis.py``.
RESULT: Gaia star within 5 arcsec (parallax/err > 5 or pm > 3 mas/yr) separates: G < 20 catches
15/38 gallery artifacts (all 10 stars, 5/13 bad_data) and loses 2/53 physical (1 AGN, 1 emission
line); G < 18 catches 12/38 and loses 0/53; control 1% / 0.4%. Fibre QA bits (coadd fiberstatus,
tile-QA QAFIBERSTATUS) fire at similar rates in every group (LOWEFFTIME 17-26%). Worst-exposure
conditions barely separate (sky g < 20: artifacts 42%, physical 34%, control 19%) and catch no
bad_sky case. Assumption: gallery92 labels (91 continuum outliers) stand for artifacts in both
branches.
FIGURES: none (analysis.txt).
"""

# %%
import glob
import os
import re

import numpy as np
from astropy.io import fits
from astropy.table import Table

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-11-10_data_quality_screen"
BITS = {
    2: "BROKENFIBER",
    8: "MISSINGPOSITION",
    9: "BADPOSITION",
    10: "POORPOSITION",
    11: "BRIGHTNEIGHBOR",
    12: "LOWTRANSMISSION",
    13: "NEARCHARGETRAP",
    14: "VARIABLETHRU",
    15: "LOWEFFTIME",
    16: "BADFIBER",
    17: "BADTRACE",
    18: "BADFLAT",
    19: "BADARC",
    20: "VARIABLE",
    21: "MANYREJECTED",
    22: "BADAMPB",
    23: "BADAMPR",
    24: "BADAMPZ",
    25: "BADPETALPOS",
    26: "BADPETALSKY",
    27: "BADPETALSTDSTAR",
    28: "BADPETALFLUXCAL",
    29: "BADPETALSNR",
    30: "BADREADNOISE",
}
ARTIFACT = ("bad_sky", "bad_data", "star", "fibre_2675")
PHYSICAL = ("AGN", "emission_line", "quiescent", "composite")
lines = []

# %% groups
tg = np.load(OUT / "targets.npz")
lab = {}
for d in glob.glob(str(RES / "2026-11-01_gallery92") + "/*/"):
    c = os.path.basename(d.rstrip("/"))
    for f in glob.glob(d + "*.png"):
        m = re.search(r"(\d{15,})", os.path.basename(f))
        if m and c != "cutouts":
            lab[int(m[1])] = c
groups = {
    "gallery artifacts": {t for t, c in lab.items() if c in ARTIFACT},
    "gallery physical": {t for t, c in lab.items() if c in PHYSICAL},
    "old candidates": set(tg["old"].tolist()),
    "new candidates": set(tg["new"].tolist()),
    "control": set(tg["control"].tolist()),
}
for k, v in groups.items():
    lines.append(f"group {k}: {len(v)}")

# %% fibre QA bits per target, over all its SV3 bright tiles
zt = Table.read(OUT / "ztile.fits")
qa = {}
for tile in np.unique(np.asarray(zt["tileid"], int)):
    f = sorted(glob.glob(str(PATHS["DATA"] / "iron_tileqa" / str(tile) / "tile-qa-*.fits")))
    if not f:
        continue
    d = fits.open(f[-1])["FIBERQA"].data
    for t, q in zip(
        np.asarray(d["TARGETID"], np.int64), np.asarray(d["QAFIBERSTATUS"], np.int64), strict=True
    ):
        qa[(tile, int(t))] = int(q)
bits = {}
for r in zt:
    t, tile = int(r["targetid"]), int(r["tileid"])
    b = int(r["coadd_fiberstatus"]) | qa.get((tile, t), 0)
    bits[t] = bits.get(t, 0) | b
lines.append(
    f"tile-QA rows found for {sum((int(r['tileid']), int(r['targetid'])) in qa for r in zt)} of {len(zt)} ztile rows"
)

# %% exposure conditions per target: worst exposure over its tiles
ex = Table.read(OUT / "exposure.fits")
by_tile = {}
for r in ex:
    by_tile.setdefault(int(r["tileid"]), []).append(r)
cond = {}
for r in zt:
    t, rows = int(r["targetid"]), by_tile.get(int(r["tileid"]), [])
    for e in rows:
        c = cond.setdefault(
            t, {"sky_g": np.inf, "transp": np.inf, "efftime": np.inf, "seeing": -np.inf}
        )
        c["sky_g"] = min(c["sky_g"], float(e["sky_mag_g_spec"]))
        c["transp"] = min(c["transp"], float(e["transparency_gfa"]))
        c["efftime"] = min(c["efftime"], float(e["efftime_spec"]))
        c["seeing"] = max(c["seeing"], float(e["seeing_etc"]))

# %% Gaia neighbours within 5 arcsec: any source, and stars (parallax / error > 5 or proper
# motion > 3 mas/yr, the motion rule of the #10 screen); the brightest G of each
gj = Table.read(OUT / "gaia5.fits")
gaia_any, gaia_star = {}, {}
for r in gj:
    t = int(r["tid"])
    g = float(r["phot_g_mean_mag"]) if np.isfinite(r["phot_g_mean_mag"]) else 99.0
    gaia_any[t] = min(gaia_any.get(t, 99.0), g)
    plx, eplx = float(r["parallax"]), float(r["parallax_error"])
    pm = np.hypot(float(r["pmra"]), float(r["pmdec"]))
    star = (np.isfinite(plx) and np.isfinite(eplx) and eplx > 0 and plx / eplx > 5) or (
        np.isfinite(pm) and pm > 3
    )
    if star:
        gaia_star[t] = min(gaia_star.get(t, 99.0), g)


# %% rates per group
def rate(group, test):
    """Return 'k/n (pct)' for targets in group where test(t) is true."""
    ids = [t for t in group if t in bits]
    k = sum(test(t) for t in ids)
    return f"{k}/{len(ids)} ({100 * k / max(len(ids), 1):.0f}%)"


lines.append("\nfibre bits (any SV3 bright tile, coadd fiberstatus or tile-QA)")
for b, name in BITS.items():
    row = "  ".join(
        f"{k[:8]:>8} {rate(v, lambda t, b=b: bits[t] >> b & 1)}" for k, v in groups.items()
    )
    if any(bits[t] >> b & 1 for t in bits):
        lines.append(f"{name:>16}: {row}")

lines.append(
    "\nexposure conditions (worst exposure of the target's tiles): group quantiles p16/p50/p84"
)
for key in ("sky_g", "transp", "efftime", "seeing"):
    parts = []
    for k, v in groups.items():
        x = np.array([cond[t][key] for t in v if t in cond])
        parts.append(f"{k[:8]:>8} " + "/".join(f"{q:.2f}" for q in np.percentile(x, [16, 50, 84])))
    lines.append(f"{key:>8}: " + "  ".join(parts))

for label, src in (("any Gaia source", gaia_any), ("Gaia star", gaia_star)):
    lines.append(f"\n{label} within 5 arcsec, by brightest G")
    for gcut in (16, 18, 20, 21):
        lines.append(
            f"   G < {gcut}: "
            + "  ".join(
                f"{k[:8]:>8} {rate(v, lambda t, g=gcut, s=src: s.get(t, 99) < g)}"
                for k, v in groups.items()
            )
        )

lines.append("\nexposure-condition cuts (worst exposure)")
for name, test in (
    ("sky_g < 20.0", lambda t: cond[t]["sky_g"] < 20.0),
    ("sky_g < 20.5", lambda t: cond[t]["sky_g"] < 20.5),
    ("transp < 0.7", lambda t: cond[t]["transp"] < 0.7),
    ("efftime < 60", lambda t: cond[t]["efftime"] < 60),
):
    lines.append(
        f"{name:>14}: " + "  ".join(f"{k[:8]:>8} {rate(v, test)}" for k, v in groups.items())
    )

lines.append("\ngallery classes caught by a Gaia star within 5 arcsec, G < 18 / G < 20")
for c in (*ARTIFACT, *PHYSICAL):
    ids = [t for t, v in lab.items() if v == c]
    a = sum(gaia_star.get(t, 99) < 18 for t in ids)
    b = sum(gaia_star.get(t, 99) < 20 for t in ids)
    lines.append(f"{c:>14}: {a}/{len(ids)} / {b}/{len(ids)}")

text = "\n".join(lines)
print(text)
(OUT / "analysis.txt").write_text(text + "\n")
