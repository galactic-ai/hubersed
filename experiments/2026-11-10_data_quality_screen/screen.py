"""Apply the data-quality screen to the old and new candidates and recount through SIMBAD.

Screen (Nikhil 2026-11-10, after analysis.py): drop a candidate with a Gaia DR3 star within 5 arcsec
brighter than G = 20 (parallax/error > 5 or proper motion > 3 mas/yr), or observed on fibre 2675
or 2676 in any SV3 bright tile (gallery92: fibre_2675 class, the petal-5 z-arm step on 2676).
SIMBAD flags as experiments/2026-11-09_mocks_v2_outliers/simbad.py (cache results/
simbad_screen_v2.csv and results/2026-11-09_mocks_v2_outliers/simbad_new.csv; every candidate is
in one of them). Flow flags are per galaxy, so screening after flagging equals screening first.
INPUTS: results/2026-11-10_data_quality_screen/{targets.npz, ztile.fits, gaia5.fits};
data/iron_tileqa (FIBER per tile and target); results/2026-11-01_gallery92 (labels);
results/map_fits_outliers310_v2/classification_v2.csv.
COMMAND: ``uv run python experiments/2026-11-10_data_quality_screen/screen.py``.
OUTPUT: results/2026-11-10_data_quality_screen/{screened.npz, screen.txt}.
RESULT: of the 2,457 targets, 50 have a Gaia star (G < 20) within 5 arcsec and 12 used fibre
2675/2676 (control 4/2000). Old: 310 after SIMBAD -> 279 after the screen (full 223, cont 67, both
11; 23 star, 8 fibre). New: 181 -> 165 (full 88, cont 81, both 4; 15 star, 1 fibre); 66 also in the
old screened list. gallery92 labels left: old bad_sky 10, bad_data 7; new bad_sky 5, bad_data 2 (all
stars and fibre_2675 gone). Assumption: the 5 arcsec star rule and the two fibres stand for the
catalogue-detectable artifacts; sky and exposure problems need repeats.
"""

# %%
import csv
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from astropy.table import Table

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-11-10_data_quality_screen"
FLOWS = {
    "old": RES / "noised_cue_meanzero_wide_flow",
    "new": RES / "2026-11-09_mocks_v2_outliers" / "flows",
}
TAGS = ("6", "10", "15", "cont10", "cont15")
G_MAX = 20.0
BAD_FIBERS = {2675, 2676}
lines = []

# %% screen flags
gj = Table.read(OUT / "gaia5.fits")
star = set()
for r in gj:
    g = float(r["phot_g_mean_mag"]) if np.isfinite(r["phot_g_mean_mag"]) else 99.0
    plx, eplx = float(r["parallax"]), float(r["parallax_error"])
    pm = np.hypot(float(r["pmra"]), float(r["pmdec"]))
    stellar = (np.isfinite(plx) and np.isfinite(eplx) and eplx > 0 and plx / eplx > 5) or (
        np.isfinite(pm) and pm > 3
    )
    if stellar and g < G_MAX:
        star.add(int(r["tid"]))

zt = Table.read(OUT / "ztile.fits")
fiber = {}
for tile in np.unique(np.asarray(zt["tileid"], int)):
    f = sorted(glob.glob(str(PATHS["DATA"] / "iron_tileqa" / str(tile) / "tile-qa-*.fits")))
    d = fits.open(f[-1])["FIBERQA"].data
    for t, fb in zip(np.asarray(d["TARGETID"], np.int64), np.asarray(d["FIBER"], int), strict=True):
        fiber.setdefault(int(t), set()).add(int(fb))
badfib = {t for t, fbs in fiber.items() if fbs & BAD_FIBERS}
screened_out = star | badfib
tg = np.load(OUT / "targets.npz")
ours = set(np.concatenate([tg["old"], tg["new"], tg["control"]]).tolist())
lines.append(
    f"screen, of the {len(ours)} targets: Gaia star G < {G_MAX} within 5'': {len(star & ours)}; "
    f"fibre 2675/2676: {len(badfib & ours)} (control {len(badfib & set(tg['control'].tolist()))})"
)

# %% SIMBAD flags (cached)
simbad = {}
with open(RES / "simbad_screen_v2.csv") as f:
    for r in csv.DictReader(f):
        simbad[int(r["target_id"])] = r["simbad_agn"] == "True" or r["simbad_star"] == "True"
with open(RES / "2026-11-09_mocks_v2_outliers" / "simbad_new.csv") as f:
    for r in csv.DictReader(f):
        simbad[int(r["target_id"])] = r["flagged"] == "True"

# %% recount old and new
keep = set(np.load(RES / "prescreened_sample_v1.npz")["keep"].tolist())
t310 = set(
    pd.read_csv(RES / "map_fits_outliers310_v2" / "classification_v2.csv")["target_id"].astype(int)
)
out = {}
for name, folder in FLOWS.items():
    fl = {}
    for t in TAGS:
        d = torch.load(folder / f"desi_outliers_flow_nsf_{t}latent_snr3.pt", weights_only=False)
        fl[t] = {int(x) for x in d["outlier_target_ids"]} & keep
    full, cont = fl["6"] & fl["10"] & fl["15"], fl["cont10"] & fl["cont15"]
    assert all(t in simbad for t in full | cont), name
    sf = {t for t in full if not simbad[t]}
    sc = {t for t in cont if not simbad[t]}
    ff, fc = sf - screened_out, sc - screened_out
    out[name] = (ff, fc)
    lines.append(
        f"{name}: after SIMBAD {len(sf | sc)} (full {len(sf)}, cont {len(sc)}); after the screen "
        f"{len(ff | fc)} (full {len(ff)}, cont {len(fc)}, both {len(ff & fc)}); removed "
        f"{len((sf | sc) & screened_out)} (star {len((sf | sc) & star)}, fibre {len((sf | sc) & badfib)})"
    )
lines.append(
    f"new screened vs old screened: both {len((out['new'][0] | out['new'][1]) & (out['old'][0] | out['old'][1]))}; "
    f"new in the 310 {len((out['new'][0] | out['new'][1]) & t310)}"
)

# %% gallery92 classes left in each screened list
lab = {}
for d in glob.glob(str(RES / "2026-11-01_gallery92") + "/*/"):
    c = os.path.basename(d.rstrip("/"))
    for f in glob.glob(d + "*.png"):
        m = re.search(r"(\d{15,})", os.path.basename(f))
        if m and c != "cutouts":
            lab[int(m[1])] = c
for name, (ff, fc) in out.items():
    left = pd.Series([lab[t] for t in ff | fc if t in lab]).value_counts().to_dict()
    lines.append(f"{name} screened, gallery92 labels left: {left}")

np.savez(
    OUT / "screened.npz",
    old_full=np.array(sorted(out["old"][0]), np.int64),
    old_cont=np.array(sorted(out["old"][1]), np.int64),
    new_full=np.array(sorted(out["new"][0]), np.int64),
    new_cont=np.array(sorted(out["new"][1]), np.int64),
)
text = "\n".join(lines)
print(text)
(OUT / "screen.txt").write_text(text + "\n")
