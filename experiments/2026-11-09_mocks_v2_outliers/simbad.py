"""SIMBAD step for the v2-mock outliers and the comparison with the 310.

QUESTION: After the SIMBAD screen, how many v2-mock candidates remain, and which of the 310 (by
branch, source class and fit classification) do they keep or drop?
HYPOTHESIS: none stated before the run.
METHOD: as the 310 run (claude-backup jobs/a3ef5882/tmp/recount.py and simbad_posthoc.py): flags of
the five flows intersected with the prescreened parent (results/prescreened_sample_v1.npz), full =
6 & 10 & 15, cont = cont10 & cont15; a candidate is dropped if its nearest SIMBAD object within
SEP_MAX has an AGN or star type (or a dC spectral type). Rules copied from scripts/simbad_screen.py
(commit 9781230). Answers cached in results/simbad_screen_v2.csv are reused; new targets are queried
and cached in results/2026-11-09_mocks_v2_outliers/simbad_new.csv. Check: the old flows must give
exactly the 310.
INPUTS: results/noised_cue_meanzero_wide_flow/ (old), results/2026-11-09_mocks_v2_outliers/flows/
(new), data/fastspec-iron-sv3-bright.fits (RA, DEC), results/map_fits_outliers310_v2/
classification_v2.csv.
SEED: none.
COMMAND: ``uv run python experiments/2026-11-09_mocks_v2_outliers/simbad.py`` (queries SIMBAD).
RESULT: old flows reproduce the 310 (345 candidates, 35 dropped). New: 205 candidates, 112 queried
(Harvard mirror), 181 after SIMBAD (full 92, cont 94, both 5). Of the 310: 76 kept, 234 dropped;
105 new. Kept by fit class: broad-line AGN 7/8 (all broad-line AGN classes 12/21), refit-clean
47/225, high-EW starburst (Ha tension) 6/43. Assumption: SIMBAD answers cached from the 2026-09
run are still valid; first row of each cone as in the old run.
FIGURES: none (simbad.txt, postsimbad_v2.npz).
"""

# %%
import csv

import astropy.units as u
import numpy as np
import pandas as pd
import torch
from astropy.coordinates import SkyCoord
from astropy.io import fits

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-11-09_mocks_v2_outliers"
OLD = RES / "noised_cue_meanzero_wide_flow"
NEW = OUT / "flows"
TAGS = ("6", "10", "15", "cont10", "cont15")
SEP_MAX = 2.0  # arcsec, as scripts/simbad_screen.py
AGN_TYPES = {"BLL", "QSO", "AGN", "Sy1", "Sy2", "SyG", "QSO_Candidate", "Bla", "LIN", "rG"}
STAR_TYPES = {"Star", "Pe*", "WD*", "HB*", "RGB*", "*", "PM*", "HV*"}
NEW_CACHE = OUT / "simbad_new.csv"
# The CDS server returned 404 on 2026-11-10; the Harvard mirror holds the same database.
SIMBAD_SERVER = "simbad.harvard.edu"

keep = set(np.load(RES / "prescreened_sample_v1.npz")["keep"].tolist())
lines = []


def candidates(folder):
    """Return the full and cont candidate sets of the flows in folder, after the prescreen."""
    fl = {}
    for t in TAGS:
        d = torch.load(folder / f"desi_outliers_flow_nsf_{t}latent_snr3.pt", weights_only=False)
        fl[t] = {int(x) for x in d["outlier_target_ids"]} & keep
    return fl["6"] & fl["10"] & fl["15"], fl["cont10"] & fl["cont15"]


# %% cached SIMBAD answers
cache = {}
with open(RES / "simbad_screen_v2.csv") as f:
    for r in csv.DictReader(f):
        cache[int(r["target_id"])] = r["simbad_agn"] == "True" or r["simbad_star"] == "True"
if NEW_CACHE.exists():
    with open(NEW_CACHE) as f:
        for r in csv.DictReader(f):
            cache[int(r["target_id"])] = r["flagged"] == "True"

# %% check: the old flows give the 310
c310 = pd.read_csv(RES / "map_fits_outliers310_v2" / "classification_v2.csv")
t310 = set(c310["target_id"].astype(int))
of, oc = candidates(OLD)
old_union = of | oc
assert all(t in cache for t in old_union), "old candidates missing from the cache"
old_keep = {t for t in old_union if not cache[t]}
assert old_keep == t310, (len(old_keep), len(old_keep ^ t310))
lines.append(f"old: candidates {len(old_union)}, after SIMBAD {len(old_keep)} = the 310")

# %% new candidates: query what is not cached
nf, nc = candidates(NEW)
new_union = nf | nc
todo = sorted(t for t in new_union if t not in cache)
lines.append(
    f"new: candidates {len(new_union)} (full {len(nf)}, cont {len(nc)}), to query {len(todo)}"
)
print(lines[-1], flush=True)
if todo:
    from astroquery.simbad import Simbad

    meta = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)["METADATA"].data
    row = {int(t): i for i, t in enumerate(meta["TARGETID"])}
    s = Simbad()
    s.server = SIMBAD_SERVER
    for fld in ("otype", "sp_type"):
        try:
            s.add_votable_fields(fld)
        except Exception:
            pass
    new_file = not NEW_CACHE.exists()
    with open(NEW_CACHE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["target_id", "otype", "sp_type", "sep_arcsec", "flagged"])
        for n, t in enumerate(todo, 1):
            i = row[t]
            here = SkyCoord(float(meta["RA"][i]), float(meta["DEC"][i]), unit="deg")
            otype, sptype, sep = "", "", np.nan
            r = s.query_region(here, radius=SEP_MAX * u.arcsec)
            if r is not None and len(r):
                o = SkyCoord(float(r[0]["ra"]), float(r[0]["dec"]), unit="deg")
                sep = here.separation(o).arcsec
                otype = str(r[0]["otype"]).strip()
                sptype = str(r[0].get("sp_type", "") or "").strip()
            flagged = bool(
                sep < SEP_MAX
                and (otype in AGN_TYPES or otype in STAR_TYPES or sptype.startswith("dC"))
            )
            w.writerow([t, otype, sptype, f"{sep:.3f}", flagged])
            f.flush()
            cache[t] = flagged
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}", flush=True)

# %% after SIMBAD, and the comparison with the 310
fk = {t for t in nf if not cache[t]}
ck = {t for t in nc if not cache[t]}
nk = fk | ck
lines.append(
    f"new after SIMBAD: {len(nk)} (full {len(fk)}, cont {len(ck)}, both {len(fk & ck)}); "
    f"dropped {len(new_union) - len(nk)}"
)
lines.append(
    f"vs the 310: kept {len(nk & t310)}, dropped {len(t310 - nk)}, new not in 310 {len(nk - t310)}"
)
c310["kept_v2"] = c310["target_id"].astype(int).isin(nk)
for col in ("branch", "source_class", "classification"):
    tab = c310.groupby(col)["kept_v2"].agg(["sum", "count"])
    lines.append(
        f"310 by {col}: "
        + ", ".join(f"{k} {int(v['sum'])}/{int(v['count'])}" for k, v in tab.iterrows())
    )
np.savez(
    OUT / "postsimbad_v2.npz",
    full=np.array(sorted(fk), np.int64),
    cont=np.array(sorted(ck), np.int64),
)

text = "\n".join(lines)
print(text)
(OUT / "simbad.txt").write_text(text + "\n")
