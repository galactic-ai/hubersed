"""Rest-frame U-V and V-J of 39633433328091196 from the C3K and MILES best fits, each library at both points.

QUESTION: does V-J of 39633433328091196 depend on the stellar library, at fixed parameters and at each fit's own point?
HYPOTHESIS: none registered.
INPUTS: colours_c3k_hr.npz, colours_miles.npz, check_<lib>.txt and check_pred_c3k_hr.npy of colours.py;
FastSpecFit ABSMAG00_U and ABSMAG00_V from data/fastspec-iron-sv3-bright.fits; CIGALE VAC values as quoted by the caller.
SEED: none.
COMMAND: from the repository root on the Mac, after colours.py,
``uv run --no-sync python experiments/2026-10-30_uvj_models_39633433328091196/analysis.py``.
RESULT: 2026-10-30, uvj_models.txt. At a fixed point the libraries differ by <= 0.07 in U-V and 0.04 in V-J;
the point moves V-J by 0.19 (dust 0.25 of 1.34 at the C3K point). Both fits give U-V 1.70-1.92, not 2.15.
Assumptions: C3K_HR with afe 0 for C3K_ER (sp_best off by 4-7%), FSPS MILES is not MILES in J, no nebular.
FIGURES: none; table in results/2026-10-30_uvj_models_39633433328091196/uvj_models.txt.
"""

# %%
import numpy as np
from astropy.io import fits

from hubersed.paths import get_paths

TID = 39633433328091196
PATHS = get_paths()
OUT = PATHS["RESULTS"] / "2026-10-30_uvj_models_39633433328091196"
ER_MAP = PATHS["RESULTS"] / "2026-10-29_continuum_39633433328091196_map" / f"{TID}_map.npz"
CIGALE = dict(uv=(2.151, 0.046), vj=(1.279, 0.037))  # DR1 CIGALE VAC v1.2, quoted, not read here
REST_BINS = ((3320, 4000), (4000, 5000), (5000, 6000), (6000, 7000), (7000, 9000))  # rest A

# %%
col = {lib: np.load(OUT / f"colours_{lib}.npz") for lib in ("c3k_hr", "miles")}
with fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True) as h:
    d = h["FASTSPEC"].data
    row = d[np.flatnonzero(d["TARGETID"] == TID)[0]]
    fsf_uv = float(row["ABSMAG00_U"] - row["ABSMAG00_V"])

lines = [f"Rest-frame AB colours of {TID} (FSPS U, V, 2MASS_J; photon counting)", ""]
lines.append(f"{'library':>8} {'point':>6} {'U-V':>7} {'V-J':>7} {'U-V d0':>7} {'V-J d0':>7}")
for lib, name in (("c3k_hr", "C3K_HR"), ("miles", "MILES")):
    for p in ("c3k", "miles"):
        c = col[lib]
        vals = [c[f"uv_{p}_dust"], c[f"vj_{p}_dust"], c[f"uv_{p}_nodust"], c[f"vj_{p}_nodust"]]
        lines.append(f"{name:>8} {p:>6} " + " ".join(f"{float(v):7.3f}" for v in vals))
lines += ["d0: dust2 = 0 (so dust1 = 0). C3K_HR stands in for C3K_ER, with afe 0 at both points."]
lines += [
    f"CIGALE VAC (quoted): U-V {CIGALE['uv'][0]} +- {CIGALE['uv'][1]}, "
    f"V-J {CIGALE['vj'][0]} +- {CIGALE['vj'][1]}",
    f"FastSpecFit ABSMAG00_U - ABSMAG00_V: {fsf_uv:.3f} (no J band)",
    "",
]

# %%
# FSPS get_mags cross-check of the filter integration
dmax = max(
    abs(float(c[f"{k}_{p}_{dd}"]) - float(c[f"fsps_{k}_{p}_{dd}"]))
    for c in col.values()
    for k in ("uv", "vj")
    for p in ("c3k", "miles")
    for dd in ("dust", "nodust")
)
lines.append(f"max |colour - fsps get_mags colour| over all cases: {dmax:.1e} mag")

# %%
# C3K_HR at the C3K point (afe 0) against the C3K_ER sp_best, binned in rest wavelength
er = np.load(ER_MAP)
pred = np.load(OUT / "check_pred_c3k_hr.npy")
g = er["good"]
rest = er["wave"] / (1 + float(er["z"]))
ratio = pred / er["sp_best"]
lines += ["", "C3K_HR (afe 0) / C3K_ER sp_best on fitted pixels, median per rest bin:"]
for lo, hi in REST_BINS:
    s = g & (rest >= lo) & (rest < hi)
    lines.append(f"  {lo}-{hi} A: {np.median(ratio[s]):.4f} ({s.sum()} px)")
for lib in ("c3k_hr", "miles"):
    lines += [""] + (OUT / f"check_{lib}.txt").read_text().strip().splitlines()

text = "\n".join(lines) + "\n"
(OUT / "uvj_models.txt").write_text(text)
print(text)
