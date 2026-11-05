"""Mgb, Fe5270 and Fe5335 of the SV3 dark spectrum of 39633433328091196 against the C3K ER dark MAPs (#17).

QUESTION: do Mgb, Fe5270 and Fe5335, which dust barely moves, favour the C3K ER dark MAP with dust
(logzsol 0.15) or the one without (logzsol 0.48)?
HYPOTHESIS: none registered.
INPUTS: <TID>_map.npz of 2026-10-30_continuum_39633433328091196_dark_map (with dust) and of
er_nodust/ in 2026-10-30_nodust_miles_39633433328091196_dark (without). Bands from
$SPS_HOME/data/allindices.dat, air to vacuum with _airtovac. Dust check with prospect fake_fsps.attenuate.
SEED: 0, for 500 noise draws of the data.
COMMAND: from the repository root on the Mac, ``SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync
python experiments/2026-10-30_lick_dark_39633433328091196/analysis.py``.
RESULT: 2026-10-30, lick_dark.txt. Inconclusive. Data sigma is 0.43-0.83 A and both MAPs are within
1.1 sigma on all three (Mgb +0.61 dust, -0.14 no dust; Fe5335 -0.80, -1.09). The two models differ
by at most 0.62 A (Mgb). Dust alone moves each index by at most 0.001 A. Assumptions: fitted pixels
only (Mgb red band 7 of 27, Fe5270 feature 35 of 70), one dark exposure ivar, MAPs also differ in SFH and afe.
FIGURES: none; table in results/2026-10-30_lick_dark_39633433328091196/lick_dark.txt.
"""

# %%
import os
from pathlib import Path

import astropy.units as u
import numpy as np
from prospect.sources.fake_fsps import attenuate

from hubersed.conversion import to_flambda
from hubersed.paths import PATHS
from hubersed.sps.utils import _airtovac

TID = 39633433328091196
RES = PATHS["RESULTS"]
OUT = RES / "2026-10-30_lick_dark_39633433328091196"
NPZ = f"{TID}_map.npz"
FITS = {
    "dust": RES / "2026-10-30_continuum_39633433328091196_dark_map" / NPZ,
    "no dust": RES / "2026-10-30_nodust_miles_39633433328091196_dark" / "er_nodust" / NPZ,
}
NAMES = ["Mgb", "Fe5270", "Fe5335"]
N_NOISE = 500


def flam(d, x):
    """Return maggies on the fit's wavelength grid in 1e-17 erg/s/cm^2/A."""
    return to_flambda(np.asarray(d["wave"], float) * u.AA, np.asarray(x, float) * u.mgy).value


def lick_bands(names):
    """Return {name: (feature, blue, red)} vacuum rest-frame bands from allindices.dat."""
    table = np.genfromtxt(
        Path(os.environ["SPS_HOME"]) / "data" / "allindices.dat",
        dtype=None,
        encoding=None,
        comments="#",
    )
    out = {}
    for r in table:
        if str(r[-1]) in names:
            edges = [_airtovac(float(r[i])) for i in range(6)]
            out[str(r[-1])] = ((edges[0], edges[1]), (edges[2], edges[3]), (edges[4], edges[5]))
    return out


def equivalent_width(rest, f, bands, good):
    """Return the equivalent width in A of ``f`` over the fitted pixels of the feature band.

    As in 2026-10-29_continuum_39633433328091196_map. The continuum is the line through the mean
    flux of the fitted pixels of the blue and red bands at their mean wavelengths.
    """
    feat, blue, red = bands
    pts = []
    for lo, hi in (blue, red):
        s = good & (rest >= lo) & (rest <= hi)
        pts.append((rest[s].mean(), f[..., s].mean(axis=-1)))
    (xb, yb), (xr, yr) = pts
    s = good & (rest >= feat[0]) & (rest <= feat[1])
    cont = yb[..., None] + (yr - yb)[..., None] * (rest[s] - xb) / (xr - xb)
    return np.sum((1 - f[..., s] / cont) * np.gradient(rest)[s], axis=-1)


# %% indices of the data and of both MAP models, on the same pixels
d = {k: np.load(p) for k, p in FITS.items()}
a, b = d["dust"], d["no dust"]
assert np.array_equal(a["good"], b["good"]), "different pixels"
assert np.allclose(a["flux"][a["good"]], b["flux"][b["good"]]), "different data"
good = a["good"]
rest = a["wave"] / (1 + float(a["z"]))
par = {k: dict(zip([str(x) for x in v["labels"]], v["best"], strict=True)) for k, v in d.items()}
data, unc = flam(a, a["flux"]), flam(a, a["unc"])
models = {k: flam(v, v["sp_best"]) for k, v in d.items()}
# the no-dust model reddened with the diffuse curve of the dust MAP (dust1 only hits young stars)
p = par["dust"]
models["no dust, reddened"] = attenuate(
    models["no dust"], rest, dust_type=4, dust_index=p["dust_index"], dust2=p["dust2"]
)[0]

bands = lick_bands(NAMES)
rng = np.random.default_rng(0)
noise = rng.normal(size=(N_NOISE, data.size)) * np.where(good, unc, 0.0)
lines = [
    f"SV3 dark, C3K ER MAPs. dust: logzsol {par['dust']['logzsol']:.3f}, afe "
    f"{par['dust']['afe']:.3f}, dust2 {p['dust2']:.3f}, dust_index {p['dust_index']:.3f}. "
    f"no dust: logzsol {par['no dust']['logzsol']:.3f}, afe {par['no dust']['afe']:.3f}.",
    f"Equivalent widths in A on the fitted pixels; vacuum bands; data sigma from {N_NOISE} noise "
    f"draws, seed 0. Fitted/all pixels of feature, blue, red band in brackets.",
    "'reddened' is the no-dust model times the dust MAP diffuse curve; its change shows what dust "
    "alone does to the index.",
]
for name in NAMES:
    frac = []
    for lo, hi in bands[name]:
        s = (rest >= lo) & (rest <= hi)
        frac.append(f"{(good & s).sum()}/{s.sum()}")
    w0 = equivalent_width(rest, data, bands[name], good)
    sig = equivalent_width(rest, data + noise, bands[name], good).std()
    lines.append(f"{name}: data {w0:.3f} +- {sig:.3f} ({', '.join(frac)})")
    for k, m in models.items():
        w = equivalent_width(rest, m, bands[name], good)
        lines.append(
            f"  {k:>18}: model {w:.3f}, data - model {w0 - w:+.3f} ({(w0 - w) / sig:+.2f} sigma)"
        )
text = "\n".join(lines) + "\n"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "lick_dark.txt").write_text(text)
print(text)
