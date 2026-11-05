"""Per-exposure spectrum-minus-photometry g-r of the blue outliers, against each exposure's sky (#17).

QUESTION: is the blue offset of the exp_targets.txt outliers in one exposure (sky-like) or in all?
HYPOTHESIS: none registered.
INPUTS: results/2026-10-31_logp_vs_colour/exposures/<TARGETID>.npz (exp_extract.py on stampede3),
sv3_bright_exposures.csv of the QA experiment, FIBERTOTFLUX of the FastSpecFit VAC, BASS g and r.
Arms are joined at 5750 and 7580 A on their own pixels; masked pixels are interpolated over.
The ivar-weighted mean of the exposures is a check against the VAC dgr.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_logp_vs_colour/exp_analysis.py``.
RESULT: 2026-10-31. Exposure mean matches the VAC dgr within 0.04. Exposure dgr varies by up to
1.3 mag within a target. In 4 of 18 the bluest exposure has the brightest sky; within-target
Spearman with sky g is +0.13 (p 0.36) and with efftime -0.02 (p 0.88). Three galaxies on tile 95
petal 5 share bluer 84639 and 84934 and a redder 85078. Within-target Spearman with seeing
-0.50 (p 9e-5).
FIGURES: results/2026-10-31_logp_vs_colour/exp_table.txt and exp_sky.png.
"""

# %%
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.stats import spearmanr

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

HERE = Path(__file__).resolve().parent
OUT = PATHS["RESULTS"] / "2026-10-31_logp_vs_colour"
QA = PATHS["RESULTS"] / "2026-10-31_spectrum_qa_cont_outliers"
FILT = PATHS["ROOT"] / "experiments" / "2026-10-30_photcolour_dark_39633433328091196" / "filters"
JOIN = (5750.0, 7580.0)
C_AA = 2.998e18  # speed of light in A/s

curves = {}
for b, name in (("g", "BASS-g"), ("r", "BASS-r")):
    tab = Table.read(FILT / f"{name}.ecsv")
    curves[b] = (np.asarray(tab["wavelength"], float), np.asarray(tab["response"], float))
sky, eff, see = {}, {}, {}
with open(QA / "sv3_bright_exposures.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        sky[int(r["expid"])] = float(r["sky_mag_g_spec"])
        eff[int(r["expid"])] = float(r["efftime_spec"])
        see[int(r["expid"])] = float(r["seeing_gfa"])
h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)
M, S = h["METADATA"].data, h["FASTSPEC"].data
vrow = {int(t): i for i, t in enumerate(M["TARGETID"])}


def stitch(d, k):
    """Return wave, flux and ivar of exposure row k with the arms joined at JOIN."""
    parts = []
    for arm, lo, hi in (("B", 0, JOIN[0]), ("R", JOIN[0], JOIN[1]), ("Z", JOIN[1], 1e5)):
        wv = d[f"{arm}_wave"]
        sel = (wv >= lo) & (wv < hi)
        iv = np.where(d[f"{arm}_mask"][k] == 0, d[f"{arm}_ivar"][k], 0.0)
        parts.append((wv[sel], d[f"{arm}_flux"][k][sel], iv[sel]))
    return [np.concatenate(x) for x in zip(*parts, strict=True)]


def ab_colour(wave, flux, ivar):
    """Return synthetic AB g - r of an f_lambda spectrum, masked pixels interpolated over."""
    good = ivar > 0
    if good.sum() < 100:  # a fully masked exposure has no colour
        return np.nan
    f = np.interp(wave, wave[good], flux[good])
    mag = {}
    for b, (wl, resp) in curves.items():
        rr = np.interp(wave, wl, resp, left=0, right=0)
        num = np.trapezoid(f * rr * wave, wave)
        den = np.trapezoid(rr * C_AA / wave, wave)
        mag[b] = -2.5 * np.log10(num / den)
    return mag["g"] - mag["r"]


# %% per exposure
rows, check = [], []
for line in (HERE / "exp_targets.txt").read_text().splitlines():
    t, _, dgr_vac, _ = line.split()
    t, dgr_vac = int(t), float(dgr_vac)
    d = dict(np.load(OUT / "exposures" / f"{t}.npz"))
    i = vrow[t]
    gr_phot = -2.5 * np.log10(float(M["FIBERTOTFLUX_G"][i]) / float(M["FIBERTOTFLUX_R"][i]))
    stacks = []
    for k in range(len(d["EXPID"])):
        wv, fl, iv = stitch(d, k)
        stacks.append((fl, iv))
        e = int(d["EXPID"][k])
        rows.append(dict(
            tid=t, expid=e, sky=sky.get(e, np.nan), eff=eff.get(e, np.nan), see=see.get(e, np.nan),
            status=str(d["fiberstatus_names"][k]) or "-",
            offset=1e3 * float(np.hypot(d["DELTA_X"][k], d["DELTA_Y"][k])),
            dgr=ab_colour(wv, fl, iv) - gr_phot,
        ))  # fmt: skip
    ivs = np.array([s[1] for s in stacks])
    fls = np.array([s[0] for s in stacks])
    wsum = ivs.sum(0)
    mean = np.where(wsum > 0, (fls * ivs).sum(0) / np.where(wsum > 0, wsum, 1), 0.0)
    check.append((t, dgr_vac, ab_colour(wv, mean, wsum) - gr_phot))

# %% table
lines = ["Check: ivar-weighted exposure mean against the VAC dgr (FIBERTOTFLUX)."]
lines += [f"  {t}: VAC {a:+.3f}, exposures {b:+.3f}" for t, a, b in check]
lines += [
    "",
    f"{'TARGETID':>18} {'EXPID':>6} {'sky g':>6} {'efftime':>7} {'seeing':>6} {'dgr':>7} {'offset um':>9}  "
    "FIBERSTATUS",
]
for r in rows:
    lines.append(f"{r['tid']:>18} {r['expid']:>6} {r['sky']:6.2f} {r['eff']:7.1f} {r['see']:6.2f} {r['dgr']:+7.3f} "
                 f"{r['offset']:9.1f}  {r['status']}")  # fmt: skip
# within-target deviations, so differences between galaxies drop out
dev = {"dgr": [], "sky": [], "eff": [], "see": []}
spread = []
for t in sorted({r["tid"] for r in rows}):
    rr = [r for r in rows if r["tid"] == t]
    v = {k: np.array([r[k] for r in rr]) for k in dev}
    ok = np.isfinite(v["dgr"]) & np.isfinite(v["sky"])
    if ok.sum() < 2:
        continue
    for k in dev:
        dev[k] += list(v[k][ok] - np.median(v[k][ok]))
    dg, sk = v["dgr"][ok], v["sky"][ok]
    spread.append((t, dg.min(), dg.max(), sk[np.argmin(dg)], sk.min()))
lines.append("")
for k, name in (("sky", "sky g"), ("eff", "efftime"), ("see", "seeing")):
    rho, p = spearmanr(dev["dgr"], dev[k])
    lines.append(f"Within-target Spearman of exposure dgr against {name}: rho {rho:+.3f} "
                 f"(p {p:.2g}), N {len(dev['dgr'])} exposures")  # fmt: skip
lines += [
    "",
    f"{'TARGETID':>18} {'dgr min':>8} {'dgr max':>8} {'sky of bluest':>13} {'brightest sky':>13}",
]
lines += [f"{t:>18} {a:+8.3f} {b:+8.3f} {c:13.2f} {e:13.2f}" for t, a, b, c, e in spread]
text = "\n".join(lines)
(OUT / "exp_table.txt").write_text(text + "\n")
print(text)

# %% figure: exposure dgr against sky g, one line per target
use_apj_style()
fig, ax = plt.subplots(figsize=(8, 6))
for t in sorted({r["tid"] for r in rows}):
    rr = sorted((r for r in rows if r["tid"] == t), key=lambda r: r["sky"])
    ax.plot([r["sky"] for r in rr], [r["dgr"] for r in rr], "o-", ms=4, lw=0.8)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel(r"exposure sky $g$ [mag arcsec$^{-2}$]", fontsize=13)
ax.set_ylabel(r"$\Delta(g-r)$ exposure $-$ fiber total [mag]", fontsize=13)
fig.savefig(OUT / "exp_sky.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'exp_sky.png'}")
