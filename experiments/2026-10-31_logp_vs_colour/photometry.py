"""Legacy DR9 Tractor photometry and cutouts of the 18 single-exposure blue outliers (#17).

Groups from exp_table.txt: "all blue" when every exposure has dgr < -0.25, "sky" when the bluest
exposure has the brightest sky and the spread is > 0.3, else "varies". Baseline: the matched
controls of these outliers (controls.npz). Tractor rows from ls_dr9.tractor (Data Lab) by LS_ID.
Cutouts from the Legacy Survey viewer (ls-dr9, 0.262"/px).
COMMAND: ``uv run --no-sync python experiments/2026-10-31_logp_vs_colour/photometry.py``.
OUTPUT: results/2026-10-31_logp_vs_colour/photometry.txt and cutouts.png.
RESULT: 2026-10-31. fracflux_g > 3 for 9 of 18 (controls: 95th percentile 0.054); cutouts show bright
blue neighbours. Within-target Spearman of exposure dgr against seeing: -0.84 (p 9e-9) with
fracflux_g > 1, -0.12 (p 0.55) without.
"""

# %%
import io
import urllib.request
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astroquery.utils.tap.core import TapPlus
from scipy.stats import spearmanr

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

RES = PATHS["RESULTS"]
OUT = RES / "2026-10-31_logp_vs_colour"
CUT = "https://www.legacysurvey.org/viewer/cutout.jpg?ra={}&dec={}&layer=ls-dr9&pixscale=0.262&size=96"
COLS = (
    "ls_id, type, shape_r, sersic, flux_g, flux_r, fiberflux_g, fiberflux_r, fibertotflux_g, "
    "fibertotflux_r, fracflux_g, fracflux_r, fracmasked_g, fracin_g, maskbits, nobs_g, psfsize_g"
)
warnings.filterwarnings("ignore", module="astroquery")

# %% groups from the per-exposure table
exp = defaultdict(list)
for line in (OUT / "exp_table.txt").read_text().splitlines():
    p = line.split()
    if len(p) >= 7 and p[0].isdigit() and p[1].isdigit():
        exp[int(p[0])].append((float(p[2]), float(p[5]), float(p[4])))  # sky g, dgr, seeing
group = {}
for t, v in exp.items():
    v = [x for x in v if np.isfinite(x[1])]
    sky, dg = np.array([x[0] for x in v]), np.array([x[1] for x in v])
    if dg.max() < -0.25:
        group[t] = "all blue"
    elif sky[np.argmin(dg)] == sky.min() and np.ptp(dg) > 0.3:
        group[t] = "sky"
    else:
        group[t] = "varies"

ctl = np.load(RES / "2026-10-31_spectrum_qa_cont_outliers" / "controls.npz")
orow = {int(t): k for k, t in enumerate(ctl["outliers"])}
controls = sorted({int(c) for t in group for c in ctl["matched"][orow[t]]})

# %% Tractor rows
h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)
M = h["METADATA"].data
vrow = {int(t): i for i, t in enumerate(M["TARGETID"])}
lsid = {t: int(M["LS_ID"][vrow[t]]) for t in list(group) + controls}
tap = TapPlus(url="https://datalab.noirlab.edu/tap", verbose=False)
ids = ",".join(str(v) for v in lsid.values())
tab = tap.launch_job(f"SELECT {COLS} FROM ls_dr9.tractor WHERE ls_id IN ({ids})").get_results()
tr = {int(r["ls_id"]): r for r in tab}
print(f"{len(tr)} Tractor rows for {len(lsid)} targets")


def colours(r):
    """Return g-r from fiber, fiber-total and total flux, and fibertot/fiber in g."""

    def gr(a, b):
        return -2.5 * np.log10(float(r[a]) / float(r[b])) if r[a] > 0 and r[b] > 0 else np.nan

    return (gr("fiberflux_g", "fiberflux_r"), gr("fibertotflux_g", "fibertotflux_r"),
            gr("flux_g", "flux_r"), float(r["fibertotflux_g"]) / float(r["fiberflux_g"]))  # fmt: skip


# %% table
hdr = (f"{'TARGETID':>18} {'group':>8} {'type':>4} {'r_e':>5} {'gr_fib':>6} {'gr_tot':>6} "
       f"{'gr_all':>6} {'tot/fib g':>9} {'fracflux g':>10} {'fracmask g':>10} {'maskbits':>8}")  # fmt: skip
lines = [hdr]
for t in sorted(group, key=lambda x: group[x]):
    r = tr[lsid[t]]
    a, b, c, d = colours(r)
    lines.append(f"{t:>18} {group[t]:>8} {str(r['type']):>4} {float(r['shape_r']):5.2f} {a:6.2f} "
                 f"{b:6.2f} {c:6.2f} {d:9.3f} {float(r['fracflux_g']):10.3f} "
                 f"{float(r['fracmasked_g']):10.3f} {int(r['maskbits']):8d}")  # fmt: skip
cv = np.array([colours(tr[lsid[t]]) for t in controls if lsid[t] in tr])
ff = np.array([float(tr[lsid[t]]["fracflux_g"]) for t in controls if lsid[t] in tr])
mb = np.array([int(tr[lsid[t]]["maskbits"]) for t in controls if lsid[t] in tr])
lines += ["", f"matched controls ({len(cv)}): median gr_fib {np.nanmedian(cv[:, 0]):.2f}, "
          f"gr_tot - gr_fib {np.nanmedian(cv[:, 1] - cv[:, 0]):+.3f}, tot/fib g "
          f"{np.nanmedian(cv[:, 3]):.3f} [95th {np.nanpercentile(cv[:, 3], 95):.3f}], fracflux g "
          f"{np.median(ff):.3f} [95th {np.percentile(ff, 95):.3f}], maskbits != 0 in "
          f"{int((mb != 0).sum())}"]  # fmt: skip
lines.append("")
for name, sel in (("fracflux_g > 1", True), ("fracflux_g <= 1", False)):
    dd, ds = [], []
    for t, v in exp.items():
        if (float(tr[lsid[t]]["fracflux_g"]) > 1) != sel:
            continue
        v = np.array([x for x in v if np.isfinite(x[1]) and np.isfinite(x[2])])
        if len(v) < 2:
            continue
        dd += list(v[:, 1] - np.median(v[:, 1]))
        ds += list(v[:, 2] - np.median(v[:, 2]))
    rho, p = spearmanr(dd, ds)
    lines.append(f"{name}: within-target Spearman of exposure dgr against seeing rho {rho:+.3f} "
                 f"(p {p:.2g}), N {len(dd)} exposures")  # fmt: skip
text = "\n".join(lines)
(OUT / "photometry.txt").write_text(text + "\n")
print(text)

# %% cutouts with the 1.5" fiber
use_apj_style()
order = sorted(group, key=lambda x: (group[x], x))
fig, axes = plt.subplots(3, 6, figsize=(18, 9.6))
for ax, t in zip(axes.ravel(), order, strict=True):
    i = vrow[t]
    url = CUT.format(float(M["RA"][i]), float(M["DEC"][i]))
    with urllib.request.urlopen(url, timeout=60) as resp:
        img = plt.imread(io.BytesIO(resp.read()), format="jpg")
    ax.imshow(img, origin="upper")
    ax.add_patch(plt.Circle((47.5, 47.5), 0.75 / 0.262, fill=False, color="w", lw=1))
    ax.set_title(f"{t}\n{group[t]}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
fig.savefig(OUT / "cutouts.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'cutouts.png'}")
