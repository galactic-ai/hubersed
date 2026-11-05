"""DR1 SV3 bright coadd spectra of the 18 blue outliers with their Legacy DR9 cutouts (#17).

Cause labels from photometry.txt (neighbour: fracflux_g > 1) and rule.txt (sky: the sky g >= 21
coadd is no longer flagged by one of the flows); otherwise "not explained". Spectra are the
spender inputs with the normalisation undone, pixels with ivar > 0 only, raw (grey) and 21 px
median filter (black). Cutouts are 25" across with the 1.5" fiber.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_logp_vs_colour/gallery.py``.
OUTPUT: results/2026-10-31_logp_vs_colour/gallery.png and cutouts/<TARGETID>.jpg.
RESULT: 2026-10-31. 7 neighbour, 2 neighbour + sky, 3 sky, 6 not explained.
"""

# %%
import io
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import medfilt
from spender.data.desi import DESI

from hubersed.io.desi import load_by_index, tids_to_indices
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

HERE = Path(__file__).resolve().parent
OUT = PATHS["RESULTS"] / "2026-10-31_logp_vs_colour"
CUT = "https://www.legacysurvey.org/viewer/cutout.jpg?ra={}&dec={}&layer=ls-dr9&pixscale=0.262&size=96"
THR = {"cont10": -15.72, "cont15": -25.24}

# %% cause per target
ff = {}
for line in (OUT / "photometry.txt").read_text().splitlines()[1:]:
    p = line.split()
    if p and p[0].isdigit():
        q = p[3:] if p[1] == "all" else p[2:]
        ff[int(p[0])] = float(q[6])
cleared = {}
for line in (OUT / "rule.txt").read_text().splitlines():
    p = line.split()
    if p and p[0].isdigit() and int(p[0]) in ff:
        c10, c15 = float(p[4]), float(p[7])
        cleared[int(p[0])] = (np.isfinite(c10) and c10 > THR["cont10"]) or (
            np.isfinite(c15) and c15 > THR["cont15"]
        )
cause = {}
for t in ff:
    parts = (["neighbour"] if ff[t] > 1 else []) + (["sky"] if cleared.get(t) else [])
    cause[t] = " + ".join(parts) or "not explained"
dgr = {
    int(x.split()[0]): float(x.split()[2])
    for x in (HERE / "exp_targets.txt").read_text().splitlines()
}
order = sorted(
    cause,
    key=lambda t: (["neighbour", "neighbour + sky", "sky", "not explained"].index(cause[t]), t),
)

# %% figure
h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)
M = h["METADATA"].data
vrow = {int(t): i for i, t in enumerate(M["TARGETID"])}
wave = DESI._wave_obs.numpy()
(OUT / "cutouts").mkdir(exist_ok=True)
use_apj_style()
fig, axes = plt.subplots(6, 3, figsize=(20, 24))
for ax, t in zip(axes.ravel(), order, strict=True):
    flux, ivar, z, _ = load_by_index(int(tids_to_indices(np.array([t]))[0]))
    flux, ivar = np.asarray(flux, float), np.asarray(ivar, float)
    good = ivar > 0
    ax.plot(wave[good], flux[good], color="0.75", lw=0.4)
    ax.plot(wave[good], medfilt(flux[good], 21), color="k", lw=0.9)
    ax.set_title(f"{t}  z {float(z):.3f}  $\\Delta(g-r)$ {dgr[t]:+.2f}\n{cause[t]}", fontsize=11)
    ax.set_xlabel(r"observed wavelength [\AA]", fontsize=10)
    ax.set_ylabel(r"$f_\lambda$ [$10^{-17}$ cgs]", fontsize=10)
    jpg = OUT / "cutouts" / f"{t}.jpg"
    for attempt in range(4):  # the cutout server sometimes times out
        if jpg.exists():
            break
        i = vrow[t]
        try:
            url = CUT.format(float(M["RA"][i]), float(M["DEC"][i]))
            with urllib.request.urlopen(url, timeout=90) as r:
                jpg.write_bytes(r.read())
        except TimeoutError:
            print(f"{t}: cutout timeout, attempt {attempt + 1}")
    ins = ax.inset_axes([0.72, 0.52, 0.27, 0.46])
    ins.imshow(plt.imread(io.BytesIO(jpg.read_bytes()), format="jpg"), origin="upper")
    ins.add_patch(plt.Circle((47.5, 47.5), 0.75 / 0.262, fill=False, color="w", lw=1))
    ins.set_xticks([])
    ins.set_yticks([])
fig.tight_layout()
fig.savefig(OUT / "gallery.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'gallery.png'}")
for c in ("neighbour", "neighbour + sky", "sky", "not explained"):
    print(f"{c}: {[t for t in order if cause[t] == c]}")
