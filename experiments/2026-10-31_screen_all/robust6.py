"""Rest-frame DR1 coadd spectra and cutouts of the six seed-robust follow-up outliers.

The six are flagged by both flows in >= 15 of 20 seeds (survivors.txt) after the broad-line and
redshift-edge groups are set aside. Pixels with ivar > 0, raw (grey) and 15 px median (black);
dotted lines mark [OII], Hbeta, [OIII] 5007, Mgb, NaD and Halpha.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_screen_all/robust6.py``.
OUTPUT: results/2026-10-31_screen_all/robust6.png.
"""

import io

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from spender.data.desi import DESI

from hubersed.io.desi import load_by_index, tids_to_indices
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

RES = PATHS["RESULTS"]
OUT = RES / "2026-10-31_screen_all"
CUTS = RES / "2026-10-31_logp_vs_colour" / "cutouts"
SIX = (39627775706927918, 39627782258426347, 39633123121563924, 39627776201853253,
       39633322464249559, 39633325949715572)  # fmt: skip
MARK = {
    "[OII]": 3727,
    r"H$\beta$": 4861,
    "[OIII]": 5007,
    "Mgb": 5175,
    "NaD": 5893,
    r"H$\alpha$": 6563,
}

info = {}
for line in (OUT / "survivors.txt").read_text().splitlines():
    p = line.split()
    if p and p[0].isdigit() and int(p[0]) in SIX:
        info[int(p[0])] = dict(dn=p[3], ha=p[4], seeds=p[9], status=" ".join(p[10:]))
wave = DESI._wave_obs.numpy()
use_apj_style()
fig, axes = plt.subplots(6, 1, figsize=(18, 26))
for ax, t in zip(axes, SIX, strict=True):
    flux, ivar, z, _ = load_by_index(int(tids_to_indices(np.array([t]))[0]))
    flux, ivar, z = np.asarray(flux, float), np.asarray(ivar, float), float(z)
    good = ivar > 0
    rest = wave[good] / (1 + z)
    ax.plot(rest, flux[good], color="0.75", lw=0.4)
    sm = medfilt(flux[good], 15)
    ax.plot(rest, sm, color="k", lw=0.8)
    for name, lam in MARK.items():
        if rest.min() < lam < rest.max():
            ax.axvline(lam, color="#2166ac", ls=":", lw=0.8)
            ax.text(lam, 0.97, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=10, color="#2166ac")  # fmt: skip
    r = info[t]
    ax.set_title(f"{t}   z {z:.3f}   Dn4000 {r['dn']}   H$\\alpha$ EW {r['ha']} \\AA   seeds {r['seeds']}/20"
                 f"   ({r['status']})", fontsize=12)  # fmt: skip
    ax.set_xlabel(r"rest wavelength [\AA]", fontsize=11)
    ax.set_ylabel(r"$f_\lambda$ [$10^{-17}$ cgs]", fontsize=11)
    ins = ax.inset_axes([0.83, 0.45, 0.16, 0.53])
    ins.imshow(
        plt.imread(io.BytesIO((CUTS / f"{t}.jpg").read_bytes()), format="jpg"), origin="upper"
    )
    ins.add_patch(plt.Circle((47.5, 47.5), 0.75 / 0.262, fill=False, color="w", lw=1))
    ins.set_xticks([])
    ins.set_yticks([])
fig.tight_layout()
fig.savefig(OUT / "robust6.png", dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'robust6.png'}")
