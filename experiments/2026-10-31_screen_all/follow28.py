"""Rest-frame DR1 coadd spectra and cutouts of the 28 follow-up outliers not yet looked at closely.

The 36 of survivors.txt less the robust six (robust6.py), 39627757533007793 and 39633421923781141.
One figure per target in survivors.txt order, styled as robust6.py: pixels with ivar > 0, raw (grey)
and 15 px median (black), dotted [OII], Hbeta, [OIII] 5007, Mgb, NaD and Halpha, and the Legacy DR9
cutout (cutouts of 2026-10-31_logp_vs_colour/gallery.py) with the 1.5'' fibre at top right. The title
adds the broad Halpha and Hbeta S/N of broad.txt.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_screen_all/follow28.py``.
OUTPUT: results/2026-10-31_screen_all/follow28/<NN>_<TARGETID>.png.
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
DONE = (39627775706927918, 39627782258426347, 39633123121563924, 39627776201853253,
        39633322464249559, 39633325949715572, 39627757533007793, 39633421923781141)  # fmt: skip
MARK = {
    "[OII]": 3727,
    r"H$\beta$": 4861,
    "[OIII]": 5007,
    "Mgb": 5175,
    "NaD": 5893,
    r"H$\alpha$": 6563,
}


def table(name, cols):
    """Return {TARGETID: row fields} of a results/2026-10-31_screen_all text table."""
    out = {}
    for line in (OUT / name).read_text().splitlines():
        p = line.split()
        if p and p[0].isdigit() and len(p[0]) == 17:
            out[int(p[0])] = cols(p)
    return out


def main():
    """Write one spectrum-and-cutout figure per target."""
    info = table(
        "survivors.txt", lambda p: dict(dn=p[3], ha=p[4], seeds=p[9], status=" ".join(p[10:]))
    )
    broad = table("broad.txt", lambda p: dict(bha=p[2], bhb=p[5]))
    tids = [t for t in info if t not in DONE]
    assert len(tids) == 28, len(tids)
    wave = DESI._wave_obs.numpy()
    dest = OUT / "follow28"
    dest.mkdir(exist_ok=True)
    use_apj_style()
    for k, t in enumerate(tids, 1):
        flux, ivar, z, _ = load_by_index(int(tids_to_indices(np.array([t]))[0]))
        flux, ivar, z = np.asarray(flux, float), np.asarray(ivar, float), float(z)
        good = ivar > 0
        rest = wave[good] / (1 + z)
        fig, ax = plt.subplots(figsize=(18, 4.6))
        ax.plot(rest, flux[good], color="0.75", lw=0.4)
        ax.plot(rest, medfilt(flux[good], 15), color="k", lw=0.8)
        for name, lam in MARK.items():
            if rest.min() < lam < rest.max():
                ax.axvline(lam, color="#2166ac", ls=":", lw=0.8)
                ax.text(lam, 0.97, name, transform=ax.get_xaxis_transform(), ha="center",
                        va="top", fontsize=10, color="#2166ac")  # fmt: skip
        r, b = info[t], broad.get(t, dict(bha="-", bhb="-"))
        ax.set_title(f"{k}/28   {t}   z {z:.3f}   Dn4000 {r['dn']}   H$\\alpha$ EW {r['ha']} \\AA{{}}   "
                     f"broad H$\\alpha$ S/N {b['bha']}, H$\\beta$ S/N {b['bhb']}   seeds {r['seeds']}/20"
                     f"   ({r['status']})", fontsize=11)  # fmt: skip
        ax.set_xlabel(r"rest wavelength [\AA]", fontsize=11)
        ax.set_ylabel(r"$f_\lambda$ [$10^{-17}$ cgs]", fontsize=11)
        ins = ax.inset_axes([0.83, 0.45, 0.16, 0.53])
        ins.imshow(plt.imread(io.BytesIO((CUTS / f"{t}.jpg").read_bytes()), format="jpg"),
                   origin="upper")  # fmt: skip
        ins.add_patch(plt.Circle((47.5, 47.5), 0.75 / 0.262, fill=False, color="w", lw=1))
        ins.set_xticks([])
        ins.set_yticks([])
        fig.savefig(dest / f"{k:02d}_{t}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {len(tids)} figures to {dest}")


if __name__ == "__main__":
    main()
