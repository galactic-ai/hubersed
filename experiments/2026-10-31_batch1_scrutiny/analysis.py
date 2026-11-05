"""Scrutiny of follow-up outliers 1-5 of follow28 (#17): lines, sky coincidence and exposures.

QUESTION: which of the five are AGN, sky artefacts or clean continuum targets for a pocoMC fit?
HYPOTHESIS: none registered. Visual calls (Nikhil, 2026-10-31): 1 crowded, 2 AGN, 3 emission-line,
4 to check, 5 fit-ready.
INPUTS: FastSpecFit iron sv3 bright; spender chunk coadds (load_spectrum); spender sky-lines.txt;
DR1 single exposures of 39627770254327945 from 2026-10-31_screen_all/sky_extract.py
(results/2026-10-31_screen_all/exposures/).
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_batch1_scrutiny/analysis.py``.
RESULT: 01, 02, 04 have broad Balmer lines (Hbeta sigma 744, 634, 652 km/s, S/N 18-24; 02 also broad
Halpha S/N 116; 01 MgII sigma 686) and are AGN. 03: [OIII]5007 sits on the OH 7775.5 A line, but
[OIII]4959 (S/N 32) holds the 1:3 ratio (42.8 vs 125.7), so the lines are real; [OIII]/Hbeta 9.7 with
Halpha out of range. 05: narrow lines in all 3 exposures, log [NII]/Ha -0.91 and [OIII]/Hb 0.48, below
Kauffmann+03 (star-forming); B-arm exposures sit at 1.31, 0.73 and 1.01 of the coadd.
Assumptions: FastSpecFit 2.5.1 fluxes; rest wavelengths from memory.
FIGURES: results/2026-10-31_batch1_scrutiny/ lines.txt, sky_03.txt, exposures_05.png.
"""

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import spender
from astropy.io import fits
from scipy.signal import medfilt

from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

OUT = PATHS["RESULTS"] / "2026-10-31_batch1_scrutiny"
FIVE = (39627763623136430, 39627835547062252, 39633322460057381, 39627854832472847,
        39627770254327945)  # fmt: skip
LINES = ("MGII_2796", "MGII_2803", "OII_3726", "OII_3729", "HBETA", "HBETA_BROAD", "OIII_4959",
         "OIII_5007", "HALPHA", "HALPHA_BROAD", "NII_6584")  # fmt: skip
REST = {
    "Hbeta": 4861.3,
    "[OIII]4959": 4958.9,
    "[OIII]5007": 5006.8,
    "Halpha": 6562.8,
}  # air, memory


def lines_table():
    """Write FastSpecFit flux, S/N and sigma of LINES for the five."""
    with fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True) as h:
        meta, fs = h["METADATA"].data, h["FASTSPEC"].data
        rows = {t: np.flatnonzero(meta["TARGETID"] == t)[0] for t in FIVE}
        out = [
            "# FastSpecFit iron sv3 bright: flux (1e-17 erg/s/cm2) / S/N / sigma (km/s); '-' if no ivar"
        ]
        for k, t in enumerate(FIVE, 1):
            r = fs[rows[t]]
            cells = []
            for ln in LINES:
                f, iv = float(r[f"{ln}_FLUX"]), float(r[f"{ln}_FLUX_IVAR"])
                cells.append(
                    f"{ln} {f:.1f}/{f * np.sqrt(iv):.1f}/{r[f'{ln}_SIGMA']:.0f}"
                    if iv > 0
                    else f"{ln} -"
                )
            out.append(f"{k:02d} {t} z {meta['Z'][rows[t]]:.4f}: " + "; ".join(cells))
    (OUT / "lines.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def sky_03(t=39633322460057381, half=3.0):
    """List the sky lines of spender sky-lines.txt within ``half`` A of target 3's line positions."""
    path = os.path.join(os.path.dirname(spender.__file__), "data", "sky-lines.txt")
    sky = np.genfromtxt(path, usecols=(0, 1))
    s = load_spectrum(t)
    z = float(s.redshift.value)
    w = np.asarray(s.spectral_axis.value, float)
    f = np.asarray(s.flux.value, float)
    iv = np.asarray(s.uncertainty.array, float)
    out = [f"# {t} z {z:.5f}: line, observed A, flux and ivar at the pixel, spender sky lines "
           f"within {half} A (A, intensity)"]  # fmt: skip
    for name, lam in REST.items():
        obs = lam * (1 + z)
        near = sky[np.abs(sky[:, 0] * 10 - obs) < half]
        i = int(np.argmin(np.abs(w - obs)))
        out.append(f"{name:>11s} {obs:8.1f}  flux {f[i]:6.2f} ivar {iv[i]:6.3f}  "
                   + (", ".join(f"{a * 10:.1f} ({b:.2f})" for a, b in near) or "none"))  # fmt: skip
    peak = int(np.argmax(np.where((w > 7700) & (w < 7850), f, -np.inf)))
    out.append(f"peak in 7700-7850 A at {w[peak]:.1f} A, flux {f[peak]:.2f}")
    (OUT / "sky_03.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def exposures_05(t=39627770254327945, smooth=15):
    """Plot target 5's exposures at its lines and around the Halpha hump, and print arm medians."""
    use_apj_style()
    e = np.load(PATHS["RESULTS"] / "2026-10-31_screen_all" / "exposures" / f"{t}.npz")
    s = load_spectrum(t)
    z = float(s.redshift.value)
    cw = np.asarray(s.spectral_axis.value, float)
    cf = np.asarray(s.flux.value, float)
    cok = ~s.mask
    print(f"{t}: EXPID {e['EXPID'].tolist()}, FIBERSTATUS {e['fiberstatus_names'].tolist()}")
    wins = [("Hbeta + [OIII]", 4830.0, 5040.0), ("Halpha hump", 6250.0, 6900.0)]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(e["EXPID"])))
    fig, ax = plt.subplots(1, len(wins), figsize=(15, 4.5))
    for a, (name, lo, hi) in zip(ax, wins, strict=True):
        olo, ohi = lo * (1 + z), hi * (1 + z)
        m = cok & (cw > olo) & (cw < ohi)
        a.plot(cw[m] / (1 + z), cf[m], color="k", lw=1.0, label="coadd", zorder=5)
        for arm in "BRZ":
            w = e[f"{arm}_wave"]
            sel = (w > olo) & (w < ohi)
            for k, eid in enumerate(e["EXPID"]):
                ok = sel & (e[f"{arm}_ivar"][k] > 0) & (e[f"{arm}_mask"][k] == 0)
                a.plot(w[ok] / (1 + z), e[f"{arm}_flux"][k][ok], color=colors[k], lw=0.6,
                       label=f"EXPID {eid}" if arm == "B" or name.startswith("Ha") and arm == "R"
                       else None)  # fmt: skip
        for lam in REST.values():
            if lo < lam < hi:
                a.axvline(lam, color="#2166ac", ls=":", lw=0.8)
        a.set_title(f"{t} {name}", fontsize=10)
        a.set_xlabel(r"rest wavelength [\AA]")
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ cgs]")
    h, lab = ax[1].get_legend_handles_labels()
    keep = dict(zip(lab, h, strict=True))
    ax[1].legend(keep.values(), keep.keys(), frameon=False, fontsize=8)
    fig.savefig(OUT / "exposures_05.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    for arm in "BRZ":
        w = e[f"{arm}_wave"]
        g = cok & (cw >= w.min()) & (cw <= w.max())
        ref = medfilt(np.interp(w, cw[g], cf[g], left=np.nan, right=np.nan), smooth)
        for k, eid in enumerate(e["EXPID"]):
            ok = (e[f"{arm}_ivar"][k] > 0) & (e[f"{arm}_mask"][k] == 0)
            x = np.full(w.size, np.nan)
            x[ok] = medfilt(np.asarray(e[f"{arm}_flux"][k][ok], float), smooth)
            print(f"{arm} EXPID {eid}: median exposure/coadd {np.nanmedian(x / ref):.3f}")
    print(f"wrote {OUT / 'exposures_05.png'}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    lines_table()
    sky_03()
    exposures_05()
