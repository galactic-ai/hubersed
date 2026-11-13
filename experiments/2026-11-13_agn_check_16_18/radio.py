"""FIRST, NVSS and LoTSS DR3 sources near selected gallery105 figures, and their MEx placement.

Targets (TARGETS): 16, 18, 22, 41, 42, 74, 77, 78, 82, 86, 89 and 102.

QUESTION: Is 16 a radio source (BL Lac test)? Where does 18 sit on the mass-excitation diagram? Is
the extended LoTSS DR3 emission Nikhil saw at 22 a catalogued source at the galaxy?
HYPOTHESIS: none registered.
INPUTS: HEASARC TAP tables first and nvss, and the ASTRON TAP table lotss_dr3.main_sources (144 MHz,
6 arcsec; LoTSS DR3, A&A 707, A198), all queried live; data/fastspec-iron-sv3-bright.fits
(RA, DEC, LOGMSTAR, OIII_5007_FLUX, HBETA_FLUX of the SV3 bright coadd).
METHOD: cone of 60 arcsec. MEx curves of Juneau et al. (2014, arXiv:1403.6832, ms_arxiv.tex
l.451-471): y = log([OIII]5007/Hbeta), x = log M*; above the upper curve is MEx-AGN, below the
lower is MEx-SF. Applied only when [OIII]5007 and Hbeta both have FastSpecFit flux S/N >= 3 (J11 uses
S/N > 3; J14 also allows an Hbeta upper limit, not used here). Figures 22, 41, 42, 74, 77 and 78 added on 2026-11-13.
L144 = 4 pi D_L^2 S (1 + z)^(-alpha - 1), with S_nu ~ nu^alpha, alpha = -0.7 assumed, Planck18.
Radio excess as Best et al. (2023, arXiv:2305.05782, lofar_sfagn_final.tex l.1827-1829, 1845-1848,
1865-1875): log L150 > 22.24 + 1.08 log SFR + 0.7, with SFR floored at 0.01 Msun/yr. SFR is FastSpecFit
SFR, not their consensus SED SFR (100 Myr, Chabrier); L144 is used as L150 (about 0.01 dex apart).
Every source within 60 arcsec is listed and run through the excess test at the target's redshift;
only a source within a few arcsec of the galaxy belongs to it (74's sources at 42-50 arcsec do not).
Assumptions: FastSpecFit LOGMSTAR on the IMF of the J14 masses (J14 states Chabrier;
FastSpecFit IMF not checked here); J14 states no upper mass limit for its cubics.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-13_agn_check_16_18/radio.py``.
RESULT: pending.
FIGURES: none (radio.txt).
"""

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18
from astropy.io import fits
from astroquery.utils.tap.core import TapPlus

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-13_agn_check_16_18"
TARGETS = {
    16: 39627788365337938,
    18: 39627788382109772,
    22: 39627800440737892,
    42: 39628357138121066,
    74: 39633158257249026,
    41: 39628351614225514,
    77: 39633282689666443,
    78: 39633290038083787,
    82: 39633329376461212,
    86: 39633402046974648,
    89: 39633407818337381,
    102: 39633425279222058,
}
HEASARC = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap"
ASTRON = "https://vo.astron.nl/tap"
TABLES = {
    "first": (HEASARC, "name, ra, dec, flux_20_cm, int_flux_20_cm, major_axis, minor_axis"),
    "nvss": (HEASARC, "name, ra, dec, flux_20_cm"),
    "lotss_dr3.main_sources": (
        ASTRON,
        "source_name, ra, dec, total_flux, e_total_flux, peak_flux, isl_rms, dc_maj, dc_min, "
        "dc_pa, s_code, resolution",
    ),
}
ALPHA = -0.7
BEST_A, BEST_B, EXCESS_DEX, SFR_FLOOR = 22.24, 1.08, 0.7, 0.01
RADIUS_DEG = 60 / 3600
UP = (410.24, -109.333, 9.71731, -0.288244)
LO = (352.066, -93.8249, 8.32651, -0.246416)
MIN_SN = 3.0


def mex(x):
    """Return the J14 upper and lower MEx curves at log M* = x."""
    up = 0.375 / (x - 10.5) + 1.14 if x <= 10 else np.polyval(UP[::-1], x)
    lo = 0.375 / (x - 10.5) + 1.14 if x <= 9.6 else np.polyval(LO[::-1], x)
    return up, lo


def main():
    """Query the radio tables and place each target on the MEx diagram."""
    taps = {url: TapPlus(url=url, verbose=False) for url in (HEASARC, ASTRON)}
    lines = []
    with fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits") as h:
        f, m = h["FASTSPEC"].data, h["METADATA"].data
        for nn, tid in TARGETS.items():
            i = int(np.flatnonzero(m["TARGETID"] == tid)[0])
            ra, dec = float(m["RA"][i]), float(m["DEC"][i])
            lines.append(f"figure {nn}, {tid}, RA {ra:.6f} DEC {dec:.6f}")
            z = float(m["Z"][i])
            dl = Planck18.luminosity_distance(z).to(u.m).value
            for t, (url, cols) in TABLES.items():
                q = (
                    f"SELECT {cols}, DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec}))"
                    f" * 3600 AS sep FROM {t} WHERE CONTAINS(POINT('ICRS', ra, dec), "
                    f"CIRCLE('ICRS', {ra}, {dec}, {RADIUS_DEG})) = 1"
                )
                r = taps[url].launch_job(q).get_results()
                lines.append(f"  {t}: {len(r)} within 60 arcsec")
                for row in r:
                    lines.append("    " + ", ".join(f"{c} {row[c]}" for c in r.colnames))
                    if t.startswith("lotss"):
                        s = float(row["total_flux"]) * 1e-29  # mJy to W/m^2/Hz
                        lum = 4 * np.pi * dl**2 * s * (1 + z) ** (-ALPHA - 1)
                        kpc = Planck18.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value
                        lines.append(
                            f"      L144 {lum:.2e} W/Hz (alpha {ALPHA}); deconvolved major axis "
                            f"{float(row['dc_maj']) * kpc:.0f} kpc at z {z:.4f}"
                        )
                        sfr = max(float(f["SFR"][i]), SFR_FLOOR)
                        lim = BEST_A + BEST_B * np.log10(sfr) + EXCESS_DEX
                        lines.append(
                            f"      radio excess (Best+23): log L {np.log10(lum):.2f} vs limit "
                            f"{lim:.2f} at SFR {sfr:.3g} -> {'yes' if np.log10(lum) > lim else 'no'}"
                            f"; SFR below which it stays an excess "
                            f"{10 ** ((np.log10(lum) - EXCESS_DEX - BEST_A) / BEST_B):.3g} Msun/yr"
                        )
            x = float(f["LOGMSTAR"][i])
            fo, fb = float(f["OIII_5007_FLUX"][i]), float(f["HBETA_FLUX"][i])
            sn = [
                fl * np.sqrt(float(f[f"{c}_FLUX_IVAR"][i]))
                for fl, c in ((fo, "OIII_5007"), (fb, "HBETA"))
            ]
            if min(sn) < MIN_SN:
                lines.append(
                    f"  MEx: not applied, S/N [OIII] {sn[0]:.1f}, Hb {sn[1]:.1f} (needs >= {MIN_SN})"
                )
            else:
                y = np.log10(fo / fb)
                up, lo = mex(x)
                cls = "MEx-AGN" if y > up else "MEx-SF" if y < lo else "MEx-intermediate"
                lines.append(
                    f"  MEx: log M* {x:.2f}, log [OIII]/Hb {y:.2f}, upper {up:.2f}, "
                    f"lower {lo:.2f} -> {cls}"
                )
            lines.append("")
    text = "\n".join(lines)
    (OUT / "radio.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
