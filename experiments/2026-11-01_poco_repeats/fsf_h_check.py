"""Check the h convention of FastSpecFit v3.0 distances, masses and absolute magnitudes.

QUESTION: are FastSpecFit distances in Mpc/h used as Mpc (h = 1), and what h converts them?
Two checks:
1. The fiducial distance table of fastspecfit 3.1.5 (data/desi_fiducial_cosmology.dat, comoving
   distance in Mpc/h) against astropy Planck18 comoving distances in Mpc. Their ratio is h.
2. FastSpecFit sets ABSMAG = m_oband - dmod - KCORR (photometry.py:282) with m from the
   dereddened Legacy Surveys flux (io.py:72-76). So m - ABSMAG01_SDSS_R - KCORR01_SDSS_R returns
   dmod for the band it used. For each object the band (g, r, z, W1) closest to the h = 1 modulus
   is taken, then compared with the h = 1 and Planck18 moduli.
If dmod is the h = 1 one, the physical stellar mass is LOGMSTAR - 2 log10 h.
INPUTS: data/fastspecfit_v3.0/fastspec-iron-sv3-bright.fits; the fastspecfit 3.1.5 table, path
given with --table (tag 3.1.5 tarball, py/fastspecfit/data/).
COMMAND: ``uv run --no-sync python experiments/2026-11-01_poco_repeats/fsf_h_check.py --table <dat>``.
OUTPUT: results/2026-11-01_poco_repeats/fsf_h_check.txt.
"""

# %%
import argparse
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18
from astropy.io import fits

from hubersed.paths import PATHS

HERE = Path(__file__).resolve().parent
FSF = PATHS["DATA"] / "fastspecfit_v3.0" / "fastspec-iron-sv3-bright.fits"
TID = 39633416978694809


def main(table):
    """Write both checks."""
    z_t, _, dc_t = np.loadtxt(table, unpack=True)
    zz = np.array([0.05, 0.1, 0.2, 0.2605, 0.4])
    ratio = np.interp(zz, z_t, dc_t) / Planck18.comoving_distance(zz).to_value(u.Mpc)
    lines = ["1. table comoving distance [Mpc/h] / Planck18 [Mpc] = h_eff"]
    lines += [f"   z {z:.4f}: {r:.5f}" for z, r in zip(zz, ratio, strict=True)]
    h = float(np.median(ratio))

    with fits.open(FSF, memmap=True) as f:
        m, s = f["METADATA"].data, f["SPECPHOT"].data
        z = np.asarray(m["Z"], float)
        absm, kc = np.asarray(s["ABSMAG01_SDSS_R"], float), np.asarray(s["KCORR01_SDSS_R"], float)
        mags = {}
        for b in ("G", "R", "Z", "W1"):
            flux = np.asarray(m[f"FLUX_{b}"], float) / np.asarray(m[f"MW_TRANSMISSION_{b}"], float)
            with np.errstate(divide="ignore", invalid="ignore"):
                mags[b] = np.where(flux > 0, 22.5 - 2.5 * np.log10(flux), np.nan)
        tid = np.asarray(s["TARGETID"])
    ok = (z > 0.01) & (z < 0.6) & np.isfinite(absm) & (absm != 0)
    dl_h1 = np.interp(z, z_t, dc_t) * (1 + z)
    dm_h1 = 5 * np.log10(np.where(ok, dl_h1, 1.0)) + 25
    dm_p18 = Planck18.distmod(np.where(ok, z, 0.1)).value
    implied = np.stack([mags[b] - absm - kc for b in mags])
    gap = np.abs(implied - dm_h1)
    pick = np.argmin(np.where(np.isfinite(gap), gap, np.inf), axis=0)
    best = np.take_along_axis(implied, pick[None], 0)[0]
    ok &= np.isfinite(best)
    d1, dp = (best - dm_h1)[ok], (best - dm_p18)[ok]
    lines += [
        "",
        f"2. {ok.sum()} objects at 0.01 < z < 0.6 with ABSMAG01_SDSS_R",
        f"   implied dmod - h=1 dmod:    median {np.nanmedian(d1):+.5f}, |diff| < 0.01 mag "
        f"for {np.nanmean(np.abs(d1) < 0.01):.4f}",
        f"   implied dmod - Planck18:   median {np.nanmedian(dp):+.5f} mag "
        f"(5 log10 h_eff = {5 * np.log10(h):+.5f})",
        "",
        f"h_eff = {h:.5f}: physical log M = LOGMSTAR + {-2 * np.log10(h):.4f}",
    ]
    i = np.flatnonzero(tid == TID)
    if i.size:
        lines.append(f"{TID}: implied dmod {best[i[0]]:.4f}, h=1 {dm_h1[i[0]]:.4f}, "
                     f"Planck18 {dm_p18[i[0]]:.4f}")  # fmt: skip
    out = PATHS["RESULTS"] / HERE.name / "fsf_h_check.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", type=Path, required=True)
    main(ap.parse_args().table)
