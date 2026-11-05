"""FastSpecFit line diagnostics of the unsorted gallery92 outliers: AGN, emission-line or neither (#17).

QUESTION: which of the gallery92 plots still at the top level of results/2026-11-01_gallery92/ are
AGN or emission-line galaxies by their FastSpecFit lines?
HYPOTHESIS: none registered.
INPUTS: data/fastspec-iron-sv3-bright.fits (FASTSPEC and METADATA, the SV3 bright coadd);
COADD_NUMEXP of the SV3 bright coadd from results/2026-11-01_repeats92/sparcl_spectra.npz; the plot
names at the top level of results/2026-11-01_gallery92/ (NN_TARGETID.png).
Rules, on FastSpecFit fluxes, as used for the repeats92 table:
- broad: broad Halpha or Hbeta with flux S/N >= 5 and sigma >= 425 km/s (FWHM >= 1000), or MgII
  with S/N >= 5 and sigma >= 1000 km/s.
- BPT ([NII]) at S/N >= 3 in Halpha, Hbeta, [OIII]5007 and [NII]6584, Kauffmann+03 and Kewley+01
  curves of hubersed.detect.line_screen.
- WHAN: log [NII]/Halpha > -0.4 with EW Halpha > 6 A is "WHAN AGN-like"; EW Halpha < 3 A is
  "passive or retired".
- class: AGN if broad or BPT AGN; "emission line" if any of Halpha, Hbeta, [OIII]5007, [OII]3726
  has S/N >= 5 and EW >= 3 A and the object is not AGN; otherwise "no strong lines".
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-01_gallery92/lines.py``.
RESULT: pending.
FIGURES: results/2026-11-01_gallery92/lines.txt.
"""

import numpy as np
from astropy.io import fits

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
GAL = RES / "2026-11-01_gallery92"
SNMIN, BROAD_SIG, MGII_SIG, EWMIN = 5.0, 425.0, 1000.0, 3.0


def main():
    """Classify every top-level gallery92 target and write the table."""
    names = sorted(p.stem for p in GAL.glob("*.png"))
    sp = np.load(RES / "2026-11-01_repeats92" / "sparcl_spectra.npz", allow_pickle=True)
    nexp = {int(t): int(n) for t, s, p, n in zip(sp["targetid"], sp["survey"], sp["program"],
                                                 sp["coadd_numexp"], strict=True)
            if (str(s), str(p)) == ("sv3", "bright")}  # fmt: skip
    with fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits") as h:
        f, m = h["FASTSPEC"].data, h["METADATA"].data
        row = {int(t): i for i, t in enumerate(m["TARGETID"])}

        def sn(i, ln):
            iv = float(f[f"{ln}_FLUX_IVAR"][i])
            return float(f[f"{ln}_FLUX"][i]) * np.sqrt(iv) if iv > 0 else 0.0

        lines = [
            "nn TARGETID z nexp | broad Ha S/N sigma, broad Hb S/N sigma, MgII S/N sigma | "
            "log NII/Ha, log OIII/Hb, BPT | EW Ha | Dn4000 | class",
            "",
        ]
        for name in names:
            nn, tid = name.split("_")
            t = int(tid)
            i = row[t]
            bha = (sn(i, "HALPHA_BROAD"), float(f["HALPHA_BROAD_SIGMA"][i]))
            bhb = (sn(i, "HBETA_BROAD"), float(f["HBETA_BROAD_SIGMA"][i]))
            mg = (sn(i, "MGII_2803"), float(f["MGII_2803_SIGMA"][i]))
            broad = (bha[0] >= SNMIN and bha[1] >= BROAD_SIG) or (bhb[0] >= SNMIN and bhb[1] >= BROAD_SIG) \
                or (mg[0] >= SNMIN and mg[1] >= MGII_SIG)  # fmt: skip
            fl = {
                ln: float(f[f"{ln}_FLUX"][i]) for ln in ("HALPHA", "HBETA", "OIII_5007", "NII_6584")
            }
            ok = all(sn(i, ln) >= 3 and fl[ln] > 0 for ln in fl)
            n2 = (
                np.log10(fl["NII_6584"] / fl["HALPHA"])
                if fl["HALPHA"] > 0 and fl["NII_6584"] > 0
                else np.nan
            )
            o3 = (
                np.log10(fl["OIII_5007"] / fl["HBETA"])
                if fl["HBETA"] > 0 and fl["OIII_5007"] > 0
                else np.nan
            )
            bpt = "-"
            if ok:
                ke = 0.61 / (n2 - 0.47) + 1.19 if n2 < 0.47 else -np.inf
                ka = 0.61 / (n2 - 0.05) + 1.30 if n2 < 0.05 else -np.inf
                bpt = "AGN" if o3 > ke else "composite" if o3 > ka else "SF"
            ewha = float(f["HALPHA_EW"][i])
            whan = ""
            if np.isfinite(n2) and n2 > -0.4 and ewha > 6:
                whan = " (WHAN AGN-like)"
            strong = any(sn(i, ln) >= SNMIN and float(f[f"{ln}_EW"][i]) >= EWMIN
                         for ln in ("HALPHA", "HBETA", "OIII_5007", "OII_3726"))  # fmt: skip
            cls = (
                "AGN" if broad or bpt == "AGN" else "emission line" if strong else "no strong lines"
            )
            lines.append(
                f"{nn} {t} {float(m['Z'][i]):.3f} {nexp.get(t, -1)} | {bha[0]:.1f} {bha[1]:.0f}, "
                f"{bhb[0]:.1f} {bhb[1]:.0f}, {mg[0]:.1f} {mg[1]:.0f} | {n2:+.2f}, {o3:+.2f}, {bpt} | "
                f"{ewha:.1f} | {float(f['DN4000'][i]):.2f} | {cls}{whan}"
            )
    text = "\n".join(lines)
    (GAL / "lines.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
