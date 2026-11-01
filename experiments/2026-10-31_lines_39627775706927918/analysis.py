"""Emission lines of 39627775706927918, robust6 target 1 (#17): real lines, and what ionizes them.

QUESTION: are the emission features left by the C3K continuum fit real lines, what ionizes them,
and are the red-end spikes sky residuals?
HYPOTHESIS: none registered.
INPUTS: FastSpecFit iron sv3 bright (data/fastspec-iron-sv3-bright.fits, fastspecfit 2.5.1);
2026-10-31_poco_robust6 max-likelihood npz; DR1 single exposures (extract.py, <TID>_exposures.npz).
Lines: Kauffmann+03 (astro-ph/0304239, sec 3), Kewley+01 (astro-ph/0106324, [NII] line) and
WHAN classes of Cid Fernandes+11 (1012.4426, sec 5), read from the TeX.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_lines_39627775706927918/analysis.py``.
RESULT: no line detected (FastSpecFit Halpha flux 0, every line S/N < 1.5; WHAN passive, BPT not
classifiable). Every residual spike, the one near Halpha too (rest ~6551 A), is in EXPID 87710 only,
not 87469, so they are single-exposure artifacts.
Against the main bright coadd (1 exposure, 1254.5 s), median SV3/main is 0.72-0.74 for 87469
(flat in wavelength) and 1.13-1.17 for 87710, which also rises above 2 below ~4700 A observed.
Assumptions: FastSpecFit 2.5.1 fluxes and EWs as given; spikes picked at chi > 5 on the coadd.
FIGURES: results/2026-10-31_lines_39627775706927918/ lines.txt, bpt_whan.png, spikes.png,
exposures_main.png (with data/coadd-main-bright-26275.fits).
"""

# %%
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from hubersed.conversion import to_flambda
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

HERE = Path(__file__).resolve().parent
OUT = PATHS["RESULTS"] / HERE.name
TID = 39627775706927918
MAXL = PATHS["RESULTS"] / "2026-10-31_poco_robust6" / str(TID) / f"{TID}_poco_maxl.npz"
LINES = ["HBETA", "OIII_5007", "OI_6300", "HALPHA", "NII_6584", "SII_6716", "SII_6731"]
RED = "#b2182b"
SN_MIN = 3.0


def fastspec_row(tid):
    """Return the FastSpecFit FASTSPEC row of ``tid``."""
    with fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True) as h:
        i = np.flatnonzero(h["METADATA"].data["TARGETID"] == tid)
        assert i.size == 1, f"{tid}: {i.size} rows"
        return h["FASTSPEC"].data[i[0]], float(h["METADATA"].data["Z"][i[0]])


def line_table(row):
    """Return flux, flux S/N, EW and EW S/N of each line in ``LINES``."""
    out = {}
    for k in LINES:
        if f"{k}_FLUX" not in row.array.names:
            continue
        f, fi = float(row[f"{k}_FLUX"]), float(row[f"{k}_FLUX_IVAR"])
        w, wi = float(row[f"{k}_EW"]), float(row[f"{k}_EW_IVAR"])
        out[k] = dict(flux=f, sn=f * np.sqrt(fi), ew=w, ew_sn=w * np.sqrt(wi))
    return out


def kauffmann03(x):
    """Return the Kauffmann+03 SF/AGN curve, log [OIII]/Hbeta, for x = log [NII]/Halpha < 0.05."""
    return 0.61 / (x - 0.05) + 1.3


def kewley01(x):
    """Return the Kewley+01 maximum-starburst curve for x = log [NII]/Halpha < 0.47."""
    return 0.61 / (x - 0.47) + 1.19


def whan_class(x, ew_ha, ew_nii):
    """Return the Cid Fernandes+11 WHAN class from log [NII]/Halpha and the two EWs in A."""
    if ew_ha < 0.5 and ew_nii < 0.5:
        return "passive"
    if ew_ha < 3:
        return "retired"
    if x < -0.4:
        return "star-forming"
    return "strong AGN" if ew_ha > 6 else "weak AGN"


def spikes(d, rest_min=6700.0, chi_min=5.0, gap=10.0, n=4):
    """Return the rest wavelengths of the ``n`` strongest red-end chi > chi_min groups."""
    g = d["good"]
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.where(g, (d["flux"] - d["sp_best"]) / d["unc"], 0.0)
    rest = d["wave"] / (1 + float(d["z"]))
    idx = np.flatnonzero((rest > rest_min) & (chi > chi_min))
    groups = np.split(idx, np.flatnonzero(np.diff(rest[idx]) > gap) + 1) if idx.size else []
    peaks = sorted(((chi[i].max(), rest[i[np.argmax(chi[i])]]) for i in groups), reverse=True)
    return [r for _, r in peaks[:n]]


# %%
def main():
    """Write the line table, the BPT and WHAN panels and the spike panels."""
    use_apj_style()
    OUT.mkdir(parents=True, exist_ok=True)
    row, zfs = fastspec_row(TID)
    tab = line_table(row)
    with open(OUT / "lines.txt", "w") as f:
        f.write(f"# {TID} FastSpecFit iron sv3 bright, z {zfs:.5f}; flux 1e-17 erg/s/cm2, EW A\n")
        f.write(f"{'line':>10s} {'flux':>10s} {'S/N':>7s} {'EW':>8s} {'EW S/N':>7s}\n")
        for k, v in tab.items():
            f.write(f"{k:>10s} {v['flux']:10.3f} {v['sn']:7.1f} {v['ew']:8.3f} {v['ew_sn']:7.1f}\n")
    print((OUT / "lines.txt").read_text())

    ha, nii, hb, o3 = (tab[k] for k in ("HALPHA", "NII_6584", "HBETA", "OIII_5007"))
    x = np.log10(nii["flux"] / ha["flux"]) if nii["flux"] > 0 and ha["flux"] > 0 else np.nan
    bpt_ok = all(v["sn"] >= SN_MIN for v in (ha, nii, hb, o3))
    y = np.log10(o3["flux"] / hb["flux"]) if bpt_ok else np.nan
    if bpt_ok:
        bpt = (
            "AGN (above Kewley+01)" if y > kewley01(x) or x >= 0.47
            else "between Kauffmann+03 and Kewley+01" if y > kauffmann03(x) or x >= 0.05
            else "star-forming (below Kauffmann+03)"
        )  # fmt: skip
    else:
        bpt = f"not classifiable: a BPT line has S/N < {SN_MIN:g}"
    whan = whan_class(x, ha["ew"], nii["ew"])
    print(f"log [NII]/Ha {x:.3f}, log [OIII]/Hb {y:.3f}; BPT: {bpt}; "
          f"WHAN: {whan} (EW Ha {ha['ew']:.2f} A, EW [NII] {nii['ew']:.2f} A)")  # fmt: skip

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    xs = np.linspace(-2.0, 0.04, 300)
    ax[0].plot(xs, kauffmann03(xs), "k--", lw=1, label="Kauffmann+03")
    xs = np.linspace(-2.0, 0.46, 300)
    ax[0].plot(xs, kewley01(xs), "k-", lw=1, label="Kewley+01")
    if bpt_ok:
        ax[0].plot(x, y, "o", color=RED, ms=8)
    else:
        ax[0].axvline(x, color=RED, lw=1.5, label="[NII]/H$\\alpha$ only")
    ax[0].set(xlim=(-2, 1), ylim=(-1.5, 1.5), xlabel=r"log [NII]/H$\alpha$",
              ylabel=r"log [OIII]/H$\beta$", title=f"BPT: {bpt}")  # fmt: skip
    ax[0].legend(frameon=False, fontsize=9, loc="lower left")
    ax[1].axvline(-0.4, color="k", lw=1)
    ax[1].hlines(np.log10(6), -0.4, 1, color="k", lw=1)
    ax[1].axhline(np.log10(3), color="k", lw=1, ls="--")
    ax[1].axhline(np.log10(0.5), color="0.5", lw=1, ls=":")
    ax[1].plot(x, np.log10(max(ha["ew"], 1e-3)), "o", color=RED, ms=8)
    ax[1].set(xlim=(-2, 1), ylim=(-1, 2.5), xlabel=r"log [NII]/H$\alpha$",
              ylabel=r"log EW(H$\alpha$) [$\AA$]", title=f"WHAN: {whan}")  # fmt: skip
    fig.suptitle(f"{TID}  FastSpecFit iron", fontsize=11)
    fig.savefig(OUT / "bpt_whan.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # red-end spikes: coadd and model, each exposure, and each exposure's ivar
    d = np.load(MAXL)
    e = np.load(OUT / f"{TID}_exposures.npz")
    z = float(d["z"])
    rests = spikes(d)
    print("red-end spikes at rest", [round(r, 1) for r in rests], "A")
    flam = to_flambda(np.asarray(d["wave"], float) * u.AA, d["flux"] * u.mgy).value
    model = to_flambda(np.asarray(d["wave"], float) * u.AA, d["sp_best"] * u.mgy).value
    centres = [6563.0] + rests  # Halpha first, as a line known to be real
    fig, ax = plt.subplots(2, len(centres), figsize=(3.6 * len(centres), 5.5), sharex="col",
                           gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05})  # fmt: skip
    ax = np.atleast_2d(ax).reshape(2, -1)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(e["EXPID"])))
    for j, c in enumerate(centres):
        lo, hi = (c - 30) * (1 + z), (c + 30) * (1 + z)
        m = (d["wave"] > lo) & (d["wave"] < hi) & d["good"]
        ax[0, j].plot(d["wave"][m], flam[m], color="k", lw=1.2, label="coadd")
        ax[0, j].plot(d["wave"][m], model[m], color=RED, lw=1, label="C3K max-likelihood")
        for arm in "BRZ":
            w = e[f"{arm}_wave"]
            a = (w > lo) & (w < hi)
            if not a.any():
                continue
            for k, eid in enumerate(e["EXPID"]):
                good = a & (e[f"{arm}_ivar"][k] > 0) & (e[f"{arm}_mask"][k] == 0)
                ax[0, j].plot(w[good], e[f"{arm}_flux"][k][good], color=colors[k], lw=0.6,
                              label=f"EXPID {eid}" if arm == "R" or j else None)  # fmt: skip
                ax[1, j].plot(w[a], e[f"{arm}_ivar"][k][a], color=colors[k], lw=0.6)
        ax[0, j].axvline(c * (1 + z), color="0.6", lw=0.6, ls=":")
        ax[0, j].set_title(f"rest {c:.0f} $\\AA$" + (r" (H$\alpha$)" if j == 0 else ""))
        ax[1, j].set_xlabel(r"observed wavelength [$\AA$]")
    ax[0, 0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ cgs]")
    ax[1, 0].set_ylabel("ivar per exposure")
    handles, labels = ax[0, -1].get_legend_handles_labels()
    keep = dict(zip(labels, handles, strict=True))
    ax[0, -1].legend(keep.values(), keep.keys(), frameon=False, fontsize=8)
    fig.savefig(OUT / "spikes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"EXPID {e['EXPID'].tolist()}, FIBERSTATUS {e['fiberstatus_names'].tolist()}; "
          f"wrote {OUT / 'bpt_whan.png'} and {OUT / 'spikes.png'}")  # fmt: skip


# %%
def exposures_vs_main(smooth=15):
    """Plot both SV3 exposures and the main-survey coadd, and each SV3 exposure over main."""
    from scipy.signal import medfilt

    use_apj_style()
    e = np.load(OUT / f"{TID}_exposures.npz")
    with fits.open(PATHS["DATA"] / "coadd-main-bright-26275.fits", memmap=True) as h:
        fm = h["FIBERMAP"].data
        i = np.flatnonzero(fm["TARGETID"] == TID)
        assert i.size == 1, f"{TID}: {i.size} main rows"
        main = {arm: (np.asarray(h[f"{arm}_WAVELENGTH"].data, float),
                      np.asarray(h[f"{arm}_FLUX"].data[i[0]], float),
                      np.asarray(h[f"{arm}_IVAR"].data[i[0]], float),
                      np.asarray(h[f"{arm}_MASK"].data[i[0]]))
                for arm in "BRZ"}  # fmt: skip
        print(f"main: COADD_NUMEXP {fm['COADD_NUMEXP'][i[0]]}, "
              f"COADD_EXPTIME {fm['COADD_EXPTIME'][i[0]]:.1f} s")  # fmt: skip
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(e["EXPID"])))
    fig, ax = plt.subplots(2, 1, figsize=(15, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05})  # fmt: skip
    for arm in "BRZ":
        w, fmain, imain, mmain = main[arm]
        okm = (imain > 0) & (mmain == 0)
        sm = np.where(okm, fmain, np.nan)
        sm[okm] = medfilt(fmain[okm], smooth)
        ax[0].plot(w, sm, color="k", lw=0.8, label="main bright coadd" if arm == "B" else None)
        for k, eid in enumerate(e["EXPID"]):
            ok = (e[f"{arm}_ivar"][k] > 0) & (e[f"{arm}_mask"][k] == 0)
            s = np.where(ok, e[f"{arm}_flux"][k], np.nan)
            s[ok] = medfilt(np.asarray(e[f"{arm}_flux"][k][ok], float), smooth)
            ax[0].plot(e[f"{arm}_wave"], s, color=colors[k], lw=0.8,
                       label=f"SV3 EXPID {eid}" if arm == "B" else None)  # fmt: skip
            assert np.allclose(e[f"{arm}_wave"], w), "SV3 and main wavelength grids differ"
            ratio = s / sm
            ax[1].plot(w, ratio, color=colors[k], lw=0.8)
            print(f"{arm} EXPID {eid}: median SV3/main {np.nanmedian(ratio):.3f}")
    ax[1].axhline(1, color="0.3", lw=0.6)
    ax[1].set_ylim(0, 3)
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    ax[1].set_ylabel("SV3 exposure / main")
    ax[1].set_xlabel(r"observed wavelength [$\AA$]")
    ax[0].set_title(f"{TID}: DR1 SV3 bright exposures and main bright coadd, "
                    f"median filter {smooth} pixels")  # fmt: skip
    ax[0].legend(frameon=False, fontsize=10, loc="upper left")
    fig.savefig(OUT / "exposures_main.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'exposures_main.png'}")


if __name__ == "__main__":
    main()
    exposures_vs_main()
