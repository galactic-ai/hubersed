"""Where the C3K max-likelihood residuals of 39633123121563924 (robust6 target 3, #17) are bad, and why.

QUESTION: which wavelength ranges carry the excess chi2 of the pocoMC max-likelihood fit, and do
they line up with arm joins, sky lines or stellar and nebular features?
HYPOTHESIS: none registered.
INPUTS: 2026-10-31_poco_robust6/<TID>/<TID>_poco_maxl.npz. Arm joins from the DR1 exposure grid
(B 3600-5800, R 5760-7620, Z 7520-9824 A observed, 2026-10-31_lines_39627775706927918). Feature
wavelengths are standard air values from memory, not checked against a line list.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-10-31_resid_39633123121563924/analysis.py``.
RESULT: excess chi2 sits at Halpha+[NII]6584 (rest 6550-6600, chi2/N 18.6 at [NII]) and [OIII]
4959, 5007 (chi2/N 17.9); all four lines are in every one of the 4 exposures (EXPID 86986, 87714,
87828, 87932, three tiles), so they are real; exposures sit at 0.92-1.13 of the coadd. FastSpecFit:
log [NII]/Ha -0.05, log [OIII]/Hb 0.98 (Hbeta S/N 3.2), above Kewley+01 (0.01): an AGN by BPT,
missed by the S/N > 5 BPT screen. The fit has no nebular model.
Assumptions: FastSpecFit 2.5.1 fluxes; bins of 25 A rest with >= 10 good pixels.
FIGURES: results/2026-10-31_resid_39633123121563924/ resid.txt, resid.png, worst_bins_exposures.png,
exposure_ratio.png.
"""

# %%
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

from hubersed.conversion import to_flambda
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39633123121563924
OUT = PATHS["RESULTS"] / "2026-10-31_resid_39633123121563924"
MAXL = PATHS["RESULTS"] / "2026-10-31_poco_robust6" / str(TID) / f"{TID}_poco_maxl.npz"
BIN = 25.0  # rest A
JOINS = [(5760.0, 5800.0), (7520.0, 7620.0)]  # observed A, arm overlaps
FEATURES = {
    "[OII]3727": 3727.4, "CaII K": 3933.7, "CaII H": 3968.5, "Hdelta": 4101.7, "G band": 4304.4,
    "Hgamma": 4340.5, "Hbeta": 4861.3, "[OIII]5007": 5006.8, "Mg b": 5175.0, "Fe5270": 5270.0,
    "Fe5335": 5335.0, "NaD": 5893.0, "[OI]6300": 6300.3, "Halpha": 6562.8, "[NII]6584": 6583.5,
    "[SII]6717": 6716.4, "[SII]6731": 6730.8, "TiO 7100": 7100.0, "CaT 8498": 8498.0,
    "CaT 8542": 8542.1, "CaT 8662": 8662.1,
}  # fmt: skip
SKY = {"OI 5577": 5577.3, "NaI 5890": 5891.6, "OI 6300": 6300.3, "OI 6364": 6363.8}  # observed A
RED = "#b2182b"


def main():
    """Bin chi2 in rest wavelength, tag each bin, write the table and the figure."""
    use_apj_style()
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(MAXL)
    z, g = float(d["z"]), d["good"]
    wave = np.asarray(d["wave"], float)
    rest = wave / (1 + z)
    chi = np.where(g, (d["flux"] - d["sp_best"]) / d["unc"], np.nan)
    ivar = np.where(g, 1 / d["unc"] ** 2, np.nan)
    # an ivar dip below 0.7 of the local median marks a sky line
    dip = np.where(g, ivar < 0.7 * median_filter(np.nan_to_num(ivar), 51), False)
    edges = np.arange(np.floor(rest[g].min() / BIN) * BIN, rest[g].max() + BIN, BIN)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = g & (rest >= lo) & (rest < hi)
        if m.sum() < 10:
            continue
        olo, ohi = lo * (1 + z), hi * (1 + z)
        tags = [k for k, v in FEATURES.items() if lo - 5 <= v < hi + 5]
        tags += [f"sky {k}" for k, v in SKY.items() if olo - 5 <= v < ohi + 5]
        tags += ["arm join" for a, b in JOINS if olo < b and ohi > a]
        rows.append(dict(lo=lo, hi=hi, olo=olo, ohi=ohi, n=int(m.sum()),
                         chi2n=float(np.mean(chi[m] ** 2)), mean=float(np.mean(chi[m])),
                         dip=float(dip[m].mean()), tags=", ".join(tags)))  # fmt: skip
    total = np.nansum(chi**2)
    order = sorted(rows, key=lambda r: -(r["chi2n"] - 1) * r["n"])
    lines = [f"# {TID} z {z:.5f}: chi2 {total:.1f} on {g.sum()} px, chi2/N {total / g.sum():.3f};"
             f" {BIN:g} A rest bins sorted by excess chi2 = (chi2/N - 1) * N",
             f"{'rest A':>11s} {'obs A':>11s} {'N':>4s} {'chi2/N':>7s} {'<chi>':>6s} "
             f"{'excess':>7s} {'dip':>5s}  tags"]  # fmt: skip
    for r in order[:15]:
        lines.append(f"{r['lo']:5.0f}-{r['hi']:5.0f} {r['olo']:5.0f}-{r['ohi']:5.0f} {r['n']:4d} "
                     f"{r['chi2n']:7.2f} {r['mean']:6.2f} {(r['chi2n'] - 1) * r['n']:7.1f} "
                     f"{r['dip']:5.2f}  {r['tags']}")  # fmt: skip
    (OUT / "resid.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    flam = to_flambda(wave * u.AA, d["flux"] * u.mgy).value
    model = to_flambda(wave * u.AA, d["sp_best"] * u.mgy).value
    mid = np.array([(r["lo"] + r["hi"]) / 2 for r in rows])
    fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1, 1], "hspace": 0.05})  # fmt: skip
    ax[0].plot(rest, np.where(g, flam, np.nan), color="0.6", lw=0.5, label="DR1 coadd")
    ax[0].plot(rest, np.where(g, model, np.nan), color=RED, lw=0.8, label="C3K max-likelihood")
    ax[0].legend(frameon=False, loc="upper right")
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ cgs]")
    ax[1].step(mid, [r["chi2n"] for r in rows], where="mid", color="k", lw=1)
    ax[1].axhline(1, color="0.5", lw=0.6)
    ax[1].set_ylabel(rf"$\chi^2/N$ per {BIN:g} $\AA$")
    ax[2].step(mid, [r["mean"] for r in rows], where="mid", color=RED, lw=1)
    ax[2].axhline(0, color="0.5", lw=0.6)
    ax[2].set_ylabel(r"mean $\chi$ per bin")
    for k, v in FEATURES.items():
        for a in ax:
            a.axvline(v, color="tab:blue", lw=0.5, ls=":")
        ax[1].text(v, ax[1].get_ylim()[1], k, rotation=90, fontsize=6, va="top", color="tab:blue")
    for a, b in JOINS:
        for x in ax:
            x.axvspan(a / (1 + z), b / (1 + z), color="0.85", lw=0)
    ax[-1].set_xlabel(r"rest wavelength [$\AA$]")
    ax[0].set_title(f"{TID}  z {z:.4f}; grey bands: arm joins; dotted: features")
    fig.savefig(OUT / "resid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'resid.txt'} and {OUT / 'resid.png'}")
    return order


# %%
def exposures(order, n=4, smooth=15):
    """Plot each DR1 exposure at the ``n`` worst bins, and each exposure over the coadd."""
    from scipy.signal import medfilt

    use_apj_style()
    d = np.load(MAXL)
    e = np.load(OUT / f"{TID}_exposures.npz")
    wave = np.asarray(d["wave"], float)
    flam = to_flambda(wave * u.AA, d["flux"] * u.mgy).value
    model = to_flambda(wave * u.AA, d["sp_best"] * u.mgy).value
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(e["EXPID"])))
    print(f"EXPID {e['EXPID'].tolist()}, TILEID {e['TILEID'].tolist()}, NIGHT {e['NIGHT'].tolist()}, "
          f"FIBERSTATUS {e['fiberstatus_names'].tolist()}")  # fmt: skip
    fig, ax = plt.subplots(2, n, figsize=(3.8 * n, 5.5), sharex="col",
                           gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05})  # fmt: skip
    for j, r in enumerate(order[:n]):
        lo, hi = r["olo"] - 25, r["ohi"] + 25
        m = (wave > lo) & (wave < hi) & d["good"]
        ax[0, j].plot(wave[m], flam[m], color="k", lw=1.2, label="coadd")
        ax[0, j].plot(wave[m], model[m], color=RED, lw=1, label="C3K max-likelihood")
        for arm in "BRZ":
            w = e[f"{arm}_wave"]
            a = (w > lo) & (w < hi)
            for k, eid in enumerate(e["EXPID"]):
                ok = a & (e[f"{arm}_ivar"][k] > 0) & (e[f"{arm}_mask"][k] == 0)
                ax[0, j].plot(w[ok], e[f"{arm}_flux"][k][ok], color=colors[k], lw=0.6,
                              label=f"EXPID {eid}" if j == 0 and arm == "B" else None)  # fmt: skip
                ax[1, j].plot(w[a], e[f"{arm}_ivar"][k][a], color=colors[k], lw=0.6)
        ax[0, j].axvspan(r["olo"], r["ohi"], color="0.9", lw=0)
        ax[0, j].set_title(f"rest {r['lo']:.0f}-{r['hi']:.0f} $\\AA$", fontsize=10)
        ax[1, j].set_xlabel(r"observed wavelength [$\AA$]")
    ax[0, 0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ cgs]")
    ax[1, 0].set_ylabel("ivar per exposure")
    ax[0, 0].legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "worst_bins_exposures.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 3.5))
    for arm in "BRZ":
        w = e[f"{arm}_wave"]
        g = d["good"]
        ref = np.interp(
            w, wave[g], flam[g], left=np.nan, right=np.nan
        )  # blank outside the fit range
        for k, eid in enumerate(e["EXPID"]):
            ok = (e[f"{arm}_ivar"][k] > 0) & (e[f"{arm}_mask"][k] == 0)
            s = np.full(w.size, np.nan)
            s[ok] = medfilt(np.asarray(e[f"{arm}_flux"][k][ok], float), smooth)
            ratio = s / medfilt(ref, smooth)
            ax.plot(w, ratio, color=colors[k], lw=0.7, label=f"EXPID {eid}" if arm == "B" else None)
            print(f"{arm} EXPID {eid}: median exposure/coadd {np.nanmedian(ratio):.3f}")
    ax.axhline(1, color="0.4", lw=0.6)
    ax.set_ylim(0, 2)
    ax.set_xlabel(r"observed wavelength [$\AA$]")
    ax.set_ylabel("exposure / coadd")
    ax.set_title(f"{TID}: each exposure over the coadd, median filter {smooth} pixels")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUT / "exposure_ratio.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'worst_bins_exposures.png'} and {OUT / 'exposure_ratio.png'}")


# %%
def fastspec_bpt():
    """Print the FastSpecFit BPT lines of the target and where it falls against Kewley+01."""
    from astropy.io import fits

    with fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True) as h:
        i = np.flatnonzero(h["METADATA"].data["TARGETID"] == TID)[0]
        r = h["FASTSPEC"].data[i]
    v = {}
    for k in ("HALPHA", "NII_6584", "HBETA", "OIII_5007"):
        v[k] = float(r[f"{k}_FLUX"])
        print(f"{k}: flux {v[k]:.2f}, S/N {v[k] * np.sqrt(r[f'{k}_FLUX_IVAR']):.1f}, "
              f"EW {r[f'{k}_EW']:.2f} A, sigma {r[f'{k}_SIGMA']:.0f} km/s")  # fmt: skip
    x, y = np.log10(v["NII_6584"] / v["HALPHA"]), np.log10(v["OIII_5007"] / v["HBETA"])
    print(
        f"log [NII]/Ha {x:.3f}, log [OIII]/Hb {y:.3f}, Kewley+01 at x {0.61 / (x - 0.47) + 1.19:.3f}"
    )


if __name__ == "__main__":
    worst = main()
    if (OUT / f"{TID}_exposures.npz").exists():
        exposures(worst)
    fastspec_bpt()
