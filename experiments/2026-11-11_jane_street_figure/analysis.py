"""Three-box figure for the Jane Street statement, one box per year, in the style of the FINESST one.

QUESTION: Can one compact figure show the better model (Year 1), the posterior the fast network must
match (Year 2) and the DESI sample (Year 3), using results only?
HYPOTHESIS: None. This is a figure, not a test.
INPUTS: the #14 export of 39627770174637084 (2026-10-18_issue14_figures/export, MILES + Cue against
UberSED free_dust, chi2_red as exported, not recomputed), posterior_free_dust.npz of
export_posterior.py (run that first), the METADATA of data/fastspec-iron-sv3-bright.fits, and the
NOIRLab DESI Year-3 slice noirlab2512b (screen JPEG, CC BY 4.0, credit DESI Collaboration/DOE/KPNO/
NOIRLab/NSF/AURA/R. Proctor), downloaded to desi_slice/ with
``curl -sL -o <out>/desi_slice/noirlab2512b_screen.jpg
https://storage.noirlab.edu/media/archives/images/screen/noirlab2512b.jpg``.
SEED: none.
COMMAND: from the repository root on the Mac,
``uv run python experiments/2026-11-11_jane_street_figure/analysis.py``.
RESULT: 2026-11-11. Both versions made; Nikhil chose figure_image for statement_v10 (Figure 1).
FIGURES: results/2026-11-11_jane_street_figure/figure_{wedge,image}.{pdf,png}, two Year 3 options.
"""

# %%
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from hubersed.conversion import to_flambda
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39627770174637084
RES = PATHS["RESULTS"]
OUT = RES / Path(__file__).resolve().parent.name
EXPORT = RES / "2026-10-18_issue14_figures" / "export"
FASTSPEC = PATHS["DATA"] / "fastspec-iron-sv3-bright.fits"
CORNER = {
    "logmass": r"$\log M_\star$",
    "logzsol": r"$\log Z_\star$",
    "gas_logz": r"$\log Z_{\rm gas}$",
}
SLICE = OUT / "desi_slice" / "noirlab2512b_screen.jpg"
CREDIT = "Image: DESI Collaboration/DOE/KPNO/\nNOIRLab/NSF/AURA/R. Proctor (CC BY 4.0)"
COLORS = {"data": "0.45", "sota": "C1", "uber": "C0"}


# %%
def load_export(name):
    """Return the #14 export ``name`` (miles_cue or ubersed) as a dict."""
    d = np.load(EXPORT / f"emission_line_{TID}_{name}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def flam(wave, maggies):
    """Return maggies as f_lambda in DESI units."""
    return to_flambda(wave * u.AA, maggies * u.mgy).value


def posterior():
    """Return the weighted free_dust posterior of the CORNER parameters."""
    d = np.load(OUT / "posterior_free_dust.npz")
    assert list(d["names"]) == list(CORNER), d["names"]
    return d["theta"], d["w"]


def bgs_positions():
    """Return RA and z of SV3 BGS galaxies with a good redshift."""
    m = fits.getdata(FASTSPEC, "METADATA")
    good = (m["ZWARN"] == 0) & (m["SPECTYPE"] == "GALAXY") & (m["SV3_BGS_TARGET"] > 0)
    good &= (m["Z"] > 0.01) & (m["Z"] < 0.6)
    return np.asarray(m["RA"][good], float), np.asarray(m["Z"][good], float)


# %%
def year1(sub, sota, uber):
    """Draw the DESI spectrum with both best fits and their chi."""
    ax, axr = sub.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0})
    sub.subplots_adjust(left=0.11, right=0.98, bottom=0.19, top=0.87)
    rest = uber["wave_rest"]
    data = np.where(np.isfinite(uber["flux"]), uber["flux"], np.nan)
    ax.plot(rest, flam(uber["wave_obs"], data), color=COLORS["data"], lw=0.4, label="DESI spectrum")
    for fitd, key, name in ((sota, "sota", "SoTA model"), (uber, "uber", "UberSED")):
        inside = np.isfinite(fitd["flux"])
        lo, hi = rest[inside].min(), rest[inside].max()
        span = (rest >= lo) & (rest <= hi)
        label = rf"{name} ($\chi^2_\nu = {float(fitd['chi2_red']):.2f}$)"
        ax.plot(
            rest[span],
            flam(fitd["wave_obs"][span], fitd["model"][span]),
            color=COLORS[key],
            lw=0.6,
            label=label,
        )
        axr.plot(rest, fitd["chi"], color=COLORS[key], lw=0.3)
    cont = np.nanpercentile(flam(uber["wave_obs"], data), [1, 99])
    ax.set_ylim(0, 1.6 * cont[1])
    ax.set_yticks([])
    ax.set_ylabel(r"$f_\lambda$")
    leg = ax.legend(loc="upper right", fontsize=5.5, frameon=True, framealpha=1, edgecolor="none")
    leg.set_zorder(5)
    axr.set_ylim(-5, 5)
    for y in (-2, 2):
        axr.axhline(y, color="0.3", ls="--", lw=0.5)
    axr.set_ylabel(r"$\chi$")
    axr.set_xlabel(r"rest-frame wavelength [\AA]")
    sub.suptitle("Year 1: a better model", fontsize=9)


def year2(sub, theta, w):
    """Draw a small corner plot of the Bayesian posterior."""
    import corner

    k = len(CORNER)
    axes = sub.subplots(k, k, gridspec_kw={"hspace": 0.08, "wspace": 0.08})
    sub.subplots_adjust(left=0.31, right=0.95, bottom=0.24, top=0.8)
    names = list(CORNER.values())
    rng = [tuple(np.quantile(theta[:, i], [0.002, 0.998])) for i in range(k)]
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(theta[:, i], bins=30, weights=w, range=rng[i], histtype="step", color="C0")
                ax.set_yticks([])
            else:
                corner.hist2d(
                    theta[:, j],
                    theta[:, i],
                    weights=w,
                    ax=ax,
                    range=[rng[j], rng[i]],
                    levels=(0.393, 0.865),
                    smooth=1.0,
                    bins=30,
                    color="C0",
                    plot_datapoints=False,
                    fill_contours=True,
                )
            ax.set_xlim(rng[j])
            ax.xaxis.set_major_locator(MaxNLocator(2))
            if i != j:
                ax.set_ylim(rng[i])
                ax.yaxis.set_major_locator(MaxNLocator(2))
            ax.tick_params(labelsize=5)
            ax.tick_params(axis="x", labelrotation=45)
            if i < k - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(names[j], fontsize=7)
            if j > 0 or i == 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(names[i], fontsize=7)
                ax.yaxis.set_label_coords(-1.05, 0.5)
    sub.suptitle("Year 2: fast inference\n(benchmark posterior)", fontsize=9)


def year3_wedge(sub, ra, z):
    """Draw the SV3 BGS galaxies as a wedge in RA and redshift."""
    ax = sub.add_subplot(projection="polar")
    sub.subplots_adjust(left=0.02, right=0.98, bottom=0.22, top=0.84)
    ax.scatter(np.deg2rad(ra), z, s=0.05, color="C0", alpha=0.3, lw=0, rasterized=True)
    ax.set_thetamin(ra.min())
    ax.set_thetamax(ra.max())
    ax.set_rmax(0.4)
    ax.set_rticks([0.1, 0.2, 0.3, 0.4])
    ax.set_rlabel_position(ra.max() + 3)
    ax.set_xticklabels([])
    ax.tick_params(labelsize=5)
    ax.text(
        0.5,
        -0.05,
        f"{len(z):,} DESI galaxies\nof ten million to come",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
    )
    sub.suptitle("Year 3: DESI", fontsize=9)


def year3_image(sub, n):
    """Show the centre of the NOIRLab DESI Year-3 slice with its credit."""
    ax = sub.add_subplot()
    sub.subplots_adjust(left=0.04, right=0.96, bottom=0.25, top=0.84)
    ax.imshow(plt.imread(SLICE)[:, 150:1130])
    ax.axis("off")
    ax.text(
        0.5,
        -0.04,
        "255,000 DESI spectra in hand,\nten million to come",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
    )
    ax.text(0.5, -0.36, CREDIT, transform=ax.transAxes, ha="center", va="top", fontsize=4)
    sub.suptitle("Year 3: DESI", fontsize=9)


# %%
def main():
    """Build the figure twice, with the wedge and with the NOIRLab image for Year 3."""
    use_apj_style()
    OUT.mkdir(parents=True, exist_ok=True)
    sota, uber = load_export("miles_cue"), load_export("ubersed")
    theta, w = posterior()
    ra, z = bgs_positions()
    for name in ("wedge", "image"):
        fig = plt.figure(figsize=(6.5, 2.5))
        subs = fig.subfigures(1, 3, width_ratios=[2.3, 1.2, 1.3], wspace=0.02)
        for s in subs:
            box = Rectangle((0.01, 0.01), 0.98, 0.98, transform=s.transSubfigure, fill=False)
            box.set(linestyle="--", linewidth=0.8, edgecolor="k")
            s.patches.append(box)
        year1(subs[0], sota, uber)
        year2(subs[1], theta, w)
        if name == "wedge":
            year3_wedge(subs[2], ra, z)
        else:
            year3_image(subs[2], len(z))
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"figure_{name}.{ext}", dpi=300)
        plt.close(fig)
        print(OUT / f"figure_{name}.png")


if __name__ == "__main__":
    main()
