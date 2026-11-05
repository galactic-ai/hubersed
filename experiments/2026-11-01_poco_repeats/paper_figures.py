"""Figures of the deep dive on one quiescent target: metallicity, dust and mass, SFH prior, summary.

QUESTION: how do the ladder runs compare with each other, with the DR1 VACs, with the spectrum's
own Halpha and Dn4000, and with literature scaling relations at the target's sigma?
Figures:
1. fig_metallicity_plane: [Fe/H] against [alpha/Fe] for the afe runs; solar-scaled runs at
   afe = 0. Grey lines of constant log Z/Zsun come from MIST2 (mist2_zh.py). Literature points are
   at sigma 140 and 155 km/s, put on this plane through the same MIST2 relation:
   - Thomas+2010 (arXiv:0912.0259) Eqs. 1-2;
   - McDermid+2015 (arXiv:1501.03723) Table "Line fit parameters", rows Re/2 and Re;
   - Conroy+2014 (arXiv:1303.6629) Table 2, stacks log sigma 2.14 and 2.23, [Fe/H] and [Mg/Fe].
2. fig_dust_mass: overlaid corner of A_V, dust_index and logmass, as ladder_corner.py.
3. fig_sfh_prior: log sigma_dyn against logsfr_ratios_1-3 for the DR1 and recoadd posteriors,
   with the prior draws of derived.py in grey.
4. fig_summary, one panel per quantity:
   - stellar mass: ours (spectrum aperture and scaled to total by the r-band flux ratio of
     photometry_check.py), FastSpecFit v3.0 (raw and h-corrected with h_eff of fsf_h_check.txt),
     CIGALE;
   - sigma;
   - A_V, with the Dias-Garcia+2019 (arXiv:1802.06813) median range for quiescent galaxies;
   - total metallicity, with the relations above;
   - age, with literature z = 0 ages minus the Planck18 lookback time;
   - SFR, with the EmFit Halpha upper limit through Kennicutt & Evans 2012 (log C = 41.27,
     Table 1, Kroupa IMF) and the Planck18 luminosity distance.
5. fig_checks: Dn4000 and g-r of data, models and FastSpecFit (index_check.py, photometry_check.py).
Literature numbers are hard-coded from the TeX locations named in the code; VAC numbers are read
from vac_rows.txt. Our numbers come from the same posteriors and draws as ladder_corner.py and
derived.py.
COMMAND: ``uv run --no-sync python experiments/2026-11-01_poco_repeats/paper_figures.py --tid <TID>``.
OUTPUT: results/2026-11-01_poco_repeats/<TID>/paper/fig_*.png.
"""

# %%
import argparse
import importlib.util
import re
import sys
from pathlib import Path

import astropy.units as u
import corner
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

HERE = Path(__file__).resolve().parent
RES = PATHS["RESULTS"] / HERE.name
SIGMAS = (140.0, 155.0)
SHORT = ["MILES", "C3K HR", "C3K HR+afe", "DR1 R matrix", "recoadd"]


def load(name, path):
    """Load the file at ``path`` as module ``name``."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def literature(sigma):
    """Return literature [Z/H], [alpha/Fe] and age (Gyr, z = 0) at ``sigma`` km/s."""
    ls = np.log10(sigma)
    out = {
        # Thomas+2010 Eqs. 1-2 (0912.0259 lines 420-422); observed scatter 0.23, 0.11, 0.07 dex
        "Thomas+10": dict(zh=-1.34 + 0.65 * ls, afe=-0.55 + 0.33 * ls, age=10 ** (-0.11 + 0.47 * ls),
                          szh=0.11, safe=0.07),
        # McDermid+2015 Table "Line fit parameters" (i), SSP, rows Re/2 and Re; rms in the table
        "McDermid+15 Re/2": dict(zh=0.45 * ls - 1.09, afe=0.33 * ls - 0.49, age=10 ** (0.93 * ls - 1.07),
                                 szh=0.10, safe=0.07),
        "McDermid+15 Re": dict(zh=0.45 * ls - 1.16, afe=0.31 * ls - 0.44, age=10 ** (0.83 * ls - 0.83),
                               szh=0.10, safe=0.08),
    }  # fmt: skip
    return out


# Conroy+2014 Table 2 (1303.6629 lines 503-504): log sigma, age Gyr, [Fe/H], [Mg/Fe]; systematic 0.05
CGVD14 = [(2.14, 6.8, -0.04, 0.09), (2.23, 7.0, -0.03, 0.12)]
DIAS_GARCIA_AV = (0.15, 0.30)  # 1802.06813: median A_V of quiescent galaxies, all below 0.6
KE12_LOGC = 41.27  # 1204.3552 table1.tex, Halpha row


def feh_at_total(logz, total, afe):
    """Return [Fe/H] that gives total log Z/Zsun ``total`` at ``afe`` on the MIST2 grid."""
    feh = np.linspace(-1.5, 0.5, 2001)
    return float(np.interp(total, logz(feh, np.full_like(feh, afe)), feh))


def q(v, w=None):
    """Return the 16, 50 and 84 per cent (weighted) quantiles of ``v``."""
    if w is None:
        return np.percentile(v, [16, 50, 84])
    o = np.argsort(v)
    c = np.cumsum(w[o]) / np.sum(w)
    return np.interp([0.16, 0.5, 0.84], c, v[o])


def fig_metallicity(rr, logz, out):
    """Draw the [Fe/H]-[alpha/Fe] plane."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    afe_grid = np.linspace(-0.2, 0.6, 50)
    for tot in (-0.4, -0.2, 0.0, 0.2, 0.4):
        ax.plot(
            [feh_at_total(logz, tot, a) for a in afe_grid], afe_grid, color="0.7", ls="--", lw=1
        )
        ax.text(
            feh_at_total(logz, tot, 0.58),
            0.585,
            f"{tot:+.1f}",
            color="0.5",
            fontsize=9,
            ha="center",
        )
    for (_, colour, _, cols, w, _), name in zip(rr, SHORT, strict=True):
        if "afe" in cols:
            corner.hist2d(cols["logzsol"], cols["afe"], weights=w, ax=ax, color=colour, levels=(0.393, 0.865),
                          plot_datapoints=False, plot_density=False, no_fill_contours=True, fill_contours=False, smooth=1.0,
                          contour_kwargs={"linewidths": 2})  # fmt: skip
            ax.plot([], [], color=colour, lw=2, label=name)
        else:
            lo, m, hi = q(cols["logzsol"], w)
            ax.errorbar(m, 0.0, xerr=[[m - lo], [hi - m]], fmt="o", color=colour, ms=7, capsize=3,
                        label=f"{name} (solar-scaled)")  # fmt: skip
    marks = {"Thomas+10": "s", "McDermid+15 Re/2": "^", "McDermid+15 Re": "v"}
    for key, mk in marks.items():
        pts = [literature(s)[key] for s in SIGMAS]
        xs = [feh_at_total(logz, p["zh"], p["afe"]) for p in pts]
        ax.errorbar(xs, [p["afe"] for p in pts], yerr=[p["safe"] for p in pts], fmt=mk, color="k",
                    mfc="none", ms=8, capsize=2, label=f"{key}, sigma 140-155")  # fmt: skip
    ax.errorbar([c[2] for c in CGVD14], [c[3] for c in CGVD14], xerr=0.05, yerr=0.05, fmt="D", color="k",
                ms=7, capsize=2, label="Conroy+14 stacks, sigma 138-170")  # fmt: skip
    ax.set_xlabel("[Fe/H] (logzsol of the afe runs)")
    ax.set_ylabel("[alpha/Fe]")
    ax.set_xlim(-0.9, 0.55)
    ax.set_ylim(-0.1, 0.62)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    ax.set_title("dashed: constant total log Z/Zsun (MIST2)", fontsize=10)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_dust_mass(rr, names, out):
    """Draw a small overlaid corner of ``names`` for every run, contours unfilled."""
    k = len(names)
    fig, axes = plt.subplots(k, k, figsize=(8, 8), gridspec_kw={"wspace": 0.08, "hspace": 0.08})
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.set_axis_off()
                continue
            for _, colour, _, cols, w, _ in rr:
                if i == j:
                    ax.hist(cols[names[j]], bins=30, weights=w, density=True, histtype="step", color=colour,
                            lw=1.8)  # fmt: skip
                else:
                    corner.hist2d(cols[names[j]], cols[names[i]], weights=w, ax=ax, color=colour, smooth=1.0,
                                  levels=(0.393, 0.865), plot_datapoints=False, plot_density=False,
                                  no_fill_contours=True, fill_contours=False,
                                  contour_kwargs={"linewidths": 1.8})  # fmt: skip
            if i == j:
                ax.set_yticks([])
            if i < k - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(names[j])
            if j > 0:
                ax.set_yticklabels([])
            elif i > 0:
                ax.set_ylabel(names[i])
    from matplotlib.lines import Line2D

    fig.legend([Line2D([], [], color=c, lw=3) for _, c, *_ in rr], SHORT, loc="upper right", fontsize=11,
               frameon=False, bbox_to_anchor=(0.95, 0.92))  # fmt: skip
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_sfh_prior(rr, tid, out):
    """Draw log sigma_dyn against the youngest SFR ratios, posterior against prior."""
    d = RES / str(tid) / "derived"
    keys = {
        "C3K ER + afe + R matrix, DR1 coadd": "dr1",
        "C3K ER + afe + R matrix, recoadd": "recoadd",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey=True)
    for row, (label, colour, _, cols, w, _) in enumerate(r for r in rr if r[0] in keys):
        dr = np.load(d / f"draws_{keys[label]}.npz")
        labels = list(dr["labels"])
        prior = dr["theta_prior"]
        for j in range(3):
            ax = axes[row, j]
            p = f"logsfr_ratios_{j + 1}"
            ax.scatter(np.log10(prior[:, labels.index("sigma_dyn")]), prior[:, labels.index(p)], s=3,
                       color="0.75", rasterized=True)  # fmt: skip
            corner.hist2d(np.log10(cols["sigma_dyn"]), cols[p], weights=w, ax=ax, color=colour,
                          levels=(0.393, 0.865), plot_datapoints=False, plot_density=False, no_fill_contours=True,
                          fill_contours=False, smooth=1.0, contour_kwargs={"linewidths": 2})  # fmt: skip
            ax.set_ylim(-4, 4)
            if row == 1:
                ax.set_xlabel("log sigma_dyn")
            if j == 0:
                ax.set_ylabel(f"{SHORT[3] if row == 0 else SHORT[4]}\nlogsfr_ratios")
            ax.set_title(p, fontsize=10)
    fig.suptitle(
        "grey: 1000 prior draws (derived.py); contours: posterior 39 and 86 per cent", fontsize=10
    )
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def vac_values(tid):
    """Return VAC numbers parsed from vac_rows.txt and h_eff from fsf_h_check.txt."""
    t = (RES / str(tid) / "vac_rows.txt").read_text()

    def num(pat):
        m = re.search(pat, t)
        assert m, pat
        return [float(x) for x in m.groups()]

    v = {
        "LOGMSTAR": num(r"LOGMSTAR\s+([\d.]+) \+-\s+([\d.]+)"),
        "AGE": num(r"AGE\s+([\d.]+) \+-\s+([\d.]+)"),
        "SFR": num(r"SFR\s+([\d.]+) \+-\s+([\d.]+)"),
        "VDISP": num(r"VDISP\s+([\d.]+) \+-\s+([\d.]+)"),
        "A_V": num(r"A_V\s+([\d.]+) \+-\s+([\d.]+)"),
        "DN4000": num(r"  DN4000\s+([\d.]+) \+-\s+([\d.]+)"),
        "DN4000_MODEL": num(r"DN4000_MODEL\s+([\d.]+) \+-\s+([\d.]+)"),
        "LOGM": num(r"LOGM\s+([\d.]+) \+- ([\d.]+)"),
        "LOGSFR": num(r"LOGSFR (-?[\d.]+) \+- ([\d.]+)"),
    }
    h = re.search(r"h_eff = ([\d.]+)", (RES / "fsf_h_check.txt").read_text())
    v["h_eff"] = float(h.group(1))
    return v


def fig_summary(rr, tid, logz, vac, flux_ratio, out):
    """Draw one panel per quantity: ours, VACs, Halpha and literature."""
    d = RES / str(tid) / "derived"
    keys = ["miles", "c3k_hr", "c3k_hr_afe", "dr1", "recoadd"]
    draws = {k: np.load(d / f"draws_{k}.npz") for k in keys}
    z = float(draws["dr1"]["z"])
    t_obs = float(draws["dr1"]["t_obs"])
    lookback = Planck18.lookback_time(z).to_value(u.Gyr)
    colours = [r[1] for r in rr]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ax = axes.ravel()

    def row(a, y, lo, m, hi, colour, label, marker="o", upper=False):
        if upper:
            a.errorbar(
                m,
                y,
                xerr=[[m * 0.5], [0]],
                xuplims=False,
                xlolims=False,
                fmt="<",
                color=colour,
                ms=8,
            )
        else:
            a.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt=marker, color=colour, ms=7, capsize=3)
        a.text(1.02, y, label, transform=a.get_yaxis_transform(), fontsize=8, va="center")

    # (a) stellar mass
    a, y = ax[0], 0
    for (_, colour, _, cols, w, _), name in zip(rr, SHORT, strict=True):
        lo, m, hi = q(cols["logmass"], w)
        row(a, y, lo, m, hi, colour, f"{name}: formed, spectrum aperture")
        y += 1
    lo, m, hi = q(rr[3][3]["logmass"], rr[3][4]) + np.log10(flux_ratio)
    row(a, y, lo, m, hi, colours[3], f"DR1 formed, x{flux_ratio:.2f} r-flux to total")
    y += 1
    lm, e = vac["LOGMSTAR"]
    row(a, y, lm - e, lm, lm + e, "k", "FastSpecFit v3.0 raw (h=1, surviving, total)", "s")
    y += 1
    dh = -2 * np.log10(vac["h_eff"])
    row(
        a,
        y,
        lm - e + dh,
        lm + dh,
        lm + e + dh,
        "k",
        f"FastSpecFit, + {dh:.2f} (h_eff {vac['h_eff']:.3f})",
        "D",
    )
    y += 1
    lm, e = vac["LOGM"]
    row(a, y, lm - e, lm, lm + e, "0.4", "CIGALE DR1 (surviving?, total, BC03)", "^")
    a.set_xlabel("log M* [Msun]")

    # (b) sigma
    a, y = ax[1], 0
    for (_, colour, _, cols, w, _), name in zip(rr, SHORT, strict=True):
        lo, m, hi = q(cols["sigma_smooth"], w)
        row(a, y, lo, m, hi, colour, name)
        y += 1
    v, e = vac["VDISP"]
    row(a, y, v - e, v, v + e, "k", "FastSpecFit v3.0 VDISP", "s")
    a.set_xlabel("sigma* [km/s]")

    # (c) A_V
    a, y = ax[2], 0
    a.axvspan(*DIAS_GARCIA_AV, color="0.9")
    for (_, colour, _, cols, w, _), name in zip(rr, SHORT, strict=True):
        lo, m, hi = q(cols["A_V"], w)
        row(a, y, lo, m, hi, colour, name)
        y += 1
    v, e = vac["A_V"]
    row(a, y, max(v - e, 0), v, v + e, "k", "FastSpecFit v3.0 (1.086 TAUV, MW removed)", "s")
    a.set_xlabel("A_V [mag] (ours include Milky Way dust); grey: quiescent median range")

    # (d) total metallicity
    a, y = ax[3], 0
    for (_, colour, _, cols, w, _), name in zip(rr, SHORT, strict=True):
        tot = logz(cols["logzsol"], cols.get("afe", np.zeros_like(cols["logzsol"])))
        lo, m, hi = q(tot, w)
        row(a, y, lo, m, hi, colour, name)
        k = keys[y]
        s_lo, s_m, s_hi = q(draws[k]["post_sfr100"])
        print(f"{name:14s} log Z/Zsun {m:+.3f} -{m - lo:.3f} +{hi - m:.3f}; t_age "
              f"{np.median(draws[k]['post_t_age']):.2f} Gyr; SFR100 {s_m:.3f} -{s_m - s_lo:.3f} "
              f"+{s_hi - s_m:.3f} Msun/yr")  # fmt: skip
        y += 1
    for key, mk in (("Thomas+10", "s"), ("McDermid+15 Re/2", "^"), ("McDermid+15 Re", "v")):
        vals = [literature(s)[key]["zh"] for s in SIGMAS]
        sc = literature(SIGMAS[0])[key]["szh"]
        m = np.mean(vals)
        row(a, y, min(vals) - sc, m, max(vals) + sc, "k", f"{key} [Z/H], sigma 140-155", mk)
        y += 1
    a.set_xlabel("total log Z/Zsun\n(ours: MIST2, Zsun 0.016358; literature: own scales)")

    # (e) ages
    a, y = ax[4], 0
    a.axvline(t_obs, color="r", lw=1)
    a.axvspan(4.26, 10.11, color="0.93")
    for k, colour, name in zip(keys, colours, SHORT, strict=True):
        lo, m, hi = q(draws[k]["post_t_age"])
        row(a, y, lo, m, hi, colour, f"{name}: t_age (mass-weighted)")
        y += 1
    lo, m, hi = q(draws["dr1"]["prior_t_age"])
    row(a, y, lo, m, hi, "0.6", "DR1 prior: t_age")
    y += 1
    v, e = vac["AGE"]
    row(a, y, v - e, v, v + e, "k", "FastSpecFit AGE (formed-mass weighted)", "s")
    y += 1
    for key, mk in (("Thomas+10", "s"), ("McDermid+15 Re", "v")):
        vals = [literature(s)[key]["age"] - lookback for s in SIGMAS]
        row(
            a,
            y,
            min(vals),
            np.mean(vals),
            max(vals),
            "0.3",
            f"{key} SSP age at z=0 minus lookback",
            mk,
        )
        y += 1
    vals = [c[1] - lookback for c in CGVD14]
    row(
        a,
        y,
        min(vals),
        np.mean(vals),
        max(vals),
        "0.3",
        "Conroy+14 light-weighted minus lookback",
        "D",
    )
    a.set_xlabel("age / lookback [Gyr]; red: age of universe; grey: oldest bin")

    # (f) SFR
    a, y = ax[5], 0
    a.set_xscale("log")
    for k, colour, name in zip(keys, colours, SHORT, strict=True):
        lo, m, hi = q(draws[k]["post_sfr100"])
        row(a, y, lo, m, hi, colour, f"{name}: SFR 100 Myr, spectrum aperture")
        y += 1
    v, e = vac["SFR"]
    row(a, y, v - e, v, v + e, "k", "FastSpecFit raw (0-30 Myr, total, h=1)", "s")
    y += 1
    lv, e = vac["LOGSFR"]
    row(
        a,
        y,
        10 ** (lv - e),
        10**lv,
        10 ** (lv + e),
        "0.4",
        "CIGALE (10 Myr, total; fails PDF flag)",
        "^",
    )
    y += 1
    from astropy.table import Table

    em = Table.read(HERE / f"emfit_{tid}.csv", format="ascii.csv", comment="#")[0]
    lha = (
        float(em["ha_n_flux"])
        * 1e-17
        * 4
        * np.pi
        * Planck18.luminosity_distance(z).to_value(u.cm) ** 2
    )
    sfr_ha = lha / 10**KE12_LOGC
    a.errorbar(sfr_ha, y, xerr=[[sfr_ha * 0.5]], xuplims=True, fmt="<", color="m", ms=9)
    a.text(1.02, y, f"EmFit Halpha upper limit {sfr_ha:.3f} (KE12, no dust)", transform=a.get_yaxis_transform(),
           fontsize=8, va="center")  # fmt: skip
    a.set_xlabel("SFR [Msun/yr]")
    for a in ax:
        a.set_yticks([])
        a.invert_yaxis()
    fig.subplots_adjust(wspace=1.15, hspace=0.3)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return sfr_ha


def fig_checks(tid, vac, out):
    """Draw Dn4000 and g-r of data, best models and FastSpecFit."""
    lp = load("ladder_plot", HERE / "ladder_plot.py")
    ic = load("index_check", HERE / "index_check.py")
    pc = load("photometry_check", HERE / "photometry_check.py")
    from astropy.io import fits
    from astropy.table import Table

    b = ic.bands()
    m = fits.getdata(pc.FSF, "METADATA")
    r = m[m["TARGETID"] == tid][0]
    sys_ = "N" if str(r["PHOTSYS"]).strip() == "N" else "S"
    curves = {
        k[-1]: Table.read(pc.FILT / f"{n}.ecsv")
        for k, n in zip("gr", pc.NAMES[sys_][:2], strict=True)
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    rng = np.random.default_rng(0)
    y = 0
    for label, path, colour in lp.steps(tid):
        if not path.exists():
            continue
        d = np.load(path)
        rest = d["wave"] / (1 + float(d["z"]))
        use = np.asarray(d["good"], bool) & np.isfinite(d["unc"])
        noisy = d["flux"] + rng.normal(size=(2000, rest.size)) * np.where(use, d["unc"], 0.0)
        dn, _ = ic.measure(rest, noisy, use, b)
        dm, _ = ic.measure(rest, np.asarray(d["sp_best"], float), use, b)
        if np.isfinite(dn).any():
            lo, med, hi = np.percentile(dn, [16, 50, 84])
            a1.errorbar(
                med, y, xerr=[[med - lo], [hi - med]], fmt="o", color="0.5", ms=6, capsize=3
            )
        a1.plot(dm, y, "*", color=colour, ms=13)
        a1.text(1.02, y, label, transform=a1.get_yaxis_transform(), fontsize=8, va="center")
        fx = np.asarray(d["sp_best"], float)
        lo_w, hi_w = d["wave"].min(), d["wave"].max()
        if label != "MILES":
            gr = pc.colour(pc.synth(d["wave"], fx, curves["g"], lo_w, hi_w),
                           pc.synth(d["wave"], fx, curves["r"], lo_w, hi_w))  # fmt: skip
            a2.plot(gr, y, "*", color=colour, ms=13)
            good = np.asarray(d["good"], bool)
            wg, fg = d["wave"][good], np.asarray(d["flux"], float)[good]
            gd = pc.colour(pc.synth(wg, fg, curves["g"], wg.min(), wg.max()),
                           pc.synth(wg, fg, curves["r"], wg.min(), wg.max()))  # fmt: skip
            if np.isfinite(gd):
                a2.plot(gd, y, "o", color="0.5", ms=7)
        a2.text(1.02, y, label, transform=a2.get_yaxis_transform(), fontsize=8, va="center")
        y += 1
    v, e = vac["DN4000"]
    a1.errorbar(v, y, xerr=e, fmt="s", color="k", capsize=3)
    a1.text(
        1.02,
        y,
        "FastSpecFit DN4000 (data)",
        transform=a1.get_yaxis_transform(),
        fontsize=8,
        va="center",
    )
    v, e = vac["DN4000_MODEL"]
    a1.errorbar(v, y + 1, xerr=e, fmt="D", color="k", capsize=3)
    a1.text(
        1.02,
        y + 1,
        "FastSpecFit DN4000_MODEL",
        transform=a1.get_yaxis_transform(),
        fontsize=8,
        va="center",
    )
    a1.set_xlabel("Dn4000 (grey: our data measurement; stars: best models)")
    fib = pc.colour(float(r["FIBERFLUX_G"]), float(r["FIBERFLUX_R"]))
    tot = pc.colour(float(r["FLUX_G"]), float(r["FLUX_R"]))
    a2.axvline(fib, color="0.3", ls="--", label=f"LS fibre {fib:.3f}")
    a2.axvline(tot, color="0.3", ls=":", label=f"LS total {tot:.3f}")
    s = fits.getdata(pc.FSF, "SPECPHOT")
    s = s[s["TARGETID"] == tid][0]
    fsf_gr = pc.colour(float(s["FLUX_SYNTH_PHOTMODEL_G"]) * float(r["MW_TRANSMISSION_G"]),
                       float(s["FLUX_SYNTH_PHOTMODEL_R"]) * float(r["MW_TRANSMISSION_R"]))  # fmt: skip
    a2.plot(fsf_gr, y, "D", color="k", ms=8)
    a2.text(1.02, y, "FastSpecFit model (total, MW-reddened)", transform=a2.get_yaxis_transform(),
            fontsize=8, va="center")  # fmt: skip
    a2.legend(fontsize=8, frameon=False, loc="lower left")
    a2.set_xlabel("g-r observed (grey: data spectrum where it covers g; stars: best models)")
    for a in (a1, a2):
        a.set_yticks([])
        a.set_ylim(y + 1.6, -0.6)
    fig.subplots_adjust(wspace=0.75)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(tid):
    """Write the five figures of ``tid``."""
    lc = load("ladder_corner", HERE / "ladder_corner.py")
    lh = load("ladder_hist", HERE / "ladder_hist.py")
    pc = load("photometry_check", HERE / "photometry_check.py")
    ana, rr = lc.runs(tid)
    logz, _ = lh.mist2_logz(RES / "mist2_zh.txt")
    use_apj_style()
    out = RES / str(tid) / "paper"
    out.mkdir(exist_ok=True)
    fig_metallicity(rr, logz, out / "fig_metallicity_plane.png")
    fig_dust_mass(rr, ["A_V", "dust_index", "logmass"], out / "fig_dust_mass.png")
    fig_sfh_prior(rr, tid, out / "fig_sfh_prior.png")
    vac = vac_values(tid)
    from astropy.io import fits
    from astropy.table import Table

    m = fits.getdata(pc.FSF, "METADATA")
    r = m[m["TARGETID"] == tid][0]
    sys_ = "N" if str(r["PHOTSYS"]).strip() == "N" else "S"
    curve = Table.read(pc.FILT / f"{pc.NAMES[sys_][1]}.ecsv")
    dr1 = np.load(RES / str(tid) / f"{tid}_poco_maxl.npz")
    good = np.asarray(dr1["good"], bool)
    spec_r = 1e9 * pc.synth(dr1["wave"][good], dr1["flux"][good], curve, dr1["wave"][good].min(),
                            dr1["wave"][good].max())  # fmt: skip
    ratio = float(r["FLUX_R"]) / spec_r
    sfr_ha = fig_summary(rr, tid, logz, vac, ratio, out / "fig_summary.png")
    fig_checks(tid, vac, out / "fig_checks.png")
    print(f"r-band total/spectrum flux ratio {ratio:.3f}; Halpha SFR limit {sfr_ha:.4f} Msun/yr")
    print(f"wrote {sorted(p.name for p in out.glob('fig_*.png'))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tid", type=int, required=True)
    main(ap.parse_args().tid)
