"""Seed gallery105.tsv with screen, repeat, colour, FastSpecFit and EmFit columns for each target.

The category column is left empty; it is filled by eye from the gallery figures.
INPUTS: targets of analysis.py; results/2026-11-10_data_quality_screen/{screened.npz, gaia5.fits,
repeats.csv}; data/fastspec-iron-sv3-bright.fits (FASTSPEC, METADATA);
results/2026-11-10_line_width_split/emfit_widths_sv3_bright.fits.
Rules, copied unchanged: Gaia star (parallax/err > 5 or pm > 3 mas/yr) and D(g-r) (FLUX_SYNTH minus
FIBERFLUX colour) as 2026-11-10_data_quality_screen screen.py and colour.py; FastSpecFit broad, BPT,
WHAN and class as 2026-11-01_gallery92/lines.py, except that EW(Halpha) and WHAN are blank when
Halpha is off the spectrum. EmFit sigmas are given only with sigma flag 0.
HdA and HgA are measured on each DR1 coadd (load_by_index) with the Worthey & Ottaviani (1997) Table 1
bandpasses: mean flux in each sideband, a straight line between the sideband midpoints as the
continuum, index = (l2 - l1)(1 - <F/F_c>) over the good pixels of the feature band (their Eq. 2),
in rest-frame A, absorption positive. The bandpasses are taken as air and moved to vacuum with
_airtovac (the paper does not say). Good pixels have ivar > 0; load_by_index zeroes spender's sky-line
mask, so a band can lose up to half its pixels. An index is NaN if any band has fewer than 25% good
pixels; <name>_good is the smallest good fraction of its three bands. Its error is the std over
200 noise draws from ivar (SEED 0). No resolution matching to Lick is done.
goto_like follows Goto (2007, arXiv:0801.1106, l.333) with EW(Hd) replaced by HdA: HdA > 5 A,
EW(Ha) < 3 A and EW([OII] 3726 + 3729) < 2.5 A, FastSpecFit EWs with emission positive; when Halpha
is off the spectrum only the [OII] cut is used (goto_ha_used False).
COMMAND: ``uv run --no-sync python experiments/2026-11-13_gallery105/seed.py``.
OUTPUT: results/2026-11-13_gallery105/gallery105.tsv, one row per figure, in figure order.
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from spender.data.desi import DESI

from hubersed.io.desi import load_by_index, tids_to_indices
from hubersed.paths import PATHS
from hubersed.sps.utils import _airtovac

RES = PATHS["RESULTS"]
OUT = RES / "2026-11-13_gallery105"
DQ = RES / "2026-11-10_data_quality_screen"
SNMIN, BROAD_SIG, MGII_SIG, EWMIN = 5.0, 425.0, 1000.0, 3.0
# Worthey & Ottaviani (1997) Table 1: (feature, blue, red) bands in A.
BANDS = {
    "hda": ((4083.50, 4122.25), (4041.60, 4079.75), (4128.50, 4161.00)),
    "hga": ((4319.75, 4363.50), (4283.50, 4319.75), (4367.25, 4419.75)),
}
MIN_GOOD, N_MC, SEED = 0.25, 200, 0
GOTO_HD, GOTO_HA, GOTO_OII = 5.0, 3.0, 2.5


def lick_index(rest, flux, ivar, bands, rng):
    """Return a Lick index, its noise-draw std and the smallest good-pixel fraction of its bands."""
    vac = [(_airtovac(lo), _airtovac(hi)) for lo, hi in bands]
    sel = [(rest >= lo) & (rest <= hi) for lo, hi in vac]
    good = min((ivar[s] > 0).mean() if s.any() else 0.0 for s in sel)
    if good < MIN_GOOD:
        return np.nan, np.nan, good
    sel = [s & (ivar > 0) for s in sel]
    (f_lo, f_hi), (b_lo, b_hi), (r_lo, r_hi) = vac
    wb, wr = (b_lo + b_hi) / 2, (r_lo + r_hi) / 2
    sig = 1 / np.sqrt(np.where(ivar > 0, ivar, np.inf))

    def index(fl):
        cb, cr = (fl[s].mean() for s in sel[1:])
        cont = cb + (cr - cb) * (rest[sel[0]] - wb) / (wr - wb)
        return float((f_hi - f_lo) * (1 - np.mean(fl[sel[0]] / cont)))

    draws = [index(flux + rng.standard_normal(flux.size) * sig) for _ in range(N_MC)]
    return index(flux), float(np.std(draws)), good


def mag(f):
    """Return -2.5 log10 of a positive flux, NaN otherwise."""
    return -2.5 * np.log10(f) if f > 0 else np.nan


def gaia_star_g():
    """Return the G of the brightest Gaia star within 5 arcsec of each target that has one."""
    best = {}
    for r in Table.read(DQ / "gaia5.fits"):
        g = float(r["phot_g_mean_mag"]) if np.isfinite(r["phot_g_mean_mag"]) else 99.0
        plx, eplx = float(r["parallax"]), float(r["parallax_error"])
        pm = np.hypot(float(r["pmra"]), float(r["pmdec"]))
        stellar = (np.isfinite(plx) and np.isfinite(eplx) and eplx > 0 and plx / eplx > 5) or (
            np.isfinite(pm) and pm > 3
        )
        if stellar:
            t = int(r["tid"])
            best[t] = min(best.get(t, 99.0), g)
    return best


def fastspec(f, m, i):
    """Return the FastSpecFit line columns of row ``i``, with the gallery92 lines.py rules."""

    def sn(ln):
        iv = float(f[f"{ln}_FLUX_IVAR"][i])
        return float(f[f"{ln}_FLUX"][i]) * np.sqrt(iv) if iv > 0 else 0.0

    bha = (sn("HALPHA_BROAD"), float(f["HALPHA_BROAD_SIGMA"][i]))
    bhb = (sn("HBETA_BROAD"), float(f["HBETA_BROAD_SIGMA"][i]))
    mg = (sn("MGII_2803"), float(f["MGII_2803_SIGMA"][i]))
    broad = (bha[0] >= SNMIN and bha[1] >= BROAD_SIG) or (bhb[0] >= SNMIN and bhb[1] >= BROAD_SIG) \
        or (mg[0] >= SNMIN and mg[1] >= MGII_SIG)  # fmt: skip
    fl = {ln: float(f[f"{ln}_FLUX"][i]) for ln in ("HALPHA", "HBETA", "OIII_5007", "NII_6584")}
    ok = all(sn(ln) >= 3 and fl[ln] > 0 for ln in fl)
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
    bpt = ""
    if ok:
        ke = 0.61 / (n2 - 0.47) + 1.19 if n2 < 0.47 else -np.inf
        ka = 0.61 / (n2 - 0.05) + 1.30 if n2 < 0.05 else -np.inf
        bpt = "AGN" if o3 > ke else "composite" if o3 > ka else "SF"
    # HALPHA_EW_IVAR is 0 when Halpha is off the spectrum (z > 0.497); then EW and WHAN are blank.
    ewha = float(f["HALPHA_EW"][i]) if float(f["HALPHA_EW_IVAR"][i]) > 0 else np.nan
    whan = "AGN-like" if np.isfinite(n2) and n2 > -0.4 and ewha > 6 else ""
    whan = "passive or retired" if ewha < 3 else whan
    strong = any(sn(ln) >= SNMIN and float(f[f"{ln}_EW"][i]) >= EWMIN
                 for ln in ("HALPHA", "HBETA", "OIII_5007", "OII_3726"))  # fmt: skip
    cls = "AGN" if broad or bpt == "AGN" else "emission line" if strong else "no strong lines"
    d_gr = (mag(float(f["FLUX_SYNTH_G"][i])) - mag(float(f["FLUX_SYNTH_R"][i]))) - (
        mag(float(m["FIBERFLUX_G"][i])) - mag(float(m["FIBERFLUX_R"][i]))
    )
    return {
        "fs_class": cls,
        "fs_broad": broad,
        "fs_bpt": bpt,
        "fs_whan": whan,
        "fs_log_nii_ha": round(n2, 2),
        "fs_log_oiii_hb": round(o3, 2),
        "fs_ew_ha": round(ewha, 1),
        "fs_ew_oii": round(
            sum(float(f[f"OII_{w}_EW"][i]) for w in (3726, 3729))
            if any(float(f[f"OII_{w}_EW_IVAR"][i]) > 0 for w in (3726, 3729))
            else np.nan,
            1,
        ),
        "fs_dn4000": round(float(f["DN4000"][i]), 2),
        "d_gr": round(d_gr, 2),
    }


def main():
    """Write the seeded table."""
    names = sorted(p.stem for p in OUT.glob("[0-9]*_*.png"))
    assert len(names) == 105, len(names)
    sc = np.load(DQ / "screened.npz")
    passed = set(sc["new_full"].tolist()) | set(sc["new_cont"].tolist())
    post = np.load(RES / "2026-11-09_mocks_v2_outliers" / "postsimbad_v2.npz")
    full, cont = set(post["full"].tolist()), set(post["cont"].tolist())
    star = gaia_star_g()
    rep = pd.read_csv(DQ / "repeats.csv")
    rep = rep[rep["flows"] == "new"]
    em = Table.read(RES / "2026-11-10_line_width_split" / "emfit_widths_sv3_bright.fits")
    erow = {int(t): i for i, t in enumerate(em["targetid"])}

    def emsig(i, c):
        s, fl = em[f"{c}_sigma"][i], em[f"{c}_sigma_flag"][i]
        return round(float(s), 1) if not np.ma.is_masked(s) and fl == 0 else np.nan

    wave = DESI._wave_obs.numpy()
    rng = np.random.default_rng(SEED)
    rows = []
    with fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits") as h:
        f, m = h["FASTSPEC"].data, h["METADATA"].data
        frow = {int(t): i for i, t in enumerate(m["TARGETID"])}
        for name in names:
            nn, tid = name.split("_")
            t = int(tid)
            r = rep[rep["target_id"] == t]
            g = star.get(t, np.nan)
            row = {
                "nn": nn,
                "target_id": t,
                "z": round(float(m["Z"][frow[t]]), 4),
                "branch": "both" if t in full and t in cont else "full" if t in full else "cont",
                "passes_dq_screen": t in passed,
                "screen_reason": ""
                if t in passed
                else "Gaia star G < 20"
                if g < 20
                else "fibre 2675/2676",
                "gaia_star_g": round(g, 2),
                "repeat_state": ";".join(
                    f"{b}:{s}" for b, s in zip(r["branch"], r["state"], strict=True)
                ),
                "n_usable": ";".join(str(n) for n in r["n_usable"]),
            }
            row |= fastspec(f, m, frow[t])
            flux, ivar, z, _ = load_by_index(int(tids_to_indices(np.array([t]))[0]))
            flux, ivar = np.asarray(flux, float), np.asarray(ivar, float)
            rest = wave / (1 + float(z))
            for k, bands in BANDS.items():
                v, e, good = lick_index(rest, flux, ivar, bands, rng)
                row[k], row[f"{k}_err"], row[f"{k}_good"] = round(v, 2), round(e, 2), round(good, 2)
            ha_used = np.isfinite(row["fs_ew_ha"])
            row["goto_like"] = bool(
                row["hda"] > GOTO_HD
                and row["fs_ew_oii"] < GOTO_OII
                and (not ha_used or row["fs_ew_ha"] < GOTO_HA)
            )
            row["goto_ha_used"] = ha_used
            i = erow.get(t)
            row |= {
                "em_ha_n_sigma": emsig(i, "ha_n") if i is not None else np.nan,
                "em_ha_b_sigma": emsig(i, "ha_b") if i is not None else np.nan,
                "em_prob_broad": round(float(em["prob_broad"][i]), 2) if i is not None else np.nan,
                "category": "",
                "notes": "",
            }
            rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "gallery105.tsv", sep="\t", index=False)
    print(d["fs_class"].value_counts().to_string())
    print(f"goto_like {d['goto_like'].sum()} (Halpha used in {d['goto_ha_used'].sum()})")
    print(f"wrote {len(d)} rows to {OUT / 'gallery105.tsv'}")


if __name__ == "__main__":
    main()
