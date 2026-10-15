"""Control sample without the [OII]7325 cut: does Cue also miss the upper branch for typical galaxies?

QUESTION: the calibration sample needs [OII]7325 S/N >= 5. Among metal-rich (N2 > -0.8) star-forming
galaxies above log M* 10 that cut keeps 86 of 25039, with higher O3, EW(Hb) and sSFR at fixed mass,
also at matched Hbeta S/N. Do metal-rich galaxies without the cut, at the same data quality, also fit
worse when forced onto the upper branch, in the same lines?

``pick`` selects the control pool: star-forming as in data.py, [OII]7325 S/N < SNR_7325, FastSpecFit
N2 > N2_MIN, Hbeta S/N >= SNR_HB, Z_MIN < z <= Z_MAX, an EmFit sv3/bright row, and EmFit N2 > N2_MIN.
It writes the pool in the lineflux_table.fits format (fluxes built as in data.py) and picks N_PICK
galaxies matched in log M* to the metal-rich galaxies of subset_20.txt and subset_ew60.txt (as in
forced.py pick): N_PICK reference masses evenly spaced in rank, each matched by a random unused
control within DM of it, else the nearest. ``compare`` reads the free and fit.py --zmin fits of the
picks and of forced_ids.txt and writes control_compare.txt: O/H, A_V, the change of the best data
log likelihood, model/data per line and of Halpha/Hbeta. SEED: SEED.

RESULT: 2026-10-15, 989 in the pool, 10 picks (fits run on ls6 idev). 8 of 10 free fits sit at O/H
7.86-8.20; 2 are already above 8.4 (8.44, 8.89, A_V ~0.7). Forced, the 8 lose dlnL 4.5 to 144 (median
~16) with the same pattern as forced_ids: [OIII] 1.11-1.20x, [SII]6716 1.11-1.26x (7 of 8), [OII]3727
0.56-0.82x (5 of 8), A_V drops and Halpha/Hbeta model/data falls to 0.81-0.96. The [OII]7325 cut does
not cause the failure.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/control.py pick``
(queries Astro Data Lab); then for each TARGETID in control/control_ids.txt, ``fit.py TARGETID
--lines strong --table results/2026-10-14_cue_lineflux_calibration/control/control_table.fits`` with
and without ``--zmin -0.31``; then ``control.py compare``, all from the repository root.

OUTPUT: control_table.fits, control_ids.txt and control_compare.txt in
results/2026-10-14_cue_lineflux_calibration/control/.
"""

import sys

import forced
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astroquery.utils.tap.core import TapPlus
from fit import LINES

from hubersed.detect.sky_screens import DL_TAP, query
from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
CTL = OUT / "control"
SNR_BPT = 3.0
SNR_7325 = 5.0
SNR_HB = 35.0
N2_MIN = -0.8
Z_MIN, Z_MAX = 0.01, 0.3
N_PICK = 10
DM = 0.1
SEED = 0
# as data.py
EMFIT = {
    "HB": ("hb_n", "hb_out", "hb_b"),
    "OIII5007": ("oiii5007", "oiii5007_out"),
    "NII6584": ("nii6583", "nii6583_out"),
    "HA": ("ha_n", "ha_out", "ha_b"),
    "SII6716": ("sii6716", "sii6716_out"),
    "SII6731": ("sii6731", "sii6731_out"),
}
FSF = {
    "OII3727": ("OII_3726", "OII_3729"),
    "NEIII3869": ("NEIII_3869",),
    "HG": ("HGAMMA",),
    "OIII4363": ("OIII_4363",),
    "OI6300": ("OI_6300",),
    "SIII6312": ("SIII_6312",),
    "OII7325": ("OII_7320", "OII_7330"),
}
FSF_OF_EMFIT = {
    "HB": "HBETA",
    "OIII5007": "OIII_5007",
    "NII6584": "NII_6584",
    "HA": "HALPHA",
    "SII6716": "SII_6716",
    "SII6731": "SII_6731",
}


def fsf_sum(t, cols):
    """Return the summed FastSpecFit flux and its error, NaN where any line has no ivar."""
    flux = sum(t[f"{c}_FLUX"] for c in cols)
    ivar = [t[f"{c}_FLUX_IVAR"] for c in cols]
    ok = np.all([iv > 0 for iv in ivar], axis=0)
    var = sum(np.where(iv > 0, 1.0 / np.where(iv > 0, iv, 1.0), 0.0) for iv in ivar)
    return np.where(ok, flux, np.nan), np.where(ok, np.sqrt(var), np.nan)


def snr(t, cols):
    """Return the FastSpecFit S/N of the summed lines, 0 where undefined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(np.divide(*fsf_sum(t, cols)))


def pool_mask(t):
    """Return the mask of the control pool before the EmFit checks."""
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(t["NII_6584_FLUX"] / t["HALPHA_FLUX"])
        y = np.log10(t["OIII_5007_FLUX"] / t["HBETA_FLUX"])
        sf = np.all(
            [snr(t, (k,)) >= SNR_BPT for k in ("HALPHA", "HBETA", "OIII_5007", "NII_6584")], axis=0
        )
        sf &= (x < 0.05) & (y < 0.61 / (x - 0.05) + 1.3)
    m = sf & (snr(t, FSF["OII7325"]) < SNR_7325) & (x > N2_MIN)
    return m & (snr(t, ("HBETA",)) >= SNR_HB) & (t["Z"] > Z_MIN) & (t["Z"] <= Z_MAX)


def build(fsf):
    """Return the pool as a table in the lineflux_table.fits format, with EmFit fluxes."""
    cols = ["targetid"]
    for comps in EMFIT.values():
        cols += [f"{c}_flux" for c in comps] + [f"{c}_flux_err" for c in comps]
    ids = ",".join(str(int(x)) for x in fsf["TARGETID"])
    q = (
        f"SELECT {', '.join(cols)} FROM desi_dr1.emfit WHERE targetid IN ({ids}) "
        "AND survey = 'sv3' AND program = 'bright'"
    )
    em = query(lambda: TapPlus(url=DL_TAP).launch_job(q).get_results())
    assert len(np.unique(em["targetid"])) == len(em), "duplicate EmFit rows"
    order = {int(x): i for i, x in enumerate(em["targetid"])}
    fsf = fsf[np.array([int(x) in order for x in fsf["TARGETID"]])]
    em = em[[order[int(x)] for x in fsf["TARGETID"]]]

    def col(name):
        return np.nan_to_num(np.ma.filled(np.ma.asarray(em[name], dtype=float), np.nan))

    out = Table()
    out["TARGETID"] = fsf["TARGETID"]
    out["Z"] = fsf["Z"]
    out["LOGMSTAR"] = fsf["LOGMSTAR"]
    out["EW_HB"] = fsf["HBETA_EW"]
    out["SNR_HB"] = snr(fsf, ("HBETA",))
    for name, comps in EMFIT.items():
        out[f"F_{name}"] = sum(col(f"{c}_flux") for c in comps)
        out[f"E_{name}"] = np.sqrt(sum(col(f"{c}_flux_err") ** 2 for c in comps))
    for name, lines in FSF.items():
        out[f"F_{name}"], out[f"E_{name}"] = fsf_sum(fsf, lines)
    for name, c in FSF_OF_EMFIT.items():
        out[f"FSF_F_{name}"], out[f"FSF_E_{name}"] = fsf_sum(fsf, (c,))
    out.meta["COMMENT"] = "fluxes 1e-17 erg/s/cm2, not dust-corrected; see control.py"
    return out


def n2(t):
    """Return the EmFit log [NII]6584/Halpha, NaN where [NII] is not positive."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t["F_NII6584"] > 0, np.log10(t["F_NII6584"] / t["F_HA"]), np.nan)


def reference():
    """Return the rows of the fitted metal-rich galaxies, selected as in forced.py pick."""
    t = Table.read(OUT / "lineflux_table.fits")
    ids = np.concatenate(
        [np.loadtxt(OUT / f, dtype=np.int64) for f in ("subset_20.txt", "subset_ew60.txt")]
    )
    t = t[np.isin(t["TARGETID"], ids)]
    return t[(t["Z"] > Z_MIN) & (n2(t) > N2_MIN)]


def describe(name, t):
    """Print medians and 16-84 ranges of the selection quantities."""
    with np.errstate(divide="ignore", invalid="ignore"):
        q = {
            "logM": t["LOGMSTAR"],
            "z": t["Z"],
            "N2": n2(t),
            "O3": np.log10(t["F_OIII5007"] / t["F_HB"]),
            "EW_HB": t["EW_HB"],
            "Ha/Hb": t["F_HA"] / t["F_HB"],
        }
    print(f"{name} ({len(t)})")
    for k, v in q.items():
        p = np.nanpercentile(v, [16, 50, 84])
        print(f"  {k:>6} {p[1]:7.3f} [{p[0]:7.3f}, {p[2]:7.3f}]")


def pick():
    """Write the control table and the mass-matched picks."""
    fsf = fits.getdata(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", "FASTSPEC")
    spender = np.load(PATHS["DATA"] / "all_target_ids.npy").astype(np.int64)
    fsf = fsf[np.isin(fsf["TARGETID"], spender)]
    fsf = fsf[pool_mask(fsf)]
    print(f"{len(fsf)} in the pool before EmFit")
    t = build(fsf)
    print(f"{len(t)} with an EmFit sv3/bright row")
    t = t[n2(t) > N2_MIN]
    print(f"{len(t)} with EmFit N2 > {N2_MIN}")
    CTL.mkdir(parents=True, exist_ok=True)
    t.write(CTL / "control_table.fits", overwrite=True)

    ref = reference()
    mref = np.sort(np.asarray(ref["LOGMSTAR"]))
    targets = mref[np.round(np.linspace(0, len(mref) - 1, N_PICK)).astype(int)]
    rng = np.random.default_rng(SEED)
    m = np.asarray(t["LOGMSTAR"])
    free = np.ones(len(t), bool)
    picks = []
    for mt in targets:
        near = np.flatnonzero(free & (np.abs(m - mt) <= DM))
        i = rng.choice(near) if len(near) else np.flatnonzero(free)[np.argmin(np.abs(m - mt)[free])]
        free[i] = False
        picks.append(i)
        print(f"ref logM {mt:.2f}: {int(t['TARGETID'][i])} logM {m[i]:.2f} N2 {n2(t)[i]:+.2f}")
    np.savetxt(CTL / "control_ids.txt", t["TARGETID"][picks], fmt="%d")
    describe("reference metal-rich", ref)
    describe("control pool", t)
    describe("control picks", t[picks])


def compare():
    """Write control_compare.txt: free against forced fits for the picks and for forced_ids.txt."""
    order = list(LINES)
    out = [
        f"free vs forced (12+log(O/H) >= {forced.ZMIN_OH}) strong-line fits; m/d = model/data, "
        "S/N >= 3; Ha/Hb is m/d(HA) / m/d(HB)",
        f"{'TARGETID':>18} {'fit':>6} {'O/H':>5} {'A_V':>5} {'dlnL':>7} {'Ha/Hb':>6}  "
        + " ".join(f"{k:>9}" for k in order),
    ]
    groups = (
        ("control", CTL, np.loadtxt(CTL / "control_ids.txt", dtype=np.int64)),
        ("forced_ids", OUT, np.loadtxt(OUT / "forced_ids.txt", dtype=np.int64)),
    )
    for name, where, ids in groups:
        # forced.load reads its module OUT when called
        forced.OUT = where
        out.append(f"# {name}")
        for tid in ids:
            free, forc = forced.load(tid, forced.FREE), forced.load(tid, forced.FORCED)
            for fit, f in (("free", free), ("forced", forc)):
                md = f["md"]
                hahb = md["HA"] / md["HB"] if "HA" in md and "HB" in md else np.nan
                cells = [f"{md[k]:9.2f}" if k in md else f"{'-':>9}" for k in order]
                out.append(
                    f"{tid:>18} {fit:>6} {f['oh'][1]:5.2f} {f['av']:5.2f} "
                    f"{f['lnl'] - free['lnl']:+7.2f} {hahb:6.2f}  " + " ".join(cells)
                )
    text = "\n".join(out)
    print(text)
    (CTL / "control_compare.txt").write_text(text + "\n")


if __name__ == "__main__":
    {"pick": pick, "compare": compare}[sys.argv[1]]()
