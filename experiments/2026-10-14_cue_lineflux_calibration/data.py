"""Build the line-flux table of the Cue calibration sample from the EmFit and FastSpecFit VACs.

Sample: the spender SV3 sample (data/all_target_ids.npy) in the iron SV3 bright FastSpecFit VAC,
star-forming (Halpha, Hbeta, [OIII]5007, [NII]6584 at S/N >= 3, below Kauffmann+03, as in
2026-10-14_auroral_snr_fraction.py), [OII]7320+7330 at S/N >= 5, and an EmFit row from the same
sv3/bright spectrum (desi_dr1.emfit on Astro Data Lab TAP).

Fluxes, all in 1e-17 erg/s/cm2 and not dust-corrected:

- EmFit totals for HB, HA, OIII5007, NII6584, SII6716, SII6731: the sum of the narrow, outflow and
  broad components EmFit fits. Errors are the component errors in quadrature.
- FastSpecFit Gaussian fluxes for OII3727 (3726 + 3729, the split is not used), NEIII3869, HG,
  OI6300, SIII6312, OIII4363, OII7325 (7320 + 7330), SIII9069 and HEI5876. Errors from FLUX_IVAR.
  A line with FLUX_IVAR 0 gets NaN ([SIII]9069 is beyond the DESI range above z ~0.08).
- FSF_<line>: FastSpecFit Gaussian fluxes of the six EmFit lines, for the FastSpecFit-only run.

[OIII]4959 and [NII]6548 are left out, since their ratio to the strong line is fixed by atomic
physics. Also saved: Z, LOGMSTAR, the FastSpecFit rest-frame HBETA_EW (A) as EW_HB, the EmFit
flags prob_broad, ha_b_sigma, oiii_dbl_flag and sii_dbl_flag, and WIDE_BROAD (prob_broad >= 80 and
ha_b_sigma > 500 km/s).

COMMAND: ``uv run python experiments/2026-10-14_cue_lineflux_calibration/data.py`` from the
repository root. It queries Astro Data Lab.

OUTPUT: lineflux_table.fits in results/2026-10-14_cue_lineflux_calibration/.
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astroquery.utils.tap.core import TapPlus

from hubersed.detect.sky_screens import DL_TAP, query
from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
TID = 39627770174637084
SNR_BPT = 3.0
SNR_7325 = 5.0
WIDE_BROAD_KMS = 500.0
# table line name, EmFit components summed
EMFIT = {
    "HB": ("hb_n", "hb_out", "hb_b"),
    "OIII5007": ("oiii5007", "oiii5007_out"),
    "NII6584": ("nii6583", "nii6583_out"),
    "HA": ("ha_n", "ha_out", "ha_b"),
    "SII6716": ("sii6716", "sii6716_out"),
    "SII6731": ("sii6731", "sii6731_out"),
}
# table line name, FastSpecFit lines summed
FSF = {
    "OII3727": ("OII_3726", "OII_3729"),
    "NEIII3869": ("NEIII_3869",),
    "HG": ("HGAMMA",),
    "OIII4363": ("OIII_4363",),
    "OI6300": ("OI_6300",),
    "SIII6312": ("SIII_6312",),
    "OII7325": ("OII_7320", "OII_7330"),
    "SIII9069": ("SIII_9069",),
    "HEI5876": ("HEI_5876",),
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


def select(t):
    """Return the mask of star-forming galaxies with [OII]7320,7330 at S/N >= SNR_7325."""
    snr = {k: np.divide(*fsf_sum(t, (k,))) for k in ("HALPHA", "HBETA", "OIII_5007", "NII_6584")}
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(t["NII_6584_FLUX"] / t["HALPHA_FLUX"])
        y = np.log10(t["OIII_5007_FLUX"] / t["HBETA_FLUX"])
        sf = np.all([np.nan_to_num(v) >= SNR_BPT for v in snr.values()], axis=0)
        sf &= (x < 0.05) & (y < 0.61 / (x - 0.05) + 1.3)
    f, e = fsf_sum(t, FSF["OII7325"])
    return sf & (np.nan_to_num(f / e) >= SNR_7325)


fsf = fits.getdata(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", "FASTSPEC")
fsf = fsf[np.isin(fsf["TARGETID"], np.load(PATHS["DATA"] / "all_target_ids.npy").astype(np.int64))]
fsf = fsf[select(fsf)]
print(f"{len(fsf)} star-forming galaxies with [OII]7325 S/N >= {SNR_7325}")

cols = [
    "targetid",
    "survey",
    "program",
    "prob_broad",
    "ha_b_sigma",
    "oiii_dbl_flag",
    "sii_dbl_flag",
]
for comps in EMFIT.values():
    cols += [f"{c}_flux" for c in comps] + [f"{c}_flux_err" for c in comps]
ids = ",".join(str(int(x)) for x in fsf["TARGETID"])
q = (
    f"SELECT {', '.join(cols)} FROM desi_dr1.emfit WHERE targetid IN ({ids}) "
    "AND survey = 'sv3' AND program = 'bright'"
)
em = query(lambda: TapPlus(url=DL_TAP).launch_job(q).get_results())
assert len(np.unique(em["targetid"])) == len(em), "duplicate EmFit rows"
print(f"{len(em)} with an EmFit sv3/bright row")

order = {int(x): i for i, x in enumerate(em["targetid"])}
keep = np.array([int(x) in order for x in fsf["TARGETID"]])
fsf = fsf[keep]
em = em[[order[int(x)] for x in fsf["TARGETID"]]]
assert np.all(np.asarray(em["targetid"], np.int64) == fsf["TARGETID"])


def col(name):
    """Return an EmFit column as floats with masked or NaN entries as 0."""
    return np.nan_to_num(np.ma.filled(np.ma.asarray(em[name], dtype=float), np.nan))


out = Table()
out["TARGETID"] = fsf["TARGETID"]
out["Z"] = fsf["Z"]
out["LOGMSTAR"] = fsf["LOGMSTAR"]
out["EW_HB"] = fsf["HBETA_EW"]
for name, comps in EMFIT.items():
    out[f"F_{name}"] = sum(col(f"{c}_flux") for c in comps)
    out[f"E_{name}"] = np.sqrt(sum(col(f"{c}_flux_err") ** 2 for c in comps))
for name, lines in FSF.items():
    out[f"F_{name}"], out[f"E_{name}"] = fsf_sum(fsf, lines)
for name, c in FSF_OF_EMFIT.items():
    out[f"FSF_F_{name}"], out[f"FSF_E_{name}"] = fsf_sum(fsf, (c,))
out["PROB_BROAD"] = col("prob_broad")
out["HA_B_SIGMA"] = col("ha_b_sigma")
out["OIII_DBL"] = np.asarray(em["oiii_dbl_flag"], bool)
out["SII_DBL"] = np.asarray(em["sii_dbl_flag"], bool)
out["WIDE_BROAD"] = (out["PROB_BROAD"] >= 80) & (out["HA_B_SIGMA"] > WIDE_BROAD_KMS)
out.meta["COMMENT"] = "fluxes 1e-17 erg/s/cm2, not dust-corrected; see data.py"

OUT.mkdir(parents=True, exist_ok=True)
out.write(OUT / "lineflux_table.fits", overwrite=True)
print(f"wrote {len(out)} rows; WIDE_BROAD {int(out['WIDE_BROAD'].sum())}")
row = out[out["TARGETID"] == TID]
assert len(row) == 1, "target not in the table"
for name in [*EMFIT, *FSF]:
    print(f"{name:>10} {float(row[f'F_{name}'][0]):9.2f} +- {float(row[f'E_{name}'][0]):7.2f}")
