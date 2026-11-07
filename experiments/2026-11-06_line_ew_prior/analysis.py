"""How large are emission-line equivalent widths in DR1 SV3 bright, for a data-based line prior?

QUESTION: per line, what are the maximum and the 99.9th and 99.99th percentiles of the rest-frame EW
in FastSpecFit v3.0 and EmFit, for our parent sample (median pixel S/N > 3) and for our outliers? These set the upper bound of a
U(0, EW_max) amplitude prior (Chang's suggestion, in EW rather than flux).
HYPOTHESIS: none registered.
INPUTS: data/fastspecfit_v3.0/fastspec-iron-sv3-bright.fits (SPECTYPE GALAXY; EW = FLUX/CONT/(1+z),
checked to 1e-6); results/2026-11-06_line_ew_prior/emfit_sv3_bright.fits (extract_emfit.py from the VAC v2.3.1 file;
fetch_emfit.py gives the same table from Data Lab when it is up), whose EW
is the summed component flux over the FastSpecFit continuum at the same line, matched on TARGETID;
outliers are `full` of results/prescreened_outliers_postsimbad.npz (231). A FastSpecFit EW counts
when EW > 0 and EW_IVAR > 0, an EmFit EW when its flux and the continuum are > 0.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-06_line_ew_prior/analysis.py``.
RESULT: 2026-11-07, ew_prior.txt, parent sample (median pixel S/N > 3, 114,136 galaxies). FastSpecFit
99.99th percentiles: Halpha 544 A, [OIII]5007 518, Hbeta 143, [NII]6584 82, [SII] 46 and 39, [OI]6300
20, [OIII]4363 12, broad Halpha 429. EmFit agrees for narrow lines; its Balmer totals and broad EWs
on the FSF continuum are unusable. Outliers exceed the 99.99th percentile in Halpha (1024),
[OIII]5007 (733), [OIII]4363 (20), [SII]6716 (54) and [ArIII]7135 (30). Assumptions: EmFit EW on the
FSF continuum; SV3 bright; EmFit v2.3.1. W_j (ew_limits.json) = max(p99.99, outlier max) per line.
FIGURES: results/2026-11-06_line_ew_prior/ ew_prior.txt, ew_limits.json (W_j) and ew_hist.png.
"""

# %%
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-06_line_ew_prior"
FSF = PATHS["DATA"] / "fastspecfit_v3.0" / "fastspec-iron-sv3-bright.fits"
OUTLIERS = PATHS["RESULTS"] / "prescreened_outliers_postsimbad.npz"
# parent sample: median pixel S/N > 3, the galaxies the outlier flows were trained on
PARENT = PATHS["DATA"] / "latents" / "spender_spec_10latent_snr3.h5"
QUANTILES = (99.9, 99.99)
# FastSpecFit lines of the free-line ions (hubersed.sps.free_lines.IONS) and the broad Balmer lines
FSF_LINES = (
    "HBETA HALPHA HBETA_BROAD HALPHA_BROAD OII_3726 OII_3729 OIII_4959 OIII_5007 NII_6548 "
    "NII_6584 SII_6716 SII_6731 OI_6300 ARIII_7135 OIII_4363 OII_7320 OII_7330 NII_5755 "
    "NEIII_3869 SIII_6312 SIII_9069"
).split()
# EmFit line: (components summed, FastSpecFit continuum column)
EMFIT_LINES = {
    "hb (n+out+b)": (("hb_n", "hb_out", "hb_b"), "HBETA_CONT"),
    "hb_b": (("hb_b",), "HBETA_CONT"),
    "oiii4959": (("oiii4959", "oiii4959_out"), "OIII_4959_CONT"),
    "oiii5007": (("oiii5007", "oiii5007_out"), "OIII_5007_CONT"),
    "nii6548": (("nii6548", "nii6548_out"), "NII_6548_CONT"),
    "nii6583": (("nii6583", "nii6583_out"), "NII_6584_CONT"),
    "ha (n+out+b)": (("ha_n", "ha_out", "ha_b"), "HALPHA_CONT"),
    "ha_b": (("ha_b",), "HALPHA_CONT"),
    "sii6716": (("sii6716", "sii6716_out"), "SII_6716_CONT"),
    "sii6731": (("sii6731", "sii6731_out"), "SII_6731_CONT"),
}

# %% FastSpecFit EWs
with fits.open(FSF, memmap=True) as h:
    meta, fs = h["METADATA"].data, h["FASTSPEC"].data
    gal = meta["SPECTYPE"] == "GALAXY"
    tid, z = np.asarray(meta["TARGETID"])[gal], np.asarray(meta["Z"])[gal]
    fsf = {}
    for line in FSF_LINES:
        ew = np.asarray(fs[f"{line}_EW"])[gal]
        fsf[line] = np.where((ew > 0) & (np.asarray(fs[f"{line}_EW_IVAR"])[gal] > 0), ew, np.nan)
    cont = {c: np.asarray(fs[c])[gal] for _, c in EMFIT_LINES.values()}
outlier = np.isin(tid, np.load(OUTLIERS)["full"])
with h5py.File(PARENT) as h:
    parent = np.isin(tid, np.asarray(h["target_ids"]))
for line in FSF_LINES:
    fsf[line] = np.where(parent, fsf[line], np.nan)
assert np.unique(tid).size == tid.size, "repeated TARGETID in FastSpecFit"

# %% EmFit EWs on the FastSpecFit continuum (skipped until fetch_emfit.py has run)
EMFIT = OUT / "emfit_sv3_bright.fits"
emfit, em_outlier, common = {}, np.zeros(0, bool), np.zeros(0)
if EMFIT.exists():
    with fits.open(EMFIT) as h:
        em = h[1].data
    etid = np.asarray(em["targetid"]).astype(np.int64)
    assert np.unique(etid).size == etid.size, "repeated TARGETID in EmFit"
    common, i_fsf, i_em = np.intersect1d(tid, etid, return_indices=True)
    emfit = {}
    for name, (comps, ccol) in EMFIT_LINES.items():
        flux = sum(np.nan_to_num(np.asarray(em[f"{c}_flux"], float)) for c in comps)[i_em]
        c = cont[ccol][i_fsf]
        emfit[name] = np.where(
            (flux > 0) & (c > 0), flux / np.where(c > 0, c, 1) / (1 + z[i_fsf]), np.nan
        )
    em_outlier = outlier[i_fsf]
    for name in emfit:
        emfit[name] = np.where(parent[i_fsf], emfit[name], np.nan)


# %% table
def stats(ew):
    """Return n, max and the QUANTILES percentiles of the finite values, or nan."""
    ok = np.isfinite(ew)
    if not ok.any():
        return 0, np.nan, [np.nan] * len(QUANTILES)
    return (
        int(ok.sum()),
        float(np.max(ew[ok])),
        [float(np.percentile(ew[ok], q)) for q in QUANTILES],
    )


lines = [
    f"Rest-frame EW (A) in the parent sample (median pixel S/N > 3, {parent.sum()} of "
    f"{gal.sum()} FastSpecFit SV3 bright GALAXY rows; {outlier.sum()} outliers). EmFit matched on "
    f"TARGETID ({common.size} rows). n = rows with a positive EW; outliers: n and max.",
    "",
    f"{'catalogue':10} {'line':16} {'n':>7} {'max':>10} {'p99.9':>8} {'p99.99':>8}   "
    f"{'n out':>5} {'max out':>8}",
]
for cat, data, mask in (("FSF", fsf, outlier), ("EmFit", emfit, em_outlier)):
    for name, ew in data.items():
        na, ma, qa = stats(ew)
        no, mo, _ = stats(ew[mask])
        lines.append(
            f"{cat:10} {name:16} {na:7d} {ma:10.1f} {qa[0]:8.1f} {qa[1]:8.1f}   {no:5d} {mo:8.1f}"
        )
# W_j of the uniform EW prior (Nikhil 2026-11-07): per FastSpecFit line, the larger of the parent
# 99.99th percentile and the largest outlier EW
limits = {}
for name, ew in fsf.items():
    _, _, q = stats(ew)
    _, mo, _ = stats(ew[outlier])
    limits[name] = float(np.nanmax([q[1], mo]))
lines += ["", "W_j = max(parent p99.99, outlier max), FastSpecFit:"]
lines += [f"  {name:16} {w:8.1f}" for name, w in limits.items()]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ew_limits.json").write_text(json.dumps(limits, indent=1) + "\n")
(OUT / "ew_prior.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# %% histograms: log EW, all galaxies and outliers, both catalogues
bins = np.logspace(-2, 4, 61)
panels = list(fsf) + [k for k in emfit]
ncol = 6
fig, axes = plt.subplots(int(np.ceil(len(panels) / ncol)), ncol, figsize=(20, 12), sharex=True)
for ax, name in zip(axes.flat, panels, strict=False):
    ew, mask = (fsf[name], outlier) if name in fsf else (emfit[name], em_outlier)
    tag = "FSF" if name in fsf else "EmFit"
    ok = np.isfinite(ew)
    ax.hist(ew[ok], bins=bins, color="0.6", label="all")
    ax.hist(ew[ok & mask], bins=bins, color="C3", label="outliers")
    _, ma, qa = stats(ew)
    ax.axvline(qa[1], color="k", ls="--", lw=0.8)
    ax.axvline(ma, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{tag} {name}", fontsize=9)
for ax in axes.flat[len(panels) :]:
    ax.set_visible(False)
axes.flat[0].legend(frameon=False, fontsize=8)
fig.supxlabel("rest-frame EW [A]; dashed 99.99th percentile, dotted max (parent sample)")
fig.tight_layout()
fig.savefig(OUT / "ew_hist.png", dpi=100)
