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
Per free column (ion_limits.json, summed ion EWs): [SII] 89 A, [OI] 26, [ArIII]7135 30, [OIII]4363 20,
[OII]7325 15, [NII]5755 11, [NeIII] 74, [SIII]6312 10, [SIII]9069 177, H broad 507; broad forbidden
columns take their ion's narrow W ([OII] 238, [OIII] 1040, [NII] 108). Cue ratios [OI]6363/6300 0.308,
[NeIII]3968/3869 0.304.
FIGURES: results/2026-11-06_line_ew_prior/ ew_prior.txt, ew_limits.json (per line), ion_limits.json
(per free column) and ew_hist.png.
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


# W per free column (Nikhil 2026-11-07): per galaxy, the sum of the ion's FastSpecFit EWs, a missing
# or undetected line counting zero; lines FastSpecFit lacks ([OI]6363, [NeIII]3968) are added with
# Cue's median ratio to the measured line; broad forbidden columns take their ion's narrow W.
def cue_ratio(missing, measured, n=500, seed=0):
    """Return Cue's median luminosity ratio of two lines over random gas parameters."""
    import importlib.resources

    from cuejax.emulator import Emulator

    path = importlib.resources.files("cuejax").joinpath("data/cue_emlines_info.dat")
    info = np.genfromtxt(path, dtype=[("wave", "f8"), ("name", "<U20")], delimiter=",")
    names = [str(x).strip() for x in info["name"]]
    emu = Emulator()
    order = list(emu.free_parameter_order)
    theta = np.tile([float(np.atleast_1d(getattr(emu, k))[0]) for k in order], (n, 1))
    rng = np.random.default_rng(seed)
    ranges = {"gas_logu": (-4, -1), "gas_logn": (1, 4), "gas_logz": (-2.2, 0.5),
              "gas_logno": (-1, np.log10(5.4)), "gas_logco": (-1, np.log10(5.4))}  # fmt: skip
    for k, (lo, hi) in ranges.items():
        theta[:, order.index(k)] = rng.uniform(lo, hi, n)
    lum = np.asarray(emu.predict_lines(theta=theta))
    return float(np.median(lum[:, names.index(missing)] / lum[:, names.index(measured)]))


r_oi = cue_ratio("[O I] 6363", "[O I] 6300")
r_ne = cue_ratio("[Ne III] 3968", "[Ne III] 3869")
IONS = {  # column: ((FastSpecFit line, weight), ...)
    "[SII]": (("SII_6716", 1.0), ("SII_6731", 1.0)),
    "[OI]": (("OI_6300", 1.0 + r_oi),),
    "[ArIII]7135": (("ARIII_7135", 1.0),),
    "[OIII]4363": (("OIII_4363", 1.0),),
    "[OII]7325": (("OII_7320", 1.0), ("OII_7330", 1.0)),
    "[NII]5755": (("NII_5755", 1.0),),
    "[NeIII]": (("NEIII_3869", 1.0 + r_ne),),
    "[SIII]6312": (("SIII_6312", 1.0),),
    "[SIII]9069": (("SIII_9069", 1.0),),
    "[OII]": (("OII_3726", 1.0), ("OII_3729", 1.0)),
    "[OIII]": (("OIII_4959", 1.0), ("OIII_5007", 1.0)),
    "[NII]": (("NII_6548", 1.0), ("NII_6584", 1.0)),
    "H broad": tuple((f"{b}_BROAD", 1.0) for b in ("HALPHA", "HBETA")),
}
ion_limits = {}
for ion, members in IONS.items():
    vals = np.array([np.nan_to_num(fsf[ln]) * w for ln, w in members]).sum(axis=0)
    seen = np.any([np.isfinite(fsf[ln]) for ln, _ in members], axis=0)
    ew = np.where(seen & (vals > 0), vals, np.nan)
    _, _, q = stats(ew)
    _, mo, _ = stats(ew[outlier])
    ion_limits[ion] = float(np.nanmax([q[1], mo]))
# broad forbidden columns use their ion's narrow W
for ion in ("[OII]", "[OIII]", "[NII]", "[SII]", "[OI]"):
    ion_limits[f"{ion} broad"] = ion_limits[ion]
lines += ["", f"Cue ratios: [OI]6363/6300 {r_oi:.4f}, [NeIII]3968/3869 {r_ne:.4f}",
          "W per free column (summed ion EWs, max(parent p99.99, outlier max)):"]  # fmt: skip
lines += [f"  {ion:16} {w:8.1f}" for ion, w in ion_limits.items()]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ion_limits.json").write_text(json.dumps(ion_limits, indent=1) + "\n")
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
