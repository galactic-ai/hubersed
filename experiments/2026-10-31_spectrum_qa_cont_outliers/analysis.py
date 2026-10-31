"""Data-quality metrics of the 94 common continuum outliers against three control sets (#17).

QUESTION: do the DR1 SV3 bright coadds of the 94 common continuum outliers disagree with their
fiber photometry, come from brighter-sky exposures, or have mis-scaled ivar more often than
control galaxies?
HYPOTHESIS: none registered.
INPUTS: sample94.npz (sample.py), controls.npz (controls.py), zpix_flags.csv (flags.py),
sv3_bright_exposures.csv (exposures.py), data/fastspec-iron-sv3-bright.fits, the DESIchunk1024
pkl spectra.
M1: dgr = -2.5 log10[(FLUX_SYNTH_G / FIBERFLUX_G) / (FLUX_SYNTH_R / FIBERFLUX_R)], drz the same
for r and z. Negative means the spectrum is bluer than the photometry.
M2: every exposure of the target's TILEID_LIST tiles; brightest and faintest sky g and their count.
M3: per arm (observed < 5800, 5800-7600, > 7600 A), 1.4826 MAD of (flux - 51 px median filter)
times sqrt(ivar) over pixels with ivar > 0. 1 means ivar matches the scatter.
SEED: none.
COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-31_spectrum_qa_cont_outliers/analysis.py``.
RESULT: 2026-10-31, qa_summary.txt. Outlier spectra are bluer than their fiber photometry: dgr
median -0.131 [16th -0.404] against -0.014 [-0.075] (matched), -0.071 (cont10) and -0.082 (cont15)
latent neighbours; drz -0.019 against +0.027. Brightest-exposure sky g has the same median (20.78)
in all groups, with a brighter 16th percentile for outliers (19.32 against 19.74 matched). sigma(chi)
is within 0.03 of the controls in every arm. With FIBERTOTFLUX, dgr_tot is -0.104 against +0.011
(matched). COADD_FIBERSTATUS and ZWARN are 0 for all 94 (spender keeps only those). Spearman of dgr
among the 94: sky_min rho +0.13 (p 0.21), pos_off -0.27 (p 0.010), pos_rms -0.26 (p 0.013),
psf2fib +0.03 (p 0.78). Assumptions: tile-level sky (COADD_NUMEXP equals the tile count for 85 of
94), unscreened controls, seed 0 flows, 8 correlation tests without correction.
FIGURES: results/2026-10-31_spectrum_qa_cont_outliers/qa_summary.txt, qa_outliers.txt,
qa_correlations.txt, qa_dist.png.
"""

# %%
import csv
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import medfilt
from scipy.stats import spearmanr
from spender.data.desi import DESI

from hubersed.io.desi import tids_to_indices
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

OUT = PATHS["RESULTS"] / "2026-10-31_spectrum_qa_cont_outliers"
KERNEL = 51
ARMS = {"b": (0, 5800), "r": (5800, 7600), "z": (7600, 1e5)}
GROUPS = ("outliers", "matched", "cont10", "cont15")

ctl = np.load(OUT / "controls.npz")
groups = {"outliers": ctl["outliers"]} | {g: ctl[g].ravel() for g in GROUPS[1:]}
every = np.unique(np.concatenate(list(groups.values())))

# %% M1, spectrum against fiber photometry
h = fits.open(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", memmap=True)
M, S = h["METADATA"].data, h["FASTSPEC"].data
vrow = {int(t): i for i, t in enumerate(M["TARGETID"])}
m = {}
for t in every.tolist():
    i = vrow[t]
    m[t] = {"z": float(M["Z"][i]), "tiles": [int(x) for x in str(M["TILEID_LIST"][i]).split()]}
    for phot, suffix in (("FIBERFLUX", ""), ("FIBERTOTFLUX", "_tot")):
        ratio = {}
        for b in "GRZ":
            fs, ff = float(S[f"FLUX_SYNTH_{b}"][i]), float(M[f"{phot}_{b}"][i])
            ratio[b] = fs / ff if fs > 0 and ff > 0 else np.nan
        m[t]["dgr" + suffix] = -2.5 * np.log10(ratio["G"] / ratio["R"])
        m[t]["drz" + suffix] = -2.5 * np.log10(ratio["R"] / ratio["Z"])

# %% M2, exposure sky from the tiles
exps = defaultdict(list)
with open(OUT / "sv3_bright_exposures.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        exps[int(r["tileid"])].append((int(r["expid"]), float(r["sky_mag_g_spec"])))
check = sorted(exps[560])
print(f"check, tile 560: {check}")  # 2026-10-30: 87837-87839, sky g 22.05 / 20.67 / 19.09
for d in m.values():
    sky = [s for tile in d["tiles"] for _, s in exps.get(tile, [])]
    d["n_exp"] = len(sky)
    d["sky_min"] = min(sky) if sky else np.nan  # brightest sky
    d["sky_max"] = max(sky) if sky else np.nan

# %% M3, ivar against the scatter, one chunk file at a time
wave = DESI._wave_obs.numpy()
gidx = tids_to_indices(every.astype(np.int64))
by_chunk = defaultdict(list)
for t, g in zip(every.tolist(), gidx.tolist(), strict=True):
    by_chunk[g // 1024].append((t, g % 1024))
for c, rows in sorted(by_chunk.items()):
    with open(PATHS["DATA"] / "desi_spectra" / f"DESIchunk1024_{c}.pkl", "rb") as f:
        s, w, _, tid, norm, *_ = pickle.load(f)
    for t, r in rows:
        assert int(tid[r]) == t, "wrong chunk row"
        n = float(norm[r])
        flux, ivar = np.asarray(s[r], float) * n, np.asarray(w[r], float) / n**2
        for arm, (lo, hi) in ARMS.items():
            good = (ivar > 0) & (wave >= lo) & (wave < hi)
            if good.sum() < KERNEL:  # too few pixels for the median filter
                m[t][f"sig_{arm}"] = np.nan
                continue
            chi = (flux[good] - medfilt(flux[good], KERNEL)) * np.sqrt(ivar[good])
            m[t][f"sig_{arm}"] = 1.4826 * np.median(np.abs(chi - np.median(chi)))

# %% DESI zpix flags and fiber diagnostics (flags.py)
with open(OUT / "zpix_flags.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        t = int(r["targetid"])
        if t not in m:
            continue
        m[t]["fiberstatus"] = int(r["coadd_fiberstatus"])
        m[t]["zwarn"] = int(r["zwarn"])
        m[t]["numexp"] = int(r["coadd_numexp"])
        # mean and rms fiber position offset in the focal plane, mm
        m[t]["pos_off"] = float(np.hypot(float(r["mean_delta_x"]), float(r["mean_delta_y"])))
        m[t]["pos_rms"] = float(np.hypot(float(r["rms_delta_x"]), float(r["rms_delta_y"])))
        m[t]["psf2fib"] = float(r["mean_psf_to_fiber_specflux"])

# %% tables
KEYS = (
    "dgr", "drz", "dgr_tot", "drz_tot", "sky_min", "sky_max", "n_exp", "numexp",
    "pos_off", "pos_rms", "psf2fib", "sig_b", "sig_r", "sig_z",
)  # fmt: skip
lines = [
    f"pool {int(ctl['pool_size'])}; 10 controls per outlier per set, no reuse within a set; "
    "controls not screened (Gaia, SGA, SIMBAD, PSF, BPT, shred).",
    "Each cell: median [16th, 84th percentile] (number finite).",
    "",
]
for g in GROUPS:
    tl = groups[g].tolist()
    nfs = sum(m[t]["fiberstatus"] != 0 for t in tl)
    nzw = sum(m[t]["zwarn"] != 0 for t in tl)
    same = sum(m[t]["numexp"] == m[t]["n_exp"] for t in tl)
    lines.append(
        f"{g:>9}: COADD_FIBERSTATUS != 0 in {nfs}, ZWARN != 0 in {nzw}, "
        f"zpix COADD_NUMEXP equals the tile count in {same} of {len(tl)}"
    )
lines.append("")
for k in KEYS:
    lines.append(k)
    for g in GROUPS:
        v = np.array([m[t][k] for t in groups[g].tolist()], float)
        v = v[np.isfinite(v)]
        q = np.percentile(v, [50, 16, 84])
        lines.append(f"  {g:>9}: {q[0]:8.3f} [{q[1]:8.3f}, {q[2]:8.3f}] ({len(v)})")
text = "\n".join(lines)
(OUT / "qa_summary.txt").write_text(text + "\n")
print(text)

hdr = f"{'TARGETID':>18} {'z':>6} " + " ".join(f"{k:>7}" for k in KEYS)
rows = [hdr] + [
    f"{t:>18} {m[t]['z']:6.3f} " + " ".join(f"{m[t][k]:7.3f}" for k in KEYS)
    for t in groups["outliers"].tolist()
]
(OUT / "qa_outliers.txt").write_text("\n".join(rows) + "\n")

# %% does dgr track the sky or the fiber diagnostics? Spearman, within each group
corr = ["Spearman rank correlation of dgr with each quantity: rho, p, N."]
for g in ("outliers", "matched"):
    for k in ("sky_min", "pos_off", "pos_rms", "psf2fib"):
        x = np.array([m[t]["dgr"] for t in groups[g].tolist()])
        y = np.array([m[t][k] for t in groups[g].tolist()])
        ok = np.isfinite(x) & np.isfinite(y)
        rho, p = spearmanr(x[ok], y[ok])
        corr.append(f"  {g:>8} dgr vs {k:>8}: rho {rho:+.3f}, p {p:.2g}, N {int(ok.sum())}")
text = "\n".join(corr)
(OUT / "qa_correlations.txt").write_text(text + "\n")
print(text)

# %% figure: cumulative distributions per metric and group
use_apj_style()
colors = {"outliers": "k", "matched": "#d95f02", "cont10": "#1b9e77", "cont15": "#7570b3"}
labels = {
    "dgr": r"$\Delta(g-r)$ spectrum $-$ fiber [mag]",
    "drz": r"$\Delta(r-z)$ spectrum $-$ fiber [mag]",
    "sky_min": r"brightest exposure sky $g$ [mag arcsec$^{-2}$]",
    "sig_b": r"$\sigma(\chi)$, $\lambda < 5800$ \AA",
    "sig_r": r"$\sigma(\chi)$, 5800--7600 \AA",
    "sig_z": r"$\sigma(\chi)$, $\lambda > 7600$ \AA",
}
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (k, lab) in zip(axes.ravel(), labels.items(), strict=True):
    for g in GROUPS:
        v = np.sort([m[t][k] for t in groups[g].tolist() if np.isfinite(m[t][k])])
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=colors[g],
                lw=2 if g == "outliers" else 1.2, label=f"{g} ({len(v)})")  # fmt: skip
    ax.set_xlabel(lab, fontsize=13)
    ax.set_ylabel("cumulative fraction", fontsize=13)
axes[0, 0].legend(frameon=False, fontsize=11, loc="center right")
fig.savefig(OUT / "qa_dist.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'qa_dist.png'}")
