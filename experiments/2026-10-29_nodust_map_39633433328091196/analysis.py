"""MAPs with and without dust of the continuum outlier 39633433328091196 (C. Hahn's test, issue #17).

QUESTION: is the best fit without dust about as good as the one with dust, for C3K ER and MILES,
and where in rest wavelength does turning dust off cost chi2?
HYPOTHESIS: none registered.
INPUTS: C3K ER with dust, <TID>_map.npz of 2026-10-29_continuum_39633433328091196_map (icx job
3657895); C3K ER without dust, er_nodust/<TID>_map.npz of fit.py (icx job 3658736); MILES with and
without dust, miles_dust/ and miles_nodust/ of fit.py (icx job 3658941); C3K ER with rest
3850-4450 and 4600-4750 A masked, with and without dust, er_dust_features/ and er_nodust_features/
(follow-up 3, icx jobs 3659211 and 3659176); both libraries with the logzsol prior capped at +0.2,
*_zmax0.2/ (follow-up 2, icx jobs 3659851, 3659852, 3659853). Copied from stampede3.
SEED: none.
COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-29_nodust_map_39633433328091196/analysis.py``.
RESULT: 2026-10-29, nodust.txt. Delta chi2 (no dust minus dust): C3K ER +112.9 (delta BIC +87.6),
with logzsol <= 0.2 +112.3, with 3850-4450 and 4600-4750 A masked +105.8; MILES +15.2 (delta BIC
-9.8), with logzsol <= 0.2 +35.6 (+10.6). The cost sits at the blue end, 4300-5300 A and the red end.
Masked, C3K's slope is free (steep low-dust start within 1.3 in ln p of the grey one). Assumptions:
MAPs from 7 starts (4 for capped C3K with dust), fixed SFH hyperparameters, DR1 coadd ivar.
FIGURES: results/2026-10-29_nodust_map_39633433328091196/nodust_<lib>.png and nodust.txt;
miles_dust_zcap.png and .txt (MILES with dust, logzsol capped or not); er_vs_miles_zcap.png.
"""

# %%
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

from hubersed.conversion import to_flambda
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39633433328091196
OUT = PATHS["RESULTS"] / "2026-10-29_nodust_map_39633433328091196"
MAPDIR = PATHS["RESULTS"] / "2026-10-29_continuum_39633433328091196_map"
PAIRS = {
    "C3K ER": (MAPDIR / f"{TID}_map.npz", OUT / "er_nodust" / f"{TID}_map.npz"),
    "MILES": (OUT / "miles_dust" / f"{TID}_map.npz", OUT / "miles_nodust" / f"{TID}_map.npz"),
    "C3K ER masked": (
        OUT / "er_dust_features" / f"{TID}_map.npz",
        OUT / "er_nodust_features" / f"{TID}_map.npz",
    ),
    "C3K ER zmax0.2": (
        OUT / "er_dust_zmax0.2" / f"{TID}_map.npz",
        OUT / "er_nodust_zmax0.2" / f"{TID}_map.npz",
    ),
    "MILES zmax0.2": (
        OUT / "miles_dust_zmax0.2" / f"{TID}_map.npz",
        OUT / "miles_nodust_zmax0.2" / f"{TID}_map.npz",
    ),
}
COLORS = {"dust": "#b2182b", "no dust": "#2166ac"}
BINS = [(lo, lo + 500) for lo in range(3300, 7300, 500)]  # rest A


def flam(d, x):
    """Return maggies on the fit's wavelength grid in 1e-17 erg/s/cm^2/A."""
    return to_flambda(np.asarray(d["wave"], float) * u.AA, np.asarray(x, float) * u.mgy).value


def chi(d):
    """Return the chi of a fit on its own data, NaN on unfitted pixels."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d["good"], (d["flux"] - d["sp_best"]) / d["unc"], np.nan)


# %% table: chi2, parameters and chi2 change by rest-frame bin
lines = []
done = {}
for lib, (p_dust, p_none) in PAIRS.items():
    if not (p_dust.exists() and p_none.exists()):
        lines += [f"{lib}: not run yet", ""]
        continue
    a, b = np.load(p_dust), np.load(p_none)
    assert np.array_equal(a["good"], b["good"]), f"{lib}: different pixels"
    assert np.allclose(a["flux"][a["good"]], b["flux"][b["good"]]), f"{lib}: different data"
    g = a["good"]
    ca, cb = chi(a), chi(b)
    x2a, x2b = float(np.sum(ca[g] ** 2)), float(np.sum(cb[g] ** 2))
    na, nb = len(a["labels"]), len(b["labels"])
    dbic = (x2b - x2a) - (na - nb) * np.log(g.sum())
    done[lib] = (a, b)
    lines += [
        f"{lib}: {g.sum()} px; chi2 with dust {x2a:.1f} ({na} free), without {x2b:.1f} ({nb} free); "
        f"delta chi2 {x2b - x2a:+.1f}, delta BIC {dbic:+.1f}",
        f"  best start with dust {int(a['start'])}, without {int(b['start'])}",
        f"  {'parameter':>16} {'dust':>10} {'no dust':>10}",
    ]
    pa = dict(zip([str(x) for x in a["labels"]], a["best"], strict=True))
    pb = dict(zip([str(x) for x in b["labels"]], b["best"], strict=True))
    for k in pa:
        nd = f"{pb[k]:10.4f}" if k in pb else f"{'fixed':>10}"
        lines.append(f"  {k:>16} {pa[k]:10.4f} {nd}")
    rest = a["wave"] / (1 + float(a["z"]))
    lines.append(f"  {'rest A':>11} {'px':>5} {'chi2/N dust':>11} {'no dust':>8} {'d chi2':>8}")
    for lo, hi in BINS:
        s = g & (rest >= lo) & (rest < hi)
        if not s.any():
            continue
        lines.append(
            f"  {lo:5d}-{hi:5d} {s.sum():5d} {np.mean(ca[s] ** 2):11.3f} {np.mean(cb[s] ** 2):8.3f} "
            f"{np.sum(cb[s] ** 2 - ca[s] ** 2):+8.1f}"
        )
    lines.append("")
text = "\n".join(lines)
(OUT / "nodust.txt").write_text(text + "\n")
print(text)

# %% figure per library: data with both MAPs, then the chi of each
use_apj_style()
for lib, (a, b) in done.items():
    rest = a["wave"] / (1 + float(a["z"]))
    g = a["good"]
    data = flam(a, a["flux"])
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.06})  # fmt: skip
    ax[0].plot(rest, np.where(g, data, np.nan), color="0.75", lw=0.5, label="DESI SV3 (DR1 coadd)")
    med = np.full(data.size, np.nan)
    med[g] = medfilt(data[g], 9)
    ax[0].plot(rest, med, color="k", lw=0.8, label="median filter, 9 pixels")
    for (key, d), a_chi in zip((("dust", a), ("no dust", b)), ax[1:], strict=True):
        ax[0].plot(rest, np.where(g, flam(d, d["sp_best"]), np.nan), color=COLORS[key], lw=0.8,
                   label=f"{lib} MAP, {key}")  # fmt: skip
        a_chi.plot(rest, chi(d), color=COLORS[key], lw=0.5)
        for h in (-2, 2):
            a_chi.axhline(h, color="0.3", ls="--", lw=1)
        a_chi.axhline(0, color="0.3", lw=0.6)
        a_chi.set_ylim(-5, 5)
        a_chi.set_ylabel(r"$\chi$", fontsize=14)
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
    ax[0].set_ylim(bottom=0)
    ax[0].legend(frameon=False, fontsize=12, loc="upper right")
    ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
    path = OUT / f"nodust_{lib.lower().replace(' ', '_').replace('.', '')}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

# %% MILES with dust, logzsol prior uncapped against capped at +0.2: data, both MAPs and both chi
a = np.load(OUT / "miles_dust" / f"{TID}_map.npz")
b = np.load(OUT / "miles_dust_zmax0.2" / f"{TID}_map.npz")
assert np.array_equal(a["good"], b["good"]), "different pixels"
g = a["good"]
rest = a["wave"] / (1 + float(a["z"]))
cap_colors = {"logzsol <= 0.5": "#b2182b", "logzsol <= 0.2": "#2166ac"}
fig, ax = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})  # fmt: skip
ax[0].plot(rest, np.where(g, flam(a, a["flux"]), np.nan), color="0.6", lw=0.5,
           label="DESI SV3 (DR1 coadd)")  # fmt: skip
for key, d in (("logzsol <= 0.5", a), ("logzsol <= 0.2", b)):
    ax[0].plot(rest, np.where(g, flam(d, d["sp_best"]), np.nan), color=cap_colors[key], lw=0.8,
               label="MILES MAP with dust, logzsol $\\leq$ " + key[-3:])  # fmt: skip
    ax[1].plot(rest, chi(d), color=cap_colors[key], lw=0.5, alpha=0.7)
for h in (-2, 2):
    ax[1].axhline(h, color="0.3", ls="--", lw=1)
ax[1].axhline(0, color="0.3", lw=0.6)
ax[1].set_ylim(-5, 5)
ax[1].set_ylabel(r"$\chi$", fontsize=14)
ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
ax[0].legend(frameon=False, fontsize=12, loc="lower right")
ax[1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
fig.savefig(OUT / "miles_dust_zcap.png", dpi=130, bbox_inches="tight")
plt.close(fig)
pa = dict(zip([str(x) for x in a["labels"]], a["best"], strict=True))
pb = dict(zip([str(x) for x in b["labels"]], b["best"], strict=True))
rows = [f"{'parameter':>16} {'<= 0.5':>10} {'<= 0.2':>10} {'change':>10}"]
rows += [f"{k:>16} {pa[k]:10.4f} {pb[k]:10.4f} {pb[k] - pa[k]:+10.4f}" for k in pa]
(OUT / "miles_dust_zcap.txt").write_text("\n".join(rows) + "\n")
print("\n".join(rows))

# %% C3K ER MAP with dust against the MILES MAP with dust and logzsol <= 0.2: data, both MAPs, chi of each
e = np.load(MAPDIR / f"{TID}_map.npz")
m = np.load(OUT / "miles_dust_zmax0.2" / f"{TID}_map.npz")
assert not np.any(m["good"] & ~e["good"]), "MILES fits pixels the ER fit drops"
assert np.allclose(e["flux"][m["good"]], m["flux"][m["good"]]), "different data"
rest = e["wave"] / (1 + float(e["z"]))
fits = (("C3K ER MAP with dust", e, "#b2182b"),
        (r"MILES MAP with dust, logzsol $\leq$ 0.2", m, "#2166ac"))  # fmt: skip
fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.06})  # fmt: skip
ax[0].plot(rest, np.where(e["good"], flam(e, e["flux"]), np.nan), color="0.6", lw=0.5,
           label="DESI SV3 (DR1 coadd)")  # fmt: skip
for (label, d, color), a_chi in zip(fits, ax[1:], strict=True):
    ax[0].plot(rest, np.where(d["good"], flam(d, d["sp_best"]), np.nan), color=color, lw=0.8,
               label=label)  # fmt: skip
    a_chi.plot(rest, chi(d), color=color, lw=0.5)
    for h in (-2, 2):
        a_chi.axhline(h, color="0.3", ls="--", lw=1)
    a_chi.axhline(0, color="0.3", lw=0.6)
    a_chi.set_ylim(-5, 5)
    a_chi.set_ylabel(r"$\chi$", fontsize=14)
ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
ax[0].legend(frameon=False, fontsize=12, loc="lower right")
ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
fig.savefig(OUT / "er_vs_miles_zcap.png", dpi=130, bbox_inches="tight")
plt.close(fig)
