"""MAPs with and without dust of the SV3 dark spectrum of 39633433328091196, against SV3 bright (#17).

QUESTION: on the SV3 dark spectrum, does turning dust off still cost chi2 for C3K ER and MILES,
and does that cost differ from the SV3 bright one?
HYPOTHESIS: none registered.
INPUTS: dark C3K ER with dust, <TID>_map.npz of 2026-10-30_continuum_39633433328091196_dark_map
(spr job 3662995); dark C3K ER without dust and dark MILES with and without dust (logzsol <= 0.2),
er_nodust/, miles_dust_zmax0.2/ and miles_nodust_zmax0.2/ of fit.py (spr job 3663513); bright
pairs from 2026-10-29_continuum_39633433328091196_map and 2026-10-29_nodust_map_39633433328091196.
Copied from stampede3.
SEED: none.
COMMAND: from the repository root on the Mac,
``uv run --no-sync python experiments/2026-10-30_nodust_miles_39633433328091196_dark/analysis.py``.
RESULT: 2026-10-30, nodust_dark.txt. Delta chi2 (no dust minus dust), dark against bright: C3K ER
+387.0 (delta BIC +361.7) against +112.9 (+87.6); MILES +292.5 (+267.6) against +35.6 (+10.6). On
dark the cost sits at 4300-5300 A and above 6300 A rest. MILES dark with dust, start 3 is 44.2 above
the best in -ln p. Assumptions: 4 starts on dark, fixed SFH hyperparameters, DR1 coadd ivar of one
dark exposure, MILES logzsol at the +0.2 cap.
FIGURES: results/2026-10-30_nodust_miles_39633433328091196_dark/nodust_dark.txt and
nodust_dark_<lib>.png.
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
RES = PATHS["RESULTS"]
OUT = RES / "2026-10-30_nodust_miles_39633433328091196_dark"
DARK_ER = RES / "2026-10-30_continuum_39633433328091196_dark_map"
BRIGHT_ER = RES / "2026-10-29_continuum_39633433328091196_map"
BRIGHT = RES / "2026-10-29_nodust_map_39633433328091196"
NPZ = f"{TID}_map.npz"
PAIRS = {
    "C3K ER": {
        "dark": (DARK_ER / NPZ, OUT / "er_nodust" / NPZ),
        "bright": (BRIGHT_ER / NPZ, BRIGHT / "er_nodust" / NPZ),
    },
    "MILES logzsol <= 0.2": {
        "dark": (OUT / "miles_dust_zmax0.2" / NPZ, OUT / "miles_nodust_zmax0.2" / NPZ),
        "bright": (BRIGHT / "miles_dust_zmax0.2" / NPZ, BRIGHT / "miles_nodust_zmax0.2" / NPZ),
    },
}
SLUG = {"C3K ER": "c3k_er", "MILES logzsol <= 0.2": "miles"}
COLORS = {"dust": "#b2182b", "no dust": "#2166ac"}
BINS = [(lo, lo + 500) for lo in range(3300, 7300, 500)]  # rest A


def flam(d, x):
    """Return maggies on the fit's wavelength grid in 1e-17 erg/s/cm^2/A."""
    return to_flambda(np.asarray(d["wave"], float) * u.AA, np.asarray(x, float) * u.mgy).value


def chi(d):
    """Return the chi of a fit on its own data, NaN on unfitted pixels."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d["good"], (d["flux"] - d["sp_best"]) / d["unc"], np.nan)


def spread(d):
    """Return -ln p of each start minus the best one, as a short string."""
    f = np.asarray(d["fun_all"], float)
    return " ".join(f"{x:.1f}" for x in f - f.min())


# %% table: chi2, parameters and chi2 change by rest-frame bin, dark and bright
lines = []
loaded = {}
for lib, spectra in PAIRS.items():
    lines.append(f"{lib}")
    params, cols, bins = {}, [], {}
    for spec, (p_dust, p_none) in spectra.items():
        a, b = np.load(p_dust), np.load(p_none)
        assert np.array_equal(a["good"], b["good"]), f"{lib} {spec}: different pixels"
        assert np.allclose(a["flux"][a["good"]], b["flux"][b["good"]]), (
            f"{lib} {spec}: different data"
        )
        g = a["good"]
        ca, cb = chi(a), chi(b)
        x2a, x2b = float(np.sum(ca[g] ** 2)), float(np.sum(cb[g] ** 2))
        na, nb = len(a["labels"]), len(b["labels"])
        dbic = (x2b - x2a) - (na - nb) * np.log(g.sum())
        loaded[(lib, spec)] = (a, b)
        lines += [
            f"  {spec}: {g.sum()} px; chi2 with dust {x2a:.1f} ({na} free), without {x2b:.1f} "
            f"({nb} free); delta chi2 {x2b - x2a:+.1f}, delta BIC {dbic:+.1f}",
            f"    best start with dust {int(a['start'])}, -ln p above best per start: {spread(a)}",
            f"    best start without dust {int(b['start'])}, -ln p above best per start: {spread(b)}",
        ]
        for key, d in (("dust", a), ("no dust", b)):
            cols.append(f"{spec} {key}")
            for k, v in zip([str(x) for x in d["labels"]], d["best"], strict=True):
                params.setdefault(k, {})[cols[-1]] = float(v)
        rest = a["wave"] / (1 + float(a["z"]))
        for lo, hi in BINS:
            s = g & (rest >= lo) & (rest < hi)
            bins.setdefault((lo, hi), {})[spec] = (
                int(s.sum()),
                float(np.sum(cb[s] ** 2 - ca[s] ** 2)),
            )
    lines.append("  " + f"{'parameter':>16}" + "".join(f" {c:>15}" for c in cols))
    for k, row in params.items():
        cells = "".join(f" {row[c]:15.4f}" if c in row else f" {'fixed':>15}" for c in cols)
        lines.append(f"  {k:>16}{cells}")
    lines.append(f"  {'rest A':>11} {'dark px':>8} {'d chi2':>8} {'bright px':>10} {'d chi2':>8}")
    for (lo, hi), row in bins.items():
        nd, dd = row["dark"]
        nbr, db = row["bright"]
        if nd == 0 and nbr == 0:
            continue
        lines.append(f"  {lo:5d}-{hi:5d} {nd:8d} {dd:+8.1f} {nbr:10d} {db:+8.1f}")
    lines.append("")
text = "\n".join(lines)
(OUT / "nodust_dark.txt").write_text(text + "\n")
print(text)

# %% figure per library: dark data with both MAPs, then the chi of each
use_apj_style()
for lib in PAIRS:
    a, b = loaded[(lib, "dark")]
    rest = a["wave"] / (1 + float(a["z"]))
    g = a["good"]
    data = flam(a, a["flux"])
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.06})  # fmt: skip
    ax[0].plot(
        rest, np.where(g, data, np.nan), color="0.75", lw=0.5, label="DESI SV3 dark (DR1 coadd)"
    )
    med = np.full(data.size, np.nan)
    med[g] = medfilt(data[g], 9)
    ax[0].plot(rest, med, color="k", lw=0.8, label="median filter, 9 pixels")
    for (key, d), a_chi in zip((("dust", a), ("no dust", b)), ax[1:], strict=True):
        ax[0].plot(rest, np.where(g, flam(d, d["sp_best"]), np.nan), color=COLORS[key], lw=0.8,
                   label=f"{lib.replace('<=', r'$\leq$')} MAP, {key}")  # fmt: skip
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
    path = OUT / f"nodust_dark_{SLUG[lib]}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
