"""How FSPS builds MILES SSPs above the top of the MILES metallicity grid.

QUESTION: above log Z/Zsun +0.2 (MILES top, Z 0.030), does FSPS give the +0.2 MILES star spectra
at the Teff and logg of the higher-Z MIST isochrone, so colour moves but line strength at fixed
Teff and logg does not?
HYPOTHESIS: yes, from the clamp in sps_setup.f90:331-335. C3K_HR, whose grid reaches +0.5,
is the control.
INPUTS: ssp_miles.npz (miles_mac build) and ssp_c3k_hr.npz (hubersed venv) from ssp.py.
SEED: none.
COMMAND: from the repository root on the Mac, with SPS_HOME=~/Astronomy_Research/fsps and
OMP_NUM_THREADS=1,
``~/Astronomy_Research/fsps_builds/miles_mac/.venv/bin/python experiments/2026-10-30_miles_zcap_test/ssp.py --name miles``,
``uv run --no-sync python experiments/2026-10-30_miles_zcap_test/ssp.py --name c3k_hr``, then
``uv run --no-sync python experiments/2026-10-30_miles_zcap_test/analysis.py``.
RESULT: 2026-10-30, ratio.txt. Partial. MILES spectra are frozen from MIST node +0.25 up (library swap
12->13 changes nothing); indices still grow via cooler MIST stars (Mgb +0.27 A, iso 12->13), but the
line-blanketing part of the blue continuum is frozen too. logzsol 0.2 mixes 20% node 0 library.
Assumptions: 10 Gyr scaled-solar SSPs, native resolution, ad hoc CaHK band.
FIGURES: results/2026-10-30_miles_zcap_test/ratio.png and ratio.txt.
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style
from hubersed.sps.utils import _airtovac

OUT = PATHS["RESULTS"] / "2026-10-30_miles_zcap_test"
RANGE = (3550.0, 7350.0)  # rest vacuum A, inside the FSPS MILES range
NORM = (5450.0, 5550.0)  # rest A window the ratios are normalised in
BIN = 100.0  # A, bin width of the continuum ratio
LICK = ("Hbeta", "Mgb", "Fe5270", "Fe5335", "G4300", "NaD")  # from SPS_HOME/data/allindices.dat
# Ca H&K band of my own (air A): feature, blue and red pseudo-continua. Not a published index.
CAHK = (3920.0, 3985.0, 3895.0, 3915.0, 3990.0, 4010.0)
PAIRS = ((0.2, 0.35), (0.2, 0.5), (0.0, 0.2))  # (reference, target) logzsol; 0 to 0.2 is in-grid

# %% load
lib = {k: np.load(OUT / f"ssp_{k}.npz") for k in ("miles", "c3k_hr")}
for k, d in lib.items():
    print(k, list(d["libraries"]), str(d["fsps_file"]))


def read_indices():
    """Return air-wavelength Lick band edges by name from SPS_HOME/data/allindices.dat."""
    path = Path(os.environ["SPS_HOME"]) / "data" / "allindices.dat"
    out = {}
    for line in path.read_text().splitlines():
        p = line.split()
        if line.startswith("#") or len(p) < 8:
            continue
        out[p[7]] = tuple(float(x) for x in p[:6])
    return out


BANDS = {k: v for k, v in read_indices().items() if k in LICK}
BANDS["CaHK"] = CAHK
BANDS = {k: tuple(_airtovac(x) for x in v) for k, v in BANDS.items()}


def ew(wave, flam, band):
    """Return a pseudo-continuum equivalent width in A, positive for absorption."""
    f0, f1, b0, b1, r0, r1 = band
    fine = np.arange(min(b0, f0), max(r1, f1), 0.05)
    f = np.interp(fine, wave, flam)

    def mean(lo, hi):
        s = (fine >= lo) & (fine <= hi)
        return f[s].mean()

    xb, xr = 0.5 * (b0 + b1), 0.5 * (r0 + r1)
    cb, cr = mean(b0, b1), mean(r0, r1)
    s = (fine >= f0) & (fine <= f1)
    cont = cb + (cr - cb) * (fine[s] - xb) / (xr - xb)
    return np.trapezoid(1 - f[s] / cont, fine[s])


def binned(wave, flam):
    """Return 100 A bin centres and bin means normalised by the mean over NORM."""
    edges = np.arange(RANGE[0], RANGE[1] + 1, BIN)
    norm = flam[(wave >= NORM[0]) & (wave <= NORM[1])].mean()
    m = [
        flam[(wave >= lo) & (wave < hi)].mean() / norm
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    return 0.5 * (edges[1:] + edges[:-1]), np.array(m)


def compare(wave, a, b):
    """Return binned b/a, the max |b/a - 1| over RANGE at native sampling, and index changes."""
    mid, ma = binned(wave, a)
    _, mb = binned(wave, b)
    s = (wave >= RANGE[0]) & (wave <= RANGE[1])
    raw = np.max(np.abs(b[s] / a[s] - 1))
    dew = {k: ew(wave, b, v) - ew(wave, a, v) for k, v in BANDS.items()}
    return mid, mb / ma, raw, dew


# %% SSP at logzsol pairs, zcontinuous 1
lines = [
    "10 Gyr dust-free Kroupa SSPs, MIST. Continuum: 100 A bin means normalised at 5450-5550 A.",
    "dEW in A (target minus reference); b3600 = 3550-3650 A bin; raw = max |ratio-1| at native sampling.",
    "bands (vacuum, from allindices.dat via _airtovac; CaHK ad hoc): "
    + "; ".join(f"{k} {v[0]:.1f}-{v[1]:.1f}" for k, v in BANDS.items()),
    "",
    f"{'lib':7} {'pair':10} {'b3600':>7} {'bin min':>7} {'bin max':>7} {'raw':>6} "
    + " ".join(f"{k:>7}" for k in BANDS),
]
ratios = {}
for k, d in lib.items():
    z = list(np.round(d["logzsol"], 2))
    ew0 = {b: ew(d["wave"], d["spec"][z.index(0.2)], v) for b, v in BANDS.items()}
    lines.append(
        f"{k:7} {'EW at 0.2':10} {'':>7} {'':>7} {'':>7} {'':>6} "
        + " ".join(f"{ew0[b]:7.3f}" for b in BANDS)
    )
    for r0, r1 in PAIRS:
        mid, rat, raw, dew = compare(d["wave"], d["spec"][z.index(r0)], d["spec"][z.index(r1)])
        ratios[(k, r0, r1)] = (mid, rat)
        lines.append(
            f"{k:7} {f'{r0}->{r1}':10} {rat[0]:7.4f} {rat.min():7.4f} {rat.max():7.4f} {raw:6.4f} "
            + " ".join(f"{dew[b]:+7.3f}" for b in BANDS)
        )

# %% decomposition with the swap grid: swap[a, b] is isochrone of node a, library of node b
lines += [
    "",
    "swap grid, nodes 11, 12, 13 = MIST log Z/Zsun 0, +0.25, +0.5",
    f"{'lib':7} {'test':28} {'b3600':>7} {'bin min':>7} {'bin max':>7} {'raw':>6} "
    + " ".join(f"{k:>7}" for k in BANDS),
]
tests = (
    ("rebuild node 12 vs FSPS", None, 1),
    ("rebuild node 13 vs FSPS", None, 2),
    ("library 12->13, iso 13", (2, 1), (2, 2)),
    ("library 12->13, iso 12", (1, 1), (1, 2)),
    ("library 11->12, iso 12", (1, 0), (1, 1)),
    ("isochrone 12->13, lib 12", (1, 1), (2, 1)),
    ("isochrone 12->13, lib 13", (1, 2), (2, 2)),
)
for k, d in lib.items():
    for name, ia, ib in tests:
        if ia is None:
            x, y = d["node"][ib], d["swap"][ib, ib]
        else:
            x, y = d["swap"][ia], d["swap"][ib]
        _, rat, raw, dew = compare(d["wave"], x, y)
        lines.append(
            f"{k:7} {name:28} {rat[0]:7.4f} {rat.min():7.4f} {rat.max():7.4f} {raw:6.4f} "
            + " ".join(f"{dew[b]:+7.3f}" for b in BANDS)
        )

# %% isochrone temperatures at 10 Gyr
lines += [
    "",
    "10 Gyr MIST isochrone: L-weighted mean Teff, turnoff Teff (max Teff of phase 0), "
    "Teff of the RGB point nearest logg 2",
]
for zm in lib["miles"]["zmet"]:
    d = lib["miles"]
    lt, ll, lw, ph, lg = (d[f"iso{zm}_{c}"] for c in ("logt", "logl", "logweight", "phase", "logg"))
    w = 10 ** (lw + ll)
    rgb = ph == 2
    i = np.argmin(np.abs(lg[rgb] - 2.0))
    lines.append(
        f"zmet {zm}: <Teff>_L {np.sum(w * 10**lt) / w.sum():7.1f} K, turnoff "
        f"{10 ** lt[ph == 0].max():7.1f} K, RGB logg {lg[rgb][i]:.2f} Teff "
        f"{10 ** lt[rgb][i]:7.1f} K"
    )
for zm in lib["miles"]["zmet"]:
    same = all(
        np.allclose(lib["miles"][f"iso{zm}_{c}"], lib["c3k_hr"][f"iso{zm}_{c}"])
        for c in ("logt", "logg", "logweight")
    )
    lines.append(f"zmet {zm}: MILES and C3K builds have the same isochrone: {same}")

text = "\n".join(lines) + "\n"
(OUT / "ratio.txt").write_text(text)
print(text)

# %% figure: binned ratios above the MILES cap, one panel per library
use_apj_style()
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for a, k in zip(ax, lib, strict=True):
    for (r0, r1), c in zip(PAIRS[:2], ("C0", "C3"), strict=True):
        mid, rat = ratios[(k, r0, r1)]
        a.plot(mid, rat, color=c, lw=1.3, drawstyle="steps-mid", label=f"{r1} / {r0}")
    a.axhline(1, color="0.3", lw=0.6)
    a.set_title(k.upper().replace("_", " "), fontsize=13)
    a.set_xlabel(r"rest vacuum wavelength [$\AA$]", fontsize=12)
ax[0].set_ylabel(f"SSP ratio, {BIN:.0f} A bins, norm. 5500 A", fontsize=12)
ax[0].legend(frameon=False, fontsize=10, title="logzsol")
fig.savefig(OUT / "ratio.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'ratio.png'}")
