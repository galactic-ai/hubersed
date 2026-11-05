"""1D posteriors of one target's free parameters for three ladder runs, on the prior range.

The runs are MILES, C3K HR + afe (degraded) and C3K ER + afe + R matrix on the DR1 coadd, loaded as
in ladder_corner.py. One panel per free parameter of the DR1 run. The x axis spans the union of the
runs' prior bounds. Bins are 30 equal steps over the 0.1 to 99.9 per cent span of the runs' weighted
samples, so narrow posteriors stay resolved on a wide axis. The logsfr_ratios prior is a Gaussian
set by the SFH hyperparameters, with no bounds, so its nine panels share one axis and one set of
bins: the widest 0.1 to 99.9 per cent span over the three runs and nine ratios. A run without a
parameter (afe in MILES) has no curve in that panel. With afe free logzsol is [Fe/H], without it [Z/H].
A last panel puts every run on log10(Z/Zsun), with Z the initial metal mass fraction of the MIST2
model (mist2_zh.py), linear in log Z between the ([Fe/H], [alpha/Fe]) grid nodes, and Zsun the Z
of the [Fe/H] = 0, [alpha/Fe] = 0 model. Solar-scaled runs take logzsol as [Fe/H] at [alpha/Fe] = 0.
Its axis spans the prior corners of logzsol and afe. ladder_metallicity.txt gives log10(Z/Zsun) of
every run, and [Fe/H] and [alpha/Fe] of the afe runs, as 16, 50 and 84 per cent quantiles.
COMMAND: ``uv run --no-sync python experiments/2026-11-01_poco_repeats/ladder_hist.py --tid <TARGETID>``.
OUTPUT: results/2026-11-01_poco_repeats/<TARGETID>/ladder_hist.png and ladder_metallicity.txt.
"""

# %%
import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator

from hubersed.plotting.style import use_apj_style

HERE = Path(__file__).resolve().parent
ZLAB = r"$\log Z/Z_\odot$"
RUNS = ("MILES", "C3K HR + afe (degraded)", "C3K ER + afe + R matrix, DR1 coadd")


def load(name, path):
    """Load the file at ``path`` as module ``name``."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mist2_logz(path):
    """Return log10(Z/Zsun) of the MIST2 grid as a function of [Fe/H] and [alpha/Fe], and Zsun."""
    t = np.loadtxt(path, skiprows=1, comments="[")
    feh, afe = np.unique(t[:, 0]), np.unique(t[:, 1])
    assert len(t) == feh.size * afe.size, "MIST2 grid is not regular"
    logz = np.full((feh.size, afe.size), np.nan)
    logz[np.searchsorted(feh, t[:, 0]), np.searchsorted(afe, t[:, 1])] = np.log10(t[:, 3])
    zsun = 10 ** logz[feh == 0.0, afe == 0.0].item()
    interp = RegularGridInterpolator((feh, afe), logz - np.log10(zsun))
    return lambda f, a: interp(np.column_stack(np.broadcast_arrays(f, a))), zsun


def table(ana, rr, path, zsun):
    """Write log10(Z/Zsun) of every run, and [Fe/H] and [alpha/Fe] of the afe runs, to ``path``."""
    q = np.array([0.16, 0.5, 0.84])
    lines = [f"{'run':36s} {'log Z/Zsun':>22} {'[Fe/H]':>22} {'[alpha/Fe]':>22}"]
    for label, _, _, cols, w, _ in rr:
        cells = []
        for key in (ZLAB, "logzsol", "afe"):
            if key == ZLAB or "afe" in cols:
                a, m, b = ana.wquantile(cols[key], w, q)
                cells.append(f"{m:+8.3f} -{m - a:.3f} +{b - m:.3f}")
            else:
                cells.append(f"{'-':>22}")
        lines.append(f"{label:36s} " + " ".join(f"{c:>22}" for c in cells))
    lines += ["", f"median -(median - q16) +(q84 - median); Zsun = {zsun:.6f}, MIST2 Z at [Fe/H] = 0,",
              "[alpha/Fe] = 0 (mist2_zh.py); [Fe/H] is logzsol of the afe runs"]  # fmt: skip
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(tid):
    """Draw the 1D posteriors of ``tid`` for the three runs of RUNS."""
    lc = load("ladder_corner", HERE / "ladder_corner.py")
    ana, rr = lc.runs(tid)
    rr = [r for r in rr if r[0] in RUNS]
    assert [r[0] for r in rr] == list(RUNS), f"runs found: {[r[0] for r in rr]}"
    names = list(rr[-1][2].theta_labels())
    logz, zsun = mist2_logz(lc.RES / "mist2_zh.txt")
    print(f"Zsun {zsun:.6f}")
    for *_, cols, _, _ in rr:
        cols[ZLAB] = logz(cols["logzsol"], cols.get("afe", 0.0))
    names.append(ZLAB)
    table(ana, rr, lc.RES / str(tid) / "ladder_metallicity.txt", zsun)
    use_apj_style()
    ncol = 5
    nrow = -(-len(names) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow))

    def span(ps):
        q = [ana.wquantile(c[p], w, np.array([0.001, 0.999])) for p in ps for *_, c, w, _ in rr
             if p in c]  # fmt: skip
        return min(a for a, _ in q), max(b for _, b in q)

    sfh = [p for p in names if p.startswith("logsfr_ratios")]
    sfh_lo, sfh_hi = span(sfh)
    for ax, p in zip(axes.ravel(), names, strict=False):
        if p in sfh:
            lo, hi = sfh_lo, sfh_hi
            edges = np.linspace(lo, hi, 31)
        elif p == ZLAB:
            zlo, zhi = lc.prior_range(rr[-1][2], "logzsol")
            alo, ahi = lc.prior_range(rr[-1][2], "afe")
            lo, hi = logz(zlo, alo).item(), logz(zhi, ahi).item()
            edges = np.linspace(*span([p]), 31)
        else:
            bounds = [b for b in (lc.prior_range(m, p) for _, _, m, *_ in rr) if b]
            lo, hi = min(b[0] for b in bounds), max(b[1] for b in bounds)
            edges = np.linspace(*span([p]), 31)
        assert np.isfinite([lo, hi]).all(), f"{p} range {lo}, {hi}"
        for _, colour, _, cols, w, _ in rr:
            if p in cols:
                ax.hist(cols[p], bins=edges, weights=w, density=True, histtype="step",
                        color=colour, lw=2)  # fmt: skip
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.set_xlabel(p, fontsize=16)
        ax.tick_params(labelsize=12)
        print(f"{p:16s} axis {lo:g} to {hi:g}, bins {edges[0]:g} to {edges[-1]:g}")
    for ax in axes.ravel()[len(names) :]:
        ax.set_axis_off()
    handles = [Line2D([], [], color=c, lw=3) for _, c, *_ in rr]
    fig.legend(handles, [r[0] for r in rr], loc="upper center", ncol=len(rr), fontsize=16,
               frameon=False, bbox_to_anchor=(0.5, 1.03))  # fmt: skip
    fig.tight_layout()
    out = lc.RES / str(tid) / "ladder_hist.png"
    fig.savefig(out, dpi=80, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tid", type=int, required=True)
    main(ap.parse_args().tid)
