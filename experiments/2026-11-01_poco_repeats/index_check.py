"""Dn4000 and HdeltaA of one target's data and of each ladder step's max-likelihood model.

QUESTION: does each step's best model reproduce the two age indices of its data? Both barely depend
on dust, so they test age apart from the age-dust trade-off seen against FastSpecFit.
Bands come from $SPS_HOME/data/allindices.dat, as 2026-10-29_continuum_recoadd/analysis.py:
- Dn4000: mean F_nu over rest 4000-4100 A divided by that over 3850-3950 A (Balogh+1999).
- HdeltaA: equivalent width in A over 4083.5-4122.25 A, against a straight line through the mean
  F_lambda of 4041.6-4079.75 and 4128.5-4161.0 A (Worthey & Ottaviani 1997).
Data and model are measured on the same fitted pixels of each step's own data, from the
<step>_maxl.npz files that ladder_plot.py reads (degraded data for the MILES and C3K HR steps).
A band needs 80 per cent of its pixels fitted. Errors are the 16-84 per cent range over 2000 draws
of Gaussian noise with the stored uncertainties. Only the max-likelihood model is measured, not the
posterior.
COMMAND: ``uv run --no-sync python experiments/2026-11-01_poco_repeats/index_check.py --tid <TID>``.
OUTPUT: results/2026-11-01_poco_repeats/<TID>/index_check.txt.
"""

# %%
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NDRAW = 2000
MIN_GOOD = 0.8


def load(name, path):
    """Load the file at ``path`` as module ``name``."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bands():
    """Return the Dn4000 and HdeltaA band edges from allindices.dat."""
    t = np.genfromtxt(Path(os.environ["SPS_HOME"]) / "data" / "allindices.dat", dtype=None,
                      encoding=None, comments="#")  # fmt: skip
    out = {
        str(r[-1]): [float(x) for x in list(r)[:6]]
        for r in t
        if str(r[-1]) in ("Dn4000", "HdeltaA")
    }
    assert set(out) == {"Dn4000", "HdeltaA"}, out
    return out


def measure(rest, fnu, use, b):
    """Return Dn4000 and HdeltaA of F_nu on rest wavelengths, NaN where a band lacks pixels."""
    flam = fnu / rest**2  # F_lambda up to a constant, enough for ratios and widths

    def sel(lo, hi):
        s = (rest >= lo) & (rest <= hi)
        return s & use, (use[s].mean() if s.any() else 0.0)

    _, _, b0, b1, r0, r1 = b["Dn4000"]
    (sb, fb), (sr, fr) = sel(b0, b1), sel(r0, r1)
    dn = (
        np.mean(fnu[..., sr], axis=-1) / np.mean(fnu[..., sb], axis=-1)
        if min(fb, fr) >= MIN_GOOD
        else np.nan
    )
    f0, f1, b0, b1, r0, r1 = b["HdeltaA"]
    (sf, ff), (sb, fb), (sr, fr) = sel(f0, f1), sel(b0, b1), sel(r0, r1)
    if min(ff, fb, fr) < MIN_GOOD:
        return dn, np.nan
    cb, cr = np.mean(flam[..., sb], axis=-1), np.mean(flam[..., sr], axis=-1)
    xb, xr = (b0 + b1) / 2, (r0 + r1) / 2
    x = rest[sf]
    cont = cb[..., None] + (cr - cb)[..., None] * (x - xb) / (xr - xb)
    ew = np.trapezoid(1 - flam[..., sf] / cont, x, axis=-1)
    return dn, ew


def main(tid):
    """Write data and model indices of every step of ``tid``."""
    lp = load("ladder_plot", HERE / "ladder_plot.py")
    b = bands()
    rng = np.random.default_rng(0)
    lines = [f"TARGETID {tid}; Dn4000 (F_nu ratio) and HdeltaA (A), median and 16-84 per cent over noise",
             f"{'step':36s} {'Dn4000 data':>22s} {'Dn4000 model':>13s} {'HdA data':>22s} {'HdA model':>10s}"]  # fmt: skip
    for label, path, _ in lp.steps(tid):
        if not path.exists():
            lines.append(f"{label:36s} not run yet")
            continue
        d = np.load(path)
        rest = d["wave"] / (1 + float(d["z"]))
        use = np.asarray(d["good"], bool) & np.isfinite(d["unc"])
        noisy = d["flux"] + rng.normal(size=(NDRAW, rest.size)) * np.where(use, d["unc"], 0.0)
        dn_mc, hd_mc = measure(rest, noisy, use, b)
        dn_m, hd_m = measure(rest, np.asarray(d["sp_best"], float), use, b)
        cells = []
        for mc, m in ((dn_mc, dn_m), (hd_mc, hd_m)):
            if np.all(np.isnan(mc)):
                cells += [f"{'no coverage':>22s}", f"{'':>13s}"]
                continue
            lo, med, hi = np.percentile(mc, [16, 50, 84])
            cells += [f"{med:9.3f} -{med - lo:.3f} +{hi - med:.3f}", f"{m:13.3f}"]
        lines.append(f"{label:36s} " + " ".join(cells))
    lines += ["", "Model values are of the max-likelihood spectrum on the same pixels as the data."]
    out = Path(lp.RES) / str(tid) / "index_check.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tid", type=int, required=True)
    main(ap.parse_args().tid)
