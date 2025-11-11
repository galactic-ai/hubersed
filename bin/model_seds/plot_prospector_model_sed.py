#!/usr/bin/env python
# Plot Prospector NP-SFH DESI-like spectra saved as NPZ (maggies)
# Works with files produced by make_bgs_npsfh_grid_specmodel.py

import os, re, glob, argparse
import numpy as np
import matplotlib.pyplot as plt

from hubersed.conversion import maggies_to_flambda

def load_index(dirpath):
    idx = os.path.join(dirpath, "index.csv")
    if not os.path.exists(idx):
        return None
    try:
        import pandas as pd
        return pd.read_csv(idx)
    except Exception:
        return None

def npz_meta(path):
    d = np.load(path, allow_pickle=True)
    meta = d["meta"].item()
    return d["wave"], d["flux"], meta


def parse_args():
    ap = argparse.ArgumentParser(description="Plot NP-SFH DESI-like model spectra (NPZ).")
    ap.add_argument("--dir", default="models_bgs_desi_npsfh_spec",
                    help="Directory with NPZ files and index.csv")
    ap.add_argument("--file", default=None,
                    help="Plot this specific NPZ file (overrides filters)")
    ap.add_argument("--zred", type=float, default=None, help="Filter by redshift (exact)")
    ap.add_argument("--logz", type=float, default=None, help="Filter by log10(Z/Zsun) (exact)")
    ap.add_argument("--dust2", type=float, default=None, help="Filter by dust2 (exact)")
    ap.add_argument("--logu", type=float, default=None, help="Filter by gas_logu (exact)")
    ap.add_argument("--sfh", type=int, default=None, help="Filter by SFH id (exact)")

    ap.add_argument("--grid", type=int, default=1,
                    help="Number of models to plot (1 = single panel). If >1, plots first N matches.")
    ap.add_argument("--unit", choices=["maggies","flam"], default="flam",
                    help="y-axis units to plot (maggies or f_lambda)")
    ap.add_argument("--norm-window", default=None,
                    help="Continuum renorm window ÅÅ, e.g. '6000,6500' (median scaling).")
    ap.add_argument("--lines", action="store_true", help="Draw common emission-line markers.")
    ap.add_argument("--sfh-inset", action="store_true", help="Add an inset with SFH mass fractions.")
    ap.add_argument("--xlim", default="3600,9824", help="x-range, e.g. '3600,9000'")
    ap.add_argument("--save", default=None, help="Save figure to this path (png/pdf).")
    ap.add_argument("--title", default=None, help="Custom title (single panel).")
    return ap.parse_args()

def find_files(args):
    if args.file:
        return [args.file] if os.path.exists(args.file) else []

    patt = os.path.join(args.dir, "spec_*.npz")
    files = sorted(glob.glob(patt))
    if not files:
        return []

    if args.file:
        return [args.file] if os.path.exists(args.file) else []


    def match(fname):
        # parse tags from filename for quick filter (also double-check meta at load)
        # filename pattern: spec_z{z}_logz{+/-X.X}_d2{Y.Y}_logu{+/-X.X}_sfh{NNN}.npz
        z = re.search(r"z([0-9.]+)_", fname)
        LZ = re.search(r"logz([+\-][0-9.]+)_", fname)
        d2 = re.search(r"_d2([0-9.]+)_", fname)
        lu = re.search(r"_logu([+\-][0-9.]+)_", fname)
        sh = re.search(r"_sfh(\d{3})\.npz$", fname)
        ok = True
        if args.zred is not None: ok &= (z and abs(float(z.group(1)) - args.zred) < 1e-6)
        if args.logz is not None: ok &= (LZ and abs(float(LZ.group(1)) - args.logz) < 1e-6)
        if args.dust2 is not None: ok &= (d2 and abs(float(d2.group(1)) - args.dust2) < 1e-6)
        if args.logu is not None: ok &= (lu and abs(float(lu.group(1)) - args.logu) < 1e-6)
        if args.sfh  is not None: ok &= (sh and int(sh.group(1)) == args.sfh)
        return ok

    files = [f for f in files if match(f)]
    return files[:max(args.grid,1)]

def parse_window(win):
    if not win: return None
    try:
        a,b = [float(x) for x in win.split(",")]
        return (min(a,b), max(a,b))
    except Exception:
        return None

def apply_norm(w, y, win):
    if not win: return y, 1.0
    lo, hi = win
    m = (w>=lo) & (w<=hi)
    if not np.any(m): return y, 1.0
    scale = np.median(y[m])
    if scale == 0 or not np.isfinite(scale): return y, 1.0
    return y/scale, scale

def add_line_markers(ax, z):
    lines = {"[O II] 3727":3727.0, "Hβ 4861":4861.33, "[O III] 5007":5006.84, "Hα 6563":6562.80}
    for name, lam0 in lines.items():
        lam = lam0*(1+z)
        if ax.get_xlim()[0] <= lam <= ax.get_xlim()[1]:
            ax.axvline(lam, ls="--", lw=0.8)
            ax.text(lam+6, ax.get_ylim()[0]+0.85*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                    name, rotation=90, va="top", fontsize=8)

def plot_sfh_inset(ax, meta):
    if "agebins" not in meta or "sfr_fraction" not in meta: return
    agebins = np.array(meta["agebins"])
    sfrf = np.array(meta["sfr_fraction"])
    # mass fraction per bin ~ sfr * Δt (here sfrf already sums to 1 by construction)
    mass_frac = sfrf / sfrf.sum()
    # label bins by age midpoint (Gyr)
    edges_gyr = 10**agebins / 1e9
    tmid = 0.5*(edges_gyr[:,0] + edges_gyr[:,1])
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    iax = inset_axes(ax, width="30%", height="35%", loc="upper right", borderpad=0.8)
    iax.bar(np.arange(len(mass_frac)), mass_frac, width=0.85)
    iax.set_xticks([])
    iax.set_ylabel("Mass frac", fontsize=8)
    iax.set_title("SFH bins", fontsize=8)
    for spine in iax.spines.values(): spine.set_linewidth(0.6)
    iax.tick_params(labelsize=8)

def main():
    args = parse_args()
    files = find_files(args)
    if not files:
        raise SystemExit("No NPZ files found (check --dir / filters / --file).")

    xlo, xhi = [float(v) for v in args.xlim.split(",")]
    win = parse_window(args.norm_window)

    # Single panel
    if len(files) == 1:
        w, f, meta = npz_meta(files[0])
        y = f if args.unit == "maggies" else maggies_to_flambda(w, f)
        y, s = apply_norm(w, y, win)

        plt.figure(figsize=(9,4))
        plt.plot(w, y, lw=1.0, label=os.path.basename(files[0]))
        plt.xlim(xlo, xhi)
        plt.xlabel("Wavelength [Å]")
        plt.ylabel("maggies" if args.unit=="maggies" else r"$f_\lambda$ [cgs]")
        title = (args.title or
                 f"z={meta['zred']:.3f}  logZ={meta['logzsol']:+.1f}  d2={meta['dust2']:.1f}  "
                 f"logU={meta['gas_logu']:+.1f}  SFH={int(meta.get('sfr_fraction') is not None)}")
        plt.title(title)

        if args.lines:
            add_line_markers(plt.gca(), meta['zred'])
        if args.sfh_inset:
            plot_sfh_inset(plt.gca(), meta)

        plt.tight_layout()
        if args.save:
            plt.savefig(args.save, dpi=160, bbox_inches="tight")
        else:
            plt.show()
        return

    # Grid (first N matches)
    n = len(files)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 2.8*nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for ax, fpath in zip(axes.ravel(), files):
        w, f, meta = npz_meta(fpath)
        y = f if args.unit == "maggies" else maggies_to_flambda(w, f)
        y, s = apply_norm(w, y, win)
        ax.plot(w, y, lw=0.9)
        ax.set_xlim(xlo, xhi)
        ax.set_title(f"z={meta['zred']:.3f}  logZ={meta['logzsol']:+.1f}  d2={meta['dust2']:.1f}  U={meta['gas_logu']:+.1f}",
                     fontsize=9)
        if args.lines: add_line_markers(ax, meta['zred'])
        if args.sfh_inset: plot_sfh_inset(ax, meta)

    # tidy up empty axes
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Wavelength [Å]")
    for r in axes:
        r[0].set_ylabel("maggies" if args.unit=="maggies" else r"$f_\lambda$ [cgs]")

    fig.suptitle(f"{n} models", y=0.99, fontsize=12)
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=160, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":
    main()
