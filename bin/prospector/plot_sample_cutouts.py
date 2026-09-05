"""Legacy cutouts for a continuum-outlier sample npz.

    uv run python bin/prospector/plot_sample_cutouts.py \
        -s results/cont_outlier_sample20_v3.npz -o results/cont_outlier20_v3_cutouts.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import numpy as np
from astropy.io import fits

from hubersed.paths import PATHS
from hubersed.plotting.cutouts import cutout_grid


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-s", "--sample", default=str(PATHS["RESULTS"] / "cont_outlier_sample20_v3.npz"))
    p.add_argument("-o", "--out", default=str(PATHS["RESULTS"] / "cont_outlier20_v3_cutouts.png"))
    p.add_argument("--vac", default=str(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits"))
    p.add_argument("--ncol", type=int, default=5)
    p.add_argument("--size", type=int, default=140, help="cutout side in pixels (0.262\"/pix)")
    p.add_argument("--dpi", type=int, default=140)
    args = p.parse_args(argv)

    tids = [int(t) for t in np.load(args.sample, allow_pickle=True)["target_ids"]]
    h = fits.open(args.vac)
    md, fs = h["METADATA"].data, h["FASTSPEC"].data
    idx = {int(t): i for i, t in enumerate(np.asarray(md["TARGETID"]).astype("<i8"))}
    i = [idx[t] for t in tids]

    fig, _ = cutout_grid(
        np.asarray(md["RA"])[i].astype("<f8"), np.asarray(md["DEC"])[i].astype("<f8"),
        labels=tids,
        sublabels=[f"z={float(md['Z'][j]):.4f}  logM={float(fs['LOGMSTAR'][j]):.2f}" for j in i],
        ncol=args.ncol, size=args.size)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}  n={len(tids)}")


if __name__ == "__main__":
    main()
