"""Plot Cue semi-strong-line O/H against direct T_e O/H and against stellar mass.

Left, Cue O/H of the semi fits (strong lines, [SIII]9069 and He I 5876, [OI] dropped) against direct
T_e O/H for the galaxies whose direct value is kept, coloured by N2. Right, both against log M*
with running medians in 0.5 dex mass bins, the mass-metallicity relation each method gives.

INPUTS: semi/semi_compare.txt of semi.py compare. SEED: none.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/semi_plot.py``.

OUTPUT: semi/semi_direct.png in results/2026-10-14_cue_lineflux_calibration/.
"""

import matplotlib.pyplot as plt
import numpy as np

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

DIR = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration" / "semi"
BINS = np.arange(7.0, 11.01, 0.5)


def read():
    """Return N2, log M*, direct O/H, kept flag and Cue semi O/H per galaxy."""
    rows = []
    for line in (DIR / "semi_compare.txt").read_text().splitlines()[2:]:
        f = line.split()
        if len(f) == 12 and f[0].isdigit():
            rows.append((float(f[1]), float(f[2]), float(f[3]), f[4] == "yes", float(f[6])))
    n2, logm, direct, keep, semi = (np.array(c) for c in zip(*rows, strict=True))
    return n2, logm, direct, keep.astype(bool), semi


def running_median(x, y):
    """Return bin centres and medians of y in BINS of x, for bins with at least 5 points."""
    idx = np.digitize(x, BINS)
    out = [
        (BINS[i - 1] + 0.25, np.median(y[idx == i]))
        for i in range(1, len(BINS))
        if (idx == i).sum() >= 5
    ]
    return np.array(out).T


def main():
    """Write the two-panel figure."""
    use_apj_style()
    n2, logm, direct, keep, semi = read()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    sc = ax1.scatter(direct[keep], semi[keep], c=n2[keep], s=10, cmap="viridis")
    lim = (7.4, 9.4)
    ax1.plot(lim, lim, color="0.4", lw=0.8)
    ax1.axvline(8.4, color="0.6", ls=":", lw=0.8)
    ax1.set(
        xlim=lim, ylim=lim, xlabel=r"direct $T_e$ 12+log(O/H)", ylabel="Cue 12+log(O/H), semi lines"
    )
    ax1.text(0.03, 0.95, f"N = {keep.sum()}", transform=ax1.transAxes, va="top")
    fig.colorbar(sc, ax=ax1, label=r"N2 = log [NII]/H$\alpha$")

    ax2.scatter(logm[keep], direct[keep], s=8, color="0.6", label=r"direct $T_e$")
    ax2.scatter(logm, semi, s=8, color="C0", alpha=0.6, label="Cue, semi lines")
    for y, m, c in ((direct[keep], logm[keep], "k"), (semi, logm, "C0")):
        xm, ym = running_median(m, y)
        ax2.plot(xm, ym, color=c, lw=2)
    ax2.set(ylim=lim, xlabel=r"log $M_*$ (FastSpecFit)", ylabel="12+log(O/H)")
    ax2.legend(frameon=False, loc="lower right")
    path = DIR / "semi_direct.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
