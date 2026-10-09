"""Best-fit spectrum and SFH of each MAP run saved by map_co.py.

Run from the repository root with
``uv run python experiments/2026-09-26_nautilus_c3k_broadline/map_co_oh_plots.py --window miles``.
"""

import argparse
import pickle
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from hubersed.conversion import to_flambda
from hubersed.fitting.chi2 import WAVE_OBS
from hubersed.paths import PATHS
from hubersed.plotting.sfh import sfh_figure
from hubersed.plotting.spectra import plot_residual, residual_chi, spectrum_figure

RUN_NAMES = ("free", "tied", "tied_fixed")
COLOR = {"free": "C0", "tied": "C1", "tied_fixed": "C2"}


def main(tid, window, out):
    with open(out / f"{tid}_map_co_{window}.pkl", "rb") as f:
        res = pickle.load(f)
    d = res["data"]
    z, flux, unc, good = d["z"], d["flux"], d["unc"], d["good"]
    runs = {k: res[k] for k in RUN_NAMES if k in res}
    rest = WAVE_OBS / (1 + z)

    def flam(maggies):
        """Convert maggies on WAVE_OBS to DESI f_lambda units, NaN outside the fitted pixels."""
        f = to_flambda(WAVE_OBS * u.AA, np.asarray(maggies, float) * u.mgy).value
        return np.where(good, f, np.nan)

    def label(name, r):
        # no underscores in labels: the ApJ style renders text with LaTeX
        return rf"MAP C/O {name}, [C/O] = {r['gas_logco']:+.2f}, $\chi^2_\nu$ = {r['chi2_red']:.3f}"

    # spectrum with chi residuals
    fig, ax = spectrum_figure(
        WAVE_OBS,
        z=z,
        figsize=(11, 6),
        data=flam(flux),
        unc=flam(unc),
        band_kw={},
        data_kw={"label": "DESI spectrum, degraded"},
        models=[
            {"flux": flam(r["spec"]), "color": COLOR[n], "lw": 0.8, "label": label(n, r)}
            for n, r in runs.items()
        ],
    )
    for n, r in runs.items():
        chi = residual_chi(flux, r["spec"], unc, good)
        plot_residual(ax[1], WAVE_OBS, z=z, chi=chi, lw=0.5, color=COLOR[n], alpha=0.8)
    ax[1].set_ylim(-8, 8)
    ax[1].set_xlim(rest[good].min(), rest[good].max())
    fig.savefig(out / f"{tid}_map_co_{window}_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # SFH of each run's best point
    first, *rest_runs = runs.items()
    edges = first[1]["sfh"]["edges_gyr"]
    fig, ax = sfh_figure(
        edges,
        first[1]["sfh"]["ssfr"],
        first[1]["sfh"]["cmf"],
        ssfr_kwargs={"label": f"MAP C/O {first[0]}", "ssfr_kw": {"color": COLOR[first[0]]}},
        cmf_kwargs={"color": COLOR[first[0]]},
    )
    for n, r in rest_runs:
        assert np.allclose(r["sfh"]["edges_gyr"], edges)
        ax[0].stairs(r["sfh"]["ssfr"], edges, color=COLOR[n], lw=1.8, label=f"MAP C/O {n}")
        ax[1].stairs(r["sfh"]["cmf"], edges, color=COLOR[n], lw=1.8)
    ax[0].legend(frameon=True, fontsize="small", loc="lower left")
    fig.savefig(out / f"{tid}_map_co_{window}_sfh.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    for n, r in runs.items():
        print(
            f"{n}: chi2_red {r['chi2_red']:.3f}, lnL {r['lnl']:.2f}, gas_logco {r['gas_logco']:+.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tid", type=int, default=39627770174637084)
    parser.add_argument("--window", choices=["full", "miles"], default="miles")
    parser.add_argument(
        "--out",
        type=Path,
        default=PATHS["RESULTS"] / "2026-09-26_nautilus_c3k_broadline" / "map_co",
    )
    a = parser.parse_args()
    main(a.tid, a.window, a.out)
