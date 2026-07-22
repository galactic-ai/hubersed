from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

LOOKBACK_LABEL = "lookback time [Gyr]"
_10MYR, _100MYR = 1e-2, 1e-1  # Gyr


def plot_ssfr(
    ax,
    edges,
    ssfr,
    *,
    ssfr_inplace=None,
    guides=(_10MYR, _100MYR),
    label=r"sSFR = SFR/$M_\star$ (final)",
    label_inplace=r"sSFR = SFR/$M_\star(\geq t)$ (mass in place)",
    ssfr_kw=None,
    inplace_kw=None,
):
    """Top panel: sSFR(t) as steps, log-log.

    ssfr : (N,) sSFR per bin, normalised by FINAL stellar mass.
    ssfr_inplace : (N,) optional second curve normalised by mass formed by that time
        (the dashed "mass in place" curve in the reference figure).
    quenched_divide : horizontal dash-dot at this sSFR (None to skip).
    guides : vertical lines at these lookback times [Gyr] (first dotted, rest dashed).
    """
    edges = np.asarray(edges, float)
    ax.stairs(np.asarray(ssfr, float), edges,
              **{"color": "C3", "lw": 1.8, "label": label, **(ssfr_kw or {})})
    if ssfr_inplace is not None:
        ax.stairs(np.asarray(ssfr_inplace, float), edges,
                  **{"color": "rebeccapurple", "lw": 1.5, "ls": "--",
                     "label": label_inplace, **(inplace_kw or {})})
    for g in guides:
        ax.axvline(g, color="0.6", ls=":" if g == guides[0] else "--", lw=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"sSFR [yr$^{-1}$]")
    ax.legend(frameon=True, fontsize="small", loc="lower left")
    return ax


def plot_cumulative_mass(ax, edges, cmf, *, guides=(_10MYR, _100MYR), **step_kw):
    """Bottom panel: cumulative mass fraction vs lookback time (log x, linear y).

    cmf : (N,) cumulative mass fraction per bin (0 at recent times -> 1 at oldest).
    """
    edges = np.asarray(edges, float)
    ax.stairs(np.asarray(cmf, float), edges, **{"color": "C0", "lw": 1.8, **step_kw})
    for g in guides:
        ax.axvline(g, color="0.6", ls=":" if g == guides[0] else "--", lw=0.8)
    for y in (0.5, 0.9):
        ax.axhline(y, color="0.85", ls=":", lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(LOOKBACK_LABEL)
    ax.set_ylabel("cumulative mass fraction")
    return ax


def sfh_figure(
    edges,
    ssfr,
    cmf,
    *,
    ssfr_inplace=None,
    extra_panels=0,
    height_ratios=None,
    figsize=(8, 6),
    apj_style=True,
    ssfr_kwargs=None,
    cmf_kwargs=None,
):
    """Stacked sSFR + cumulative-mass panels sharing the lookback-time x-axis.

    Returns (fig, axes): axes[0]=sSFR, axes[1]=cumulative mass, axes[2:]=`extra_panels`
    blank Axes (sharing x) for manual additions. apj_style=False keeps active style.
    """
    if apj_style:
        from hubersed.plotting.style import use_apj_style
        use_apj_style()

    n = 2 + int(extra_panels)
    if height_ratios is None:
        height_ratios = [2, 1] + [1] * int(extra_panels)
    fig, axes = plt.subplots(
        n, 1, sharex=True, figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.05},
        squeeze=False,
    )
    axes = axes[:, 0]
    plot_ssfr(axes[0], edges, ssfr, ssfr_inplace=ssfr_inplace, **(ssfr_kwargs or {}))
    plot_cumulative_mass(axes[1], edges, cmf, **(cmf_kwargs or {}))
    axes[0].tick_params(labelbottom=False)
    return fig, axes


if __name__ == "__main__":  # synthetic smoke demo
    edges = np.array([1e-3, 1e-2, 3e-2, 1e-1, 3.5e-1, 9e-1, 2.2, 5.2, 13.0])
    ssfr = np.array([3e-12, 3e-10, 1.6e-11, 2.2e-11, 8e-11, 2e-10, 2.2e-10, 3.4e-11])
    inplace = ssfr * np.array([1, 1, 1, 1, 1, 1, 1.05, 3.8])
    masses = np.array([0.005, 0.005, 0.01, 0.02, 0.09, 0.27, 0.35, 0.25])
    cmf = np.cumsum(masses)

    fig, ax = sfh_figure(edges, ssfr, cmf, ssfr_inplace=inplace)
    fig.savefig("/tmp/sfh_demo.pdf", bbox_inches="tight")
    print("saved /tmp/sfh_demo.pdf")
