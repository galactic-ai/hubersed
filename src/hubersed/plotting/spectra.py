from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

DESI_FLUX_LABEL = r"$f_\lambda\ [10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}]$"
REST_WAVE_LABEL = r"rest wavelength [$\mathrm{\AA}$]"


def _rest(wave, z):
    w = np.asarray(wave, float)
    return w / (1.0 + z) if z is not None else w


def residual_chi(data, model, unc, mask=None):
    """Signed residual in sigma: (data - model) / unc. NaN where unc<=0 or masked."""
    data, model, unc = (np.asarray(a, float) for a in (data, model, unc))
    out = np.full(data.shape, np.nan)
    good = np.isfinite(unc) & (unc > 0)
    if mask is not None:
        good &= np.asarray(mask, bool)
    out[good] = (data[good] - model[good]) / unc[good]
    return out


def plot_spectrum(
    ax,
    wave,
    *,
    z=None,
    data=None,
    unc=None,
    medfilt=None,
    models=None,
    data_kw=None,
    medfilt_kw=None,
    band_kw=None,
):
    """Top panel: data (optional +/-unc band), median-filtered data, and models.

    Parameters
    ----------
    ax : matplotlib Axes
    wave : (N,) wavelength. Rest-frame unless `z` given (then observed / (1+z)).
    data, unc : (N,) observed flux and 1-sigma uncertainty. Pass band_kw (even {})
        to draw the +/-unc band.
    medfilt : (N,) pre-computed median-filtered data, or None. Compute it yourself
        (scipy.signal.medfilt) so this module stays IO-free.
    models : list of dicts, each {"flux": (N,) [, "wave", "label", color, ls, ...]}.
        Non-"flux"/"wave" keys pass to ax.plot. "wave" defaults to the data wave.
    """
    w = _rest(wave, z)

    if data is not None:
        if unc is not None and band_kw is not None:
            u = np.asarray(unc, float)
            ax.fill_between(
                w, data - u, data + u,
                **{"alpha": 0.25, "lw": 0, "color": "0.6", **band_kw},
            )
        ax.plot(w, data, **{"color": "0.4", "lw": 0.5, "label": "DESI spectrum",
                            **(data_kw or {})})

    if medfilt is not None:
        ax.plot(w, medfilt, **{"color": "k", "lw": 1.0, "label": "median filter",
                               **(medfilt_kw or {})})

    for m in models or []:
        m = dict(m)
        flux = m.pop("flux")
        mw = _rest(m.pop("wave"), z) if "wave" in m else w
        ax.plot(mw, flux, **{"lw": 1.2, **m})

    ax.set_ylabel(DESI_FLUX_LABEL)
    if data is not None or medfilt is not None or models:
        ax.legend(frameon=False, fontsize="small")
    return ax


def plot_residual(
    ax,
    wave,
    *,
    z=None,
    chi=None,
    data=None,
    model=None,
    unc=None,
    mask=None,
    levels=(2, 5),
    line_kw=None,
    **plot_kw,
):
    """Bottom panel: residual chi vs rest wavelength.

    Give `chi` directly, or (`data`, `model`, `unc`) to compute it. Grey dashed
    guides at +/- each value in `levels` (default +/-2, +/-5), plus a solid zero.
    """
    w = _rest(wave, z)
    if chi is None:
        if data is None or model is None or unc is None:
            raise ValueError("pass chi=..., or data + model + unc")
        chi = residual_chi(data, model, unc, mask)
    ax.plot(w, chi, **{"color": "0.3", "lw": 0.5, **plot_kw})
    ax.axhline(0.0, color="0.5", lw=0.8)
    lk = {"color": "0.5", "ls": "--", "lw": 0.8, **(line_kw or {})}
    for lv in levels:
        ax.axhline(+lv, **lk)
        ax.axhline(-lv, **lk)
    ax.set_xlabel(REST_WAVE_LABEL)
    ax.set_ylabel(r"$\chi = (d - m)/\sigma$")
    return ax


def spectrum_figure(
    wave,
    *,
    extra_panels=0,
    height_ratios=None,
    figsize=(9, 5),
    apj_style=True,
    **spectrum_kwargs,
):
    """Stacked spectrum + residual sharing the rest-wavelength x-axis.

    Returns (fig, axes): axes[0]=spectrum, axes[1]=residual, axes[2:]=`extra_panels`
    blank Axes (also sharing x) for manual additions, e.g. a PolyOptCal response.

    Only spectrum kwargs are forwarded to plot_spectrum. Draw the residual yourself
    on axes[1] so *which* model it is against stays explicit. apj_style=False keeps
    the active style.
    """
    if apj_style:
        from hubersed.plotting.style import use_apj_style
        use_apj_style()

    n = 2 + int(extra_panels)
    if height_ratios is None:
        height_ratios = [3, 1] + [1] * int(extra_panels)
    fig, axes = plt.subplots(
        n, 1, sharex=True, figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.05},
        squeeze=False,
    )
    axes = axes[:, 0]
    plot_spectrum(axes[0], wave, **spectrum_kwargs)
    axes[0].tick_params(labelbottom=False)
    return fig, axes


if __name__ == "__main__":  # synthetic smoke demo
    from scipy.signal import medfilt as _mf

    rng = np.random.default_rng(0)
    w = np.linspace(3600, 9800, 2000)
    model = 10 * np.exp(-((w - 5000) / 1500) ** 2) + 2
    data = model + rng.normal(0, 0.4, w.size)
    unc = np.full_like(w, 0.4)

    fig, ax = spectrum_figure(
        w, z=0.02,
        data=data, unc=unc, band_kw={},
        medfilt=_mf(data, 21),
        models=[{"flux": model, "label": "MAP model", "color": "C3"}],
    )
    plot_residual(ax[1], w, z=0.02, data=data, model=model, unc=unc)
    fig.savefig("/tmp/spectrum_demo.pdf", bbox_inches="tight")
    print("saved /tmp/spectrum_demo.pdf")
