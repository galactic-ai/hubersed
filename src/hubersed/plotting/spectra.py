"""Plot spectra, models and residuals against rest wavelength."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

DESI_FLUX_LABEL = r"$f_\lambda\ [10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}]$"
REST_WAVE_LABEL = r"rest wavelength [$\mathrm{\AA}$]"


def _rest(wave, z):
    """Divide wavelength by 1 + z, or return it unchanged when z is None."""
    w = np.asarray(wave, float)
    return w / (1.0 + z) if z is not None else w


def residual_chi(data, model, unc, mask=None):
    """Return the residual in units of the uncertainty, (data - model) / unc.

    Parameters
    ----------
    data, model, unc : np.ndarray
        Observed flux, model flux and one sigma uncertainty, in the same units.
    mask : np.ndarray, optional
        True for pixels to keep.

    Returns
    -------
    np.ndarray
        The residual. NaN where ``unc`` is not positive and finite, or where ``mask`` is False.
    """
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
    """Draw a spectrum, its median-filtered version, and any number of models.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    wave : np.ndarray
        Wavelength in Angstrom. If ``z`` is given it is taken as observed and shifted to
        rest frame, otherwise it is used as it is.
    z : float, optional
        Redshift.
    data, unc : np.ndarray, optional
        Observed flux and its one sigma uncertainty. The y label assumes DESI units of
        1e-17 erg/s/cm^2/A.
    medfilt : np.ndarray, optional
        Median-filtered data, computed by the caller, for example with
        ``scipy.signal.medfilt``.
    models : list of dict, optional
        One dict per model with a ``flux`` key, an optional ``wave`` key, and any other
        keys passed to ``ax.plot``, such as ``label`` or ``color``.
    data_kw, medfilt_kw : dict, optional
        Extra keyword arguments for the data and median-filter lines.
    band_kw : dict, optional
        Keyword arguments for the uncertainty band. The band is drawn only when this is
        given, so pass ``{}`` for the default look.

    Returns
    -------
    matplotlib.axes.Axes
        The same axes.
    """
    w = _rest(wave, z)

    if data is not None:
        if unc is not None and band_kw is not None:
            u = np.asarray(unc, float)
            ax.fill_between(
                w,
                data - u,
                data + u,
                **{"alpha": 0.25, "lw": 0, "color": "0.6", **band_kw},
            )
        ax.plot(w, data, **{"color": "0.4", "lw": 0.5, "label": "DESI spectrum", **(data_kw or {})})

    if medfilt is not None:
        ax.plot(
            w, medfilt, **{"color": "k", "lw": 1.0, "label": "median filter", **(medfilt_kw or {})}
        )

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
    """Draw the residual in units of sigma against rest wavelength.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    wave : np.ndarray
        Wavelength in Angstrom, shifted to rest frame when ``z`` is given.
    z : float, optional
        Redshift.
    chi : np.ndarray, optional
        Residual to plot. If not given it is computed from ``data``, ``model``, ``unc``
        and ``mask`` with ``residual_chi``.
    data, model, unc, mask : np.ndarray, optional
        Inputs for ``residual_chi``.
    levels : tuple of float
        Dashed guide lines are drawn at plus and minus each level, with a solid line at zero.
    line_kw : dict, optional
        Extra keyword arguments for the guide lines.
    **plot_kw
        Extra keyword arguments for the residual line.

    Returns
    -------
    matplotlib.axes.Axes
        The same axes.

    Raises
    ------
    ValueError
        If neither ``chi`` nor all of ``data``, ``model`` and ``unc`` are given.
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
    """Make a figure with the spectrum on top and an empty residual panel below.

    The residual is left for the caller to draw with ``plot_residual``, so it is always
    clear which model it compares against.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength in Angstrom, passed to ``plot_spectrum``.
    extra_panels : int
        Number of empty panels added below, sharing the x axis.
    height_ratios : list of float, optional
        Panel heights. The default is 3 for the spectrum and 1 for each other panel.
    figsize : tuple of float
        Figure size in inches.
    apj_style : bool
        Apply ``use_apj_style`` first. False keeps the active style.
    **spectrum_kwargs
        Passed to ``plot_spectrum``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : np.ndarray of matplotlib.axes.Axes
        Spectrum panel, residual panel, then the extra panels.
    """
    if apj_style:
        from hubersed.plotting.style import use_apj_style

        use_apj_style()

    n = 2 + int(extra_panels)
    if height_ratios is None:
        height_ratios = [3, 1] + [1] * int(extra_panels)
    fig, axes = plt.subplots(
        n,
        1,
        sharex=True,
        figsize=figsize,
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
    model = 10 * np.exp(-(((w - 5000) / 1500) ** 2)) + 2
    data = model + rng.normal(0, 0.4, w.size)
    unc = np.full_like(w, 0.4)

    fig, ax = spectrum_figure(
        w,
        z=0.02,
        data=data,
        unc=unc,
        band_kw={},
        medfilt=_mf(data, 21),
        models=[{"flux": model, "label": "MAP model", "color": "C3"}],
    )
    plot_residual(ax[1], w, z=0.02, data=data, model=model, unc=unc)
    fig.savefig("/tmp/spectrum_demo.pdf", bbox_inches="tight")
    print("saved /tmp/spectrum_demo.pdf")
