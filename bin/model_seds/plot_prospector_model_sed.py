"""Plot model SEDs from make_prospector_model_sed.py output.

Usage
-----
# Plot 50 random spectra colored by stellar mass:
python plot_prospector_model_sed.py --param stellar_masses --n 50

# Plot median spectra in 5 bins of redshift:
python plot_prospector_model_sed.py --param redshifts --mode bins --nbins 5

# Plot prior distributions for all parameters:
python plot_prospector_model_sed.py --mode priors

Available --param values:
    redshifts, stellar_masses, stellar_metallicities,
    sigma_regs, tau_eqs, tau_ins, sigma_dyns, tau_dyns,
    ns, tau_dust_2s, tau_dust_1s,
    u_mins, gamma_es, q_pahs,
    gas_metallicities, gas_ionization_parameters, sigma_smooths
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from hubersed.paths import PATHS
from hubersed.style import *

DATA_PATH = PATHS["DATA"]
RESULTS_PATH = PATHS["RESULTS"]
HDF5_FILE = DATA_PATH / "prospector_stochastic_model_seds_500000.h5"

PARAM_LABELS = {
    "redshifts":                 r"Redshift $z$",
    "stellar_masses":            r"$\log M_\star / M_\odot$",
    "stellar_metallicities":     r"$\log Z_\star / Z_\odot$",
    "sigma_regs":                r"$\sigma_\mathrm{reg}$",
    "tau_eqs":                   r"$\tau_\mathrm{eq}$ [Gyr]",
    "tau_ins":                   r"$\tau_\mathrm{in}$ [Gyr]",
    "sigma_dyns":                r"$\sigma_\mathrm{dyn}$",
    "tau_dyns":                  r"$\tau_\mathrm{dyn}$ [Gyr]",
    "ns":                        r"Dust index $n$",
    "tau_dust_2s":               r"$\tau_{\mathrm{dust},2}$ (diffuse)",
    "tau_dust_1s":               r"Dust ratio $\tau_1/\tau_2$",
    "u_mins":                    r"$U_\mathrm{min}$",
    "gamma_es":                  r"$\gamma_e$",
    "q_pahs":                    r"$q_\mathrm{PAH}$ [%]",
    "gas_metallicities":         r"$\log Z_\mathrm{gas} / Z_\odot$",
    "gas_ionization_parameters": r"$\log U_\mathrm{gas}$",
    "sigma_smooths":             r"$\sigma_v$ [km/s]",
}

LOG_PARAMS = {"sigma_regs", "sigma_dyns", "gamma_es"}


def load_data(param, n_load=None):
    with h5py.File(HDF5_FILE, "r") as hf:
        wave: np.ndarray = np.asarray(hf["wavelength"])
        param_vals: np.ndarray = np.asarray(hf[f"priors/{param}"])
        n_total = len(param_vals)

        if n_load is None:
            idx = np.arange(n_total)
        else:
            idx = np.random.choice(n_total, size=min(n_load, n_total), replace=False)
            idx.sort()

        fluxes: np.ndarray = np.asarray(hf["fluxes"][idx, :])  # type: ignore[index]
        param_vals = param_vals[idx]

    return wave, fluxes, param_vals, idx


def normalize_flux(fluxes):
    """Normalize each spectrum to its median (ignoring zeros)."""
    medians = np.median(fluxes, axis=1, keepdims=True)
    medians = np.where(medians == 0, 1.0, medians)
    return fluxes / medians


def plot_colored(param, n=100, output=None):
    """Plot n random spectra colored by param value."""
    wave, fluxes, param_vals, _ = load_data(param, n_load=n)
    fluxes = normalize_flux(fluxes)

    is_log = param in LOG_PARAMS
    vmin, vmax = param_vals.min(), param_vals.max()
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax) if is_log else mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(10, 5))

    # sort by param so lower values are drawn first (higher on top)
    order = np.argsort(param_vals)
    for i in order:
        color = cmap(norm(param_vals[i]))
        ax.plot(wave, fluxes[i], color=color, alpha=0.4, lw=0.6)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(PARAM_LABELS.get(param, param) or param, fontsize=12)

    ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=12)
    ax.set_ylabel("Normalized flux", fontsize=12)
    ax.set_title(f"Model SEDs colored by {PARAM_LABELS.get(param, param)} (n={len(param_vals)})", fontsize=12)
    ax.set_xlim(wave.min(), wave.max())

    plt.tight_layout()
    _save_or_show(fig, output, f"colored_{param}")


def plot_bins(param, nbins=5, output=None):
    """Plot median spectrum in bins of param."""
    wave, fluxes, param_vals, _ = load_data(param)
    fluxes = normalize_flux(fluxes)

    is_log = param in LOG_PARAMS
    if is_log:
        edges = np.logspace(np.log10(param_vals.min()), np.log10(param_vals.max()), nbins + 1)
    else:
        edges = np.linspace(param_vals.min(), param_vals.max(), nbins + 1)

    cmap = plt.get_cmap("plasma")
    colors = [cmap(i / (nbins - 1)) for i in range(nbins)]

    fig, ax = plt.subplots(figsize=(10, 5))

    for k in range(nbins):
        mask = (param_vals >= edges[k]) & (param_vals < edges[k + 1])
        if mask.sum() == 0:
            continue
        median_spec = np.median(fluxes[mask], axis=0)
        lo = np.percentile(fluxes[mask], 16, axis=0)
        hi = np.percentile(fluxes[mask], 84, axis=0)

        label_val = (edges[k] + edges[k + 1]) / 2
        label = f"{PARAM_LABELS.get(param, param)} = {label_val:.3g}  (n={mask.sum()})"

        ax.plot(wave, median_spec, color=colors[k], lw=1.5, label=label)
        ax.fill_between(wave, lo, hi, color=colors[k], alpha=0.15)

    ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=12)
    ax.set_ylabel("Normalized flux", fontsize=12)
    ax.set_title(f"Median SEDs in bins of {PARAM_LABELS.get(param, param)}", fontsize=12)
    ax.set_xlim(wave.min(), wave.max())
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    _save_or_show(fig, output, f"bins_{param}")


def plot_priors(output=None):
    """Plot histograms of all prior parameter distributions."""
    with h5py.File(HDF5_FILE, "r") as hf:
        params: dict[str, np.ndarray] = {k: np.asarray(hf[f"priors/{k}"]) for k in PARAM_LABELS}

    ncols = 4
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flat

    for ax, (key, vals) in zip(axes, params.items()):
        is_log = key in LOG_PARAMS
        if is_log:
            bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 50)
            ax.set_xscale("log")
        else:
            bins = 50
        ax.hist(vals, bins=bins, color="steelblue", alpha=0.8, edgecolor="none")
        ax.set_xlabel(PARAM_LABELS[key], fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)

    # hide unused axes
    for ax in axes:
        ax.set_visible(False)

    fig.suptitle("Prior parameter distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    _save_or_show(fig, output, "priors")


def _save_or_show(fig, output, default_name):
    if output:
        path = output
    else:
        path = RESULTS_PATH / f"{default_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved to {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--param", default="stellar_masses", choices=list(PARAM_LABELS.keys()),
                        help="Parameter to color/bin spectra by")
    parser.add_argument("--mode", default="colored", choices=["colored", "bins", "priors"],
                        help="Plot mode: colored spectra, binned medians, or prior histograms")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of random spectra to draw (colored mode only)")
    parser.add_argument("--nbins", type=int, default=5,
                        help="Number of bins (bins mode only)")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: auto-named PNG in current dir)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.mode == "colored":
        plot_colored(args.param, n=args.n, output=args.output)
    elif args.mode == "bins":
        plot_bins(args.param, nbins=args.nbins, output=args.output)
    elif args.mode == "priors":
        plot_priors(output=args.output)


if __name__ == "__main__":
    main()
