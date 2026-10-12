"""The ER model at the maximum likelihood point of the HR split 45 run, against the ER fit.

QUESTION: how does the ER model fit the ER data at the parameters of the HR max-L point, next to
the ER max-L point? How much of the difference comes from the dust and gas parameters alone, and
what does dust_ratio alone do?
The two runs have the same model and the same 29 parameters, and differ in the library
resolution, the data degradation, the sky mask and the fitted pixels.

Four points go through the ER model on the ER data: the ER max-L point, the HR max-L point, the
ER max-L point with only the dust parameters (dust2, dust_ratio, dust_index) and the gas
parameters (gas_logz, gas_logu, gas_lognH, gas_logno, gas_logco) set to their HR max-L values, and
the HR max-L point with only dust_ratio set to its ER max-L value.

INPUTS: the ER ``_pred.npz`` of predict.py, and
results/2026-09-30_nautilus_c3k_broadline_split45/39627770174637084_broad_shared_split45_unmask_rest9000_seed0.h5
with its .json. SEED: none (deterministic).

Run from the repository root on the Mac with the C3K_ER build (no AFE_FLAG; [alpha/Fe] is 0 in
the fit), as jacobian.py. Command:
``PYTHONPATH=<site with the C3K_ER fsps>:experiments/2026-10-09_nautilus_c3k_er_broadline
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-09_nautilus_c3k_er_broadline/hr_point.py``.

OUTPUT: in results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/,
hr_point_spectrum.png and hr_point_line_windows.png (ER and HR max-L points),
hr_swap_spectrum.png and hr_swap_line_windows.png (ER, HR and the dust and gas swap),
hr_dustratio_spectrum.png and hr_dustratio_line_windows.png (ER, HR and HR with the ER
dust_ratio), and hr_point_windows.txt
(chi2 of each point in the line windows of jacobian.py). The line window figures are for slides.
The log likelihood and chi2 of each point are printed.
"""

import importlib.util
import json
import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from fit import loglike, process_state
from jacobian import WINDOWS
from nautilus import Sampler

from hubersed.conversion import to_flambda
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

TID = 39627770174637084
FB, SPLIT, REST_MAX = "shared", 45.0, 9000.0
OUT = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline"
FIG = OUT / str(TID)
STEM = f"{TID}_er_broad_{FB}_split{SPLIT:g}_rest{REST_MAX:g}_seed0"
HR_DIR = PATHS["RESULTS"] / "2026-09-30_nautilus_c3k_broadline_split45"
HR_STEM = f"{TID}_broad_shared_split45_unmask_rest9000_seed0"
HR_FIT = PATHS["ROOT"] / "experiments" / "2026-09-30_nautilus_c3k_broadline_split45" / "fit.py"
SWAP = [
    "dust2",
    "dust_ratio",
    "dust_index",
    "gas_logz",
    "gas_logu",
    "gas_lognH",
    "gas_logno",
    "gas_logco",
]
# point name, short name for the chi axis labels, colour and line style
POINTS = {
    "ER max-L": ("ER", "#b2182b", "-"),
    "HR max-L": ("HR", "#2166ac", "--"),
    "ER + HR dust, gas": ("swap", "#1b7837", ":"),
    "HR + ER dust ratio": ("dust ratio", "#762a83", "-."),
}
# rest-frame vacuum windows of the line_windows.png of analysis.py, with LaTeX titles
SLIDE_WINDOWS = {
    r"H$\beta$": (4845, 4880),
    r"[O\,III]\,5007": (4995, 5022),
    r"[O\,I]\,6300": (6292, 6312),
    r"H$\alpha$ + [N\,II]": (6540, 6595),
    r"[S\,II]": (6708, 6742),
}


def load_module(path, name):
    """Import a fit.py from another experiment under its own module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hr_best():
    """Return the HR max-L point in physical units, its labels and its stored log likelihood."""
    f30 = load_module(HR_FIT, "fit_0930")
    cfg = json.loads((HR_DIR / f"{HR_STEM}.json").read_text())
    m = f30.make_model(cfg["z"], cfg["forbidden_broad"], cfg["sigma_split"])
    s = Sampler(
        lambda x: x,
        lambda x: 0.0,
        n_dim=m.ndim,
        n_live=1000,
        filepath=str(HR_DIR / f"{HR_STEM}.h5"),
        resume=True,
    )
    unit, _, log_l = s.posterior()
    return m.prior_transform(unit[np.argmax(log_l)]), list(m.theta_labels()), float(log_l.max())


def spectrum_plot(rest, flam, flux, spec, chi, good, keys, path):
    """Plot the data and the model of each point, with one chi panel per point below."""
    n_pix = int(good.sum())
    fig, ax = plt.subplots(
        1 + len(keys),
        1,
        figsize=(11, 5 + 1.5 * len(keys)),
        sharex=True,
        gridspec_kw={"height_ratios": [3] + [1] * len(keys), "hspace": 0.05},
    )
    ax[0].plot(rest, flam(flux), color="0.3", lw=0.6, label="DESI spectrum (DESI resolution)")
    for key in keys:
        c2 = np.nansum(chi[key] ** 2) / n_pix
        ax[0].plot(
            rest,
            flam(spec[key]),
            color=POINTS[key][1],
            lw=0.7,
            label=rf"C3K_ER model at {key}, $\chi^2$/pixel = {c2:.3f}",
        )
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    ax[0].legend(frameon=False, loc="upper right")
    for a, key in zip(ax[1:], keys, strict=True):
        a.plot(rest, chi[key], color=POINTS[key][1], lw=0.5)
        for h in (-5, -2, 2, 5):
            a.axhline(h, color="0.5", ls="--", lw=0.6)
        a.axhline(0, color="0.3", lw=0.6)
        a.set_ylim(-8, 8)
        a.set_ylabel(rf"$\chi$, {POINTS[key][0]}")
    ax[-1].set_xlabel(r"rest wavelength [$\AA$]")
    ax[-1].set_xlim(rest[good].min(), rest[good].max())
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


def windows_plot(rest, flam, flux, unc, spec, chi, keys, path, chi_lim=None):
    """Plot the five slide line windows with LaTeX text and large fonts, one chi row per point.

    ``chi_lim`` fixes the y range of every chi row. By default the rows autoscale.

    The ApJ style must be active. It applies to the whole process, so the caller sets it after
    the spectrum plots.
    """
    n = len(keys)
    fig, axes = plt.subplots(
        1 + n,
        len(SLIDE_WINDOWS),
        figsize=(24, 6 + 1.6 * n),
        gridspec_kw={"height_ratios": [3] + [1.2] * n, "hspace": 0.08, "wspace": 0.28},
    )
    for j, (name, (w_lo, w_hi)) in enumerate(SLIDE_WINDOWS.items()):
        k = (rest > w_lo) & (rest < w_hi)
        a = axes[0, j]
        f, e = flam(flux)[k], flam(unc)[k]
        a.fill_between(rest[k], f - e, f + e, color="0.8", step="mid", lw=0)
        a.step(rest[k], f, "k", where="mid", lw=1.2, label="DESI")
        for key in keys:
            short, col, ls = POINTS[key]
            a.plot(rest[k], flam(spec[key])[k], color=col, lw=2, ls=ls, label=key)
        a.set_title(name, fontsize=22)
        for i, key in enumerate(keys, start=1):
            ar = axes[i, j]
            ar.step(rest[k], chi[key][k], color=POINTS[key][1], where="mid", lw=1.4)
            ar.axhline(0, color="0.3", lw=0.8)
            for h in (-3, 3):
                ar.axhline(h, color="0.6", lw=0.8, ls=":")
            if chi_lim is not None:
                ar.set_ylim(*chi_lim)
            if j == 0:
                ar.set_ylabel(rf"$\chi$, {POINTS[key][0]}", fontsize=16)
        for a in axes[:, j]:
            a.set_xlim(w_lo, w_hi)
            a.tick_params(labelsize=15)
        for a in axes[:-1, j]:
            a.tick_params(labelbottom=False)
        axes[-1, j].set_xlabel(r"rest wavelength [\AA]", fontsize=18)
    axes[0, 0].set_ylabel(
        r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]", fontsize=18
    )
    # one legend row under the title, since the line peaks fill the panels
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="upper center",
        ncol=len(names),
        fontsize=17,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    extra = ""
    if "ER + HR dust, gas" in keys:
        extra = " (swap: ER max-L with the HR dust and gas values)"
    if "HR + ER dust ratio" in keys:
        extra = r" (dust ratio: HR max-L with the ER dust\_ratio)"
    fig.suptitle(rf"{TID}, C3K\_ER model on the ER data{extra}", fontsize=22, y=1.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


def main():
    d = np.load(OUT / f"{STEM}_pred.npz")
    labels = [str(x) for x in d["labels"]]
    z, good, flux, unc = float(d["z"]), d["good"], d["flux"], d["unc"]
    model, obs, sps = process_state(TID, FB, SPLIT, REST_MAX)
    assert list(model.theta_labels()) == labels

    theta_hr, labels_hr, lnl_hr_own = hr_best()
    assert labels_hr == labels
    swap = d["best"].copy()
    for lab in SWAP:
        swap[labels.index(lab)] = theta_hr[labels.index(lab)]
    hr_dr = theta_hr.copy()
    hr_dr[labels.index("dust_ratio")] = d["best"][labels.index("dust_ratio")]

    def predict(theta):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(model.predict(theta, observations=obs, sps=sps)[0][0], float)

    pts = {
        "ER max-L": d["best"],
        "HR max-L": theta_hr,
        "ER + HR dust, gas": swap,
        "HR + ER dust ratio": hr_dr,
    }
    spec, chi = {}, {}
    n_pix = int(good.sum())
    for key, theta in pts.items():
        lnl = loglike(theta, TID, FB, SPLIT, REST_MAX)
        spec[key] = predict(theta)
        chi[key] = np.where(good, (flux - spec[key]) / unc, np.nan)
        print(
            f"{key}: lnL in the ER model {lnl:.2f}, chi2 {np.nansum(chi[key] ** 2):.1f}, "
            f"chi2/pixel {np.nansum(chi[key] ** 2) / n_pix:.3f} over {n_pix} pixels"
        )
        if key == "ER max-L":
            assert np.isclose(lnl, float(d["max_lnl"]), atol=0.1), (lnl, float(d["max_lnl"]))
    print(f"HR max-L lnL in the HR fit (other data): {lnl_hr_own:.2f}")

    # chi2 of each point in each line window of jacobian.py, the rest counted as continuum
    rest = d["wave"] / (1 + z)
    keys = list(pts)
    rows, used = [], np.zeros_like(good)
    for name, (w_lo, w_hi) in [*WINDOWS.items(), ("continuum", (None, None))]:
        k = good & ~used if w_lo is None else good & (rest >= w_lo) & (rest < w_hi)
        used |= k
        c2 = [float(np.sum(chi[key][k] ** 2)) for key in keys]
        rows.append(
            f"{name:12s} {int(k.sum()):5d} {c2[0]:9.1f}"
            + "".join(f" {c:9.1f} {c - c2[0]:+9.1f}" for c in c2[1:])
        )
    head = f"{'window':12s} {'n_pix':>5s} {'chi2 ER':>9s}" + "".join(
        f" {'chi2 ' + tag:>9s} {tag + '-ER':>9s}" for tag in ("HR", "SW", "DR")
    )
    note = (
        "columns: ER max-L, HR max-L (HR), ER max-L with the HR dust and gas values (SW), "
        "HR max-L with the ER dust_ratio (DR)"
    )
    table = "\n".join([note, head, *rows])
    print(table)
    (FIG / "hr_point_windows.txt").write_text(table + "\n")

    width = max(len(lab) for lab in labels)
    print(f"{'parameter':{width}s}" + "".join(f"  {t:>10s}" for t in ("ER", "HR", "SW", "DR")))
    for i, lab in enumerate(labels):
        print(f"{lab:{width}s}" + "".join(f"  {pts[key][i]:10.4f}" for key in keys))

    def flam(maggies):
        f = to_flambda(d["wave"] * u.AA, np.asarray(maggies, float) * u.mgy).value
        return np.where(good, f, np.nan)

    sets = {
        "hr_point": ["ER max-L", "HR max-L"],
        "hr_swap": ["ER max-L", "HR max-L", "ER + HR dust, gas"],
        "hr_dustratio": ["ER max-L", "HR max-L", "HR + ER dust ratio"],
    }
    for tag, ks in sets.items():
        spectrum_plot(rest, flam, flux, spec, chi, good, ks, FIG / f"{tag}_spectrum.png")
    use_apj_style()
    for tag, ks in sets.items():
        lim = (-7, 7) if tag == "hr_dustratio" else None
        windows_plot(rest, flam, flux, unc, spec, chi, ks, FIG / f"{tag}_line_windows.png", lim)


if __name__ == "__main__":
    main()
