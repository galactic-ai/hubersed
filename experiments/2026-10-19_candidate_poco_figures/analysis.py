"""QUESTION: How well do the pocoMC free_dust fits of the free-line candidates match their data,
over the full spectrum and in the line windows, next to their quick MAPs?
HYPOTHESIS: None registered before this analysis.
INPUTS: pmc_final.state of poco.py copied from ls6 into
results/2026-10-15_broadline_free_amplitude/poco_<TID>_free_dust_neff1024/, <TID>_map_free.npz of
2026-10-18_candidate_map (free variant, SFH hyperparameters fixed at #14's), and the 10-13 _pred.npz
of 39627770174637084 for those hyperparameters. Same layout as 2026-10-18_issue14_figures.py: data
and the highest-posterior pocoMC particle, then chi panels from -5 to 5 with dotted lines at -2 and
2. The flux axis is not clipped. Each chi panel gives chi2/(N - free parameters) and the pixels
beyond 5.
SEED: none.
COMMAND: from the repository root on the Mac with the C3K_ER build, once per target,
FIT_TID=<TID> PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-15_broadline_free_amplitude:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-19_candidate_poco_figures/analysis.py
RESULT: 2026-10-19. chi2_nu 1.350 (pocoMC) and 1.456 (MAP) on 5541 px for 39633278973513279, 0.963
and 1.005 on 5397 px for 39633001071509599.
FIGURES: <TID>_residuals.png and <TID>_windows.png in results/2026-10-19_candidate_poco_figures/.
"""

# %%

import argparse

import astropy.units as u
import numpy as np

from hubersed.conversion import to_flambda
from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-10-19_candidate_poco_figures"
VARIANT = "free_dust"
N_TOP = 300
COLORS = {"poco": "#b2182b", "map": "#2166ac"}


def desi(wave, maggies):
    """Return maggies as DESI flux density, 1e-17 erg/s/cm^2/A."""
    return to_flambda(wave * u.AA, maggies * u.mgy).value


def rows_of(tid):
    """Return the pocoMC best and quick MAP rows, each with its model spectrum and chi."""
    import fit
    from compare_poco import load_sampler, map_point

    assert fit.TID == tid, "set FIT_TID"
    state = RES / "2026-10-15_broadline_free_amplitude" / f"poco_{tid}_{VARIANT}_neff1024"
    s = load_sampler(state / "pmc_final.state")
    usamp, w, logl, _ = s.posterior()
    model, obs, cue = fit.process_state(VARIANT)
    theta = np.array([model.prior_transform(x) for x in usamp])
    top = np.argsort(logl)[-N_TOP:]
    lnp = np.array([logl[i] + model.prior_product(theta[i]) for i in top])
    best = theta[top[np.argmax(lnp)]]
    map_npz = RES / "2026-10-18_candidate_map" / f"{tid}_map_free.npz"
    theta_map = map_point(model, map_npz)
    n_map = len(np.load(map_npz)["labels"])

    o = obs[0]
    wave = np.asarray(o.wavelength, float)
    flux = np.asarray(o.flux, float)
    unc = np.asarray(o.uncertainty, float)
    mask = np.asarray(o.mask, bool) & np.isfinite(unc) & (unc > 0)
    z = float(np.atleast_1d(model.params["zred"])[0])
    rows = {}
    for key, theta_k, n_free, name in (
        ("poco", best, model.ndim, r"pocoMC free\_dust, highest posterior"),
        ("map", theta_map, n_map, r"quick MAP, free variant, SFH hyperparameters of \#14"),
    ):
        spec = np.asarray(model.predict(theta_k, observations=obs, sps=cue)[0][0], float)
        chi = np.where(mask, (flux - spec) / unc, np.nan)
        rows[key] = dict(model=spec, chi=chi, n_free=n_free, name=name)
    data = dict(wave=wave, rest=wave / (1 + z), flux=flux, mask=mask, z=z)
    return data, rows


def chi_label(row, mask, sel=None):
    """Return the chi2_nu text of one row, with the pixel count and the pixels beyond 5."""
    m = mask if sel is None else mask & sel
    chi = row["chi"][m]
    n = int(m.sum())
    red = float(np.sum(chi**2)) / (n - row["n_free"])
    off = int(np.sum(np.abs(chi) > 5))
    return rf"$\chi^2_\nu$ = {red:.3f} ({n} px, {off} beyond $\pm$5)"


def residual_figure(tid, data, rows, path):
    """Draw the data and the pocoMC best fit, then the chi of each row from -5 to 5."""
    import matplotlib.pyplot as plt

    rest, mask = data["rest"], data["mask"]
    fig, ax = plt.subplots(
        1 + len(rows),
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3] + [1] * len(rows), "hspace": 0.06},
    )
    ax[0].plot(
        rest, np.where(mask, desi(data["wave"], data["flux"]), np.nan), color="0.2", lw=0.6,
        label="DESI (recoadd)",
    )  # fmt: skip
    ax[0].plot(
        rest, np.where(mask, desi(data["wave"], rows["poco"]["model"]), np.nan),
        color=COLORS["poco"], lw=0.8, label=rows["poco"]["name"],
    )  # fmt: skip
    ax[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=14)
    ax[0].legend(frameon=False, fontsize=13, loc="upper right")
    ax[0].set_title(f"{tid}, z = {data['z']:.5f}", fontsize=15)
    for a, (key, row) in zip(ax[1:], rows.items(), strict=True):
        a.plot(rest, row["chi"], color=COLORS[key], lw=0.5)
        for h in (-2, 2):
            a.axhline(h, color="0.3", ls=":", lw=1)
        a.axhline(0, color="0.3", lw=0.6)
        a.set_ylim(-5, 5)
        a.set_ylabel(r"$\chi$", fontsize=14)
        a.text(
            0.005, 0.88, f"{row['name']}: " + chi_label(row, mask), transform=a.transAxes,
            fontsize=12, va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
        )  # fmt: skip
    ax[-1].set_xlabel(r"rest wavelength [$\AA$]", fontsize=14)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def windows_figure(tid, data, rows, path):
    """Draw the line windows with the data, each row's model and chi, and chi2/N per window."""
    import matplotlib.pyplot as plt
    from map import WINDOWS

    rest, mask = data["rest"], data["mask"]
    fig, ax = plt.subplots(
        2,
        len(WINDOWS),
        figsize=(3.2 * len(WINDOWS), 6.0),
        gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05},
        sharex="col",
    )
    for i, (key, (a, b)) in enumerate(WINDOWS.items()):
        win = (rest >= a) & (rest < b)
        m = win & mask
        ax[0, i].plot(
            rest[win], np.where(mask[win], desi(data["wave"][win], data["flux"][win]), np.nan),
            color="0.2", lw=1.0,
        )  # fmt: skip
        handles = []
        for run, row in rows.items():
            if not m.any():
                continue
            c2n = float(np.sum(row["chi"][m] ** 2) / m.sum())
            (h,) = ax[0, i].plot(
                rest[win], np.where(mask[win], desi(data["wave"][win], row["model"][win]), np.nan),
                color=COLORS[run], lw=1.1, label=rf"$\chi^2/N$ = {c2n:.1f}",
            )  # fmt: skip
            handles.append(h)
            ax[1, i].plot(rest[win], row["chi"][win], color=COLORS[run], lw=0.9)
        ax[0, i].set_title(key, fontsize=14)
        lo, hi = ax[0, i].get_ylim()
        ax[0, i].set_ylim(lo, lo + 1.45 * (hi - lo))  # room for the chi2 legend above the lines
        ax[0, i].legend(handles=handles, frameon=False, fontsize=10, loc="upper left")
        for h in (-2, 2):
            ax[1, i].axhline(h, color="0.3", ls=":", lw=0.8)
        ax[1, i].axhline(0, color="0.3", lw=0.6)
        ax[1, i].set_ylim(-5, 5)
        ax[1, i].set_xlabel(r"rest [$\AA$]", fontsize=12)
    ax[0, 0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ cgs $\AA^{-1}$]", fontsize=13)
    ax[1, 0].set_ylabel(r"$\chi$", fontsize=13)
    handles = [plt.Line2D([], [], color="0.2", lw=2)] + [
        plt.Line2D([], [], color=COLORS[k], lw=3) for k in rows
    ]
    names = [f"DESI recoadd, {tid}"] + [r["name"] for r in rows.values()]
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3, fontsize=13,
               frameon=False)  # fmt: skip
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# %%
def main():
    """Write the two figures of the target in FIT_TID."""
    import fit

    from hubersed.plotting.style import use_apj_style

    argparse.ArgumentParser(description=__doc__).parse_args()
    use_apj_style()
    OUT.mkdir(exist_ok=True)
    tid = fit.TID
    data, rows = rows_of(tid)
    for row in rows.values():
        print(row["name"], chi_label(row, data["mask"]))
    residual_figure(tid, data, rows, OUT / f"{tid}_residuals.png")
    windows_figure(tid, data, rows, OUT / f"{tid}_windows.png")


if __name__ == "__main__":
    main()
