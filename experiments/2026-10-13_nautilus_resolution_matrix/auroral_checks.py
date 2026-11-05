"""Line fluxes, attenuation and [SIII]9069 of the n_H profile points of map_unmask.py.

QUESTION: the model over-predicts [OII]7320,7330 and [SIII]6312 (map_unmask_windows.txt). Is that
a temperature problem of Cue, or does Prospector's attenuation curve differ from the one the data
need between 3727 and 7325?

Three checks at the profile points n_H 1.6 and 2.4 (profile_nH<V>.npz, [SII]6731 fitted). The line
flux tables also give the max-L point of the rmatrix nautilus run (NAUTILUS, all 29 parameters
free, [SII]6731 masked in that fit), the global best fit with nothing fixed:

- observed line fluxes. The model's attenuated line fluxes against the FastSpecFit fluxes, each
  divided by its own Halpha. No dust correction enters.
- attenuation. Prospector's effective A(lambda) on the lines, from the fluxes with and without dust
  (dust2 = 0, so dust1 = dust2 * dust_ratio is 0 too), against CCM89 (R_V 3.1) at the E(B-V) of
  the FastSpecFit Balmer decrement for an intrinsic Halpha/Hbeta of 2.86.
- [SIII]6312/9069, which depends on temperature and not on density. 9069 and 9532 are outside the
  fitted range (rest < 9000 A), so this plots data and model there without fitting.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_nautilus_resolution_matrix/auroral_checks.py``

OUTPUT: auroral_checks.txt and auroral_checks_siii.png in
results/2026-10-13_nautilus_resolution_matrix/.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyneb as pn
from astropy.io import fits
from map_unmask import DIR, OUT, build
from model_setup import TID, make_model, make_params

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

NH = (1.6, 2.4)
NAUTILUS = OUT / f"{TID}_rmatrix_mask_er_broad_shared_split45_rest9000_seed0_pred.npz"
MAGGIE_TO_1E17 = 3631e-23 / 1e-17
HA_HB_INTRINSIC = 2.86
# name, Cue line names summed, FastSpecFit columns summed
LINES = [
    ("[OII]3726,3729", ("[O II] 3726", "[O II] 3729"), ("OII_3726", "OII_3729")),
    ("Hbeta", ("Ba-beta 4861",), ("HBETA",)),
    ("[OIII]4363", ("[O III] 4363",), ("OIII_4363",)),
    ("[OIII]5007", ("[O III] 5007",), ("OIII_5007",)),
    ("[OI]6300", ("[O I] 6300",), ("OI_6300",)),
    ("[SIII]6312", ("[S III] 6312",), ("SIII_6312",)),
    ("Halpha", ("Ba-alpha 6563",), ("HALPHA",)),
    ("[NII]6584", ("[N II] 6584",), ("NII_6584",)),
    ("[SII]6716", ("[S II] 6716",), ("SII_6716",)),
    ("[SII]6731", ("[S II] 6731",), ("SII_6731",)),
    ("[OII]7320,7330", ("[O II] 7323", "[O II] 7332"), ("OII_7320", "OII_7330")),
    ("[SIII]9069", ("[S III] 9069",), ("SIII_9069",)),
    ("[SIII]9532", ("[S III] 9532",), ("SIII_9532",)),
]
SIII_WINDOWS = {"[SIII]6312": (6300, 6325), "[SIII]9069": (9055, 9085), "[SIII]9532": (9520, 9548)}


def line_fluxes(model, theta, obs, cue):
    """Return the model's line fluxes in 1e-17 erg/s/cm2 and its spectrum at ``theta``."""
    sp = model.predict(theta, observations=obs, sps=cue)[0][0]
    names = np.char.strip(np.asarray(model.emline_info["name"]).astype(str))
    elum = model._eline_lum * model.flux_norm() / (1 + model._zred) * MAGGIE_TO_1E17
    out = {}
    for key, cue_names, _ in LINES:
        assert all((names == n).sum() == 1 for n in cue_names), cue_names
        out[key] = float(sum(elum[names == n][0] for n in cue_names))
    return out, sp


def main():
    model, obs, cue, _ = build(True)
    lab = model.theta_labels()
    o = obs[0]
    z = float(np.atleast_1d(model.params["zred"])[0])
    rest = o.wavelength / (1 + z)

    f = fits.getdata(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", "FASTSPEC")
    i = np.flatnonzero(f["TARGETID"] == TID)[0]
    data, err = {}, {}
    for key, _, cols in LINES:
        data[key] = float(sum(f[c + "_FLUX"][i] for c in cols))
        err[key] = float(np.sqrt(sum(1 / f[c + "_FLUX_IVAR"][i] for c in cols)))
    rc = pn.RedCorr(law="CCM89", R_V=3.1)
    rc.setCorr(data["Halpha"] / data["Hbeta"] / HA_HB_INTRINSIC, 6564.61, 4862.68)
    ebv = float(rc.E_BV)

    obsd, intr, spectra = {}, {}, {}
    for nh in NH:
        d = np.load(DIR / f"profile_nH{nh:.1f}.npz")
        fit = dict(zip([str(x) for x in d["labels"]], d["theta"], strict=True))
        fit["gas_lognH"] = nh
        theta = np.array([fit[k] for k in lab])
        obsd[nh], spectra[nh] = line_fluxes(model, theta, obs, cue)
        dust = {
            k: float(np.atleast_1d(model.params[k])[0]) for k in ("dust1", "dust2", "dust_index")
        }
        nodust = theta.copy()
        nodust[model.theta_index["dust2"]] = 0.0
        intr[nh] = line_fluxes(model, nodust, obs, cue)[0]
    nd = np.load(NAUTILUS, allow_pickle=True)
    free = make_model(make_params(z), "rmatrix")
    assert free.theta_labels() == [str(x) for x in nd["labels"]]
    obsd["max-L"] = line_fluxes(free, nd["best"], obs, cue)[0]
    cols = [*NH, "max-L"]

    out = [
        f"{TID}, FastSpecFit Halpha/Hbeta {data['Halpha'] / data['Hbeta']:.3f}, E(B-V) {ebv:.3f}"
    ]
    out.append(f"Prospector dust at n_H {NH[-1]}: {dust}")
    out += ["", "observed line fluxes / own Halpha (model attenuated, no dust correction anywhere)"]
    out.append(
        f"{'line':>15} {'data':>8} {'err':>7} "
        + " ".join(f"{('n_H ' + str(c)) if c != 'max-L' else 'naut maxL':>9}" for c in cols)
    )
    for key, _, _ in LINES:
        r = data[key] / data["Halpha"]
        e = err[key] / data["Halpha"]
        out.append(
            f"{key:>15} {r:8.4f} {e:7.4f} "
            + " ".join(f"{obsd[c][key] / obsd[c]['Halpha']:9.4f}" for c in cols)
        )
    out += ["", "model / data of those ratios"]
    for key, _, _ in LINES:
        r = data[key] / data["Halpha"]
        out.append(
            f"{key:>15} " + " ".join(f"{obsd[c][key] / obsd[c]['Halpha'] / r:9.3f}" for c in cols)
        )
    out += [
        "",
        "A(lambda) - A(Halpha) on the lines, mag: Prospector effective vs CCM89 at that E(B-V)",
    ]
    ha = 6564.61
    names = np.char.strip(np.asarray(model.emline_info["name"]).astype(str))
    for key, cue_names, _ in LINES:
        lam = float(np.mean([model.emline_info["wave"][names == n][0] for n in cue_names]))
        a_pr = [
            -2.5 * np.log10(obsd[nh][key] / intr[nh][key])
            + 2.5 * np.log10(obsd[nh]["Halpha"] / intr[nh]["Halpha"])
            for nh in NH
        ]
        a_ccm = 2.5 * np.log10(rc.getCorr(lam) / rc.getCorr(ha))
        out.append(
            f"{key:>15} {lam:8.1f}  CCM89 {a_ccm:+.3f}  " + " ".join(f"{a:+.3f}" for a in a_pr)
        )
    s = {c: obsd[c]["[SIII]6312"] / obsd[c]["[SIII]9069"] for c in cols}
    sd = data["[SIII]6312"] / data["[SIII]9069"]
    out += [
        "",
        f"[SIII]6312/9069 observed: data {sd:.4f}, model "
        + ", ".join(f"{c} {s[c]:.4f}" for c in cols),
    ]
    out.append(
        f"[SIII]9532/9069: data {data['[SIII]9532'] / data['[SIII]9069']:.3f}, model {obsd[NH[0]]['[SIII]9532'] / obsd[NH[0]]['[SIII]9069']:.3f}"
    )
    text = "\n".join(out)
    print(text)
    (OUT / "auroral_checks.txt").write_text(text + "\n")

    use_apj_style()
    fig, ax = plt.subplots(1, len(SIII_WINDOWS), figsize=(3.6 * len(SIII_WINDOWS), 3.2))
    colors = {1.6: "#2166ac", 2.4: "#e08214"}
    for j, (key, (a, b)) in enumerate(SIII_WINDOWS.items()):
        w = (rest >= a) & (rest < b)
        ax[j].plot(rest[w], o.flux[w], color="0.2", lw=1.0, label="DESI")
        for nh in NH:
            ax[j].plot(rest[w], spectra[nh][w], color=colors[nh], lw=0.9, label=rf"$\log n_H$ {nh}")
        fitted = o.mask[w]
        if not fitted.all():
            ax[j].fill_between(
                rest[w],
                0,
                1,
                where=~fitted,
                color="0.9",
                transform=ax[j].get_xaxis_transform(),
                zorder=0,
            )
        ax[j].set_title(key)
        ax[j].set_xlabel(r"rest wavelength [$\AA$]")
    ax[0].set_ylabel("flux [maggies]")
    ax[0].legend(frameon=False, fontsize=6)
    fig.savefig(OUT / "auroral_checks_siii.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
