"""MAP fits and n_H profiles of the rmatrix mask setup of map.py with [SII]6731 fitted.

QUESTION: [SII]6731 sits on a spender sky-line mask, so the fits have no [SII] density diagnostic.
Does fitting it move log n_H toward the [SII] value (log n_e about 2.4, 2026-10-14_sii_density.py)
from the 1.12 of the rmatrix nautilus run?

Same model, data and start as ``map.py rmatrix mask``, except load_data restores the pixels within
+-6 A rest of [SII]6731 (14 pixels). Start 0 is map.py's start. Starts 1 to 5 are jittered as
map_fit does (2% of each prior range, rng seed 0). Start 6 is start 0 with log n_H at the [SII]
value, 2.4. One start per process.

The starts end along the U-n_H ridge, so the profile action fixes log n_H at one PROFILE_NH value
and fits the rest from the best unmasked start. -lnp then lacks the constant n_H prior term, so
compare it only between profile points.

COMMAND: from the repository root on the Mac with the C3K_ER build, for k in 0 to 6,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_nautilus_resolution_matrix/map_unmask.py run k [--maxfev N] [--masked]``,
then ``... map_unmask.py report``. For the profile, ``... map_unmask.py profile --nh V`` for each
V in PROFILE_NH, then report. ``... map_unmask.py windows`` splits the profile chi2 by line window.
``--masked`` keeps [SII]6731 masked, as the control.

The model over-predicts the auroral [OII]7320,7330 and [SIII]6312 lines, which hold n_H down. So
``profile --nh V --drop D --k K`` repeats the profile with them masked (D in DROPS), from two starts:
the unmasked profile point at V (K 0) and a 2% jitter of it (K 1, rng seed 1). Then
``... map_unmask.py profiles`` compares the variants.

OUTPUT: map_unmask/<unmask|masked>_start<k>.npz, map_unmask/profile_nH<V>.npz,
map_unmask/profile_<D>_nH<V>_s<K>.npz, map_unmask.txt, map_unmask_profiles.txt,
map_unmask_windows.txt and map_unmask_windows.png in
results/2026-10-13_nautilus_resolution_matrix/.
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from map import HYPERS, OUT, START
from model_setup import (
    MISSING_LINES,
    WAVE_OBS,
    get_cue,
    load_data,
    make_model,
    make_obs,
    make_params,
)

from hubersed.fitting.map_fits import jitter_scale, map_fit, safe_lnprior
from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style

N_STARTS = 6
NH_SII = 2.4
PROFILE_NH = (1.2, 1.6, 2.0, 2.4, 2.8)
JITTER_FRAC = 0.02
SEED = 0
DIR = OUT / "map_unmask"
GAS = ("gas_logz", "gas_logu", "gas_lognH", "gas_logno")
# rest-frame windows in A for the chi2 split; pixels outside all of them are continuum
WINDOWS = {
    "[OII]3726,3729": (3718, 3740),
    "[NeIII]3869": (3862, 3876),
    "Hdelta": (4095, 4110),
    "Hgamma": (4333, 4348),
    "[OIII]4363": (4359, 4370),
    "Hbeta": (4850, 4876),
    "[OIII]4959": (4950, 4970),
    "[OIII]5007": (4997, 5025),
    "[OI]6300": (6292, 6310),
    "[SIII]6312": (6310, 6318),
    "Halpha+[NII]": (6540, 6595),
    "[SII]6716": (6708, 6724),
    "[SII]6731": (6724, 6740),
    "[OII]7320,7330": (7314, 7338),
}
# windows of WINDOWS masked in each profile variant
DROPS = {
    "none": (),
    "no7325": ("[OII]7320,7330",),
    "noaur": ("[OII]7320,7330", "[SIII]6312"),
    "noaur4363": ("[OII]7320,7330", "[SIII]6312", "[OIII]4363"),
}
PROFILE_PARS = ("gas_logz", "gas_logu", "gas_logno", "gas_logco", "dust2", "dust_index")
PLOT_WINDOWS = {
    "[OII]3726,3729": (3715, 3742),
    "Hbeta": (4850, 4876),
    "[OI]6300 + [SIII]6312": (6292, 6320),
    "Halpha+[NII]": (6540, 6592),
    "[SII]6716,6731": (6706, 6742),
    "[OII]7320,7330": (7312, 7340),
}
PLOT_NH = {1.6: "#2166ac", 2.0: "#4dac26", 2.4: "#e08214", 2.8: "#b2182b"}
RATIOS = {
    "[SII]6716/6731": ("[S II] 6716", "[S II] 6731"),
    "[OII]3729/3726": ("[O II] 3729", "[O II] 3726"),
}


def build(unmask, fixed_nh=None, drop="none"):
    """Return the model, observation, Cue source and map.py start point.

    ``fixed_nh`` fixes gas_lognH at that value. ``drop`` masks the DROPS windows of that variant.
    """
    d = np.load(START, allow_pickle=True)
    best = dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))
    z, flux, unc, good = load_data(tuple(MISSING_LINES), ("[SII]6731",) if unmask else ())
    assert np.isclose(z, float(d["z"]))
    rest = WAVE_OBS / (1 + z)
    for w in DROPS[drop]:
        a, b = WINDOWS[w]
        good = good & ~((rest >= a) & (rest < b))
    unc = np.where(good, unc, np.inf)
    params = make_params(z)
    for k in HYPERS:
        params[k]["isfree"] = False
        params[k]["init"] = float(best[k])
    if fixed_nh is not None:
        params["gas_lognH"]["isfree"] = False
        params["gas_lognH"]["init"] = fixed_nh
    model = make_model(params, "rmatrix")
    theta0 = np.array([best[k] for k in model.theta_labels()])
    return model, make_obs(flux, unc, good, "rmatrix"), get_cue(), theta0


def starts(model, theta0):
    """Return the start points: theta0, jittered draws inside the prior, then theta0 at NH_SII."""
    rng = np.random.default_rng(SEED)
    scale = jitter_scale(model, theta0, JITTER_FRAC)
    out = [theta0.copy()]
    while len(out) < N_STARTS:
        st = theta0 + rng.normal(0.0, scale)
        if np.isfinite(safe_lnprior(model, st)):
            out.append(st)
    st = theta0.copy()
    st[model.theta_index["gas_lognH"]] = NH_SII
    assert np.isfinite(safe_lnprior(model, st))
    return out + [st]


def run(k, maxfev, unmask):
    """Fit from start k and save the result."""
    model, obs, cue, theta0 = build(unmask)
    st = starts(model, theta0)[k]
    tag = f"{'unmask' if unmask else 'masked'}_start{k}"
    t0 = time.time()
    best, _ = map_fit(model, obs, cue, n_seeds=0, maxfev=maxfev, theta0=st, tag=tag)
    DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        DIR / f"{tag}.npz",
        labels=np.array(model.theta_labels()),
        theta_start=st,
        theta=best.x,
        fun=best.fun,
        nfev=best.nfev,
        success=best.success,
        maxfev=maxfev,
        npix=int(obs[0].mask.sum()),
        seconds=time.time() - t0,
    )


def profile(nh, maxfev, drop="none", k=0):
    """Fit with log n_H fixed at ``nh`` and save the result.

    With drop "none" the start is the best unmasked start. Otherwise start 0 is the unmasked
    profile point at ``nh`` and start 1 is a jitter of it.
    """
    model, obs, cue, _ = build(True, fixed_nh=nh, drop=drop)
    if drop == "none":
        src = min(
            (np.load(f) for f in DIR.glob("unmask_start*.npz")), key=lambda d: float(d["fun"])
        )
        name = f"profile_nH{nh:.1f}.npz"
    else:
        src = np.load(DIR / f"profile_nH{nh:.1f}.npz")
        name = f"profile_{drop}_nH{nh:.1f}_s{k}.npz"
    prev = dict(zip([str(x) for x in src["labels"]], src["theta"], strict=True))
    st = np.array([prev[x] for x in model.theta_labels()])
    if k == 1:
        rng = np.random.default_rng(SEED + 1)
        scale = jitter_scale(model, st, JITTER_FRAC)
        while True:
            jit = st + rng.normal(0.0, scale)
            if np.isfinite(safe_lnprior(model, jit)):
                st = jit
                break
    t0 = time.time()
    best, _ = map_fit(model, obs, cue, n_seeds=0, maxfev=maxfev, theta0=st, tag=f"{drop} nH {nh}")
    np.savez(
        DIR / name,
        drop=drop,
        k=k,
        nh=nh,
        labels=np.array(model.theta_labels()),
        theta_start=st,
        theta=best.x,
        fun=best.fun,
        nfev=best.nfev,
        success=best.success,
        maxfev=maxfev,
        seconds=time.time() - t0,
    )


def chi2(model, obs, cue, theta):
    """Return chi2 of the model at ``theta`` on the fitted pixels."""
    sp = model.predict(theta, observations=obs, sps=cue)[0][0]
    o = obs[0]
    return float(np.sum(((o.flux - sp)[o.mask] / o.uncertainty[o.mask]) ** 2))


def report():
    """Write chi2, reduced chi2 and the gas parameters of every fit to map_unmask.txt.

    d(chi2) is from the lowest chi2 in each table. A profile fit's theta gets its fixed n_H back,
    so it is scored with the free model.
    """
    out = []
    head = f"{'chi2':>9} {'chi2_red':>9} {'d(chi2)':>8} {'-lnp':>11} {'nfev':>6} "
    head += " ".join(f"{g:>10}" for g in GAS)
    for tag in ("masked", "unmask"):
        files = sorted(DIR.glob(f"{tag}_start*.npz"))
        if not files and not (tag == "unmask" and list(DIR.glob("profile_nH*.npz"))):
            continue
        model, obs, cue, _ = build(tag == "unmask")
        lab = model.theta_labels()
        npix = int(obs[0].mask.sum())
        tables = [("start", [(f.stem.split("start")[1], np.load(f)) for f in files])]
        if tag == "unmask":
            prof = [(f"{float(d['nh']):.1f}", d) for d in map(np.load, DIR.glob("profile_nH*.npz"))]
            tables.append(("n_H", sorted(prof)))
        for name, rows in tables:
            if not rows:
                continue
            res = []
            for key, d in rows:
                fit = dict(zip([str(x) for x in d["labels"]], d["theta"], strict=True))
                fit.setdefault("gas_lognH", float(d["nh"]) if "nh" in d else np.nan)
                theta = np.array([fit[k] for k in lab])
                res.append((key, d, theta, chi2(model, obs, cue, theta), npix - len(d["labels"])))
            cmin = min(r[3] for r in res)
            out.append(f"{tag}, by {name} ({npix} pixels; dof = pixels - free parameters)")
            out.append(f"{name:>6} " + head)
            order = res if name == "n_H" else sorted(res, key=lambda r: r[3])
            for key, d, theta, c2, dof in order:
                out.append(
                    f"{key:>6} {c2:9.1f} {c2 / dof:9.4f} {c2 - cmin:8.2f} {float(d['fun']):11.2f} "
                    f"{int(d['nfev']):6d} " + " ".join(f"{theta[lab.index(g)]:10.3f}" for g in GAS)
                )
            out.append("")
    text = "\n".join(out)
    print(text)
    (OUT / "map_unmask.txt").write_text(text)


def profiles():
    """Write the profile of each DROPS variant to map_unmask_profiles.txt.

    chi2 is on each variant's own pixels, so d(chi2) is only compared within a variant. Each n_H
    point keeps its lower-chi2 start, and "starts" is the chi2 gap between its two starts.
    """
    model, _, cue, _ = build(True)
    lab = model.theta_labels()
    out = []
    for drop in DROPS:
        obs = build(True, drop=drop)[1]
        npix = int(obs[0].mask.sum())
        pattern = "profile_nH*.npz" if drop == "none" else f"profile_{drop}_nH*_s*.npz"
        rows = {}
        for f in DIR.glob(pattern):
            d = np.load(f)
            fit = dict(zip([str(x) for x in d["labels"]], d["theta"], strict=True))
            fit["gas_lognH"] = float(d["nh"])
            theta = np.array([fit[x] for x in lab])
            rows.setdefault(float(d["nh"]), []).append((chi2(model, obs, cue, theta), theta, d))
        if not rows:
            continue
        cmin = min(min(r[0] for r in v) for v in rows.values())
        out.append(f"variant {drop}: masked {DROPS[drop] or 'nothing extra'}, {npix} pixels")
        out.append(
            f"{'n_H':>5} {'chi2':>8} {'chi2_red':>9} {'d(chi2)':>8} {'starts':>7} "
            + " ".join(f"{x:>10}" for x in PROFILE_PARS)
        )
        for nh in sorted(rows):
            v = sorted(rows[nh], key=lambda r: r[0])
            c2, theta, d = v[0]
            gap = f"{v[1][0] - c2:7.1f}" if len(v) > 1 else f"{'-':>7}"
            out.append(
                f"{nh:5.1f} {c2:8.1f} {c2 / (npix - len(d['labels'])):9.4f} {c2 - cmin:8.2f} {gap} "
                + " ".join(f"{theta[lab.index(x)]:10.3f}" for x in PROFILE_PARS)
            )
        out.append("")
    text = "\n".join(out)
    print(text)
    (OUT / "map_unmask_profiles.txt").write_text(text)


def windows():
    """Write the profile chi2 by line window and the model density ratios to map_unmask_windows.txt.

    Each column is one profile point, with chi2 relative to the lowest-chi2 point. The data ratios
    are EmFit (narrow plus outflow) for [SII] and FastSpecFit for [OII].
    """
    model, obs, cue, _ = build(True)
    lab = model.theta_labels()
    o = obs[0]
    rest = o.wavelength / (1 + float(np.atleast_1d(model.params["zred"])[0]))
    sel = {k: o.mask & (rest >= a) & (rest < b) for k, (a, b) in WINDOWS.items()}
    sel["continuum"] = o.mask & ~np.any(list(sel.values()), axis=0)
    names = np.char.strip(np.asarray(model.emline_info["name"]).astype(str))
    prof = sorted((np.load(f) for f in DIR.glob("profile_nH*.npz")), key=lambda d: float(d["nh"]))
    cols, ratios, spectra = [], [], {}
    for d in prof:
        fit = dict(zip([str(x) for x in d["labels"]], d["theta"], strict=True))
        fit["gas_lognH"] = float(d["nh"])
        sp = model.predict(np.array([fit[k] for k in lab]), observations=obs, sps=cue)[0][0]
        c = ((o.flux - sp) / o.uncertainty) ** 2
        cols.append({k: float(c[m].sum()) for k, m in sel.items()})
        spectra[float(d["nh"])] = sp
        lum = model._eline_lum
        ratios.append(
            {k: float(lum[names == a][0] / lum[names == b][0]) for k, (a, b) in RATIOS.items()}
        )
    best = int(np.argmin([sum(c.values()) for c in cols]))
    out = [f"chi2 by window, relative to n_H {float(prof[best]['nh']):.1f} (rest A; pixels fitted)"]
    out.append(
        f"{'window':>16} {'pixels':>6} "
        + " ".join(f"{'n_H ' + str(float(d['nh'])):>9}" for d in prof)
    )
    for k, m in sel.items():
        out.append(
            f"{k:>16} {int(m.sum()):6d} " + " ".join(f"{c[k] - cols[best][k]:9.1f}" for c in cols)
        )
    out.append(
        f"{'total':>16} {int(o.mask.sum()):6d} "
        + " ".join(f"{sum(c.values()) - sum(cols[best].values()):9.1f}" for c in cols)
    )
    tid = 39627770174637084
    emfit = (
        Path.home()
        / "Documents/UberSED/knowledge/_evidence/2026-09-30_kin_vs_ion/emfit_rows_2obj.csv"
    )
    row = next(r for r in csv.DictReader(emfit.open()) if r["targetid"] == str(tid))
    sii = (float(row["sii6716_flux"]) + float(row["sii6716_out_flux"])) / (
        float(row["sii6731_flux"]) + float(row["sii6731_out_flux"])
    )
    f = fits.getdata(PATHS["DATA"] / "fastspec-iron-sv3-bright.fits", "FASTSPEC")
    i = np.flatnonzero(f["TARGETID"] == tid)[0]
    oii = float(f["OII_3729_FLUX"][i] / f["OII_3726_FLUX"][i])
    data = {"[SII]6716/6731": sii, "[OII]3729/3726": oii}
    out += ["", f"model line ratios (data: [SII] {sii:.3f} EmFit, [OII] {oii:.3f} FastSpecFit)"]
    for k in RATIOS:
        out.append(f"{k:>16} {data[k]:6.3f} " + " ".join(f"{r[k]:9.3f}" for r in ratios))
    text = "\n".join(out)
    print(text)
    (OUT / "map_unmask_windows.txt").write_text(text + "\n")
    plot_windows(o, rest, spectra, OUT / "map_unmask_windows.png")


def plot_windows(o, rest, spectra, path):
    """Plot the data and the profile models in the line windows, with chi below on -8 to 8.

    Pixels that are not fitted are left empty. The legend gives each model's chi2 in the window.
    """
    use_apj_style()
    fig, ax = plt.subplots(
        2,
        len(PLOT_WINDOWS),
        figsize=(3.4 * len(PLOT_WINDOWS), 5.5),
        gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05},
        sharex="col",
    )
    for i, (key, (a, b)) in enumerate(PLOT_WINDOWS.items()):
        win = (rest >= a) & (rest < b)
        fit = o.mask[win]
        x = rest[win]
        ax[0, i].plot(x, np.where(fit, o.flux[win], np.nan), color="0.2", lw=1.0, label="DESI")
        for nh, color in PLOT_NH.items():
            sp = spectra[nh]
            chi = (o.flux[win] - sp[win]) / o.uncertainty[win]
            c2 = float(np.sum(chi[fit] ** 2))
            ax[0, i].plot(
                x,
                np.where(fit, sp[win], np.nan),
                color=color,
                lw=0.9,
                label=rf"$\log n_H$ {nh:.1f}, $\chi^2$ {c2:.0f}",
            )
            ax[1, i].plot(x, np.where(fit, chi, np.nan), color=color, lw=0.8)
        ax[0, i].set_title(key)
        ax[0, i].legend(frameon=False, fontsize=6)
        for h in (-2, 2):
            ax[1, i].axhline(h, color="0.5", ls="--", lw=0.6)
        ax[1, i].axhline(0, color="0.3", lw=0.6)
        ax[1, i].set_ylim(-8, 8)
        ax[1, i].set_xlabel(r"rest wavelength [$\AA$]")
    ax[0, 0].set_ylabel("flux [maggies]")
    ax[1, 0].set_ylabel(r"$\chi$")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("run", "profile", "profiles", "report", "windows"))
    ap.add_argument("k", nargs="?", type=int, choices=range(N_STARTS + 1), default=0)
    ap.add_argument("--maxfev", type=int, default=120000)
    ap.add_argument("--masked", action="store_true")
    ap.add_argument("--nh", type=float, choices=PROFILE_NH)
    ap.add_argument("--drop", choices=tuple(DROPS), default="none")
    ap.add_argument("--k", dest="pk", type=int, choices=(0, 1), default=0)
    a = ap.parse_args()
    if a.action == "report":
        report()
    elif a.action == "windows":
        windows()
    elif a.action == "profiles":
        profiles()
    elif a.action == "profile":
        profile(a.nh, a.maxfev, a.drop, a.pk)
    else:
        run(a.k, a.maxfev, not a.masked)
