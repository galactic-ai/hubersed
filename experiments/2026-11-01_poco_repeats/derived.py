"""Literature quantities of one target for the five ladder runs, from posterior and prior draws.

QUESTION: what stellar mass, ages, formation times, recent sSFR and D, metallicity and sigma
does each ladder run give, and how far are they from their priors?
The quantities follow the papers in UberSED knowledge/discussion_2026-11-04_deep_dive_metrics.md.
- Stellar mass: formed (logmass) and surviving (logmass + log10 of the FSPS surviving fraction).
- t_age: formed-mass-weighted lookback age (Tacchella+2022).
- t50 and t95: lookback times when 50 and 95 per cent of the mass had formed (Suess+2022,
  Slob+2024), also as redshift.
- Light-weighted age: the FSPS compute_light_ages spectrum averaged over rest 5580 to 6820 A
  (Suess+2022).
- SFR over the last 100 Myr over surviving mass, and SFR over 10 Myr for the CIGALE comparison.
- D = sSFR_100 t_H with the mass formed (Tacchella+2022: quiescent below 1/20, transitioning
  1/20 to 1/3), and the share of draws in each class. Quenching times are not computed: on our
  bins they follow the 1-10 Myr bins (Nikhil, 2026-11-04; Suess+2022 on bin-set quenching times).
- Metallicity: [Fe/H] and [alpha/Fe] of the afe runs, log Z/Zsun of all (ladder_hist.mist2_logz).
- sigma_smooth: the stellar velocity dispersion; instrumental resolution is handled separately.
Draws: 1000 weighted posterior resamples (seed 0) and 1000 prior draws through the model's
prior_transform (seed 1), which draws logsfr_ratios from the Gaussian set by the hyperparameters.
Ages of the universe use Planck18 (hubersed.sps.utils.universe_age_gyr), as the age bins do.
COMMAND: ``uv run --no-sync python experiments/2026-11-01_poco_repeats/derived.py draws --tid <TID>``
on the laptop. Then, on stampede3, derived.slurm runs ``derived.py fsps --tid <TID> --run <RUN>``
once per run with that run's build. Then ``derived.py table --tid <TID>`` on the laptop.
OUTPUT: results/2026-11-01_poco_repeats/<TID>/derived/: draws_<RUN>.npz, fsps_<RUN>.npz,
derived.txt.
"""

# %%
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

from hubersed.paths import PATHS

HERE = Path(__file__).resolve().parent
RES = PATHS["RESULTS"] / HERE.name
NDRAW = 1000
RUNS = {
    "miles": "MILES",
    "c3k_hr": "C3K HR (degraded)",
    "c3k_hr_afe": "C3K HR + afe (degraded)",
    "dr1": "C3K ER + afe + R matrix, DR1 coadd",
    "recoadd": "C3K ER + afe + R matrix, recoadd",
}
LW_WINDOW = (5580.0, 6820.0)


def load(name, path):
    """Load the file at ``path`` as module ``name``."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def out_dir(tid):
    """Return the output folder of ``tid``, created if needed."""
    d = RES / str(tid) / "derived"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cosmic_to_z():
    """Return a function from cosmic time in Gyr to redshift, Planck18 as the age bins."""
    from hubersed.sps.utils import universe_age_gyr

    zz = np.concatenate([np.linspace(0, 1, 2001), np.geomspace(1.0005, 30, 4000)])
    tt = np.asarray(universe_age_gyr(zz), float)
    order = np.argsort(tt)
    return lambda t: np.interp(t, tt[order], zz[order], left=np.nan, right=np.nan)


def sfh_quantities(model, theta, t_obs, t_to_z):
    """Return the SFH-only quantities of one parameter vector, in Gyr, Msun and Msun/yr."""
    model.set_parameters(np.asarray(theta, float))
    ab = 10 ** np.asarray(model.params["agebins"], float)  # lookback edges, yr
    m = np.asarray(model.params["mass"], float)  # formed per bin, Msun
    t0, t1 = ab[:, 0], ab[:, 1]
    sfr = m / (t1 - t0)
    mtot = m.sum()
    q = {"logmass_formed": np.log10(mtot)}
    q["t_age"] = float(np.sum(m * (t0 + t1) / 2) / mtot / 1e9)
    # fraction of the mass formed at lookback >= t, linear inside each bin
    edges = np.append(t0, t1[-1])
    frac_older = np.append(np.cumsum(m[::-1])[::-1], 0.0) / mtot
    for p in (50, 95):
        t = np.interp(p / 100, frac_older[::-1], edges[::-1]) / 1e9
        q[f"t{p}"] = float(t)
        q[f"z{p}"] = float(t_to_z(t_obs - t))
    for name, span in (("sfr100", 1e8), ("sfr10", 1e7)):
        overlap = np.clip(np.minimum(t1, span) - t0, 0, None)
        q[name] = float(np.sum(sfr * overlap) / span)
    q["D"] = q["sfr100"] * t_obs * 1e9 / mtot
    return q


def draws(tid):
    """Write posterior and prior draws and their SFH-only quantities for every run."""
    lc = load("ladder_corner", HERE / "ladder_corner.py")
    lh = load("ladder_hist", HERE / "ladder_hist.py")
    ana, rr = lc.runs(tid)
    assert [r[0] for r in rr] == list(RUNS.values()), [r[0] for r in rr]
    logz, _ = lh.mist2_logz(RES / "mist2_zh.txt")
    t_to_z = cosmic_to_z()
    from hubersed.sps.utils import universe_age_gyr

    for key, (label, _, model, _, w, theta) in zip(RUNS, rr, strict=True):
        z = float(np.atleast_1d(model.params["zred"])[0])
        t_obs = float(universe_age_gyr(z))
        post = theta[np.random.default_rng(0).choice(len(w), size=NDRAW, p=w)]
        u = np.random.default_rng(1).uniform(size=(NDRAW, model.ndim))
        prior = np.array([model.prior_transform(x) for x in u])
        labels = list(model.theta_labels())
        out = {"labels": np.array(labels), "z": z, "t_obs": t_obs}
        for kind, th in (("post", post), ("prior", prior)):
            qs = [sfh_quantities(model, t, t_obs, t_to_z) for t in th]
            out[f"theta_{kind}"] = th
            for k in qs[0]:
                out[f"{kind}_{k}"] = np.array([q[k] for q in qs])
            col = {p: th[:, i] for i, p in enumerate(labels)}
            afe = col.get("afe", np.zeros(len(th)))
            out[f"{kind}_logzsol"] = col["logzsol"]
            out[f"{kind}_afe"] = afe
            out[f"{kind}_logz_zsun"] = logz(col["logzsol"], afe)
            out[f"{kind}_sigma_smooth"] = col["sigma_smooth"]
        path = out_dir(tid) / f"draws_{key}.npz"
        np.savez(path, **out)
        print(f"{label}: wrote {path}", flush=True)


def commit_room_gib():
    """Return CommitLimit minus Committed_AS of this node in GiB (strict overcommit on TACC)."""
    with open("/proc/meminfo") as f:
        kb = {line.split(":")[0]: int(line.split()[1]) for line in f}
    return (kb["CommitLimit"] - kb["Committed_AS"]) / 2**20


def fsps(tid, key, n=None, build=0):
    """Write the FSPS surviving fraction and light-weighted ages of the draws of run ``key``.

    With ``n`` only the first n draws of each kind run, written to a probe file, for timing. With
    ``build`` the C3K builds make all their SSP slots first, with at most that many helpers, each
    charged ~15 GiB of commit as in fit.py (GIB_HELPER); MILES has no build_ssps.
    """
    if key == "dr1":
        poco = load("quiescent_fit", HERE / "fit.py").load_poco(tid)
    else:
        poco = load("fit_ladder", HERE / "fit_ladder.py").load_ladder(tid, key)
    model, _, sps = poco.state()
    if build and getattr(sps.ssp, "build_ssps", None) is not None and key != "miles":
        k = max(1, min(build, int((commit_room_gib() - 10) // 15)))
        t0 = time.time()
        sps.ssp.build_ssps(processes=k)
        print(f"{key}: built SSP slots with {k} helpers in {time.time() - t0:.0f} s", flush=True)
    d = np.load(out_dir(tid) / f"draws_{key}.npz")
    assert list(d["labels"]) == list(model.theta_labels()), "draws were made with another model"
    out = {}
    for kind in ("post", "prior"):
        mfrac, lw, mw, t0 = [], [], [], time.time()
        for i, theta in enumerate(d[f"theta_{kind}"][:n]):
            model.set_parameters(theta)
            _, _, mf = sps.get_galaxy_spectrum(**model.params)
            tmax = 10 ** np.max(model.params["agebins"]) / 1e9
            sps.ssp.params["compute_light_ages"] = True
            wave, ages = sps.ssp.get_spectrum(tage=tmax, peraa=False)
            mw.append(float(sps.ssp.stellar_mass))
            sps.ssp.params["compute_light_ages"] = False
            band = (wave >= LW_WINDOW[0]) & (wave <= LW_WINDOW[1])
            lw.append(float(np.trapezoid(ages[band], wave[band]) / np.ptp(wave[band])))
            mfrac.append(float(mf))
            if i in (0, 9, 99) or (i + 1) % 250 == 0:
                print(
                    f"{key} {kind} {i + 1} draws, {(time.time() - t0) / (i + 1):.2f} s each",
                    flush=True,
                )
        out[f"{kind}_mfrac"] = np.array(mfrac)
        out[f"{kind}_lw_age"] = np.array(lw)
        out[f"{kind}_mw_age_fsps"] = np.array(mw)
    path = out_dir(tid) / (f"fsps_{key}.npz" if n is None else f"fsps_{key}_probe{n}.npz")
    np.savez(path, **out)
    print(f"wrote {path}", flush=True)
    with open("/proc/self/status") as f:
        print("".join(line for line in f if line.split(":")[0] in ("VmData", "VmHWM")), flush=True)


ROWS = (
    ("logmass_formed", "log M formed [Msun]", 1),
    ("logmass_surv", "log M surviving [Msun]", 1),
    ("t_age", "t_age, mass-weighted [Gyr]", 1),
    ("lw_age", "light-weighted age 5580-6820 A [Gyr]", 1),
    ("t50", "t50 lookback [Gyr]", 1),
    ("z50", "z at t50", 1),
    ("t95", "t95 lookback [Gyr]", 1),
    ("z95", "z at t95", 1),
    ("log_ssfr100", "log sSFR 100 Myr [1/yr]", 1),
    ("log_sfr10", "log SFR 10 Myr [Msun/yr]", 1),
    ("D", "D = sSFR100 t_H (M formed)", 1),
    ("f_quiescent", "share with D < 1/20", 1),
    ("f_below_third", "share with D < 1/3", 1),
    ("logz_zsun", "log Z/Zsun (MIST2)", 1),
    ("logzsol", "logzsol ([Fe/H] if afe free)", 1),
    ("afe", "[alpha/Fe]", 1),
    ("sigma_smooth", "sigma [km/s]", 1),
)


def table(tid):
    """Write 16, 50 and 84 per cent quantiles of every quantity, posterior and prior, per run."""
    d_out = out_dir(tid)
    cols = {}
    for key in RUNS:
        d = dict(np.load(d_out / f"draws_{key}.npz"))
        f = dict(np.load(d_out / f"fsps_{key}.npz"))
        for kind in ("post", "prior"):
            c = {k[len(kind) + 1 :]: v for k, v in d.items() if k.startswith(kind + "_")}
            c["logmass_surv"] = c["logmass_formed"] + np.log10(f[f"{kind}_mfrac"])
            c["lw_age"] = f[f"{kind}_lw_age"]
            with np.errstate(divide="ignore"):
                c["log_ssfr100"] = np.log10(c["sfr100"]) - c["logmass_surv"]
                c["log_sfr10"] = np.log10(c["sfr10"])
            c["f_quiescent"] = np.full(1, np.mean(c["D"] < 1 / 20))
            c["f_below_third"] = np.full(1, np.mean(c["D"] < 1 / 3))
            if key in ("miles", "c3k_hr"):
                c["afe"] = np.full_like(c["afe"], np.nan)
            cols[(key, kind)] = c
    lines = [f"{'quantity':38s} " + " ".join(f"{k:>26s}" for k in RUNS)]
    for name, desc, _ in ROWS:
        for kind in ("post", "prior"):
            cells = []
            for key in RUNS:
                v = cols[(key, kind)][name]
                ok = np.isfinite(v)
                if ok.mean() < 0.5:
                    cells.append(
                        f"{'-' if not ok.any() else f'{100 * ok.mean():.0f}% defined':>26s}"
                    )
                    continue
                if v.size == 1:
                    cells.append(f"{v[0]:9.3f}".rjust(26))
                    continue
                a, m, b = np.percentile(v[ok], [16, 50, 84])
                tag = "" if ok.all() else f" {100 * ok.mean():.0f}%"
                cells.append(f"{m:+9.3f} -{m - a:.3f} +{b - m:.3f}{tag}".rjust(26))
            lines.append(f"{(desc if kind == 'post' else '  prior'):38s} " + " ".join(cells))
    lines += [
        "",
        "median -(median - q16) +(q84 - median) over 1000 draws; 'prior' rows use 1000 prior draws.",
        "A percentage marks the share of draws where the quantity is finite. Share rows are single",
        "numbers. afe is fixed or absent for miles and c3k_hr. See the docstring for definitions.",
    ]
    path = d_out / "derived.txt"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("draws", "fsps", "table"))
    ap.add_argument("--tid", type=int, required=True)
    ap.add_argument("--run", choices=list(RUNS))
    ap.add_argument("--n", type=int, default=None, help="fsps only: first n draws, for a probe")
    ap.add_argument("--build", type=int, default=0, help="fsps only: SSP build helpers, 0 for lazy")
    a = ap.parse_args()
    if a.action == "draws":
        draws(a.tid)
    elif a.action == "fsps":
        fsps(a.tid, a.run, a.n, a.build)
    else:
        table(a.tid)
