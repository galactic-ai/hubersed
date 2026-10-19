"""Compare the pocoMC posteriors of the free and free_dust fits of 39627770174637084.

QUESTION: what changes when dust_ratio gets a TopHat(0, 2) prior and frac_nodust is freed, and is
the new dust split set by the data or by the prior?

HYPOTHESIS: dust2, dust_index and frac_nodust trade off, so the raw values differ but the effective
attenuation the spectrum sees stays close. Mass and SFH follow the effective attenuation.

The raw dust parameters cannot be compared directly, so the script derives what the light sees.
Prospect's add_dust (fork, sources/fake_fsps.py) applies, to stars and Cue lines alike,
T = [young exp(-dust1 (l/5500)^-1) + old] [(1 - f) exp(-tau2(l)) + f], with tau2 the Kriek and
Conroy curve times dust2 and f = frac_nodust. A_old = -2.5 log10 of the second factor. Nebular
lines are all young, so A_neb = A_old + 1.086 dust1 (l/5500)^-1. Prior draws of free_dust show
which derived values the data pin down.

INPUTS: the two pmc_final.state files of poco.py, copied from ls6. SEED: 0 for the prior draws.

COMMAND: from the repository root on the Mac,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-15_broadline_free_amplitude:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-18_poco_free_vs_free_dust.py``.

RESULT: 2026-10-18. The hypothesis fails. A_old(5500) falls from 0.69 to 0.33 mag and A_neb(Ha)
from 0.81 to 0.56 mag, but the nebular E(Hb - Ha) stays at 0.22-0.23 mag. In free_dust the covered
30% of the light is opaque blueward of ~4500 A, so A_old is flat at -2.5 log10(0.70) = 0.39 mag
there. free gets a similar grey slope at its dust_index ceiling of 0.4. With less attenuation the
formed log mass falls 0.14 dex, SFR over 100 Myr halves (0.105 to 0.058 Msun/yr) and the
mass-weighted age falls from 7.0 to 6.2 Gyr. Gas and line widths move less than 1 sigma (O/H 8.679
to 8.659). free_dust raises log Z by 32.8 and the best log L by 29.3. Both runs stop their MCMC at
the step cap (140 and 300), so neither is a converged reference.

FIGURES: marginals.png (shared and derived parameters) and dust_sfh.png (effective attenuation
curves and sSFR) in results/2026-10-18_poco_free_vs_free_dust/, with summary.txt.
"""

import numpy as np
from prospect.models.transforms import logsfr_ratios_to_masses

from hubersed.paths import PATHS

TID = 39627770174637084
RES = PATHS["RESULTS"]
POCO = RES / "2026-10-15_broadline_free_amplitude"
STATES = {
    "free": POCO / f"poco_{TID}_free_neff512_ls6" / "pmc_final.state",
    "free_dust": POCO / f"poco_{TID}_free_dust_neff1024" / "pmc_final.state",
}
COLORS = {"free": "#2166ac", "free_dust": "#b2182b", "prior": "0.6"}
OUT = RES / "2026-10-18_poco_free_vs_free_dust"
DUST1_INDEX = -1.0  # python-fsps default (fsps.py:531), read by nebssp_basis.get_spectrum
LAM = {"3727": 3728.5, "4000": 4000.0, "Hb": 4862.7, "5500": 5500.0, "Ha": 6564.6}
CURVE = np.geomspace(3300.0, 9000.0, 120)
N_PRIOR = 20000
SHARED = [
    "logmass", "logzsol", "dust2", "dust_ratio", "dust_index", "sigma_smooth", "sigma_reg",
    "tau_eq", "sigma_dyn", "tau_dyn", "gas_logz", "gas_logu", "gas_lognH", "gas_logno",
    "gas_logco", "eline_sigma", "eline_sigma_broad",
]  # fmt: skip


def kc13_tau(lam, dust2, dust_index):
    """Return the Kriek and Conroy diffuse optical depth, as dust_type 4 in the prospect fork."""
    x = 1e4 / lam
    cal = np.where(
        lam > 6300.0,
        1.17 * (-1.857 + 1.04 * x) + 1.78,
        1.17 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3) + 1.78,
    )
    cal = np.clip(cal / 0.44 / 4.05, 0.0, np.inf)
    eb = 0.85 - 1.9 * dust_index
    drude = eb * (lam * 350.0) ** 2 / ((lam**2 - 2175.0**2) ** 2 + (lam * 350.0) ** 2)
    return dust2 * (cal + drude / 4.05) * (lam / 5500.0) ** dust_index


def attenuation(p, lam, dust1_index):
    """Return A_old and A_neb in mag at ``lam`` for parameter columns ``p``, one row per sample."""
    lam = np.atleast_1d(lam)[None, :]
    d2, n = p["dust2"][:, None], p["dust_index"][:, None]
    f = p.get("frac_nodust", np.zeros(len(d2)))[:, None]
    a_old = -2.5 * np.log10((1 - f) * np.exp(-kc13_tau(lam, d2, n)) + f)
    dust1 = (p["dust_ratio"] * p["dust2"])[:, None]
    a_neb = a_old + 2.5 / np.log(10) * dust1 * (lam / 5500.0) ** dust1_index
    return a_old, a_neb


def sfh_columns(model, theta):
    """Return SFR over 1-10 and 1-100 Myr, sSFR_100 and the mass-weighted lookback age."""
    labels = list(model.theta_labels())
    ab = np.asarray(model.params["agebins"], float)
    t = 10**ab
    mid = t.mean(axis=1) / 1e9
    i_m = labels.index("logmass")
    i_r = [i for i, k in enumerate(labels) if k.startswith("logsfr_ratios")]
    out = {"sfr10": [], "sfr100": [], "ssfr100": [], "age_mw": []}
    for th in theta:
        m = logsfr_ratios_to_masses(logmass=th[i_m], logsfr_ratios=th[i_r], agebins=ab)
        for key, top in (("sfr10", 1e7), ("sfr100", 1e8)):
            over = np.clip(np.minimum(t[:, 1], top) - t[:, 0], 0.0, None) / np.diff(t, axis=1)[:, 0]
            out[key].append(np.sum(m * over) / (top - 1e6))
        out["ssfr100"].append(out["sfr100"][-1] / (0.6 * m.sum()))  # 0.6 as in sfh_from_theta
        out["age_mw"].append(np.sum(m * mid) / m.sum())
    return {k: np.array(v) for k, v in out.items()}


def derived(model, theta):
    """Return named columns of ``theta`` plus the derived dust and SFH values."""
    labels = list(model.theta_labels())
    p = {k: theta[:, labels.index(k)] for k in labels if k in model.config_dict}
    p["dust1"] = p["dust_ratio"] * p["dust2"]
    assert "dust1_index" not in model.params
    a_old, a_neb = attenuation(p, list(LAM.values()), DUST1_INDEX)
    for j, name in enumerate(LAM):
        p[f"A_old_{name}"] = a_old[:, j]
        p[f"A_neb_{name}"] = a_neb[:, j]
    p["E_neb_Hb_Ha"] = p["A_neb_Hb"] - p["A_neb_Ha"]
    p["S_old_4000_5500"] = p["A_old_4000"] / p["A_old_5500"]
    p["12+log(O/H)"] = p["gas_logz"] + 8.71
    return p


def wq(x, w, q=(0.16, 0.5, 0.84)):
    """Return weighted quantiles of ``x``."""
    o = np.argsort(x)
    return np.interp(q, np.cumsum(w[o]) / w.sum(), x[o])


def load_run(variant):
    """Return the model, theta, weights, log L and run summary of one pocoMC state."""
    import fit
    from compare_poco import load_sampler
    from model_setup import MISSING_LINES, load_data

    s = load_sampler(STATES[variant])
    u, w, logl, _ = s.posterior()
    w = w / w.sum()
    z = load_data(tuple(MISSING_LINES))[0]
    model = fit.make_model(z, variant)
    theta = np.array([model.prior_transform(x) for x in u])
    info = dict(iteration=s.t, calls=s.calls, logz=float(s.logz), ess=1 / np.sum(w**2))
    info["best_logl"] = float(np.max(logl))
    return model, theta, w, logl, info


def prior_draws(model, n, seed=0):
    """Return ``n`` prior draws of ``model``."""
    rng = np.random.default_rng(seed)
    return np.array([model.prior_transform(x) for x in rng.uniform(size=(n, model.ndim))])


def summary(runs, prior):
    """Return the comparison table as text."""
    lines = []
    for v, r in runs.items():
        i = r["info"]
        lines.append(
            f"{v}: iteration {i['iteration']}, calls {i['calls']}, log Z {i['logz']:.2f}, "
            f"ESS {i['ess']:.0f}, best log L {i['best_logl']:.2f}"
        )
    a, b = runs["free"]["info"], runs["free_dust"]["info"]
    lines.append(
        f"free_dust - free: d log Z {b['logz'] - a['logz']:.2f}, "
        f"d best log L {b['best_logl'] - a['best_logl']:.2f}"
    )
    lines.append("")
    lines.append("16/50/84%. shift = (free_dust - free) median / free (84-16)/2.")
    head = f"{'quantity':>18} {'free':>27} {'free_dust':>27} {'shift':>6} {'free_dust prior':>27}"
    lines.append(head)
    keys = [k for k in runs["free_dust"]["p"] if not k.startswith("logsfr_ratios")]
    for k in keys:
        row = f"{k:>18}"
        meds = {}
        for v in ("free", "free_dust"):
            p, w = runs[v]["p"], runs[v]["w"]
            if k in p:
                q = wq(p[k], w)
                meds[v] = (q[1], 0.5 * (q[2] - q[0]))
                row += f" {q[0]:8.4g} {q[1]:8.4g} {q[2]:8.4g}"
            else:
                row += f" {'':>8} {'fixed 0':>8} {'':>8}"
        if len(meds) == 2 and meds["free"][1] > 0:
            row += f" {(meds['free_dust'][0] - meds['free'][0]) / meds['free'][1]:6.1f}"
        else:
            row += f" {'':>6}"
        if k in prior:
            q = np.percentile(prior[k], [16, 50, 84])
            row += f" {q[0]:8.4g} {q[1]:8.4g} {q[2]:8.4g}"
        lines.append(row)
    return "\n".join(lines)


def marginals_figure(runs, prior, path):
    """Draw 1-D weighted marginals of both runs, with the free_dust prior for derived dust values."""
    import matplotlib.pyplot as plt

    keys = SHARED + [
        "frac_nodust", "dust1", "A_old_5500", "A_old_4000", "S_old_4000_5500", "A_neb_Ha",
        "E_neb_Hb_Ha", "sfr10", "sfr100", "ssfr100", "age_mw", "12+log(O/H)",
    ]  # fmt: skip
    ncol = 6
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow))
    for ax, k in zip(axes.ravel(), keys, strict=False):
        allx = np.concatenate([r["p"][k] for r in runs.values() if k in r["p"]])
        lo, hi = np.percentile(allx, [0.2, 99.8])
        log = k in ("sfr10", "sfr100", "ssfr100") and lo > 0
        bins = np.geomspace(lo, hi, 40) if log else np.linspace(lo, hi, 40)
        if k in prior and k.startswith(("A_", "E_", "S_", "dust1", "frac")):
            ax.hist(prior[k], bins=bins, density=True, histtype="stepfilled",
                    color=COLORS["prior"], alpha=0.35, label="free_dust prior")  # fmt: skip
        for v, r in runs.items():
            if k in r["p"]:
                ax.hist(r["p"][k], bins=bins, weights=r["w"], density=True, histtype="step",
                        color=COLORS[v], lw=1.6, label=v)  # fmt: skip
        if log:
            ax.set_xscale("log")
        ax.set_yticks([])
        ax.set_xlabel(k, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(keys) :]:
        ax.axis("off")
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    axes.ravel()[len(SHARED)].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{TID} pocoMC free (blue) vs free_dust (red)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}")


def dust_sfh_figure(runs, path):
    """Draw the A_old and A_neb curves (median and 16-84%) and the sSFR history of both runs."""
    import matplotlib.pyplot as plt

    from hubersed.fitting.map_fits import sfh_from_theta

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    rng = np.random.default_rng(0)
    for v, r in runs.items():
        idx = rng.choice(len(r["w"]), size=2000, p=r["w"])
        sub = {k: x[idx] for k, x in r["p"].items()}
        a_old, a_neb = attenuation(sub, CURVE, DUST1_INDEX)
        for a, curve, name in ((ax[0], a_old, "A_old"), (ax[1], a_neb, "A_neb")):
            lo, med, hi = np.percentile(curve, [16, 50, 84], axis=0)
            a.fill_between(CURVE, lo, hi, color=COLORS[v], alpha=0.25, lw=0)
            a.plot(CURVE, med, color=COLORS[v], lw=2, label=v)
            a.set_ylabel(f"{name} [mag]")
        sfhs = [sfh_from_theta(r["model"], r["theta"][i]) for i in idx[:1000]]
        edges = sfhs[0]["edges_gyr"]
        vals = np.array([s["ssfr"] for s in sfhs])
        lo, med, hi = np.percentile(vals, [16, 50, 84], axis=0)
        ax[2].stairs(hi, edges, baseline=lo, fill=True, color=COLORS[v], alpha=0.25, lw=0)
        ax[2].stairs(med, edges, color=COLORS[v], lw=2, label=v)
    for a in ax[:2]:
        a.set_xlabel("rest wavelength [A]")
        for lam in (LAM["Hb"], LAM["Ha"]):
            a.axvline(lam, color="0.5", ls=":", lw=1)
    ax[0].set_title("old stars, (1-f) exp(-tau2) + f")
    ax[1].set_title("nebular lines and young stars, + dust1")
    ax[2].set(xscale="log", yscale="log", xlabel="lookback time [Gyr]", ylabel="sSFR [1/yr]")
    ax[2].set_title("sSFR, current mass 0.6 x formed")
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    """Load both runs, write the table and draw the figures."""
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {}
    for v in STATES:
        model, theta, w, logl, info = load_run(v)
        p = derived(model, theta)
        p.update(sfh_columns(model, theta))
        runs[v] = dict(model=model, theta=theta, w=w, logl=logl, info=info, p=p)
    m = runs["free_dust"]["model"]
    th = prior_draws(m, N_PRIOR)
    prior = derived(m, th)
    prior.update(sfh_columns(m, th[:4000]))
    text = summary(runs, prior)
    print(text)
    (OUT / "summary.txt").write_text(text + "\n")
    marginals_figure(runs, prior, OUT / "marginals.png")
    dust_sfh_figure(runs, OUT / "dust_sfh.png")


if __name__ == "__main__":
    main()
