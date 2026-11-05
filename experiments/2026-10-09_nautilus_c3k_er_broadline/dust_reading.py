"""Dust model reading of 37084: optical depths, line response and where the dust is constrained.

QUESTION: why does the HR max-L point with only dust_ratio set to its ER max-L value overshoot
the emission lines in the ER model, and which wavelengths constrain dust2, dust_ratio and
dust_index?

The model attenuates the young CSP (ages up to dust_tesc, 10^7 yr) by dust1 and dust2 and the old
CSP by dust2 only (prospect fake_fsps.add_dust). Cue lines and Cue nebular continuum are computed
for each CSP from its own ionizing spectrum and get the same split. dust1 = dust2 * dust_ratio.

Six products:

- the optical depths tau2 (Kriek and Conroy curve) and tau1 (power law, index -1) at the lines,
  at the ER and HR max-L points, from prospect's own ``attenuate``;
- the factor exp(-delta tau1) that setting dust_ratio from 0.558 to 0.461 at the HR point gives
  the young lines;
- the dust_ratio column of the line Jacobian of jacobian.py against the young-only expectation
  -dust2 * (5500 / lambda) / ln 10, which gives the young share of each line;
- Fisher correlations and conditional widths of the dust parameters, and the share of their
  Fisher information in rest-frame wavelength bins, from jacobian.npz;
- the linear chi2 change from the ER max-L point to the HR point and to the HR point with the ER
  dust_ratio, against the values hr_point.py measured (hard-coded here);
- the information shares, widths and correlations in the basis dust1 (dust2 fixed) and dust2
  (dust1 fixed), and the young share of the model continuum, f_young = -J_ratio / (dust2 (5500 /
  lambda) m), on continuum pixels.

The Fisher numbers are local and linear at the ER max-L point and ignore the prior. The four SFH
hyperparameters have no pixel information and are dropped before the inversion.

INPUTS: jacobian.npz of jacobian.py, the ER .json and _pred.npz, and the HR split 45 chain through
``hr_point.hr_best``. SEED: none (deterministic).

Run from the repository root on the Mac with the C3K_ER build, as hr_point.py. Command:
``PYTHONPATH=<site with the C3K_ER fsps>:experiments/2026-10-09_nautilus_c3k_er_broadline
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-09_nautilus_c3k_er_broadline/dust_reading.py``.

OUTPUT: results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/dust_reading.txt.
"""

import json

import numpy as np
from hr_point import hr_best
from prospect.sources.fake_fsps import attenuate

from hubersed.paths import PATHS

TID = 39627770174637084
OUT = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline"
FIG = OUT / str(TID)
STEM = f"{TID}_er_broad_shared_split45_rest9000_seed0"
DUST1_INDEX = -1.0  # FSPS default, not set by the model
# rest-frame vacuum wavelengths
LINES = {
    "[OII]3727": 3728.5,
    "Hdelta": 4102.9,
    "Hbeta": 4862.7,
    "[OIII]5007": 5008.2,
    "Halpha": 6564.6,
    "[SII]6716": 6718.3,
    "cont 8500": 8500.0,
}
EDGES = [3500, 3700, 3740, 4000, 4300, 4800, 4900, 5000, 5030, 5500, 6000, 6540, 6600, 6700]
EDGES += [6750, 7500, 9100]


def taus(lam, dust2, dust_ratio, dust_index):
    """Return tau2 and tau1 at rest wavelengths ``lam`` from prospect's ``attenuate``."""
    one = np.ones_like(lam)
    _, ext2 = attenuate(one, lam, dust_type=4, dust_index=dust_index, dust2=dust2)
    ext1, _ = attenuate(one, lam, dust_type=4, dust1_index=DUST1_INDEX, dust1=dust2 * dust_ratio)
    return -np.log(ext2), -np.log(ext1)


def main():
    jz = np.load(FIG / "jacobian.npz")
    raw = [str(x) for x in jz["labels"]]
    labels = [f"{x}_{raw[:i].count(x) + 1}" if raw.count(x) > 1 else x for i, x in enumerate(raw)]
    ix = {x: i for i, x in enumerate(labels)}
    er = jz["best"]
    hr, _, _ = hr_best()
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    say("parameter        ER max-L   HR max-L   (HR-ER)/post_sd")
    for k in ("dust2", "dust_ratio", "dust_index", "gas_logz", "gas_logu", "gas_lognH"):
        i = ix[k]
        say(f"{k:14s} {er[i]:10.4f} {hr[i]:10.4f} {(hr[i] - er[i]) / jz['post_sd'][i]:+10.2f}")

    lam = np.array(list(LINES.values()))
    say("\noptical depths in e-folds: tau2 (all light), tau1 (young only), young total")
    pts = {"ER": er, "HR": hr}
    for name, th in pts.items():
        d2, dr, n = th[ix["dust2"]], th[ix["dust_ratio"]], th[ix["dust_index"]]
        t2, t1 = taus(lam, d2, dr, n)
        say(f"{name}: dust2 {d2:.4f}, dust_ratio {dr:.4f}, dust1 {d2 * dr:.4f}, dust_index {n:.4f}")
        for (ln, _), a, b in zip(LINES.items(), t2, t1, strict=True):
            say(f"   {ln:11s} tau2 {a:.3f}  tau1 {b:.3f}  young total {a + b:.3f}")
        t2b, t1b = taus(np.array([4862.7, 6564.6]), d2, dr, n)
        tt = t2b + t1b
        say(f"   young Balmer decrement factor exp[tau(Hb) - tau(Ha)] {np.exp(tt[0] - tt[1]):.4f}")

    d2h, drh, nh = hr[ix["dust2"]], hr[ix["dust_ratio"]], hr[ix["dust_index"]]
    _, t1_hr = taus(lam, d2h, drh, nh)
    _, t1_dr = taus(lam, d2h, er[ix["dust_ratio"]], nh)
    say("\nHR max-L with the ER dust_ratio: young line factor exp(tau1_HR - tau1_new)")
    for (ln, _), a, b in zip(LINES.items(), t1_hr, t1_dr, strict=True):
        say(f"   {ln:11s} x{np.exp(a - b):.4f}")

    say("\nline Jacobian d log10 L / d dust_ratio at ER max-L vs the young-only expectation")
    ew, el, ej = jz["eline_wave"], jz["eline_lum"], jz["eline_jac"][ix["dust_ratio"]]
    hb = el[np.argmin(np.abs(ew - 4862.7))]
    opt = (ew > 3500) & (ew < 9100)
    for i in np.argsort(-np.where(opt, el, 0))[:10]:
        expect = -er[ix["dust2"]] * (5500.0 / ew[i]) ** -DUST1_INDEX / np.log(10)
        say(
            f"   {ew[i]:8.2f}  L/L(Hb) {el[i] / hb:6.3f}  jac {ej[i]:+.4f}  young-only "
            f"{expect:+.4f}  young share {ej[i] / expect:.3f}"
        )

    fisher = jz["fisher"]
    keep = [j for j in range(len(labels)) if fisher[j, j] > 0]
    say("\ndropped from the Fisher matrix (no pixel information): ")
    say("   " + ", ".join(labels[j] for j in range(len(labels)) if j not in keep))
    lk = [labels[j] for j in keep]
    f = fisher[np.ix_(keep, keep)]
    cov = np.linalg.inv(f)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    say("\nFisher correlations (local, no prior), six largest")
    for p in ("dust2", "dust_ratio", "dust_index"):
        i = lk.index(p)
        order = [j for j in np.argsort(-np.abs(corr[i])) if j != i][:6]
        say(f"   {p}: " + ", ".join(f"{lk[j]} {corr[i, j]:+.2f}" for j in order))
        say(
            f"      sigma conditional {1 / np.sqrt(f[i, i]):.4f}, marginal {sd[i]:.4f}, "
            f"posterior {jz['post_sd'][keep[i]]:.4f}"
        )
    i = lk.index("dust_ratio")
    shift = er[ix["dust_ratio"]] - hr[ix["dust_ratio"]]
    k_sig = shift * np.sqrt(f[i, i])
    say(f"   dust_ratio shift HR -> ER value {shift:+.4f} = {k_sig:+.1f} conditional sigma")
    say(f"   linear conditional delta chi2 {k_sig**2:.0f}")

    # linear chi2 change from ER max-L, -2 g.d + d.F.d, with the gradient g at ER max-L
    pred = np.load(OUT / f"{STEM}_pred.npz")
    good = pred["good"]
    assert np.allclose(pred["unc"][good], jz["unc"])
    resid = (pred["flux"] - pred["sp_best"])[good] / jz["unc"]
    grad = (jz["jac"][keep] / jz["unc"]) @ resid
    hr_dr = hr.copy()
    hr_dr[ix["dust_ratio"]] = er[ix["dust_ratio"]]
    say(
        "\nlinear delta chi2 from ER max-L (hr_point.py measured: HR +231.4, HR + ER ratio +1376.9)"
    )
    lin = {}
    for name, th in (("HR", hr), ("HR + ER ratio", hr_dr)):
        dv = (th - er)[keep]
        lin[name] = -2 * grad @ dv + dv @ f @ dv
        say(f"   {name}: {lin[name]:+.0f}")
    say("   the full HR point is outside the linear range (gas and line profile shifts of 4 to 15")
    say("   posterior sd), so the per-parameter linear terms are not used")
    say(
        f"   cost of the dust_ratio move at the HR point: linear {lin['HR + ER ratio'] - lin['HR']:+.0f}, "
        f"measured {1376.9 - 231.4:+.0f}"
    )

    z = json.loads((OUT / f"{STEM}.json").read_text())["z"]
    rest = jz["wave"] / (1 + z)
    say(f"\nshare (percent) of Fisher information J^2/unc^2 per rest-frame bin, z {z:.5f}")
    say("   bins: " + " ".join(f"{a}-{b}" for a, b in zip(EDGES[:-1], EDGES[1:], strict=True)))
    for p in ("dust2", "dust_ratio", "dust_index", "logzsol", "logmass"):
        info = (jz["jac"][ix[p]] / jz["unc"]) ** 2
        row = [
            info[(rest >= a) & (rest < b)].sum() for a, b in zip(EDGES[:-1], EDGES[1:], strict=True)
        ]
        say(f"   {p:11s} " + " ".join(f"{100 * r / info.sum():5.1f}" for r in row))

    # the same Jacobian in the basis (dust1, dust2 at fixed dust1), with dust_ratio = dust1 / dust2
    d2, dr = er[ix["dust2"]], er[ix["dust_ratio"]]
    j_d1 = jz["jac"][ix["dust_ratio"]] / d2
    j_d2 = jz["jac"][ix["dust2"]] - dr / d2 * jz["jac"][ix["dust_ratio"]]
    say("\nsame shares in the basis dust1 (dust2 fixed) and dust2 (dust1 fixed)")
    for name, jrow in (("dust1", j_d1), ("dust2|dust1", j_d2)):
        info = (jrow / jz["unc"]) ** 2
        row = [
            info[(rest >= a) & (rest < b)].sum() for a, b in zip(EDGES[:-1], EDGES[1:], strict=True)
        ]
        say(f"   {name:11s} " + " ".join(f"{100 * r / info.sum():5.1f}" for r in row))
        cont = jz["region"] == "continuum"
        say(f"      continuum pixels {100 * info[cont].sum() / info.sum():.1f}%")
    t = np.eye(len(keep))
    i2, ir = lk.index("dust2"), lk.index("dust_ratio")
    # columns of d(old)/d(new): new parameters (dust2|dust1, dust1) sit in the slots of (dust2, ratio)
    t[ir, i2] = -dr / d2
    t[ir, ir] = 1 / d2
    f1 = t.T @ f @ t
    c1 = np.linalg.inv(f1)
    s1 = np.sqrt(np.diag(c1))
    r1 = c1 / np.outer(s1, s1)
    names = [x.replace("dust_ratio", "dust1").replace("dust2", "dust2|dust1") for x in lk]
    for k in (ir, i2):
        order = [j for j in np.argsort(-np.abs(r1[k])) if j != k][:6]
        say(
            f"   {names[k]}: sigma conditional {1 / np.sqrt(f1[k, k]):.4f}, marginal {s1[k]:.4f}; "
            + ", ".join(f"{names[j]} {r1[k, j]:+.2f}" for j in order)
        )

    # young share of the continuum: d m / d dust_ratio = -dust2 (5500 / lam) m_young
    pred_good = pred["sp_best"][good]
    f_young = -jz["jac"][ix["dust_ratio"]] / (d2 * (5500.0 / rest) * pred_good)
    say(
        "\nyoung (< 10 Myr) share of the attenuated model continuum, median per bin, continuum pixels"
    )
    cont = jz["region"] == "continuum"
    vals = []
    for a, b in zip(EDGES[:-1], EDGES[1:], strict=True):
        k = cont & (rest >= a) & (rest < b)
        vals.append(f"{a}-{b}:{np.median(f_young[k]):.3f}" if k.any() else f"{a}-{b}:  -  ")
    say("   " + " ".join(vals))

    (FIG / "dust_reading.txt").write_text("\n".join(out) + "\n")
    print(f"wrote {FIG / 'dust_reading.txt'}")


if __name__ == "__main__":
    main()
