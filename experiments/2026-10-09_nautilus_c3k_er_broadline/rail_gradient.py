"""Split the likelihood gradient on the railed parameters of the ER fit by rest-frame window.

QUESTION: which line windows push gas_logco to its ceiling (and dust_index to its ceiling and
eline_sigma_forb to its floor) in the #14 ER fit of 39627770174637084?

At the maximum likelihood point the gradient g_i = sum J_i (data - model) / unc^2 of a railed
parameter i is the slope of the profile likelihood, if the free parameters are at their optimum
(their gradients near zero, checked here). Two splits over windows:

- raw: the sum of J_i (data - model) / unc^2 over each window;
- residualized: J_i minus its projection on the other parameters' Jacobians,
  J_i - J_o F_oo^-1 F_oi, so the split shows only the part of the pull that no other parameter
  can take over. The SFH ratios and hyperparameters are held fixed in the projection (the SFH
  directions are nearly degenerate without the prior).

Gradients are shown times the Fisher width of the parameter (sigma_cond = F_ii^-1/2), so 1 means a
pull of one local sigma. Local and linear; uses jacobian.npz from jacobian.py and the maximum
likelihood spectrum from predict.py.

INPUTS: jacobian.npz and the ``_pred.npz``. SEED: none. No FSPS needed.

Run from the repository root with
``uv run python experiments/2026-10-09_nautilus_c3k_er_broadline/rail_gradient.py``.

OUTPUT: results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/rail_gradient.txt.
"""

import numpy as np

from hubersed.paths import PATHS

TID = 39627770174637084
OUT = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline"
STEM = f"{TID}_er_broad_shared_split45_rest9000_seed0"
RAILED = ["gas_logco", "dust_index", "eline_sigma_forb"]
HYPERS = {"sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn"}
# rest-frame bins for splitting the continuum part; 3646 A is the Balmer jump
CONT_BINS = [
    (3320, 3646),
    (3646, 4000),
    (4000, 4500),
    (4500, 5500),
    (5500, 6500),
    (6500, 7500),
    (7500, 9000),
]
# weak Cue lines that jacobian.py's windows leave in the continuum; they get their own windows here
EXTRA_WINDOWS = {
    "[NII]5755": (5750, 5762),
    "[SIII]6312": (6310, 6318),
    "[OI]6363": (6360, 6371),
    "HeI6678": (6674, 6686),
    "HeI7065": (7061, 7073),
    "[ArIII]7136": (7132, 7144),
    "[OII]7320/30": (7316, 7337),
}


def main():
    j = np.load(OUT / str(TID) / "jacobian.npz")
    p = np.load(OUT / f"{STEM}_pred.npz")
    labels = [str(x) for x in j["labels"]]
    good = p["good"]
    r = (p["flux"] - p["sp_best"])[good] / j["unc"]  # residual in units of unc
    jw = j["jac"] / j["unc"]  # Jacobian in units of unc
    rest = j["wave"] / (1 + float(p["z"]))
    region = j["region"].astype(object)
    for w, (a, b) in EXTRA_WINDOWS.items():
        region[(region == "continuum") & (rest >= a) & (rest < b)] = w
    names = [str(x) for x in j["windows"]][:-1] + list(EXTRA_WINDOWS) + ["continuum"]
    fisher = j["fisher"]

    grad = jw @ r
    free = [i for i, lab in enumerate(labels) if lab not in HYPERS]
    sig = 1 / np.sqrt(np.diag(fisher)[free])
    lines = ["Gradient times Fisher width (sigma_cond) for every free parameter at best:"]
    lines += [
        f"{labels[i]:>18s} {g * s:+8.3f}" for i, g, s in zip(free, grad[free], sig, strict=True)
    ]

    others_all = [i for i in free if not labels[i].startswith("logsfr")]
    for name in RAILED:
        i = labels.index(name)
        o = [k for k in others_all if k != i]
        coef = np.linalg.solve(fisher[np.ix_(o, o)], fisher[o, i])
        jres = jw[i] - coef @ jw[o]
        s_i = 1 / np.sqrt(fisher[i, i])
        s_res = 1 / np.sqrt(np.sum(jres**2))
        raw = np.array([np.sum((jw[i] * r)[region == w]) for w in names]) * s_i
        res = np.array([np.sum((jres * r)[region == w]) for w in names]) * s_res
        lines.append(
            f"\n{name}: total raw {raw.sum():+.3f} (x sigma_cond {s_i:.4g}), total residualized "
            f"{res.sum():+.3f} (x sigma {s_res:.4g}, the width once the other parameters adjust)"
        )
        lines.append(f"{'window':>12s} {'raw':>8s} {'resid':>8s}")
        for k in np.argsort(-np.abs(res)):
            lines.append(f"{names[k]:>12s} {raw[k]:+8.3f} {res[k]:+8.3f}")
        lines.append("  continuum pixels split by rest wavelength:")
        for a, b in CONT_BINS:
            c = (region == "continuum") & (rest >= a) & (rest < b)
            lines.append(
                f"  {a:5.0f}-{b:5.0f} A {c.sum():5d} px {s_i * np.sum((jw[i] * r)[c]):+8.3f} "
                f"{s_res * np.sum((jres * r)[c]):+8.3f}"
            )

    text = "\n".join(lines) + "\n"
    (OUT / str(TID) / "rail_gradient.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
