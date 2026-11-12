"""Numerical checks of the integrated likelihood for a bounded equivalent-width line prior.

QUESTION: does the closed form of UberSED knowledge/free_line_prior_derivation_2026-11-06.md
(section 5: flat-prior result, minus sum ln(W c), plus ln P_B) equal brute-force integration of the
likelihood over the prior box, including a line whose amplitude sits within about one standard
deviation of its upper bound, and does the Jacobian term hold when integrating over EW directly?
HYPOTHESIS: agreement to 1e-4 in ln L with the exact P_B; the diagonal P_B fails for weak,
correlated lines.
INPUTS: none (toy model: 200 pixels, sigma 0.3, two unit-width lines at -0.6 and +0.6, amplitude
correlation -0.70, continuum c = (2, 2.5), EW limits W = (3, 3), so bounds b = (6, 7.5)).
SEED: numpy default_rng 1 for the box cases, 2 for the EW cases.
COMMAND: ``uv run --no-sync python experiments/2026-11-07_line_prior_check/analysis.py``.
RESULT: 2026-11-07, checks.txt. The closed form with the exact P_B equals brute force to five
decimals in all cases, including a_hat 5.980 +- 0.100 below the bound 6 (ln P_B = -0.543); the
diagonal P_B is off by 0.337 when both correlated lines are near zero; the EW-space integrals equal
the closed form for c = (2, 2.5) and (0.7, 4). Assumptions: toy model, two correlated lines.
FIGURES: results/2026-11-07_line_prior_check/checks.txt.
"""

# %%
import numpy as np
from scipy import integrate, stats

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-07_line_prior_check"
X = np.linspace(-10.0, 10.0, 200)
SIGMA = np.full(X.size, 0.3)
G = np.stack([np.exp(-0.5 * (X + 0.6) ** 2), np.exp(-0.5 * (X - 0.6) ** 2)], axis=1)
W = np.array([3.0, 3.0])
C_BOX = np.array([2.0, 2.5])
CASES = {
    "one strong, one weak line": [0.3, 4.0],
    "both lines near zero": [0.05, 0.02],
    "line 1 at its upper bound": [6.0, 1.0],
}


def setup(alpha_true, seed):
    """Return data, amplitude covariance, MLE amplitudes and the log likelihood function."""
    rng = np.random.default_rng(seed)
    d = G @ np.asarray(alpha_true) + rng.normal(0.0, SIGMA)
    inv = 1.0 / SIGMA**2
    cov = np.linalg.inv(G.T @ (G * inv[:, None]))
    a_hat = cov @ (G.T @ (inv * d))

    def lnlike(a):
        r = d - G @ a
        return -0.5 * np.sum(inv * r**2) - 0.5 * np.sum(np.log(2 * np.pi * SIGMA**2))

    return cov, a_hat, lnlike


def closed_form(cov, a_hat, lnlike, c, diagonal=False):
    """Return ln L(theta) of section 5 and ln P_B, with the exact or the diagonal P_B."""
    b = W * c
    if diagonal:
        s = np.sqrt(np.diag(cov))
        pb = np.prod(stats.norm.cdf((b - a_hat) / s) - stats.norm.cdf(-a_hat / s))
    else:
        pb = stats.multivariate_normal(a_hat, cov).cdf(b, lower_limit=np.zeros(2))
    flat = lnlike(a_hat) + 0.5 * np.log(np.linalg.det(2 * np.pi * cov))
    return flat - np.sum(np.log(W)) - np.sum(np.log(c)) + np.log(pb), np.log(pb), flat


# %% box cases: brute force over the amplitude box against the closed form
lines = [
    "ln L(theta) of the bounded EW prior: brute-force integration vs the closed form (section 5).",
    "",
    f"{'case':28} {'s1/s2':>11} {'a_hat':>15} {'brute':>11} {'exact P_B':>11} {'diag P_B':>11} "
    f"{'ln P_B':>9} {'flat':>11}",
]
for name, truth in CASES.items():
    cov, a_hat, lnlike = setup(truth, seed=1)
    b = W * C_BOX
    val, _ = integrate.dblquad(
        lambda a2, a1, f=lnlike, m=a_hat: np.exp(f(np.array([a1, a2])) - f(m)),
        0.0, b[0], 0.0, b[1], epsabs=1e-13, epsrel=1e-11,
    )  # fmt: skip
    brute = np.log(val) + lnlike(a_hat) - np.sum(np.log(b))
    exact, lnpb, flat = closed_form(cov, a_hat, lnlike, C_BOX)
    diag, _, _ = closed_form(cov, a_hat, lnlike, C_BOX, diagonal=True)
    s = np.sqrt(np.diag(cov))
    lines.append(
        f"{name:28} {s[0]:.3f}/{s[1]:.3f} {a_hat[0]:7.3f},{a_hat[1]:7.3f} {brute:11.5f} "
        f"{exact:11.5f} {diag:11.5f} {lnpb:9.5f} {flat:11.5f}"
    )
corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
lines += ["", f"amplitude correlation {corr:+.3f}; bounds b = {W * C_BOX}", ""]

# %% Jacobian: integrate directly over EW, alpha = EW * c, for two continua
lines.append("Jacobian check: direct integration over EW ~ U(0, 3) vs the closed form.")
cov, a_hat, lnlike = setup([0.3, 4.0], seed=2)
for c in (np.array([2.0, 2.5]), np.array([0.7, 4.0])):
    val, _ = integrate.dblquad(
        lambda e2, e1, c=c: np.exp(lnlike(np.array([e1, e2]) * c) - lnlike(a_hat)),
        0.0, W[0], 0.0, W[1], epsabs=1e-13, epsrel=1e-11,
    )  # fmt: skip
    direct = np.log(val) + lnlike(a_hat) - np.sum(np.log(W))
    exact, _, _ = closed_form(cov, a_hat, lnlike, c)
    lines.append(f"  c = {c}: direct {direct:.5f}, closed form {exact:.5f}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "checks.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
