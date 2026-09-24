"""Read alf runs and check their convergence.

The results table and the comparison with Prospector MAP fits are printed by
``scripts/read_alf_sample.py``. Line numbers refer to alf commit 4ef7bb8.

Notes
-----
Each ``.mcmc`` row holds -2 ln P, the 46 parameters in str2arr.f90 order, and 6
mass-to-light ratios (alf.f90:655-656). ``load_run`` checks this layout against the
``.sum`` file, so a column offset fails loudly.

[X/Fe] is taken per chain sample, as in alf's scripts/read_alf.py. Mg and Fe are
correlated, so adding their marginal errors in quadrature would overstate the error.

Convergence is judged on the whole ensemble by comparing the two halves of the chain.
Per-walker Gelman-Rubin is not useful because walkers accept only a few percent of moves.
"""

import numpy as np
from scipy.interpolate import interp1d

# .mcmc columns (alf.f90:655-656, str2arr.f90:25-75). Column 0 is -2 ln P. The
# mass-to-light ratios are in r, I and K (alf_vars.f90:183-184).
LABELS = [
    "m2lnP",
    "velz",
    "sigma",
    "logage",
    "zH",
    "FeH",
    "a",
    "C",
    "N",
    "Na",
    "Mg",
    "Si",
    "K",
    "Ca",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Co",
    "Ni",
    "Cu",
    "Sr",
    "Ba",
    "Eu",
    "Teff",
    "IMF1",
    "IMF2",
    "logfy",
    "sigma2",
    "velz2",
    "logm7g",
    "hotteff",
    "loghot",
    "fy_logage",
    "logemline_h",
    "logemline_oii",
    "logemline_oiii",
    "logemline_sii",
    "logemline_ni",
    "logemline_nii",
    "logtrans",
    "jitter",
    "logsky",
    "IMF3",
    "IMF4",
    "h3",
    "h4",
    "ML_r",
    "ML_i",
    "ML_k",
    "MW_r",
    "MW_i",
    "MW_k",
]
# .sum rows (alf.f90:745-746). cl98 is the 97.5 percent row.
SUM_ROWS = [
    "mean",
    "chi2min",
    "error",
    "cl2.5",
    "cl16",
    "cl50",
    "cl84",
    "cl98",
    "lo_prior",
    "hi_prior",
]
ELEMENTS = ["a", "C", "N", "Na", "Mg", "Si", "Ca", "Ti"]

# Library correction tables from alf's scripts/read_alf.py:254-277 (m11). They apply to
# a (the O proxy), Mg, and Ca, Ti, Si. C, N and Na get none (read_alf.py:304).
_LIB_FEH = [-1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2]
_LIB_OFE = [0.6, 0.5, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.0, 0.0]
_LIB_MGFE = [0.4, 0.4, 0.4, 0.4, 0.34, 0.22, 0.14, 0.11, 0.05, 0.04]
_LIB_CAFE = [0.32, 0.3, 0.28, 0.26, 0.26, 0.17, 0.12, 0.06, 0.0, 0.0]
ERR_FLOOR = 0.1  # dex, smallest half-width allowed for an [X/Fe] interval


def _lib_corr(elem, zh_chain):
    """Return the library correction added to [X/Fe] for each chain sample.

    Parameters
    ----------
    elem : str
        Element label from LABELS.
    zh_chain : ndarray
        zH samples.

    Returns
    -------
    ndarray or float
        Correction in dex, or 0 for elements without one.

    Notes
    -----
    Outside the table the correction is extrapolated linearly, as in alf's
    scripts/read_alf.py:281-292.
    """
    if elem == "a":
        tab = _LIB_OFE
    elif elem == "Mg":
        tab = _LIB_MGFE
    elif elem in ("Ca", "Ti", "Si"):
        tab = _LIB_CAFE
    else:
        return 0.0
    return interp1d(_LIB_FEH, tab, kind="linear", fill_value="extrapolate")(zh_chain)


def load_run(stem):
    """Read one alf run and check its column layout.

    Parameters
    ----------
    stem : str
        Path to the run without the ``.mcmc`` or ``.sum`` suffix.

    Returns
    -------
    chain : dict of str to ndarray
        Chain samples keyed by LABELS.
    summary : dict of str to dict
        ``.sum`` rows keyed by SUM_ROWS, each keyed by LABELS.

    Raises
    ------
    SystemExit
        If the column count or the chain medians disagree with the ``.sum`` file.
    """
    M = np.loadtxt(f"{stem}.mcmc")
    A = np.loadtxt(f"{stem}.sum")
    if M.shape[1] != len(LABELS):
        raise SystemExit(
            f"{stem}.mcmc has {M.shape[1]} columns, expected {len(LABELS)}. "
            "alf's parameter set changed; update LABELS from alf.f90:655."
        )
    C = {k: M[:, i] for i, k in enumerate(LABELS)}
    S = {r: dict(zip(LABELS, A[i], strict=True)) for i, r in enumerate(SUM_ROWS)}

    # Compare chain medians with the .sum 50th percentile. Column 0 is left out because
    # the .sum percentile rows store 0.0 there.
    for k in ("sigma", "logage", "zH", "FeH", "Mg"):
        got, want = float(np.median(C[k])), float(S["cl50"][k])
        if not np.isclose(got, want, rtol=2e-3, atol=1e-3):
            raise SystemExit(
                f"{stem}: column alignment FAILED on {k}: "
                f"chain median {got:.4f} vs .sum cl50 {want:.4f}"
            )
    return C, S


def read_header(stem):
    """Read the run settings from the header of an alf ``.sum`` file.

    Parameters
    ----------
    stem : str
        Path to the run without the ``.sum`` suffix.

    Returns
    -------
    dict of str to str
        Values of the ``key = value`` header lines, such as ``Nwalkers`` and ``fit_type``.
    """
    out = {}
    with open(f"{stem}.sum") as f:
        for line in f:
            if line.startswith("#") and "=" in line:
                key, value = line[1:].split("=", 1)
                out[key.strip()] = value.strip()
    return out


def convergence(C, *, nwalkers):
    """Check that the walker ensemble is stationary.

    Parameters
    ----------
    C : dict of str to ndarray
        Chain from ``load_run``.
    nwalkers : int
        Walkers in the run, from ``read_header``. alf writes one row per walker per step.

    Returns
    -------
    dict
        Width ratio of the two chain halves, lnP drift, fraction of moved steps, and ``ok``.
    """
    n = len(C["m2lnP"])
    nc = n // nwalkers
    out = {"nsteps": nc, "nwalkers": nwalkers}
    W = np.column_stack([C[k] for k in LABELS]).reshape(nc, nwalkers, len(LABELS))
    half = nc // 2
    ratios = []
    for k in ("logage", "zH", "FeH", "Mg", "sigma"):
        i = LABELS.index(k)
        a, b = W[:half, :, i], W[half:, :, i]
        ratios.append(b.std() / a.std() if a.std() > 0 else np.nan)
    out["width_ratio"] = float(np.median(ratios))
    lp = -0.5 * C["m2lnP"]
    out["lnp_drift"] = float(np.median(lp[half * nwalkers :]) - np.median(lp[: half * nwalkers]))
    out["lnp_scale"] = float(np.median(np.abs(lp)))
    out["moved"] = float((np.diff(W[:, :, LABELS.index("Mg")], axis=0) != 0).mean())
    # stationary ensemble: width ratio near 1, drift negligible against the lnP scale
    out["ok"] = bool(
        0.9 < out["width_ratio"] < 1.1
        and abs(out["lnp_drift"]) < 0.001 * max(out["lnp_scale"], 1.0)
    )
    return out
