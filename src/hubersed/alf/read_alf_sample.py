"""Read alf runs, check them, and compare their metallicity with Prospector MAP fits.

Run ``uv run python -m hubersed.alf.read_alf_sample --alf-results $ALF_HOME/results``.
Line numbers refer to alf commit 4ef7bb8.

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

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np

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
    ``np.interp`` holds the end values outside the table, while alf's read_alf.py
    extrapolates linearly.
    """
    if elem == "a":
        tab = _LIB_OFE
    elif elem == "Mg":
        tab = _LIB_MGFE
    elif elem in ("Ca", "Ti", "Si"):
        tab = _LIB_CAFE
    else:
        return 0.0
    return np.interp(zh_chain, _LIB_FEH, tab)


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
    S = {r: dict(zip(LABELS, A[i])) for i, r in enumerate(SUM_ROWS)}

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


def main(argv=None):
    """Print the alf results table and optionally write npz files.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. By default they come from sys.argv.

    Returns
    -------
    int
        Exit status.
    """
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--alf-results", required=True)
    p.add_argument("--tag", default="", help="only read runs whose stem ends with this")
    p.add_argument(
        "--prospector",
        default="results/cont_map_fits20",
        help="MAP-fit dir to compare logzsol against",
    )
    p.add_argument("-o", "--out", default=None)
    p.add_argument(
        "--sample-in",
        default=None,
        help="an existing sample npz to clone, e.g. results/cont_outlier_sample20.npz",
    )
    p.add_argument(
        "--sample-out",
        default=None,
        help="write a sample npz holding ONLY the galaxies with an alf run, "
        "with the logzsol column replaced by alf's [Z/H]. Feed it to "
        "run_map_fits_outliers.py --fix-from-sample logzsol.",
    )
    a = p.parse_args(argv)
    if bool(a.sample_in) != bool(a.sample_out):
        raise SystemExit("--sample-in and --sample-out go together")

    stems = sorted(s[:-4] for s in glob.glob(str(Path(a.alf_results) / f"*{a.tag}*.sum")))
    if not stems:
        raise SystemExit(f"no .sum files under {a.alf_results}")

    # Prospector logzsol keyed by TARGETID
    pros = {}
    for f in glob.glob(str(Path(a.prospector) / "3*.pkl")):
        r = pickle.load(open(f, "rb"))
        if isinstance(r, dict) and r.get("status") == "ok":
            d = dict(zip(r["labels"], np.asarray(r["theta"], float)))
            pros[int(r["target_id"])] = (d["logzsol"], r["stats"]["chi2_red"])

    rows = []
    print(
        f"{'TARGETID':>19}{'conv':>6}{'Wratio':>8}{'moved':>7}{'zH':>8}{'FeH':>8}"
        + "".join(f"{'[' + e + '/Fe]':>13}" for e in ELEMENTS)
        + f"{'pros_lgZ':>10}{'dZ':>8}"
    )
    for s in stems:
        # TARGETIDs have 16 or more digits. Shorter numbers in a file name are
        # nicknames and are never matched.
        tid = next(
            (
                int(t)
                for t in Path(s).name.replace("_", " ").replace(".", " ").split()
                if t.isdigit() and len(t) >= 16
            ),
            None,
        )
        C, S = load_run(s)
        cv = convergence(C, nwalkers=int(read_header(s)["Nwalkers"]))
        xfe = {
            e: np.percentile(C[e] - C["FeH"] + _lib_corr(e, C["zH"]), [16, 50, 84])
            for e in ELEMENTS
        }
        # error floor: widen any percentile pair narrower than ERR_FLOOR
        for e in ELEMENTS:
            lo, med, hi = xfe[e]
            half = max((hi - lo) / 2, ERR_FLOOR)
            xfe[e] = np.array([med - half, med, med + half])
        # rail check vs alf priors from the .sum lo/hi_prior rows
        railed = [
            e
            for e in ELEMENTS + ["zH", "FeH"]
            if abs(np.median(C[e]) - S["lo_prior"][e]) < 0.02
            or abs(np.median(C[e]) - S["hi_prior"][e]) < 0.02
        ]
        if railed:
            print(f"  RAIL WARNING {Path(s).name}: {railed}")
        zh, feh = np.median(C["zH"]), np.median(C["FeH"])
        pz, pchi = pros.get(tid, (np.nan, np.nan)) if tid is not None else (np.nan, np.nan)
        rows.append(
            dict(
                tid=tid,
                name=Path(s).name,
                zH=zh,
                FeH=feh,
                pros_logzsol=pz,
                pros_chi2=pchi,
                conv=cv,
                jitter=float(np.median(C["jitter"])),
                logage=float(np.median(C["logage"])),
                **{f"{e}_Fe": xfe[e] for e in ELEMENTS},
            )
        )
        who = str(tid) if tid is not None else Path(s).name[:19]
        print(
            f"{who:>19}{'ok' if cv['ok'] else 'BAD':>6}{cv['width_ratio']:>8.3f}"
            f"{cv['moved']:>7.3f}{zh:>8.3f}{feh:>8.3f}"
            + "".join(f"{xfe[e][1]:>+7.3f}+-{(xfe[e][2] - xfe[e][0]) / 2:<4.2f}" for e in ELEMENTS)
            + (f"{pz:>10.3f}{zh - pz:>8.3f}" if np.isfinite(pz) else f"{'--':>10}{'--':>8}")
        )
        if tid is None:
            print(
                f"{'':>19}  ^ no 16+ digit TARGETID in the filename; not matched to a "
                "Prospector fit. Name alf inputs desi_<TARGETID>.dat."
            )

    ok = [r for r in rows if r["conv"]["ok"]]
    print(f"\n{len(ok)}/{len(rows)} converged by the ensemble-stationarity test")
    if not ok:
        return 0
    from math import comb

    print(f"\n{'element':>10}{'median [X/Fe]':>15}{'n>0':>7}{'sign-test p':>13}")
    for e in ELEMENTS:
        v = np.array([r[f"{e}_Fe"][1] for r in ok])
        k, n = int((v > 0).sum()), len(v)
        pv = 2 * sum(comb(n, i) * 0.5**n for i in range(max(k, n - k), n + 1))
        print(f"{e:>10}{np.median(v):>+15.4f}{k:>4}/{n:<3}{min(pv, 1.0):>13.4f}")
    d = np.array([r["zH"] - r["pros_logzsol"] for r in ok if np.isfinite(r["pros_logzsol"])])
    if d.size:
        k, n = int((d < 0).sum()), len(d)
        pv = 2 * sum(comb(n, i) * 0.5**n for i in range(max(k, n - k), n + 1))
        print(
            f"\nalf [Z/H] minus Prospector logzsol: median {np.median(d):+.4f} dex, "
            f"lower in {k}/{n}, sign-test p = {min(pv, 1.0):.4f}"
        )
        print("  alf can fit a non-solar abundance pattern; Prospector cannot. If Prospector")
        print("  is compensating for alpha-enhancement with overall Z, this is negative.")
    if a.sample_out:
        S = np.load(a.sample_in, allow_pickle=True)
        have = {r["tid"]: r["zH"] for r in rows if r["tid"] is not None}
        # Subset the sample by TARGETID, not by row position.
        src = np.asarray(S["target_ids"], np.int64)
        idx = [i for i, t in enumerate(src) if int(t) in have]
        missing = sorted(have.keys() - {int(t) for t in src})
        if missing:
            raise SystemExit(f"alf runs with no row in {a.sample_in}: {missing}")
        out = {
            k: (np.asarray(S[k])[idx] if np.asarray(S[k]).shape[:1] == src.shape else S[k])
            for k in S.files
        }
        out["logzsol"] = np.array([have[int(src[i])] for i in idx], float)
        assert np.array_equal(out["target_ids"], src[idx]), "TARGETID order lost"
        out["provenance"] = np.array(
            f"{len(idx)} galaxies with an alf run, subset of {a.sample_in} by TARGETID. "
            f"The logzsol column is alf's [Z/H] per galaxy (median of the chain), NOT a "
            f"seed -- intended for run_map_fits_outliers.py --fix-from-sample logzsol. "
            f"alf runs read from {a.alf_results} with tag '{a.tag}'. "
            f"Every other column is copied unchanged from the source sample."
        )
        np.savez(a.sample_out, **out)
        print(
            f"\nwrote {a.sample_out}: {len(idx)} galaxies, "
            f"logzsol = alf [Z/H] in [{out['logzsol'].min():+.3f}, "
            f"{out['logzsol'].max():+.3f}]"
        )

    if a.out:
        np.savez(
            a.out,
            **{
                k: np.array([r[k] for r in rows], dtype=object)
                for k in ("tid", "zH", "FeH", "pros_logzsol", "jitter", "logage")
            },
            elements=ELEMENTS,
            xfe=np.array([[r[f"{e}_Fe"] for e in ELEMENTS] for r in rows]),
        )
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
