"""Read a directory of alf runs, check convergence, and compare to the Prospector fits.

    uv run python bin/prospector/read_alf_sample.py \
        --alf-results ~/Astronomy_Research/alf/results \
        --prospector results/cont_map_fits20

Why this exists rather than a notebook cell
-------------------------------------------
Reading ``.mcmc`` by guessing the column offset produced a sign-flipped abundance pattern
once already (``[Mg/Fe] = -0.21`` instead of ``+0.135``). The layout is fixed by
``alf.f90:655``::

    WRITE(12,...) -2.0*lp_emcee_in(j), pos_emcee_in(:,j), m2l, m2lmw

i.e. column 0 is -2 ln P, then the 46 free parameters in ``STR2ARR`` order, then
``m2l`` (3) and ``m2lmw`` (3) = 53 columns. This module ASSERTS that alignment against
values the ``.sum`` file reports independently before computing anything. Do not remove
that assertion.

Two further traps, both hit on the first run:

* ``[X/Fe]`` must come from the CHAIN, not from the marginal errors. ``corr(Mg, FeH)``
  is +0.62, so quadrature on the marginals gives 0.072 where the chain gives 0.044 --
  1.9 sigma instead of 3.1 sigma. The difference is better constrained than either term.
* Gelman-Rubin across emcee walkers is MISLEADING here. With ``facc ~ 0.1`` a walker
  moves ~10 times in 100 production steps, so within-walker variance collapses and
  R-hat blows up whether or not the ensemble is right. The valid check is whether the
  ENSEMBLE is stationary: width ratio between chain halves near 1, and flat ln P.
"""

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np

# alf.f90:655 column order. First entry is -2 ln P, not a parameter.
LABELS = [
    "m2lnP", "velz", "sigma", "logage", "zH", "FeH", "a", "C", "N", "Na", "Mg", "Si",
    "K", "Ca", "Ti", "V", "Cr", "Mn", "Co", "Ni", "Cu", "Sr", "Ba", "Eu", "Teff",
    "IMF1", "IMF2", "logfy", "sigma2", "velz2", "logm7g", "hotteff", "loghot",
    "fy_logage", "logemline_h", "logemline_oii", "logemline_oiii", "logemline_sii",
    "logemline_ni", "logemline_nii", "logtrans", "jitter", "logsky", "IMF3", "IMF4",
    "h3", "h4", "ML_v", "ML_i", "ML_k", "MW_v", "MW_i", "MW_k",
]
# .sum row order, from its own header comment
SUM_ROWS = ["mean", "chi2min", "error", "cl2.5", "cl16", "cl50", "cl84", "cl98",
            "lo_prior", "hi_prior"]
ELEMENTS = ["a", "C", "N", "Na", "Mg", "Si", "Ca", "Ti"]


def load_run(stem):
    """Return (chain dict, sum dict) with the column alignment ASSERTED, not assumed."""
    M = np.loadtxt(f"{stem}.mcmc")
    A = np.loadtxt(f"{stem}.sum")
    if M.shape[1] != len(LABELS):
        raise SystemExit(f"{stem}.mcmc has {M.shape[1]} columns, expected {len(LABELS)}. "
                         "alf's parameter set changed; update LABELS from alf.f90:655.")
    C = {k: M[:, i] for i, k in enumerate(LABELS)}
    S = {r: dict(zip(LABELS, A[i])) for i, r in enumerate(SUM_ROWS)}

    # The assertion. Compare the chain median to the .sum 50th percentile for five
    # parameters the .sum reports independently. NB: do NOT include column 0 -- the
    # .sum's CL rows carry chi2 = 0.0, which is what defeated the first attempt.
    for k in ("sigma", "logage", "zH", "FeH", "Mg"):
        got, want = float(np.median(C[k])), float(S["cl50"][k])
        if not np.isclose(got, want, rtol=2e-3, atol=1e-3):
            raise SystemExit(f"{stem}: column alignment FAILED on {k}: "
                             f"chain median {got:.4f} vs .sum cl50 {want:.4f}")
    return C, S


def convergence(C, nwalkers=256):
    """Ensemble stationarity, which is the meaningful test for an emcee ensemble."""
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
    out["lnp_drift"] = float(np.median(lp[half * nwalkers:]) - np.median(lp[:half * nwalkers]))
    out["lnp_scale"] = float(np.median(np.abs(lp)))
    out["moved"] = float((np.diff(W[:, :, LABELS.index("Mg")], axis=0) != 0).mean())
    # stationary ensemble: width ratio near 1, drift negligible against the lnP scale
    out["ok"] = bool(0.9 < out["width_ratio"] < 1.1
                     and abs(out["lnp_drift"]) < 0.001 * max(out["lnp_scale"], 1.0))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--alf-results", required=True)
    p.add_argument("--tag", default="", help="only read runs whose stem ends with this")
    p.add_argument("--prospector", default="results/cont_map_fits20",
                   help="MAP-fit dir to compare logzsol against")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--sample-in", default=None,
                   help="an existing sample npz to clone, e.g. "
                        "results/cont_outlier_sample20.npz")
    p.add_argument("--sample-out", default=None,
                   help="write a sample npz holding ONLY the galaxies with an alf run, "
                        "with the logzsol column replaced by alf's [Z/H]. Feed it to "
                        "run_map_fits_outliers.py --fix-from-sample logzsol.")
    a = p.parse_args(argv)
    if bool(a.sample_in) != bool(a.sample_out):
        raise SystemExit("--sample-in and --sample-out go together")

    stems = sorted(s[:-4] for s in glob.glob(str(Path(a.alf_results) / f"*{a.tag}*.sum")))
    if not stems:
        raise SystemExit(f"no .sum files under {a.alf_results}")

    # Prospector logzsol, matched BY TARGETID parsed out of the alf filename.
    pros = {}
    for f in glob.glob(str(Path(a.prospector) / "3*.pkl")):
        r = pickle.load(open(f, "rb"))
        if isinstance(r, dict) and r.get("status") == "ok":
            d = dict(zip(r["labels"], np.asarray(r["theta"], float)))
            pros[int(r["target_id"])] = (d["logzsol"], r["stats"]["chi2_red"])

    rows = []
    print(f"{'TARGETID':>19}{'conv':>6}{'Wratio':>8}{'moved':>7}{'[Z/H]':>8}{'[Fe/H]':>8}"
          + "".join(f"{'[' + e + '/Fe]':>13}" for e in ELEMENTS)
          + f"{'pros_lgZ':>10}{'dZ':>8}")
    for s in stems:
        # Match by TARGETID, never by nickname or position (CLAUDE.md). DESI TARGETIDs
        # are 17-18 digits; anything shorter in the filename is a log shorthand like
        # "42580" and must NOT be matched against the catalogue.
        tid = next((int(t) for t in Path(s).name.replace("_", " ").replace(".", " ").split()
                    if t.isdigit() and len(t) >= 16), None)
        C, S = load_run(s)
        cv = convergence(C)
        xfe = {e: np.percentile(C[e] - C["FeH"], [16, 50, 84]) for e in ELEMENTS}
        zh, feh = np.median(C["zH"]), np.median(C["FeH"])
        pz, pchi = pros.get(tid, (np.nan, np.nan)) if tid is not None else (np.nan, np.nan)
        rows.append(dict(tid=tid, name=Path(s).name, zH=zh, FeH=feh, pros_logzsol=pz,
                         pros_chi2=pchi, conv=cv, jitter=float(np.median(C["jitter"])),
                         logage=float(np.median(C["logage"])),
                         **{f"{e}_Fe": xfe[e] for e in ELEMENTS}))
        who = str(tid) if tid is not None else Path(s).name[:19]
        print(f"{who:>19}{'ok' if cv['ok'] else 'BAD':>6}{cv['width_ratio']:>8.3f}"
              f"{cv['moved']:>7.3f}{zh:>8.3f}{feh:>8.3f}"
              + "".join(f"{xfe[e][1]:>+7.3f}+-{(xfe[e][2]-xfe[e][0])/2:<4.2f}"
                        for e in ELEMENTS)
              + (f"{pz:>10.3f}{zh - pz:>8.3f}" if np.isfinite(pz)
                 else f"{'--':>10}{'--':>8}"))
        if tid is None:
            print(f"{'':>19}  ^ no 16+ digit TARGETID in the filename; not matched to a "
                  "Prospector fit. Name alf inputs desi_<TARGETID>.dat.")

    ok = [r for r in rows if r["conv"]["ok"]]
    print(f"\n{len(ok)}/{len(rows)} converged by the ensemble-stationarity test")
    if not ok:
        return 0
    from math import comb
    print(f"\n{'element':>10}{'median [X/Fe]':>15}{'n>0':>7}{'sign-test p':>13}")
    for e in ELEMENTS:
        v = np.array([r[f"{e}_Fe"][1] for r in ok])
        k, n = int((v > 0).sum()), len(v)
        pv = 2 * sum(comb(n, i) * 0.5 ** n for i in range(max(k, n - k), n + 1))
        print(f"{e:>10}{np.median(v):>+15.4f}{k:>4}/{n:<3}{min(pv,1.0):>13.4f}")
    d = np.array([r["zH"] - r["pros_logzsol"] for r in ok if np.isfinite(r["pros_logzsol"])])
    if d.size:
        k, n = int((d < 0).sum()), len(d)
        pv = 2 * sum(comb(n, i) * 0.5 ** n for i in range(max(k, n - k), n + 1))
        print(f"\nalf [Z/H] minus Prospector logzsol: median {np.median(d):+.4f} dex, "
              f"lower in {k}/{n}, sign-test p = {min(pv,1.0):.4f}")
        print("  alf can fit a non-solar abundance pattern; Prospector cannot. If Prospector")
        print("  is compensating for alpha-enhancement with overall Z, this is negative.")
    if a.sample_out:
        S = np.load(a.sample_in, allow_pickle=True)
        have = {r["tid"]: r["zH"] for r in rows if r["tid"] is not None}
        # Subset and reorder the ORIGINAL sample BY TARGETID, never by row position.
        src = np.asarray(S["target_ids"], np.int64)
        idx = [i for i, t in enumerate(src) if int(t) in have]
        missing = sorted(have.keys() - {int(t) for t in src})
        if missing:
            raise SystemExit(f"alf runs with no row in {a.sample_in}: {missing}")
        out = {k: (np.asarray(S[k])[idx] if np.asarray(S[k]).shape[:1] == src.shape
                   else S[k]) for k in S.files}
        out["logzsol"] = np.array([have[int(src[i])] for i in idx], float)
        assert np.array_equal(out["target_ids"], src[idx]), "TARGETID order lost"
        out["provenance"] = np.array(
            f"{len(idx)} galaxies with an alf run, subset of {a.sample_in} by TARGETID. "
            f"The logzsol column is alf's [Z/H] per galaxy (median of the chain), NOT a "
            f"seed -- intended for run_map_fits_outliers.py --fix-from-sample logzsol. "
            f"alf runs read from {a.alf_results} with tag '{a.tag}'. "
            f"Every other column is copied unchanged from the source sample.")
        np.savez(a.sample_out, **out)
        print(f"\nwrote {a.sample_out}: {len(idx)} galaxies, "
              f"logzsol = alf [Z/H] in [{out['logzsol'].min():+.3f}, "
              f"{out['logzsol'].max():+.3f}]")

    if a.out:
        np.savez(a.out, **{k: np.array([r[k] for r in rows], dtype=object)
                           for k in ("tid", "zH", "FeH", "pros_logzsol", "jitter", "logage")},
                 elements=ELEMENTS,
                 xfe=np.array([[r[f"{e}_Fe"] for e in ELEMENTS] for r in rows]))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
