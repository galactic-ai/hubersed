"""Read alf runs, check them, and compare their metallicity with Prospector MAP fits.

Run ``uv run python scripts/read_alf_sample.py --alf-results $ALF_HOME/results``.
The reading and checks are in ``hubersed.alf.alf_output``.
"""

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np

from hubersed.alf.alf_output import (
    ELEMENTS,
    ERR_FLOOR,
    _lib_corr,
    convergence,
    load_run,
    read_header,
)


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
        "with the logzsol column replaced by alf's [Z/H]. run_map_fits_outliers "
        "starts logzsol at that value.",
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
            d = dict(zip(r["labels"], np.asarray(r["theta"], float), strict=True))
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
            f"The logzsol column is alf's [Z/H] per galaxy (median of the chain). "
            f"run_map_fits_outliers starts logzsol there. "
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
