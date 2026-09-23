"""Does alf's metallicity move toward Prospector's when alf is forced solar-scaled?

Run ``uv run python experiments/2026-08-25_compare_alf_solar_scaled.py``. It needs alf runs
tagged ``quiescent`` (free abundances) and ``solarscaled`` (fit_type=2). Line numbers refer to
alf commit 4ef7bb8.

Prospector is scaled-solar. If the gap closes when alf is too, the abundance pattern drives
the disagreement. If it stays, other differences do.

This repeats a test from Beverage et al. 2025 (arXiv:2407.02556, section 5.4). For z = 1-3
quiescent galaxies, Prospector metallicities scattered by 0.32 dex around those from alfα
(their code based on alf) and sat 0.41 dex lower. Fitting Prospector without photometry, or
with a single-burst SFH, did not remove the gap. Also forcing alfα solar-scaled improved the
agreement but left large scatter. They found Prospector tracks alfα's [Fe/H] better than its
total metallicity. Unlike their last test, Prospector here keeps a free SFH.

With fit_type=2 alf fits only velz, sigma, logage and zH (alf_vars.f90:24,147) and skips
every element response (getmodel.f90:278), so zH is a scaled-solar total metallicity. This
needs the fork commit 4ef7bb8, otherwise every step fails the prior check.

fit_type=2 reports a plain chi2, and fit_type=0 reports -2 ln L with jitter terms
(func.f90:118-127). Only the solar-scaled chi2 is comparable to Prospector's.
"""

import argparse
import glob
import pickle
from math import comb
from pathlib import Path

import numpy as np

from hubersed.alf.read_alf_sample import load_run


def alf_set(results_dir, tag):
    out = {}
    for s in sorted(glob.glob(str(Path(results_dir) / f"*{tag}*.sum"))):
        stem = s[:-4]
        bits = [b for b in Path(stem).name.replace("_", " ").split() if b.isdigit()]
        tid = next((int(b) for b in bits if len(b) >= 16), None)
        if tid is None:
            continue
        C, _ = load_run(stem)
        out[tid] = {
            k: float(np.median(C[k])) for k in ("zH", "FeH", "Mg", "logage", "sigma", "m2lnP")
        }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--alf-results", default=None, help="default $ALF_HOME/results")
    p.add_argument("--free-tag", default="quiescent")
    p.add_argument("--solar-tag", default="solarscaled")
    p.add_argument("--prospector", default="results/cont_map_fits20_tauin")
    a = p.parse_args(argv)
    rd = a.alf_results or str(Path(__import__("os").environ["ALF_HOME"]) / "results")

    free, sol = alf_set(rd, a.free_tag), alf_set(rd, a.solar_tag)
    if not sol:
        raise SystemExit(
            f"no runs matching '{a.solar_tag}' in {rd}. Run $ALF_HOME/run_solar_scaled11.sh first."
        )
    pros = {}
    for f in glob.glob(str(Path(a.prospector) / "3*.pkl")):
        r = pickle.load(open(f, "rb"))
        if isinstance(r, dict) and r.get("status") == "ok":
            d = dict(zip(r["labels"], np.asarray(r["theta"], float)))
            s = r["sfh"]
            e = np.asarray(s["edges_gyr"], float)
            m = np.asarray(s.get("ssfr_inplace", s.get("ssfr")), float)
            pros[int(r["target_id"])] = (
                d["logzsol"],
                float(np.sum(m * 0.5 * (e[:-1] + e[1:])) / np.sum(m)),
            )

    tids = sorted(set(free) & set(sol) & set(pros))
    print(f"{len(tids)} galaxies with all three fits\n")
    hdr = (
        f"{'TARGETID':>19}{'alf free [Z/H]':>15}{'alf SS [Z/H]':>14}{'pros logzsol':>14}"
        f"{'|free-pros|':>12}{'|SS-pros|':>11}{'closer?':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    d_free, d_sol, dage = [], [], []
    for t in tids:
        zf, zs, zp = free[t]["zH"], sol[t]["zH"], pros[t][0]
        a1, a2 = abs(zf - zp), abs(zs - zp)
        d_free.append(a1)
        d_sol.append(a2)
        dage.append(np.log10(pros[t][1] / 10 ** sol[t]["logage"]))
        print(
            f"{t:>19}{zf:>15.3f}{zs:>14.3f}{zp:>14.3f}{a1:>12.3f}{a2:>11.3f}"
            f"{'YES' if a2 < a1 else 'no':>9}"
        )

    d_free, d_sol = np.array(d_free), np.array(d_sol)
    n = len(d_free)
    k = int((d_sol < d_free).sum())
    pv = 2 * sum(comb(n, i) * 0.5**n for i in range(max(k, n - k), n + 1))
    print(
        f"\n  median |alf - Prospector|:  free {np.median(d_free):.3f} dex"
        f"  ->  solar-scaled {np.median(d_sol):.3f} dex"
    )
    print(f"  closer in {k}/{n}   sign test p = {min(pv, 1.0):.4f}")
    print(
        f"  alf [Z/H] shift when forced solar-scaled: median "
        f"{np.median([sol[t]['zH'] - free[t]['zH'] for t in tids]):+.3f} dex"
    )
    print(
        f"\n  VERDICT: {'ABUNDANCE PATTERN' if np.median(d_sol) < np.median(d_free) - 0.03 else 'NUMERICS (gap survives)'}"
        " dominates the metallicity disagreement"
    )

    da = np.array(dage)
    print(
        f"\n  age check, Prospector vs solar-scaled alf: median dlog10 = {np.median(da):+.3f} dex"
        f"  ({10 ** np.median(da):.2f}x), scatter {da.std():.3f}"
    )
    print("  (free-abundance alf gave -0.127 dex / 0.75x at 4.3 sigma, 2026-08-20l)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
