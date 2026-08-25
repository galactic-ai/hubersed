"""Beverage+2025 discriminator: does alf's metallicity move toward Prospector's when
alf is forced solar-scaled?

    uv run python bin/prospector/compare_alf_solar_scaled.py

The question
------------
alf (free abundances) and Prospector disagree on metallicity. Two competing causes:

  (a) the ABUNDANCE PATTERN. Prospector is scaled-solar and cannot represent
      [Mg/Fe] > 0, so its single logzsol is forced to some compromise. Take alf's
      abundance freedom away and its [Z/H] should collapse toward Prospector's.
  (b) the NUMERICS. ztinterp kinks, the optimizer, the library resolution. Removing
      alf's abundance freedom should then do nothing to the gap.

Beverage et al. 2025 report that forcing alfalpha solar-scaled was the ONLY control that
improved their 0.32 dex Prospector-vs-alf scatter, out of removing photometry, removing
SFH freedom, and matching the SSP assumption. So (a) has a published precedent and this
is a direct replication on our sample.

What "solar-scaled" means here
------------------------------
alf ``fit_type=2`` keeps only npowell=4 parameters -- velz, sigma, logage, zH
(alf.f90:645-650, alf_vars.f90:147, str2arr.f90:25-28). feh and every [X/Fe] are pinned
to zero, so zH is a single total metallicity with scaled-solar composition, which is
exactly what Prospector's logzsol is.

Read the chi2 with care
-----------------------
func.f90:117 applies the jitter and log(2 pi sigma^2) terms ONLY for fit_type=0. So the
solar-scaled runs report a REAL chi2 while the free-abundance runs report -2lnL. The two
are not comparable to each other; the solar-scaled one IS comparable to Prospector's.
"""

import argparse
import glob
import pickle
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bin.alf.read_alf_sample import load_run  # noqa: E402


def alf_set(results_dir, tag):
    out = {}
    for s in sorted(glob.glob(str(Path(results_dir) / f"*{tag}*.sum"))):
        stem = s[:-4]
        bits = [b for b in Path(stem).name.replace("_", " ").split() if b.isdigit()]
        tid = next((int(b) for b in bits if len(b) >= 16), None)
        if tid is None:
            continue
        C, _ = load_run(stem)
        out[tid] = {k: float(np.median(C[k])) for k in
                    ("zH", "FeH", "Mg", "logage", "sigma", "m2lnP")}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--alf-results", default=None,
                   help="default $ALF_HOME/results")
    p.add_argument("--free-tag", default="quiescent")
    p.add_argument("--solar-tag", default="solarscaled")
    p.add_argument("--prospector", default="results/cont_map_fits20_tauin")
    a = p.parse_args(argv)
    rd = a.alf_results or str(Path(__import__("os").environ["ALF_HOME"]) / "results")

    free, sol = alf_set(rd, a.free_tag), alf_set(rd, a.solar_tag)
    if not sol:
        raise SystemExit(f"no runs matching '{a.solar_tag}' in {rd}. "
                         "Run $ALF_HOME/run_solar_scaled11.sh first.")
    pros = {}
    for f in glob.glob(str(Path(a.prospector) / "3*.pkl")):
        r = pickle.load(open(f, "rb"))
        if isinstance(r, dict) and r.get("status") == "ok":
            d = dict(zip(r["labels"], np.asarray(r["theta"], float)))
            s = r["sfh"]
            e = np.asarray(s["edges_gyr"], float)
            m = np.asarray(s.get("ssfr_inplace", s.get("ssfr")), float)
            pros[int(r["target_id"])] = (
                d["logzsol"], float(np.sum(m * 0.5 * (e[:-1] + e[1:])) / np.sum(m)))

    tids = sorted(set(free) & set(sol) & set(pros))
    print(f"{len(tids)} galaxies with all three fits\n")
    hdr = (f"{'TARGETID':>19}{'alf free [Z/H]':>15}{'alf SS [Z/H]':>14}{'pros logzsol':>14}"
           f"{'|free-pros|':>12}{'|SS-pros|':>11}{'closer?':>9}")
    print(hdr); print("-" * len(hdr))
    d_free, d_sol, dage = [], [], []
    for t in tids:
        zf, zs, zp = free[t]["zH"], sol[t]["zH"], pros[t][0]
        a1, a2 = abs(zf - zp), abs(zs - zp)
        d_free.append(a1); d_sol.append(a2)
        dage.append(np.log10(pros[t][1] / 10 ** sol[t]["logage"]))
        print(f"{t:>19}{zf:>15.3f}{zs:>14.3f}{zp:>14.3f}{a1:>12.3f}{a2:>11.3f}"
              f"{'YES' if a2 < a1 else 'no':>9}")

    d_free, d_sol = np.array(d_free), np.array(d_sol)
    n = len(d_free); k = int((d_sol < d_free).sum())
    pv = 2 * sum(comb(n, i) * 0.5 ** n for i in range(max(k, n - k), n + 1))
    print(f"\n  median |alf - Prospector|:  free {np.median(d_free):.3f} dex"
          f"  ->  solar-scaled {np.median(d_sol):.3f} dex")
    print(f"  closer in {k}/{n}   sign test p = {min(pv,1.0):.4f}")
    print(f"  alf [Z/H] shift when forced solar-scaled: median "
          f"{np.median([sol[t]['zH'] - free[t]['zH'] for t in tids]):+.3f} dex")
    print(f"\n  VERDICT: {'ABUNDANCE PATTERN' if np.median(d_sol) < np.median(d_free) - 0.03 else 'NUMERICS (gap survives)'}"
          " dominates the metallicity disagreement")

    da = np.array(dage)
    print(f"\n  age check, Prospector vs solar-scaled alf: median dlog10 = {np.median(da):+.3f} dex"
          f"  ({10**np.median(da):.2f}x), scatter {da.std():.3f}")
    print("  (free-abundance alf gave -0.127 dex / 0.75x at 4.3 sigma, 2026-08-20l)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
