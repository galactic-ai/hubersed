import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata

from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
TAGS = ("6latent", "10latent", "15latent", "cont10latent", "cont15latent")
Q = 0.001                      # same 0.1% mock quantile rule as the single-seed script


def jac(a, b):
    u = len(a | b)
    return len(a & b) / u if u else np.nan


def combiners(lp):
    """Three ways to turn S log-densities into one, all applied identically to mock and DESI.

    mix  = log of the MIXTURE (1/S) sum_s p_s. The only one that is itself a normalised
           density, but it is dominated by the single most optimistic member, so one
           member calling an object normal rescues it.
    mean = geometric mean of the densities (unnormalised). Compromise.
    med  = median log p. Robust to a single member that blows up on a point.
    """
    S = lp.shape[0]
    return {"mix": logsumexp(lp, axis=0) - np.log(S),
            "mean": lp.mean(0),
            "med": np.median(lp, 0)}


def sec1_members(d, tid):
    S = len(d["seeds"])
    n = np.array([(d["lp_desi"][i] <= d["thr"][i]).sum() for i in range(S)])
    print(f"\n[1] {S} members  (DESI N={len(tid)})")
    print("    seed  NLLtrain  NLLvalid     gap      thr   n_out    pct   KSmax   C2ST")
    for i, s in enumerate(d["seeds"]):
        print(f"    {int(s):4d}  {d['nll_train'][i]:8.3f}  {d['nll_valid'][i]:8.3f}  "
              f"{d['nll_valid'][i] - d['nll_train'][i]:+6.3f}  {d['thr'][i]:8.3f}  "
              f"{n[i]:5d}  {100 * n[i] / len(tid):5.3f}  {d['ks_max'][i]:5.3f}  {d['c2st'][i]:5.3f}")
    print(f"    n_out  mean {n.mean():.1f}  std {n.std(ddof=1):.1f}  min {n.min()}  max {n.max()}"
          f"   spread/mean = {(n.max() - n.min()) / n.mean():.2f}")
    print(f"    thr    mean {d['thr'].mean():.3f}  std {d['thr'].std(ddof=1):.3f}")
    print(f"    C2ST   mean {np.nanmean(d['c2st']):.3f}  max {np.nanmax(d['c2st']):.3f}"
          "   (0.5 = flow reproduces the mocks; >0.6 = it does not)")
    return n


def sec2_sets(d, tid):
    """How much of a single-seed outlier list is seed noise."""
    S = len(d["seeds"])
    sets = [set(tid[d["lp_desi"][i] <= d["thr"][i]].tolist()) for i in range(S)]
    js = np.array([jac(a, b) for a, b in combinations(sets, 2)])
    inter = set.intersection(*sets)
    union = set.union(*sets)
    print(f"\n[2] outlier-set reproducibility across seeds")
    print(f"    pairwise Jaccard: mean {js.mean():.3f}  min {js.min():.3f}  max {js.max():.3f}")
    print(f"    seed0 vs seed1 specifically: J = {jac(sets[0], sets[1]):.3f}  "
          f"|0\\1| = {len(sets[0] - sets[1])}  |1\\0| = {len(sets[1] - sets[0])}")
    print(f"    union over all {S} seeds        : {len(union)}")
    print(f"    intersection over all {S} seeds : {len(inter)}  "
          f"({100 * len(inter) / max(len(union), 1):.1f}% of union)")
    print(f"    median single-seed list size   : {np.median([len(s) for s in sets]):.0f}")
    return sets, union, inter


def sec3_votes(sets, union, S):
    votes = {}
    for s in sets:
        for t in s:
            votes[t] = votes.get(t, 0) + 1
    v = np.array(list(votes.values()))
    print(f"\n[3] vote counts over the {len(union)} objects flagged by at least one seed")
    print("      votes k    n(=k)   n(>=k)   frac(>=k)")
    for k in range(1, S + 1):
        ge = int((v >= k).sum())
        print(f"    {k:9d}  {int((v == k).sum()):7d}  {ge:7d}   {ge / len(union):8.3f}")
    return votes


def sec4_ranks(d):
    """Rank agreement is a stronger test than set overlap: it does not depend on where
    the threshold lands, so it separates 'the ranking is unstable' from 'the ranking is
    fine but the cut sits in a dense region'."""
    S = d["lp_desi"].shape[0]
    R = np.vstack([rankdata(d["lp_desi"][i]) for i in range(S)])
    C = np.corrcoef(R)
    off = C[np.triu_indices(S, 1)]
    print(f"\n[4] Spearman rho of DESI log p between seeds: mean {off.mean():.4f}  "
          f"min {off.min():.4f}  max {off.max():.4f}")
    pct = R / R.shape[1]
    return pct


def sec5_ensemble(d, tid, sets, union, inter):
    cm, cd = combiners(d["lp_mock"]), combiners(d["lp_desi"])
    print(f"\n[5] ensemble scores (threshold = {100 * Q:.1f}% quantile of the same combiner "
          "applied to the mocks)")
    ens = {}
    for k in ("mix", "mean", "med"):
        t = float(np.quantile(cm[k], Q))
        s = set(tid[cd[k] <= t].tolist())
        ens[k] = s
        jm = np.mean([jac(s, x) for x in sets])
        print(f"    {k:>4}  thr {t:9.3f}  n_out {len(s):5d}   J vs union {jac(s, union):.3f}  "
              f"J vs intersection {jac(s, inter):.3f}  mean J vs single seeds {jm:.3f}")
    return ens


def sec6_splithalf(d, tid, js_single, rng, n_rep=20):
    """Does ensembling actually buy reproducibility? Build two ensembles from disjoint
    halves of the seeds and compare their sets. If split-half J is no better than
    single-seed pairwise J, the ensemble is not more stable, only differently unstable."""
    S = d["lp_desi"].shape[0]
    if S < 4:
        return
    h = S // 2
    out = {k: [] for k in ("mix", "mean", "med")}
    for _ in range(n_rep):
        p = rng.permutation(S)
        A, B = p[:h], p[h:2 * h]
        for k in out:
            sa = combiners(d["lp_desi"][A])[k]
            sb = combiners(d["lp_desi"][B])[k]
            ta = float(np.quantile(combiners(d["lp_mock"][A])[k], Q))
            tb = float(np.quantile(combiners(d["lp_mock"][B])[k], Q))
            out[k].append(jac(set(tid[sa <= ta].tolist()), set(tid[sb <= tb].tolist())))
    print(f"\n[6] split-half reproducibility, {n_rep} random {h}v{h} splits")
    print(f"    single-seed pairwise J (from [2]) : {js_single:.3f}   <- the baseline to beat")
    for k, v in out.items():
        v = np.array(v)
        print(f"    {k:>4}-ensemble split-half J        : {v.mean():.3f} +/- {v.std(ddof=1):.3f}"
              f"   (gain {v.mean() - js_single:+.3f})")


def sec7_sample20(d, tid, votes, ens, pct, sample):
    if sample is None:
        return
    S = d["lp_desi"].shape[0]
    idx = {int(t): i for i, t in enumerate(tid)}
    print(f"\n[7] the {len(sample)} targets of cont_outlier_sample20.npz under this tag")
    print("        TARGETID  votes/S   pct_min   pct_med   pct_max   in mix/mean/med")
    nrob = 0
    for t in sample:
        i = idx.get(int(t))
        if i is None:
            print(f"    {int(t):17d}   not in this DESI latent file")
            continue
        p = pct[:, i]
        k = votes.get(int(t), 0)
        flags = "".join("Y" if int(t) in ens[c] else "." for c in ("mix", "mean", "med"))
        nrob += k == S
        print(f"    {int(t):17d}  {k:3d}/{S}  {p.min():9.5f} {np.median(p):9.5f} "
              f"{p.max():9.5f}   {flags}")
    print(f"    unanimous ({S}/{S}) under this tag: {nrob}/{len(sample)}")


def sec8_vs_stored(d, tid, tag):
    """Seed 0 here (GPU) vs the stored seed-0 run (CPU). Same seed, same code, different
    hardware -- so any disagreement is pure float noise and is a floor on how much
    instability is NOT attributable to the seed."""
    f = RES / "wide_flow_corrected" / f"desi_outliers_flow_nsf_{tag}_snr3.pt"
    if not f.exists():
        return
    import torch
    st = torch.load(f, weights_only=False)
    a = set(int(x) for x in st["outlier_target_ids"])
    b = set(tid[d["lp_desi"][0] <= d["thr"][0]].tolist())
    print(f"\n[8] stored seed-0 ({f.parent.name}/) vs seed-0 here")
    print(f"    stored n={len(a)} thr={st['threshold']:.3f}   |   here n={len(b)} "
          f"thr={d['thr'][0]:.3f}   J = {jac(a, b):.3f}")
    print("    -> at fixed seed the two should agree. Any gap here is float/hardware noise,")
    print("       i.e. a floor under section [2] that is NOT attributable to the seed.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=RES / "flow_ensemble")
    p.add_argument("--tags", nargs="*", default=None)
    p.add_argument("--method", default="nsf")
    p.add_argument("--sample", type=Path, default=RES / "cont_outlier_sample20.npz")
    a = p.parse_args()

    sample = None
    if a.sample.exists():
        sample = np.asarray(np.load(a.sample, allow_pickle=True)["target_ids"]).astype("<i8")

    tags = a.tags or [t for t in TAGS if (a.dir / f"ens_{a.method}_{t}.npz").exists()]
    ens_sets = {}
    for tag in tags:
        f = a.dir / f"ens_{a.method}_{tag}.npz"
        d = dict(np.load(f, allow_pickle=False))
        tid = d["desi_tid"].astype("<i8")
        S = d["lp_desi"].shape[0]
        print("\n" + "=" * 78)
        print(f"{tag}   dim={len(d['scaler_mean'])}  seeds={S}  file={f.name}")
        print(f"  hyper: {json.loads(str(d['hyper']))}")
        print(f"  mock:  {str(d['mock_file'])}")
        print(f"  DESI:  {str(d['desi_file'])}   encoder {str(d['encoder'])}")
        print("=" * 78)
        sec1_members(d, tid)
        sets, union, inter = sec2_sets(d, tid)
        js_single = float(np.mean([jac(x, y) for x, y in combinations(sets, 2)]))
        votes = sec3_votes(sets, union, S)
        pct = sec4_ranks(d)
        ens = sec5_ensemble(d, tid, sets, union, inter)
        sec6_splithalf(d, tid, js_single, np.random.default_rng(0))
        sec7_sample20(d, tid, votes, ens, pct, sample)
        sec8_vs_stored(d, tid, tag)
        ens_sets[tag] = ens["mean"]

    if len(ens_sets) > 1:
        print("\n" + "=" * 78)
        print("[9] cross-tag agreement of the MEAN-ensemble outlier sets (Jaccard)")
        ks = list(ens_sets)
        print("              " + "".join(f"{k:>15}" for k in ks))
        for i in ks:
            print(f"    {i:>10}  " + "".join(f"{jac(ens_sets[i], ens_sets[j]):15.3f}" for j in ks))


if __name__ == "__main__":
    main()
