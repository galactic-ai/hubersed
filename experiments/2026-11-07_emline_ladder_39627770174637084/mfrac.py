"""Surviving stellar mass of 1000 posterior draws of one ladder step (``--step``).

The draws are weighted posterior samples with seed 0, as in fit.py predict. mfrac is the ratio of
surviving to formed stellar mass that the model's predict returns, so the surviving mass is
logmass + log10(mfrac). One process, no pool.

COMMAND: with the step's FSPS build and PYTHONPATH as for fit.py, from the repository root,
``uv run --no-sync python experiments/2026-11-07_emline_ladder_39627770174637084/mfrac.py --step 7``.
On the cluster, ``STEP=7 sbatch --export=ALL <this folder>/mfrac.slurm``.

OUTPUT: results/2026-11-07_emline_ladder_39627770174637084/<STEMS[step]>/mfrac.npz with the chain
index, logmass, mfrac and logmass_surv of each draw.
"""

import argparse
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
from prospect.io.read_results import results_from

from hubersed.io import desi

N_DRAWS = 1000
if os.environ.get("SPEC_DATA"):  # staged Parquet dataset; data/ is not mounted on compute nodes
    desi.load_spectrum.__defaults__ = (
        Path(os.environ["SPEC_DATA"]) / "datasets" / "desi_sv3_bright_v1",
    )


def load_fit():
    """Return the ladder fit.py module, loaded by path."""
    spec = importlib.util.spec_from_file_location("ladder_fit", Path(__file__).with_name("fit.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def chain_labels(res):
    """Return the chain's parameter names, with vector parameters as name_1 ... name_n."""
    labels = []
    for name in res["chain"].dtype.names:
        n = int(np.prod(res["chain"].dtype[name].shape or (1,)))
        labels += [name] if n == 1 else [f"{name}_{i + 1}" for i in range(n)]
    return labels


def main():
    """Draw the posterior samples, compute mfrac for each and write mfrac.npz."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--n", type=int, default=N_DRAWS)
    args = parser.parse_args()
    fit = load_fit()
    folder = fit.OUT / fit.STEMS[args.step]
    res, _, _ = results_from(str(folder / f"{fit.STEMS[args.step]}.h5"))
    chain, weights = res["unstructured_chain"], res["weights"]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(chain), size=args.n, p=weights / weights.sum())

    t = time.time()
    observations, model, cue = fit.build(args.step)
    labels = model.theta_labels()
    assert labels == chain_labels(res), "chain columns differ from the model's parameters"
    print(f"step {args.step}: build {time.time() - t:.0f} s, {args.n} draws", flush=True)

    mfrac = np.empty(args.n)
    t = time.time()
    for k, i in enumerate(pick):
        mfrac[k] = float(np.squeeze(model.predict(chain[i], observations, sps=cue)[1]))
        if k == 0:
            print(f"first call {time.time() - t:.1f} s", flush=True)
    logmass = chain[pick, labels.index("logmass")]
    surv = logmass + np.log10(mfrac)
    np.savez(folder / "mfrac.npz", index=pick, logmass=logmass, mfrac=mfrac, logmass_surv=surv)
    q = np.percentile(surv, [16, 50, 84])
    print(
        f"step {args.step}: {args.n} calls in {time.time() - t:.0f} s; mfrac median "
        f"{np.median(mfrac):.3f}; logmass_surv {q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}]; wrote mfrac.npz"
    )


if __name__ == "__main__":
    main()
