"""pocoMC fit of the C3K ER continuum model to one of the six robust continuum outliers (#17).

Changes from 2026-10-31_poco_c3k_er_dark_39633433328091196/fit.py, which this file loads by path:
the data are the target's SV3 bright DR1 coadd (load_spectrum) with its DR1 resolution matrix
(resolution_matrix.py), the output folder is results/2026-10-31_poco_robust6/<TARGETID>/, the check
step has no MAP to compare with, and a resumed state keeps this output folder (load_state also
restores output_dir, a bug noted in handoff_2026-10-31_poco_dark_39633433328091196.md). Model, priors
and sampler settings are those of that fit: continuum only, C3K ER, free afe and SFH hyperparameters,
rest 3320 to 9000 A, n_effective 1024, n_active 512.
COMMAND: on stampede3 through prep.slurm (check) and prod.slurm (runfork), for example
``ibrun <venv>/bin/python experiments/2026-10-31_poco_robust6/fit.py runfork --tid <TARGETID>``.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS

HERE = Path(__file__).resolve().parent
OUT = PATHS["RESULTS"] / HERE.name
NAME = "fitpoco_robust6"  # registered so pickled and dill-saved functions resolve in every process


def load_poco(tid):
    """Return today's pocoMC fit module, pointed at ``tid``."""
    if NAME in sys.modules:
        return sys.modules[NAME]
    spec = importlib.util.spec_from_file_location(
        NAME, HERE.parent / "2026-10-31_poco_c3k_er_dark_39633433328091196" / "fit.py"
    )
    poco = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = poco
    spec.loader.exec_module(poco)
    out = OUT / str(tid)
    poco.base.TID = tid
    poco.base.load_spectrum = load_spectrum  # the dark fit swapped in its dark spectrum
    poco.base.MATRIX = out / "resolution_matrix.npz"
    poco.base.OUT = out
    poco.OUT = out
    poco.check = check
    poco.sample = sample
    return poco


def check(n_calls):
    """Build the model and time calls at prior draws; every value must be finite."""
    poco = sys.modules[NAME]
    t = time.time()
    model, obs, sps = poco.state()
    print(f"build {time.time() - t:.0f} s, memory {poco.memory()}", flush=True)
    print(f"TID {poco.base.TID}, z {poco._STATE['z']:.5f}, fitted px {poco._STATE['good'].sum()}, "
          f"{model.ndim} free: {model.theta_labels()}")  # fmt: skip
    for k in model.free_params:
        print(f"  {k:16s} {model.config_dict[k]['prior']}")
    rng = np.random.default_rng(0)
    times, vals = [], []
    for _ in range(n_calls):
        u = rng.uniform(size=model.ndim)
        t = time.time()
        vals.append(poco.loglike_unit(u))
        times.append(time.time() - t)
    print(f"{n_calls} calls at prior draws: times {np.round(times, 1).tolist()} s, values "
          f"{np.round(vals, 2).tolist()}, memory {poco.memory()}", flush=True)  # fmt: skip
    assert np.all(np.isfinite(vals)), "non-finite lnlike at a prior draw"


def sample(args, pool):
    """Build the sampler on rank 0 and run or resume it, keeping this run's output folder."""
    import pocomc as pc
    from scipy.stats import uniform

    poco = sys.modules[NAME]
    folder = poco.OUT / f"poco_neff{args.n_effective}{args.tag}"
    folder.mkdir(parents=True, exist_ok=True)

    class Sampler(pc.Sampler):
        """pocoMC's Sampler, keeping this process's likelihood and output folder on load."""

        def load_state(self, path):
            keep_like, keep_dir = self.log_likelihood, self.output_dir
            super().load_state(path)
            self.log_likelihood, self.output_dir = keep_like, keep_dir

    ndim = poco.state(with_sps=False)[0].ndim
    sampler = Sampler(
        prior=pc.Prior(ndim * [uniform(0.0, 1.0)]),
        likelihood=poco.loglike_unit,
        n_dim=ndim,
        n_effective=args.n_effective,
        n_active=args.n_active,
        n_max_steps=args.n_max_steps,
        pool=pool,
        output_dir=str(folder),
        output_label="pmc",
        random_state=0,
    )
    resume = poco.latest_state(folder)
    print(f"{ndim} parameters, {pool.size - 1} workers, resume from {resume}", flush=True)
    t = time.time()
    sampler.run(n_total=args.n_total, n_evidence=args.n_evidence, save_every=args.save_every,
                resume_state_path=resume)  # fmt: skip
    print(f"done in {time.time() - t:.0f} s, log Z {sampler.evidence()}", flush=True)


def main():
    """Parse the step and run it."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("check", "run", "runfork"))
    ap.add_argument("--tid", type=int, required=True)
    ap.add_argument("--n-calls", type=int, default=3, help="check: calls at prior draws")
    ap.add_argument("--n-effective", type=int, default=1024)
    ap.add_argument("--n-active", type=int, default=512)
    ap.add_argument("--n-total", type=int, default=4096)
    ap.add_argument("--n-evidence", type=int, default=4096)
    ap.add_argument("--n-max-steps", type=int, default=None, help="None is 10 * n_dim // 2")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--init-width", type=int, default=2, help="FSPS builds at once per node")
    ap.add_argument("--tag", default="", help="suffix of the output folder, for test runs")
    ap.add_argument("--build", type=int, default=None, help="runfork: build_ssps processes")
    ap.add_argument("--pool", type=int, default=None, help="runfork: workers per node")
    ap.add_argument("--no-free", action="store_true", help="runfork: keep speclib (Mac tests)")
    args = ap.parse_args()
    poco = load_poco(args.tid)
    if args.step == "check":
        check(args.n_calls)
    elif args.step == "run":
        poco.run(args)
    else:
        poco.run_fork(args)


if __name__ == "__main__":
    main()
