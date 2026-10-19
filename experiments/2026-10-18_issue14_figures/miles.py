"""Recompute the best fit of the 09-24 MILES + Cue run of 39627770174637084 with its own code.

With the current lockfile the 09-24 max-L point gives chi2_red 3.42, not the 2.52 the run reported,
and the model is 6.6% off in flux (median). The fit used fsps 0.4.7 (MILES), prospector 9aced93,
cue d6068d2 (the line list before the NIST correction of 2026-09-27) and FSPS data 82a8735. This
script runs in that environment and saves the max-L spectrum for 2026-10-18_issue14_figures.py.
``--repo`` points PATHS at the current checkout for data/ and results/, read only.

RESULT: 2026-10-18. chi2_red 2.517 (chi2 11130.5) on 4448 px, matching the 2.52 of 09-24.

COMMAND: build the environment of hubersed d4fb8ef in a scratch folder. Write each file of
``git ls-tree -r --name-only d4fb8ef -- src pyproject.toml uv.lock README.md
experiments/2026-09-24_nautilus_tauin`` with ``git show d4fb8ef:<file>`` into <old>, then
``UV_PROJECT_ENVIRONMENT=<venv> uv sync --frozen --no-dev --python 3.13`` in <old>. Write FSPS
82a8735 (python-fsps 0.4.7's libfsps), without SPECTRA/C3K, doc and src, the same way into <sps>.
Then from the repository root, ``SPS_HOME=<sps> <venv>/bin/python
experiments/2026-10-18_issue14_miles.py --repo .``.

OUTPUT: results/2026-10-18_issue14_figures/miles_0924.npz.
"""

import argparse
import importlib.util
from pathlib import Path

import hubersed.paths as hp


def main():
    """Point PATHS at the current checkout, rebuild the 09-24 model and save its max-L spectrum."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    hp.PATHS.update(DATA=repo / "data", RESULTS=repo / "results")  # before data modules load

    import numpy as np

    from hubersed.fitting.map_fits import chi2_parts, get_sps

    tid = 39627770174637084
    old = Path(hp.__file__).resolve().parents[2]
    path = old / "experiments" / "2026-09-24_nautilus_tauin" / "fit.py"
    spec = importlib.util.spec_from_file_location("fit_0924", path)
    f24 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f24)

    pd = np.load(
        repo / "results" / "2026-09-25_nautilus_tauin_c3k_hr" / f"{tid}_MILES_plotdata.npz"
    )
    z, flux, unc, good, in_miles = f24.load_data(tid)
    mask = good & in_miles
    model = f24.make_model(z)
    assert list(model.theta_labels()) == [str(x) for x in pd["labels"]]
    obs = f24.make_obs(flux, unc, mask)
    sp, stats = chi2_parts(model, pd["best"], obs, get_sps()["cue"], ~mask)
    print(f"hubersed from {old}")
    print(
        f"MILES max-L: chi2_red {stats['chi2_red']:.3f}, chi2 {stats['chi2']:.1f}, {mask.sum()} px"
    )
    out = repo / "results" / "2026-10-18_issue14_figures" / "miles_0924.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, flux=flux, unc=unc, mask=mask, model=sp, z=z, n_free=model.ndim)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
