"""Model spectra of 39633433328091196 from one library at both the MILES and the C3K ER best points.

QUESTION (C. Hahn, 2026-10-29, tests 1 and 2): at the same parameters, how do the MILES and C3K
models of 39633433328091196 differ?

Points: the 10-08 MILES nautilus max-L (miles_miles_neboff_nodeg_pred.npz) and the C3K ER MAP of
icx job 3657895 (<TID>_map.npz) with its fixed SFH hyperparameters (hr_maxl.npz). Parameters a
library's model lacks are dropped (afe for MILES), and parameters a point lacks are set to 0 (afe
for C3K at the MILES point). ``--lib er`` uses the C3K ER model, data and R matrix of
2026-10-29_continuum_39633433328091196_map/fit.py; ``--lib miles`` the MILES model and data of
2026-10-08_continuum_outlier/fit.py (native data, MILES window, nebular off). The script asserts
that each library at its own point gives back its saved best spectrum.

COMMAND: ``--lib er`` on a stampede3 compute node through spectra_er.slurm; ``--lib miles`` on a
stampede3 compute node through miles_s3.slurm (or ls6 through ls6.slurm). SPEC_DATA can name a
folder with the staged spender files.
OUTPUT: results/2026-10-29_c3k_vs_miles_dust/spectra_<lib>.npz.
"""

import argparse
import importlib.util
import os
import warnings
from pathlib import Path

import numpy as np

import hubersed.io.desi as desi_io
from hubersed.paths import PATHS

TID = 39633433328091196
EXPS = Path(__file__).resolve().parents[1]
OUT = PATHS["RESULTS"] / "2026-10-29_c3k_vs_miles_dust"
MAPDIR = PATHS["RESULTS"] / "2026-10-29_continuum_39633433328091196_map"
MILES_PRED = (
    PATHS["RESULTS"] / "2026-10-08_continuum_outlier" / f"{TID}_miles_miles_neboff_nodeg_pred.npz"
)
HYPERS = ("sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn")
if os.environ.get("SPEC_DATA"):  # staged spender files, as on stampede3 compute nodes
    desi_io.DATA_PATH = Path(os.environ["SPEC_DATA"])


def load(name, path):
    """Return the module at ``path``, loaded by path since the experiment files share names."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def as_dict(path):
    """Return the saved best point of a fit as a dict of label to value."""
    d = np.load(path)
    return dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))


def points():
    """Return the MILES and C3K ER best points, the latter with its fixed SFH hyperparameters."""
    er = as_dict(MAPDIR / f"{TID}_map.npz")
    hr = as_dict(MAPDIR / "hr_maxl.npz")
    er.update({k: hr[k] for k in HYPERS})
    return {"miles": as_dict(MILES_PRED), "er": er}


def build(lib):
    """Return the model, observation, source and fitted data of one library."""
    if lib == "er":
        target = load("fit_target", EXPS / "2026-10-29_continuum_39633433328091196_map" / "fit.py")
        model, obs, sps, _, data = target.base.build()
        return model, obs, sps, data
    f08 = load("fit_1008", EXPS / "2026-10-08_continuum_outlier" / "fit.py")
    z, flux, unc, good, target_kms = f08.load_data(TID, "miles", "miles", False)
    sps = f08.make_sps(False, False)
    obs = f08.make_obs(z, flux, unc, good, target_kms, sps)
    return f08.make_model(z, False, "none"), obs, sps, dict(z=z, flux=flux, unc=unc, good=good)


def evaluate(model, obs, sps, point):
    """Return the model spectrum at ``point`` and the free labels the point lacks (set to 0)."""
    free = model.theta_labels()
    for k, v in point.items():
        if k in model.params and k not in free:
            model.params[k] = np.atleast_1d(float(v))
    missing = [k for k in free if k not in point]
    theta = np.array([point.get(k, 0.0) for k in free], float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = np.asarray(model.predict(theta, observations=obs, sps=sps)[0][0], float)
    return spec, missing


def main():
    """Evaluate one library at both points, check its own point and save."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lib", choices=("er", "miles"), required=True)
    lib = ap.parse_args().lib
    model, obs, sps, data = build(lib)
    g = data["good"]
    out = dict(z=data["z"], flux=data["flux"], unc=data["unc"], good=g)
    for name, point in points().items():
        spec, missing = evaluate(model, obs, sps, point)
        chi2 = float(np.sum(((data["flux"][g] - spec[g]) / data["unc"][g]) ** 2))
        out[f"spec_at_{name}"] = spec
        out[f"chi2_at_{name}"] = chi2
        print(f"{lib} at {name} point: chi2 {chi2:.1f} on {g.sum()} px, set to 0: {missing}")
    saved = np.load(MAPDIR / f"{TID}_map.npz" if lib == "er" else MILES_PRED)["sp_best"]
    own = out["spec_at_er" if lib == "er" else "spec_at_miles"]
    print(f"own point against saved best: max rel diff {np.max(np.abs(own[g] / saved[g] - 1)):.2e}")
    assert np.allclose(own[g], saved[g], rtol=1e-6, atol=0), "own point does not give back sp_best"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"spectra_{lib}.npz", **out)
    print(f"wrote {OUT / f'spectra_{lib}.npz'}")


if __name__ == "__main__":
    main()
