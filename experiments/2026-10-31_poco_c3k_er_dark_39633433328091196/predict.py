"""Save the model spectrum of the max-likelihood pocoMC sample of the C3K ER dark fit (#17).

The sample with the largest stored lnlike in pmc_final.state (poco_neff1024_forktest) goes through
the prior transform and the model of fit.py with C3K_ER. Its lnlike is recomputed and must match the
stored one. The output has the keys of the MAP npz files (wave, z, flux, unc, good, labels, best,
sp_best), so analysis.py can plot it like the dark MAP.

FSPS must be the C3K_ER build with AFE_FLAG first on PYTHONPATH, with SPS_HOME holding the ER
spectra. Run on stampede3 through predict.slurm.
OUTPUT: results/2026-10-31_poco_c3k_er_dark_39633433328091196/<TID>_poco_maxl.npz.
"""

import importlib.util
import warnings
from pathlib import Path

import dill
import numpy as np
import pocomc as pc
from prospect.fitting import lnprobfn

from hubersed.sps.parameter_file import WAVE_OBS

HERE = Path(__file__).resolve().parent


def load_fit():
    """Return fit.py of this folder, loaded by path."""
    spec = importlib.util.spec_from_file_location("fitpoco", HERE / "fit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    """Pick the max-likelihood sample, check its lnlike and save its spectrum."""
    fit = load_fit()
    sampler = pc.Sampler.__new__(pc.Sampler)
    with open(fit.OUT / "poco_neff1024_forktest" / "pmc_final.state", "rb") as f:
        sampler.__dict__.update(dill.load(f))
    usamp, _, logl, _ = sampler.posterior()
    i = int(np.argmax(logl))
    model, obs, sps = fit.state()
    theta = model.prior_transform(usamp[i])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lnl = float(lnprobfn(theta, model=model, observations=obs, sps=sps, nested=True))
        spec = np.asarray(model.predict(theta, observations=obs, sps=sps)[0][0], float)
    print(
        f"max-likelihood sample {i}: stored lnlike {logl[i]:.6f}, recomputed {lnl:.6f}", flush=True
    )
    assert np.isclose(lnl, logl[i], rtol=0, atol=1e-4), "lnlike does not reproduce"
    z, flux, unc, good = fit.base.load_data()
    path = fit.OUT / f"{fit.base.TID}_poco_maxl.npz"
    np.savez(
        path,
        wave=WAVE_OBS,
        z=z,
        flux=flux,
        unc=unc,
        good=good,
        labels=np.array(model.theta_labels()),
        best=theta,
        sp_best=spec,
        lnlike=lnl,
    )
    for k, v in zip(model.theta_labels(), theta, strict=True):
        print(f"  {k:16s} {v:10.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
