"""Save the model spectrum of the max-likelihood pocoMC sample of one robust6 target.

As 2026-10-31_poco_c3k_er_dark_39633433328091196/predict.py: the sample with the largest stored lnlike
in pmc_final.state goes through the prior transform and the model of fit.py with C3K_ER; its lnlike
is recomputed and must match. Run on stampede3 through predict.slurm.
OUTPUT: results/2026-10-31_poco_robust6/<TARGETID>/<TARGETID>_poco_maxl.npz.
"""

import argparse
import importlib.util
import warnings
from pathlib import Path

import dill
import numpy as np
import pocomc as pc
from prospect.fitting import lnprobfn

from hubersed.sps.parameter_file import WAVE_OBS

HERE = Path(__file__).resolve().parent


def main(tid):
    """Pick the max-likelihood sample, check its lnlike and save its spectrum."""
    spec = importlib.util.spec_from_file_location("robust6_fit", HERE / "fit.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    poco = wrapper.load_poco(tid)
    sampler = pc.Sampler.__new__(pc.Sampler)
    with open(poco.OUT / "poco_neff1024" / "pmc_final.state", "rb") as f:
        sampler.__dict__.update(dill.load(f))
    usamp, _, logl, _ = sampler.posterior()
    i = int(np.argmax(logl))
    model, obs, sps = poco.state()
    theta = model.prior_transform(usamp[i])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lnl = float(lnprobfn(theta, model=model, observations=obs, sps=sps, nested=True))
        sp = np.asarray(model.predict(theta, observations=obs, sps=sps)[0][0], float)
    print(f"{tid} max-likelihood sample {i}: stored lnlike {logl[i]:.6f}, recomputed {lnl:.6f}",
          flush=True)  # fmt: skip
    assert np.isclose(lnl, logl[i], rtol=0, atol=1e-4), "lnlike does not reproduce"
    z, flux, unc, good = poco.base.load_data()
    path = poco.OUT / f"{tid}_poco_maxl.npz"
    np.savez(path, wave=WAVE_OBS, z=z, flux=flux, unc=unc, good=good,
             labels=np.array(model.theta_labels()), best=theta, sp_best=sp, lnlike=lnl)  # fmt: skip
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tid", type=int, required=True)
    main(ap.parse_args().tid)
