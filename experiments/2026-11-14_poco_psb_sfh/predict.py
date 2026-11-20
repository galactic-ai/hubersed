"""Save the max-likelihood spectrum and the log evidence of each fit of fit.py.

As 2026-11-01_poco_repeats/predict.py, with pocoMC's log Z. For the Cue models sp_cont is the same
spectrum with add_neb_emission off. TARGETIDs and --model restrict it.
COMMAND: through predict.slurm, or at the end of prod.slurm.
OUTPUT: results/2026-11-14_poco_psb_sfh/<TID>/<model>/maxl.npz.
"""

import importlib.util
import sys
import warnings
from pathlib import Path

import dill
import fsps
import numpy as np
import pocomc as pc
from prospect.fitting import lnprobfn

from hubersed.sps.parameter_file import WAVE_OBS
from hubersed.sps.ssp_cache import load_ssp_cache

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fitpsb", HERE / "fit.py")
fit = importlib.util.module_from_spec(_spec)
sys.modules["fitpsb"] = fit
_spec.loader.exec_module(fit)


def load_cache():
    """Load the SSP cache into this process with the population flags of fit.cached_pool."""
    sp = fsps.StellarPopulation(zcontinuous=1)
    for k in ("add_igm_absorption", "add_dust_emission", "add_neb_emission", "nebemlineinspec"):
        sp.params[k] = False
    sp.params["sfh"] = 0
    n = load_ssp_cache(sp, fit.CACHE)
    fit._LOADED_KEY["key"] = fsps.fsps._ssp_cache_key
    print(f"{n} SSP slots loaded from {fit.CACHE}", flush=True)


MODELS = ("A", "B", "AC", "BC", "AC0", "AC_main", "BC_main", "L", "L_recoadd")


def main(only=None, models=None):
    """Predict every finished fit of targets.txt, or only the TARGETIDs and models given."""
    load_cache()
    tids = [int(x.split()[0]) for x in (HERE / "targets.txt").read_text().splitlines()]
    tids = [t for t in tids if not only or t in only]
    for tid in tids:
        for name in [m for m in MODELS if not models or m in models]:
            folder = fit.OUT / str(tid) / name
            if not (folder / "pmc_final.state").is_file():
                continue
            sampler = pc.Sampler.__new__(pc.Sampler)
            with open(folder / "pmc_final.state", "rb") as f:
                sampler.__dict__.update(dill.load(f))
            logz, logz_err = sampler.evidence()
            usamp, _, logl, _ = sampler.posterior()
            i = int(np.argmax(logl))
            observations, model, sps = fit.build(tid, name)
            theta = model.prior_transform(usamp[i])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lnl = float(lnprobfn(theta, model=model, observations=observations, sps=sps,
                                     nested=True))  # fmt: skip
                sp = np.asarray(model.predict(theta, observations, sps=sps)[0][0], float)
                extra = {}
                if "C" in fit.split_name(name)[0]:
                    model.params["add_neb_emission"] = np.array([False])
                    extra["sp_cont"] = np.asarray(
                        model.predict(theta, observations, sps=sps)[0][0], float
                    )
                    model.params["add_neb_emission"] = np.array([True])
            held = fsps.fsps._ssp_cache_key == fit._LOADED_KEY["key"]
            print(f"{tid} {name}: max-lnlike sample {i}, stored {logl[i]:.6f}, recomputed "
                  f"{lnl:.6f}; key held {held}; log Z {logz:.3f} +- {logz_err:.3f}", flush=True)  # fmt: skip
            assert np.isclose(lnl, logl[i], rtol=0, atol=1e-4), "lnlike does not reproduce"
            assert held, "the SSP cache key changed"
            z, flux, unc, good, _ = fit.load_data(tid, fit.split_name(name)[1])
            np.savez(folder / "maxl.npz", wave=WAVE_OBS, z=z, flux=flux, unc=unc, good=good,
                     labels=np.array(model.theta_labels()), best=theta, sp_best=sp, lnlike=lnl,
                     logz=logz, logz_err=logz_err,
                     sigma_floor=float(model.config_dict["sigma_smooth"]["prior"].params["mini"]),
                     **extra)  # fmt: skip
    print("predict done")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tids", nargs="*", type=int, help="TARGETIDs (default all of targets.txt)")
    ap.add_argument("--model", nargs="*", choices=MODELS, help="models (default all finished)")
    a = ap.parse_args()
    main(set(a.tids), set(a.model or ()))
