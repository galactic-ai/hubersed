"""Predict the pocoMC max-likelihood point of one target with the median DESI LSF, ladder step 4.

As 2026-10-28_continuum_c3k_er_rmatrix_map/predict_median_lsf.py, with the poco_repeats fit in
place of the MAP: the parameters are ``best`` of <TARGETID>_poco_maxl.npz (predict.py), with no
refit. The model has the parameters of the fit (free afe and SFH hyperparameters) as a
HyperSpecModel, so prospect smooths the C3K_ER spectrum from the library resolution to a Gaussian of
the median DESI resolution (hubersed.sps.lsf.desi_resolution) instead of applying the R matrix. Off
the fitted pixels and in prospect's padding the target is kept 10 km/s above the library, as there.
COMMAND: on stampede3 through predict_lsf.slurm, with the C3K_ER AFE_FLAG build.
OUTPUT: results/2026-11-01_poco_repeats/<TARGETID>/median_lsf/<TARGETID>_median_lsf_pred.npz, with
the data, ``best`` and ``labels`` of the maxl file, ``sp_best`` from this model and
``sp_best_rmatrix`` from the maxl file.
"""

import argparse
import importlib.util
import warnings
from pathlib import Path

import numpy as np
from prospect.models.sedmodel import HyperSpecModel

from hubersed.sps import rebin
from hubersed.sps.config import build_continuum_model
from hubersed.sps.lsf import desi_resolution
from hubersed.sps.parameter_file import WAVE_OBS, build_obs
from hubersed.sps.utils import universe_age_gyr

HERE = Path(__file__).resolve().parent
LSF = rebin.C_KMS / (2.355 * desi_resolution(WAVE_OBS.astype(np.float64)))


def make_model(poco, z):
    """Return the pocoMC fit's model as a HyperSpecModel, as the dark fit.py make_model."""
    _, t = build_continuum_model(z)
    t["add_neb_emission"] = {"N": 1, "isfree": False, "init": False}
    t["tau_in"]["isfree"] = False
    t["tau_in"]["init"] = universe_age_gyr(z)
    for name, (prior, init) in poco.HYPER_PRIORS.items():
        t[name]["isfree"] = True
        t[name]["prior"] = prior
        if init is not None:
            t[name]["init"] = init
    t["afe"] = dict(poco.make_model(z).config_dict["afe"])
    return HyperSpecModel(t)


def make_obs(z, flux, unc, good, sps):
    """Return the observation with the median DESI LSF as its resolution in km/s."""
    lib = np.interp(WAVE_OBS, sps.wavelengths * (1 + z), sps.spectral_resolution)
    assert np.all(LSF[good] > lib[good]), "library coarser than the LSF on fitted pixels"
    res = np.where(good, LSF, np.maximum(LSF, np.hypot(lib, 10.0)))
    obs = build_obs(spec=flux, unc=unc, mask=good, resolution=res, wavelength=WAVE_OBS)
    for spec in obs:
        wave = spec.padded_wavelength
        lib_pad = np.interp(wave, sps.wavelengths * (1 + z), sps.spectral_resolution)
        pad = (wave < WAVE_OBS[0]) | (wave > WAVE_OBS[-1])
        spec.padded_resolution[pad] = np.maximum(
            spec.padded_resolution[pad], np.hypot(lib_pad[pad], 10.0)
        )
    return obs


def main(tid):
    """Predict at the max-likelihood point and save it."""
    spec = importlib.util.spec_from_file_location("quiescent_fit", HERE / "fit.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    poco = wrapper.load_poco(tid)
    rmodel, _, sps = poco.state()
    z, good = poco._STATE["z"], poco._STATE["good"]
    m = np.load(poco.OUT / f"{tid}_poco_maxl.npz")
    assert np.array_equal(m["good"], good) and float(m["z"]) == z, "data differ from the fit"
    flux, unc = m["flux"], m["unc"]
    model = make_model(poco, z)
    labels = [str(x) for x in model.theta_labels()]
    assert labels == [str(x) for x in m["labels"]] == list(rmodel.theta_labels()), "labels differ"
    obs = make_obs(z, flux, unc, good, sps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds, _ = model.predict(np.asarray(m["best"], float), observations=obs, sps=sps)
    sp = np.asarray(preds[0], float)
    chi = (flux[good] - sp[good]) / unc[good]
    chi_r = (flux[good] - m["sp_best"][good]) / unc[good]
    print(f"{tid}: chi2 median LSF {np.sum(chi**2):.1f}, R matrix {np.sum(chi_r**2):.1f}, "
          f"{good.sum()} px", flush=True)  # fmt: skip
    out = poco.OUT / "median_lsf"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tid}_median_lsf_pred.npz"
    np.savez(path, wave=WAVE_OBS, z=z, flux=flux, unc=unc, good=good, labels=np.array(labels),
             best=m["best"], sp_best=sp, sp_best_rmatrix=m["sp_best"])  # fmt: skip
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tid", type=int, required=True)
    main(ap.parse_args().tid)
