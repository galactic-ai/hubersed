"""Predict the SV3 continuum MAP of 39627757533007793 with the median DESI LSF instead of its R matrix.

The parameters are the best point of <TID>_map.npz (fit.py collect, continuum only), with no refit.
The model has the parameters of fit.py make_model, as a HyperSpecModel. prospect smooths the C3K_ER
spectrum from the library resolution to a Gaussian of the median DESI resolution
(hubersed.sps.lsf.desi_resolution), as 2026-10-08_continuum_outlier/fit.py does without degrading,
but with the library resolution kept. Off the fitted pixels and in prospect's padding the target is
kept 10 km/s above the library, as there.

COMMAND: on a stampede3 compute node, through predict_median_lsf.slurm, with fit.py next to this file.

OUTPUT: results/2026-10-28_continuum_c3k_er_rmatrix_map/<TID>_median_lsf_pred.npz, with the keys of
<TID>_map.npz for the data, ``best`` and ``labels``, and ``sp_best`` from this model.
"""

import fit
import numpy as np
from prospect.models.priors import TopHat
from prospect.models.sedmodel import HyperSpecModel

from hubersed.sps import rebin
from hubersed.sps.config import build_continuum_model
from hubersed.sps.lsf import desi_resolution
from hubersed.sps.parameter_file import WAVE_OBS, build_obs
from hubersed.sps.utils import universe_age_gyr

LSF = rebin.C_KMS / (2.355 * desi_resolution(WAVE_OBS.astype(np.float64)))


def make_model(z):
    """Return fit.py's continuum model (free afe, SFH hyperparameters fixed) as a HyperSpecModel."""
    best = fit.hr_best()
    _, t = build_continuum_model(z)
    t["add_neb_emission"] = {"N": 1, "isfree": False, "init": False}
    t["tau_in"]["isfree"] = False
    t["tau_in"]["init"] = universe_age_gyr(z)
    for name in fit.HYPERS:
        t[name]["isfree"] = False
        t[name]["init"] = best[name]
    t["afe"] = {"N": 1, "isfree": True, "init": 0.0, "prior": TopHat(mini=-0.2, maxi=0.6)}
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


def main():
    """Predict at the MAP best point and save it."""
    z, flux, unc, good = fit.load_data()
    m = np.load(fit.OUT / f"{fit.TID}_map.npz")
    assert np.array_equal(m["good"], good) and float(m["z"]) == z, "data differ from the MAP run"
    sps = fit.get_sps(nebular=False)
    model = make_model(z)
    labels = [str(x) for x in model.theta_labels()]
    assert labels == [str(x) for x in m["labels"]], "parameters differ from the MAP run"
    obs = make_obs(z, flux, unc, good, sps)
    preds, _ = model.predict(np.asarray(m["best"], float), observations=obs, sps=sps)
    sp = np.asarray(preds[0], float)
    chi = (flux[good] - sp[good]) / unc[good]
    chi_r = (flux[good] - m["sp_best"][good]) / unc[good]
    print(f"chi2 median LSF {np.sum(chi**2):.1f}, R matrix {np.sum(chi_r**2):.1f}, {good.sum()} px")
    out = fit.OUT / f"{fit.TID}_median_lsf_pred.npz"
    np.savez(out, wave=WAVE_OBS, z=z, flux=flux, unc=unc, good=good, labels=np.array(labels),
             best=m["best"], sp_best=sp, sp_best_rmatrix=m["sp_best"])  # fmt: skip
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
