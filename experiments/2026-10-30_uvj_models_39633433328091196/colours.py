"""Rest-frame U-V and V-J of 39633433328091196 from the C3K ER and MILES best fits, each library at both points.

QUESTION: does V-J of 39633433328091196 depend on the stellar library? Points as in
2026-10-29_c3k_vs_miles_dust/spectra.py, except that the MILES point is the logzsol <= 0.2 MAP
(2026-10-29_nodust_map_39633433328091196, miles_dust_zmax0.2) with the SFH hyperparameters of the
10-08 max-L, as setup_miles fixes them. Parameters a model lacks are dropped (afe for MILES), and
parameters a point lacks are set to 0. afe is 0 for C3K at both points, since the Mac C3K_HR build
(stand-in for C3K_ER) has no AFE_FLAG. Each case is also run with dust2 = 0, so dust1 = 0.

Steps. ``inputs`` (hubersed venv) evaluates the C3K model of 2026-10-28 make_model and the MILES
model of 2026-10-08 make_model at both points and saves the FSPS parameters and tabular SFH that
prospect hands to FSPS. ``spectra --lib c3k_hr|miles`` (the venv of that build, numpy and fsps
only) replays them through python-fsps and integrates the rest-frame spectra through the FSPS U,
V and 2MASS_J curves (allfilters.dat, AB, photon counting as getmags.f90). ``check --lib`` (hubersed
venv) puts the replayed spectra through the prospect model and compares with the saved sp_best.

COMMAND: from the repository root on the Mac, with SPS_HOME=~/Astronomy_Research/fsps and
OMP_NUM_THREADS=1, ``uv run --no-sync python <this file> inputs``, then ``uv run --no-sync python
<this file> spectra --lib c3k_hr`` and ``~/Astronomy_Research/fsps_builds/miles_mac/.venv/bin/python
<this file> spectra --lib miles``, then ``uv run --no-sync python <this file> check --lib c3k_hr``
and ``check --lib miles``.
OUTPUT: results/2026-10-30_uvj_models_39633433328091196/ fsps_inputs.npz, colours_<lib>.npz and
check_<lib>.txt.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np

TID = 39633433328091196
NAME = "2026-10-30_uvj_models_39633433328091196"
ROOT = Path(__file__).resolve().parents[2]
EXPS = ROOT / "experiments"
RESULTS = ROOT / "results"
OUT = RESULTS / NAME
ER_DIR = RESULTS / "2026-10-29_continuum_39633433328091196_map"
ER_MAP = ER_DIR / f"{TID}_map.npz"
MILES_MAP = (
    RESULTS / "2026-10-29_nodust_map_39633433328091196" / "miles_dust_zmax0.2" / f"{TID}_map.npz"
)
MILES_PRED = RESULTS / "2026-10-08_continuum_outlier" / f"{TID}_miles_miles_neboff_nodeg_pred.npz"
HYPERS = ("sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn")
MODELS = ("c3k", "miles")  # the prospect model, C3K ER setup or MILES setup
POINTS = ("c3k", "miles")  # the best point, C3K ER MAP or MILES logzsol <= 0.2 MAP
DUST = ("dust", "nodust")
LIB_MODEL = {"c3k_hr": "c3k", "miles": "miles"}  # FSPS build to the model it stands in for
FILTERS = {"U": (2, "u"), "V": (1, "v"), "J": (10, "2mass_j")}  # FILTER_LIST index, fsps name
WAVE_SAVE = (2000.0, 30000.0)  # rest A of the spectra saved


def key(model, point, dust):
    """Return the name of one case."""
    return f"{model}_at_{point}_{dust}"


def as_dict(path):
    """Return the saved best point of a fit as a dict of label to value."""
    d = np.load(path)
    return {str(k): float(v) for k, v in zip(d["labels"], d["best"], strict=True)}


def points():
    """Return the C3K ER and MILES best points, each with its fixed SFH hyperparameters."""
    c3k = as_dict(ER_MAP)
    hr = as_dict(ER_DIR / "hr_maxl.npz")
    c3k.update({k: hr[k] for k in HYPERS})
    miles = as_dict(MILES_MAP)
    pred = as_dict(MILES_PRED)
    miles.update({k: pred[k] for k in HYPERS})
    return {"c3k": c3k, "miles": miles}


def load(name, path):
    """Return the module at ``path``, loaded by path since the experiment files share names."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def models(z):
    """Return the C3K ER model of the 10-29 MAP and the MILES model of the 10-08 run."""
    target = load("fit_target", EXPS / "2026-10-29_continuum_39633433328091196_map" / "fit.py")
    f08 = load("fit_1008", EXPS / "2026-10-08_continuum_outlier" / "fit.py")
    return (
        {"c3k": target.base.make_model(z), "miles": f08.make_model(z, False, "none")},
        target,
        f08,
    )


def theta_of(model, point, dust):
    """Set the fixed parameters ``point`` has and return theta, with afe 0 and dust2 0 if asked.

    Returns
    -------
    theta : np.ndarray
        Free parameters in the model's order.
    zeroed : list of str
        Free labels set to 0 (the point lacks them, or afe, or dust2 without dust).
    """
    free = model.theta_labels()
    for k, v in point.items():
        if k in model.params and k not in free:
            model.params[k] = np.atleast_1d(float(v))
    p = dict(point)
    p["afe"] = 0.0  # no AFE_FLAG in the Mac C3K_HR build
    if dust == "nodust":
        p["dust2"] = 0.0
    zeroed = [k for k in free if k not in point or (k in ("afe", "dust2") and p[k] == 0.0)]
    return np.array([p.get(k, 0.0) for k in free], float), zeroed


def fsps_inputs(model, sps, theta):
    """Return what prospect hands FSPS at ``theta`` and prospect's own rest-frame spectrum.

    Returns
    -------
    params : dict
        Every FSPS parameter after ``get_galaxy_spectrum``, as plain Python values.
    sfh : dict
        Tabular SFH time (Gyr) and SFR, tage, and the total formed mass the spectrum is divided by.
    wave, spec : np.ndarray
        prospect's rest-frame wavelength (A) and spectrum (Lsun/Hz per Msun formed).
    """
    model.set_parameters(theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wave, spec, _ = sps.get_galaxy_spectrum(**model.params)
    time, sfr, tmax = sps.convert_sfh(sps.params["agebins"], sps.params["mass"])
    params = {}
    for k in sps.ssp.params.all_params:
        v = np.asarray(sps.ssp.params[k])
        assert v.size == 1, (k, v)
        params[k] = v.item()
    sfh = dict(time=np.asarray(time, float).tolist(), sfr=np.asarray(sfr, float).tolist())
    sfh.update(tage=float(tmax), mtot=float(np.sum(sps.params["mass"])))
    return params, sfh, np.asarray(wave, float), np.asarray(spec, float)


def step_inputs():
    """Save the FSPS inputs of every case, and prospect's spectra from the C3K_HR build."""
    import fsps

    from hubersed.io.desi import load_spectrum
    from hubersed.paths import PATHS
    from hubersed.sps.parameter_file import build_sps

    assert PATHS["RESULTS"] == RESULTS
    assert fsps.fsps.driver.get_nafe() == 1, "this build has afe slots; revisit afe = 0"
    z = float(load_spectrum(TID).redshift.value)
    mods, _, _ = models(z)
    sps = build_sps(zcontinuous=1, zero_library_resolution=False)
    assert sps.ssp.libraries[1] in ("c3k_hr", b"c3k_hr"), sps.ssp.libraries
    pts = points()
    inputs, out = {}, dict(z=z)
    for m in MODELS:
        for p in POINTS:
            for d in DUST:
                theta, zeroed = theta_of(mods[m], pts[p], d)
                params, sfh, wave, spec = fsps_inputs(mods[m], sps, theta)
                k = key(m, p, d)
                inputs[k] = dict(params=params, sfh=sfh, theta=theta.tolist(), zeroed=zeroed)
                inputs[k]["labels"] = mods[m].theta_labels()
                out[f"prospect_wave_{k}"], out[f"prospect_spec_{k}"] = wave, spec
                print(
                    f"{k}: set to 0 {zeroed}, dust1 {params['dust1']:.4f}, dust2 {params['dust2']:.4f}"
                )
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "fsps_inputs.npz", inputs=json.dumps(inputs), points=json.dumps(pts), **out)
    print(f"wrote {OUT / 'fsps_inputs.npz'}")


def read_filters():
    """Return the FSPS U, V and 2MASS_J curves from allfilters.dat as (wave, response) arrays."""
    blocks, names = [], []
    with open(Path(os.environ["SPS_HOME"]) / "data" / "allfilters.dat") as f:
        for line in f:
            if line.startswith("#"):
                blocks.append([])
                names.append(line.strip())
            elif line.strip():
                blocks[-1].append([float(x) for x in line.split()[:2]])
    out = {}
    for band, (index, _) in FILTERS.items():
        out[band] = np.array(blocks[index - 1]).T
        print(f"{band}: allfilters.dat block {index}, {names[index - 1]}")
    return out


def ab_mag(wave, fnu, curve):
    """Return the AB magnitude up to a constant, photon counting, as getmags.f90 does.

    The constant is the same for every band, so differences are AB colours.
    """
    r = np.interp(wave, curve[0], curve[1], left=0.0, right=0.0)
    return -2.5 * np.log10(np.trapezoid(fnu * r / wave, wave) / np.trapezoid(r / wave, wave))


def step_spectra(lib):
    """Replay every case of this library's model through the FSPS build in this process."""
    import fsps

    sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1)
    assert sp.libraries[1] in (lib, lib.encode()), sp.libraries
    d = np.load(OUT / "fsps_inputs.npz")
    inputs = json.loads(str(d["inputs"]))
    curves = read_filters()
    model = LIB_MODEL[lib]
    out = {}
    for p in POINTS:
        for dust in DUST:
            k = key(model, p, dust)
            for name, v in inputs[k]["params"].items():
                sp.params[name] = v
            sfh = inputs[k]["sfh"]
            sp.set_tabular_sfh(np.array(sfh["time"]), np.array(sfh["sfr"]))
            wave, spec = sp.get_spectrum(tage=sfh["tage"], peraa=False)
            spec = spec / sfh["mtot"]
            mags = {b: ab_mag(wave, spec, c) for b, c in curves.items()}
            fmags = sp.get_mags(
                tage=sfh["tage"], redshift=0.0, bands=[v[1] for v in FILTERS.values()]
            )
            uv, vj = mags["U"] - mags["V"], mags["V"] - mags["J"]
            fuv, fvj = fmags[0] - fmags[1], fmags[1] - fmags[2]
            print(
                f"{lib} at {p}, {dust}: U-V {uv:.4f} V-J {vj:.4f} (fsps get_mags {fuv:.4f} {fvj:.4f})"
            )
            sel = (wave >= WAVE_SAVE[0]) & (wave <= WAVE_SAVE[1])
            out[f"wave_{p}_{dust}"], out[f"spec_{p}_{dust}"] = wave[sel], spec[sel]
            out[f"full_spec_{p}_{dust}"] = spec
            out[f"uv_{p}_{dust}"], out[f"vj_{p}_{dust}"] = uv, vj
            out[f"fsps_uv_{p}_{dust}"], out[f"fsps_vj_{p}_{dust}"] = fuv, fvj
            out[f"mags_{p}_{dust}"] = np.array([mags[b] for b in FILTERS])
    out["full_wave"] = wave
    out["libraries"] = np.array(
        [str(x.decode() if isinstance(x, bytes) else x) for x in sp.libraries]
    )
    np.savez(OUT / f"colours_{lib}.npz", **out)
    print(f"wrote {OUT / f'colours_{lib}.npz'}")


def step_check(lib):
    """Put the replayed own-point spectrum through the prospect model and compare with sp_best."""
    from prospect.sources import FastStepBasis

    from hubersed.sps.parameter_file import build_sps

    d = np.load(OUT / "fsps_inputs.npz")
    z = float(d["z"])
    col = np.load(OUT / f"colours_{lib}.npz")
    mods, target, f08 = models(z)
    model = mods[LIB_MODEL[lib]]
    pts = json.loads(str(d["points"]))
    lines = []
    if lib == "c3k_hr":
        for p in POINTS:
            for dust in DUST:
                ref = d[f"prospect_spec_{key('c3k', p, dust)}"]
                rel = np.max(np.abs(col[f"full_spec_{p}_{dust}"] / ref - 1))
                lines.append(
                    f"replay against prospect, C3K_HR at {p} {dust}: max rel diff {rel:.2e}"
                )
        zr, flux, unc, good = target.base.load_data()
        obs = target.base.make_obs(flux, unc, good)
        sps = build_sps(zcontinuous=1, zero_library_resolution=False)
        saved = np.load(ER_MAP)
    else:
        zr, flux, unc, good, kms = f08.load_data(TID, "miles", "miles", False)
        sps = f08.make_sps(False, False)
        obs = f08.make_obs(zr, flux, unc, good, kms, sps)
        saved = np.load(MILES_MAP)
    assert np.isclose(zr, z)
    wave, spec = col["full_wave"], col["full_spec_" + LIB_MODEL[lib] + "_dust"]

    class Replay(type(sps)):
        """The source of this build, returning the replayed spectrum of the other build."""

        def get_galaxy_spectrum(self, **params):
            _, _, mfrac = FastStepBasis.get_galaxy_spectrum(self, **params)
            return wave, spec, mfrac

        @property
        def spectral_resolution(self):
            base = super().spectral_resolution
            return np.interp(wave, self.ssp.wavelengths, base) if np.ndim(base) else base

    sps.__class__ = Replay
    theta, _ = theta_of(model, pts[LIB_MODEL[lib]], "dust")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = np.asarray(model.predict(theta, observations=obs, sps=sps)[0][0], float)
    g = saved["good"]
    assert np.array_equal(g, good)
    rel = np.abs(pred[g] / saved["sp_best"][g] - 1)
    lines.append(
        f"{lib} own point through the prospect model against sp_best of "
        f"{(ER_MAP if lib == 'c3k_hr' else MILES_MAP).relative_to(RESULTS)}: median rel diff "
        f"{np.median(rel):.2e}, max {rel.max():.2e} on {g.sum()} px"
    )
    text = "\n".join(lines) + "\n"
    (OUT / f"check_{lib}.txt").write_text(text)
    print(text)
    np.save(OUT / f"check_pred_{lib}.npy", pred)
    if lib == "miles":
        assert rel.max() < 1e-3, "MILES replay does not give back sp_best"


def main():
    """Parse the step and run it."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("inputs", "spectra", "check"))
    ap.add_argument("--lib", choices=tuple(LIB_MODEL), default=None)
    a = ap.parse_args()
    if a.step == "inputs":
        step_inputs()
    elif a.step == "spectra":
        step_spectra(a.lib)
    else:
        step_check(a.lib)


if __name__ == "__main__":
    main()
