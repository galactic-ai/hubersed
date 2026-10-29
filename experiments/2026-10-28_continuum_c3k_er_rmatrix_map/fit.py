"""MAP of the continuum outlier 39627757533007793 with C3K_ER, free afe and its resolution matrix.

Changes from 2026-10-08_continuum_outlier run c3k_hr_full_nebon_afefree: the library is C3K_ER
(R = 10000 at rest 3300.5-10000 A) instead of C3K_HR, the data are not degraded, the line spread
is the target's DR1 resolution matrix (ResolutionMatrixModel, as 2026-10-21_continuum_poco), the
model is the stellar continuum only (no Cue, no broad lines), the fitted pixels are rest 3320 to
9000 A, and the fit is a MAP instead of nautilus. sigma_reg, tau_eq, sigma_dyn and tau_dyn are
fixed at the max-L point of the afefree run, and start 0 is that point. The other priors are those
of the 10-08 run.

``--nebular on`` adds Cue nebular emission (build_full_cue_model, no broad lines), with the Cue
priors of the 10-08 run. Its start 0 is the continuum MAP best point (<TID>_map.npz) for the
stellar parameters and the afefree max-L for the gas parameters and eline_sigma. Its outputs carry
the tag _cue. The resolution matrix is the same, including its zero rows at ivar 0 pixels.

Steps: ``check`` builds the model, checks the setup and times warm likelihood calls. ``fit
--start K`` runs start K of ``--n-starts`` (start 0 unjittered, the others jittered by ``--jitter``
times each prior range, seed 0) and saves map_s<K>.npz. ``collect`` keeps the lowest -lnp and saves
<TID>_map.npz for analysis.py, with a parameter table in map.txt.

FSPS must be the C3K_ER build with AFE_FLAG first on PYTHONPATH, with SPS_HOME holding the ER
spectra. SPEC_DATA can name a folder with the staged spender files. Run on a stampede3 compute
node through test.slurm or map.slurm, for example ``<venv>/bin/python
experiments/2026-10-28_continuum_c3k_er_rmatrix_map/fit.py fit --start 0 --maxfev 150000``.
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
from prospect.fitting import lnprobfn
from prospect.models.priors import TopHat

import hubersed.io.desi as desi_io
from hubersed.conversion import DESI_FLAM, ivar_to_maggies, to_maggies
from hubersed.fitting.map_fits import get_sps as get_cue_sps
from hubersed.fitting.map_fits import jitter_scale, map_fit, safe_lnprior
from hubersed.io.desi import load_spectrum
from hubersed.paths import PATHS
from hubersed.sps.config import build_continuum_model, build_full_cue_model
from hubersed.sps.parameter_file import WAVE_OBS, build_sps
from hubersed.sps.resolution_matrix import load_resolution_npz
from hubersed.sps.resolution_model import ResolutionMatrixModel, build_resolution_obs
from hubersed.sps.utils import universe_age_gyr

TID = 39627757533007793
OUT = PATHS["RESULTS"] / "2026-10-28_continuum_c3k_er_rmatrix_map"
MATRIX = OUT / "resolution_matrix.npz"
HR_RUN = (
    PATHS["RESULTS"] / "2026-10-08_continuum_outlier" / f"{TID}_c3k_hr_full_nebon_afefree_pred.npz"
)
HYPERS = ("sigma_reg", "tau_eq", "sigma_dyn", "tau_dyn")
REST_MIN, REST_MAX = 3320.0, 9000.0
if os.environ.get("SPEC_DATA"):
    desi_io.DATA_PATH = Path(os.environ["SPEC_DATA"])


def load_data():
    """Return the redshift, the flux and uncertainty in maggies, and the fitted pixels.

    Returns
    -------
    z : float
        Redshift.
    flux, unc : np.ndarray
        Flux and its uncertainty in maggies on ``WAVE_OBS``. Unfitted pixels have zero flux and
        infinite uncertainty.
    good : np.ndarray of bool
        Unmasked pixels with positive ivar in the coadd and the matrix file, at rest REST_MIN to
        REST_MAX.
    """
    s = load_spectrum(TID)
    z = float(s.redshift.value)
    wave = WAVE_OBS * u.AA
    flux = to_maggies(wave, s.flux).value
    iv = ivar_to_maggies(wave, s.uncertainty.quantity).value
    rest = WAVE_OBS / (1 + z)
    new = load_resolution_npz(MATRIX)
    good = (
        ~s.mask
        & np.isfinite(flux)
        & np.isfinite(iv)
        & (iv > 0)
        & (new["ivar"] > 0)
        & (rest >= REST_MIN)
        & (rest < REST_MAX)
    )
    check = to_maggies(wave, new["flux"] * DESI_FLAM).value
    assert np.allclose(check[good], flux[good], rtol=1e-5, atol=0), "matrix file differs"
    unc = np.where(good, 1 / np.sqrt(np.where(good, iv, 1.0)), np.inf)
    return z, np.where(good, flux, 0.0), unc, good


def hr_best():
    """Return the max-L point of the C3K HR afefree run as a dict of label to value."""
    d = np.load(HR_RUN)
    assert np.allclose(d["best"], d["points"][np.argmax(d["log_l"])]), "best is not the max-L point"
    return dict(zip([str(x) for x in d["labels"]], d["best"], strict=True))


def make_model(z, nebular=False):
    """Return the model with free afe and the SFH hyperparameters fixed at the HR max-L.

    Without ``nebular`` it is the stellar continuum only. With it, Cue nebular emission is added
    as in 2026-10-08_continuum_outlier/fit.py.
    """
    best = hr_best()
    cont_model, t = build_continuum_model(z)
    if nebular:
        _, t = build_full_cue_model(t, cont_model.theta, cont_model)
    else:
        t["add_neb_emission"] = {"N": 1, "isfree": False, "init": False}
    t["tau_in"]["isfree"] = False
    t["tau_in"]["init"] = universe_age_gyr(z)
    for name in HYPERS:
        t[name]["isfree"] = False
        t[name]["init"] = best[name]
    t["afe"] = {"N": 1, "isfree": True, "init": 0.0, "prior": TopHat(mini=-0.2, maxi=0.6)}
    return ResolutionMatrixModel(t)


def make_obs(flux, unc, good):
    """Return the observation list with the target's DR1 resolution matrix."""
    new = load_resolution_npz(MATRIX)
    return build_resolution_obs(
        flux, unc, good, new["res"], resolution_wave=new["wave"], wavelength=WAVE_OBS
    )


def get_sps(nebular=False):
    """Return the source, checked to be C3K_ER with five afe slots.

    With ``nebular`` it is the Cue source of map_fits.get_sps, as in
    2026-10-13_nautilus_resolution_matrix, with the library resolution kept.
    """
    import fsps

    sps = (
        get_cue_sps(zero_library_resolution=False)["cue"]
        if nebular
        else build_sps(zcontinuous=1, zero_library_resolution=False)
    )
    assert sps.ssp.libraries[1] in ("c3k_er", b"c3k_er"), sps.ssp.libraries
    assert fsps.fsps.driver.get_nafe() == 5, "FSPS was built without AFE_FLAG"
    return sps


def start_values(nebular):
    """Return start 0 as a dict of label to value.

    Without ``nebular`` it is the HR max-L. With it, the continuum MAP best point overrides the
    HR max-L for the parameters it has.
    """
    best = hr_best()
    if nebular:
        m = np.load(OUT / f"{TID}_map.npz")
        best.update(zip([str(x) for x in m["labels"]], m["best"], strict=True))
    return best


def build(nebular=False):
    """Return the model, observation, source, start 0 and the fitted data."""
    z, flux, unc, good = load_data()
    model = make_model(z, nebular)
    obs = make_obs(flux, unc, good)
    sps = get_sps(nebular)
    best = start_values(nebular)
    labels = model.theta_labels()
    missing = [k for k in labels if k not in best]
    assert not missing, f"no start value for {missing}"
    theta0 = np.array([best[k] for k in labels], float)
    return model, obs, sps, theta0, dict(z=z, flux=flux, unc=unc, good=good)


def pick_start(model, theta0, start, n_starts, jitter):
    """Return start ``start`` of ``n_starts``: theta0 for 0, else a jittered draw inside the prior."""
    if start == 0:
        return theta0
    rng = np.random.default_rng(0)
    scale = jitter_scale(model, theta0, jitter)
    starts = []
    for _ in range(400 * n_starts):
        st = theta0 + rng.normal(0.0, scale)
        if np.isfinite(safe_lnprior(model, st)):
            starts.append(st)
        if len(starts) == n_starts - 1:
            return starts[start - 1]
    raise RuntimeError(f"only {len(starts)} jittered starts inside the prior")


def lnp(model, obs, sps, theta):
    """Return the log posterior at ``theta``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lnprobfn(theta, model=model, observations=obs, sps=sps, nested=False)


def peak_memory_gib():
    """Return the peak resident memory of this process in GiB."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 2**20
    return float("nan")


def check(n_starts, jitter, n_calls, nebular=False):
    """Check the setup and time warm likelihood calls at the starts."""
    t = time.time()
    model, obs, sps, theta0, data = build(nebular)
    print(f"build {time.time() - t:.0f} s, peak memory {peak_memory_gib():.1f} GiB", flush=True)
    print(f"z {data['z']:.5f}, fitted px {data['good'].sum()}, free {model.theta_labels()}")
    print(f"fixed hypers {[(k, float(model.params[k][0])) for k in HYPERS]}")
    for k, v in zip(model.theta_labels(), theta0, strict=True):
        print(f"  {k:16s} start {v:9.4f}")
    print(f"log prior at start 0 {safe_lnprior(model, theta0):.3f}")
    starts = [pick_start(model, theta0, k, n_starts, jitter) for k in range(n_starts)]
    t = time.time()
    lp0 = lnp(model, obs, sps, theta0)
    print(f"lnp at start 0 {lp0:.2f}, first call {time.time() - t:.1f} s", flush=True)
    assert np.isfinite(lp0)
    times = []
    for i in range(n_calls):
        t = time.time()
        lnp(model, obs, sps, starts[i % n_starts])
        times.append(time.time() - t)
    times = np.array(times)
    print(
        f"{n_calls} warm calls over {n_starts} starts: median {np.median(times):.3f} s, "
        f"max {times.max():.3f} s, peak memory {peak_memory_gib():.1f} GiB"
    )


def fit(start, n_starts, jitter, maxfev, tag, nebular=False):
    """Run one start and save it to map_s<start><tag>.npz."""
    model, obs, sps, theta0, _ = build(nebular)
    theta_start = pick_start(model, theta0, start, n_starts, jitter)
    best, _ = map_fit(
        model, obs, sps, n_seeds=0, maxfev=maxfev, theta0=theta_start, tag=f"s{start}"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / f"map_s{start}{tag}.npz",
        labels=np.array(model.theta_labels()),
        theta0=theta_start,
        theta=best.x,
        fun=best.fun,
        nfev=best.nfev,
        success=best.success,
        maxfev=maxfev,
        start=start,
        n_starts=n_starts,
        jitter=jitter,
    )
    print(f"s{start}: -lnp {best.fun:.2f}, nfev {best.nfev}, success {best.success}", flush=True)
    print(f"peak memory {peak_memory_gib():.1f} GiB")


def collect(n_starts, tag, nebular=False):
    """Keep the start with the lowest -lnp, save its spectrum and write map.txt.

    A start with no output file (for example killed by the time limit) is listed as missing.
    """
    paths = {k: OUT / f"map_s{k}{tag}.npz" for k in range(n_starts)}
    missing = [k for k, p in paths.items() if not p.exists()]
    runs = {k: np.load(p) for k, p in paths.items() if p.exists()}
    runs = {k: r for k, r in runs.items() if np.isfinite(float(r["fun"]))}
    k_best = min(runs, key=lambda k: float(runs[k]["fun"]))
    r = runs[k_best]
    model, obs, sps, theta0, data = build(nebular)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = np.asarray(model.predict(r["theta"], observations=obs, sps=sps)[0][0], float)
    np.savez(
        OUT / f"{TID}_map{tag}.npz",
        wave=WAVE_OBS,
        z=data["z"],
        flux=data["flux"],
        unc=data["unc"],
        good=data["good"],
        labels=r["labels"],
        best=r["theta"],
        sp_best=spec,
        start=k_best,
        fun_all=np.array([float(runs[k]["fun"]) for k in sorted(runs)]),
    )
    name = "C3K_ER + afe + R matrix" + (" + Cue" if nebular else "")
    lines = [f"MAP of {TID}, {name}, best start {k_best} of {n_starts}"]
    lines += [f"missing starts {missing}", ""]
    lines += [
        f"start {k}: -lnp {float(runs[k]['fun']):.2f}, nfev {int(runs[k]['nfev'])} of "
        f"{int(runs[k]['maxfev'])}, success {bool(runs[k]['success'])}"
        for k in sorted(runs)
    ]
    lines += ["", f"{'parameter':>16} {'start 0':>10} {'MAP':>10}"]
    for name, a, b in zip(r["labels"], theta0, r["theta"], strict=True):
        lines.append(f"{name!s:>16} {a:10.4f} {b:10.4f}")
    text = "\n".join(lines) + "\n"
    (OUT / f"map{tag}.txt").write_text(text)
    print(text)


def main():
    """Parse the step and run it."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("check", "fit", "collect"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n-starts", type=int, default=7)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--maxfev", type=int, default=150000)
    ap.add_argument("--n-calls", type=int, default=21, help="warm calls timed by check")
    ap.add_argument("--tag", default="", help="suffix of the output files, for test runs")
    ap.add_argument("--nebular", choices=("on", "off"), default="off")
    a = ap.parse_args()
    neb = a.nebular == "on"
    tag = ("_cue" if neb else "") + a.tag
    if a.step == "check":
        check(a.n_starts, a.jitter, a.n_calls, neb)
    elif a.step == "fit":
        fit(a.start, a.n_starts, a.jitter, a.maxfev, tag, neb)
    else:
        collect(a.n_starts, tag, neb)


if __name__ == "__main__":
    main()
