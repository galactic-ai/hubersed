"""pocoMC fit of one model ladder step to one poco_repeats target (#17).

The steps are runs 1d, 2b-off and 3n-free of 2026-10-08_continuum_outlier/fit.py, whose load_data,
make_model, make_sps, make_obs and check_setup this file loads by path, with nebular emission off in
all three. miles: MILES on rest 3601.8 to 7400.8 A, data not degraded, library resolution zeroed,
median DESI LSF. c3k_hr and c3k_hr_afe: C3K HR on its full window, data degraded to R 3000 plus
10 km/s, afe fixed at 0 and free, both with the C3K HR AFE_FLAG build. The SFH hyperparameters are
free under the #14 priors. recoadd: the fit.py model, priors and C3K_ER source, with flux, ivar
and resolution matrix from the target's SV3 recoadd (recoadd.slurm,
recoadd/resolution_matrix_rows_ivar.npz) and the fitted pixels of
2026-10-29_continuum_recoadd/fit.py load_data: spender's positive-ivar pixels where the recoadded
ivar is positive, rest 3320 to 9000 A. Sampler, fork pool and settings are those of fit.py
runfork, n_effective 1024 and n_active 512. The MILES build has no build_ssps and a static
speclib, so its pool forks without the build and without the free; each worker builds its own SSP
slots.
OUTPUT: results/2026-11-01_poco_repeats/<TARGETID>/<step>/poco_neff1024 and, from predict,
<TARGETID>_<step>_maxl.npz there.
COMMAND: through prod_ladder.slurm (check, then runfork) and predict_ladder.slurm, for example
``ibrun <venv>/bin/python experiments/2026-11-01_poco_repeats/fit_ladder.py runfork --tid <TARGETID>
--step miles --no-free``.
"""

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STEPS = {
    "miles": dict(lib="miles", window="miles", degrade=False, afe="none"),
    "c3k_hr": dict(lib="c3k_hr", window="full", degrade=True, afe="zero"),
    "c3k_hr_afe": dict(lib="c3k_hr", window="full", degrade=True, afe="free"),
    "recoadd": dict(lib="c3k_er"),
}


def recoadd_data(base, tid):
    """Return z, flux, unc and good of the SV3 recoadd, as 2026-10-29_continuum_recoadd load_data."""
    import astropy.units as u

    from hubersed.conversion import DESI_FLAM, ivar_to_maggies, to_maggies
    from hubersed.io.desi import load_spectrum
    from hubersed.sps.parameter_file import WAVE_OBS
    from hubersed.sps.resolution_matrix import load_resolution_npz

    s = load_spectrum(tid)
    z, keep = float(s.redshift.value), np.asarray(s.uncertainty.array) > 0
    new = load_resolution_npz(base.MATRIX)
    wave = WAVE_OBS * u.AA
    flux = to_maggies(wave, new["flux"] * DESI_FLAM).value
    iv = ivar_to_maggies(wave, new["ivar"] * DESI_FLAM**-2).value
    rest = WAVE_OBS / (1 + z)
    good = (keep & np.isfinite(flux) & np.isfinite(iv) & (iv > 0) & (rest >= base.REST_MIN)
            & (rest < base.REST_MAX))  # fmt: skip
    unc = np.where(good, 1 / np.sqrt(np.where(good, iv, 1.0)), np.inf)
    return z, np.where(good, flux, 0.0), unc, good


def load_module(name, path):
    """Load the file at ``path`` as module ``name``, registered so its functions pickle."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_ladder(tid, step):
    """Return the pocoMC fit module of fit.py with the model, data and source of ``step``."""
    wrapper = load_module("quiescent_fit", HERE / "fit.py")
    poco = wrapper.load_poco(tid)
    if step == "recoadd":
        poco.OUT = poco.OUT / step
        poco.base.MATRIX = poco.OUT / "resolution_matrix_rows_ivar.npz"
        poco.base.load_data = lambda: recoadd_data(poco.base, tid)
        poco.LADDER = dict(step=step, **STEPS[step])
        return poco
    f08 = load_module("fit_1008", HERE.parent / "2026-10-08_continuum_outlier" / "fit.py")
    s = STEPS[step]
    cache = {}

    def state(with_sps=True):
        """Return this process's model, observation and source, built on first use."""
        if "model" not in cache:
            z, flux, unc, good, target = f08.load_data(tid, s["lib"], s["window"], s["degrade"])
            model = f08.make_model(z, False, s["afe"])
            cache.update(z=z, flux=flux, unc=unc, good=good, target=target, model=model)
        if with_sps and "sps" not in cache:
            sps = f08.make_sps(False, s["degrade"])
            assert sps.ssp.libraries[1] in (s["lib"], s["lib"].encode()), sps.ssp.libraries
            if not hasattr(sps.ssp, "build_ssps"):
                sps.ssp.build_ssps = lambda processes=None: None  # slots are built on first use
            obs = f08.make_obs(cache["z"], cache["flux"], cache["unc"], cache["good"],
                               cache["target"], sps)  # fmt: skip
            cache.update(sps=sps, obs=obs)
        return cache["model"], cache.get("obs"), cache.get("sps")

    poco.state = state
    poco._STATE = cache
    poco.OUT = poco.OUT / step
    poco.LADDER = dict(f08=f08, step=step, **s)
    return poco


def check(poco, n_calls):
    """Run the 10-08 setup checks, then fit.py's check at prior draws."""
    L = poco.LADDER
    if "f08" in L:
        L["f08"].check_setup(poco.base.TID, L["lib"], L["window"], False, L["degrade"], L["afe"])
    sys.modules["quiescent_fit"].check(n_calls)
    with open("/proc/self/status") as f:
        print("".join(line for line in f if line.split(":")[0] in ("VmData", "VmHWM")), flush=True)


def predict(poco):
    """Save the model spectrum of the max-likelihood sample of the step's pocoMC run."""
    import dill
    import pocomc as pc
    from prospect.fitting import lnprobfn

    from hubersed.sps.parameter_file import WAVE_OBS

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
    tid, step = poco.base.TID, poco.LADDER["step"]
    print(f"{tid} {step} max-likelihood sample {i}: stored lnlike {logl[i]:.6f}, recomputed "
          f"{lnl:.6f}", flush=True)  # fmt: skip
    assert np.isclose(lnl, logl[i], rtol=0, atol=1e-4), "lnlike does not reproduce"
    c = poco._STATE
    if "flux" not in c:  # recoadd keeps fit.py's state, which holds z and good only
        c = dict(zip(("z", "flux", "unc", "good"), poco.base.load_data(), strict=True))
    path = poco.OUT / f"{tid}_{step}_maxl.npz"
    np.savez(path, wave=WAVE_OBS, z=c["z"], flux=c["flux"], unc=c["unc"], good=c["good"],
             labels=np.array(model.theta_labels()), best=theta, sp_best=sp, lnlike=lnl)  # fmt: skip
    print(f"wrote {path}")


def main():
    """Parse the arguments and run the step."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("check", "runfork", "predict"))
    ap.add_argument("--tid", type=int, required=True)
    ap.add_argument("--step", choices=tuple(STEPS), required=True)
    ap.add_argument("--n-calls", type=int, default=3, help="check: calls at prior draws")
    ap.add_argument("--n-effective", type=int, default=1024)
    ap.add_argument("--n-active", type=int, default=512)
    ap.add_argument("--n-total", type=int, default=4096)
    ap.add_argument("--n-evidence", type=int, default=4096)
    ap.add_argument("--n-max-steps", type=int, default=None, help="None is 10 * n_dim // 2")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--init-width", type=int, default=2)
    ap.add_argument("--tag", default="", help="suffix of the output folder, for test runs")
    ap.add_argument("--build", type=int, default=None, help="runfork: build_ssps processes")
    ap.add_argument("--pool", type=int, default=None, help="runfork: workers per node")
    ap.add_argument("--no-free", action="store_true", help="runfork: keep speclib")
    args = ap.parse_args()
    if args.step == "miles" and args.action == "runfork":
        assert args.no_free, "the MILES build has a static speclib, run it with --no-free"
    poco = load_ladder(args.tid, args.step)
    poco.OUT.mkdir(parents=True, exist_ok=True)
    if args.action == "check":
        check(poco, args.n_calls)
    elif args.action == "runfork":
        poco.run_fork(args)
    else:
        predict(poco)


if __name__ == "__main__":
    main()
