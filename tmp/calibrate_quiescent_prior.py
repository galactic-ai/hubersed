"""
Forward-matching calibration of a data-driven QUIESCENT SFH prior component.

Goal (non-circular, observable-space): tune a quiescent SFH mu-family so that the
FORWARD-MODELED mock DN4000 distribution reproduces the OBSERVED BGS quiescent DN4000
distribution (results/quiescent_target_dn4000.npz). We never fit inferred SFHs; we match
what the data directly measure (the 4000A break), through the Cue/FSPS forward model --
the same model build as make_cue_model_sed.py, so DN4000 is consistent with mock generation.

Quiescent component:
  logsfr_ratios = mu_decl(z, tau) + draw from MVN(0, ACF(PSD))     (tau ~ family below)
where mu_decl is tmp/quiescent_sfh.declining_logsfr_ratios. tau is drawn per-mock from
[--tau-min, --tau-max] (log-uniform) so the prior SPANS the quiescent axis rather than
pinning one sSFR. Everything else (mass, logzsol, dust, gas, sigma) is drawn as in the
existing generator's prior ranges.

Outputs: mock DN4000 array + a comparison (KS statistic + overlaid histograms) vs the
observed target. Sweep (tau_min, tau_max, tform_frac, mixture handled upstream) to minimize
the DN4000 KS / match the high-break tail (DN4000 ~ 2.0-2.2).

CAVEATS (state in any writeup):
 - DN4000 alone is age x Z degenerate; this calibrates age*Z, not age. Anchor Z
   (--logzsol-min/max from the mass-Z relation, or a metallicity index) or the match is
   not unique.
 - DN4000 saturates ~2.1-2.2; the deep recent-sSFR floor is NOT constrained by this match
   (prior choice, report as upper limit).
 - Matches the OBSERVED (selection-convolved) distribution -- correct for encoder COVERAGE,
   not a volume-complete population statement.

Run on a machine with FSPS+Cue (NOT the sandbox):
  python tmp/calibrate_quiescent_prior.py --n 5000 --tau-min 0.4 --tau-max 3.0 \
      --tform-frac 0.95 --workers 8
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import sys
import copy
import argparse
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import cache

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- DN4000 (narrow, Balogh+1999): f_nu ratio of [4000,4100] / [3850,3950] AA rest ---
DN_BLUE = (3850.0, 3950.0)
DN_RED = (4000.0, 4100.0)
C_AA_S = 2.99792458e18  # c in AA/s, for f_lambda -> f_nu


def dn4000_narrow(wave_obs, flux_flam, z):
    """Narrow Dn(4000) from an observed-frame f_lambda spectrum. Converts to f_nu and
    integrates the two bands in the rest frame."""
    wr = wave_obs / (1.0 + z)
    fnu = flux_flam * wr**2 / C_AA_S  # f_lambda -> f_nu (up to const, cancels in ratio)

    def band(lo, hi):
        m = (wr >= lo) & (wr <= hi)
        return (
            np.nan
            if m.sum() < 3
            else np.trapz(fnu[m], wr[m]) / (wr[m].max() - wr[m].min())
        )

    b = band(*DN_BLUE)
    r = band(*DN_RED)
    return np.nan if (not np.isfinite(b) or b <= 0) else r / b


# ---- model build: mirror make_cue_model_sed.build_base_template ----
@cache
def _imports():
    from prospect.models import priors, transforms
    from prospect.models.sedmodel import HyperSpecModel
    from prospect.models.templates import TemplateLibrary, adjust_stochastic_params
    from prospect.observation import Spectrum
    from prospect.sources import NebStepBasis
    from hubersed.prospector.utils import make_stochastic_agebins, universe_age_gyr
    from hubersed.prospector.lsf import build_desi_resolution_matrix, DESI_WAV

    sys.path.insert(0, os.path.dirname(__file__))
    from quiescent_sfh import declining_logsfr_ratios

    return dict(
        priors=priors,
        transforms=transforms,
        HyperSpecModel=HyperSpecModel,
        TemplateLibrary=TemplateLibrary,
        adjust=adjust_stochastic_params,
        Spectrum=Spectrum,
        NebStepBasis=NebStepBasis,
        agebins=make_stochastic_agebins,
        uage=universe_age_gyr,
        Rmat=build_desi_resolution_matrix,
        DESI_WAV=DESI_WAV,
        decl=declining_logsfr_ratios,
    )


@cache
def _sps():
    return _imports()["NebStepBasis"]()


@cache
def _Rmat():
    I = _imports()
    return I["Rmat"](I["DESI_WAV"])


def _base_template():
    I = _imports()
    TL, pri, tr = I["TemplateLibrary"], I["priors"], I["transforms"]
    t = copy.deepcopy(TL["stochastic_sfh"])
    t.update(copy.deepcopy(TL["dust_emission"]))
    t.update(copy.deepcopy(TL["cue_stellar_nebular"]))
    t["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}
    t["dust_type"]["init"] = 0
    t["dust1"] = {
        "N": 1,
        "isfree": False,
        "depends_on": tr.dustratio_to_dust1,
        "init": 0.0,
    }
    t["dust_ratio"] = {
        "N": 1,
        "isfree": True,
        "init": 1.0,
        "prior": pri.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3),
    }
    t["dust_index"] = {
        "N": 1,
        "isfree": True,
        "init": 0.0,
        "prior": pri.TopHat(mini=-1.0, maxi=0.4),
    }
    t["sigma_smooth"] = {"N": 1, "isfree": False, "init": 200.0}
    t["smoothtype"] = {"N": 1, "isfree": False, "init": "vel"}
    t["fftsmooth"] = {"N": 1, "isfree": False, "init": True}
    t["eline_sigma"] = {"N": 1, "isfree": False, "init": 100.0}
    return t


def _obs():
    I = _imports()
    n = I["DESI_WAV"].size
    o = I["Spectrum"](
        wavelength=np.asarray(I["DESI_WAV"], float),
        flux=np.ones(n),
        uncertainty=np.ones(n),
        mask=np.ones(n, bool),
    )
    o.rectify()
    return o


def _one_quiescent(args):
    """Draw one quiescent mock, forward-model, return DN4000."""
    seed, tau_min, tau_max, tform_frac, logzmin, logzmax = args
    I = _imports()
    rng = np.random.default_rng(seed)
    z = float(rng.uniform(0.01, 0.6))
    tau = float(
        10 ** rng.uniform(np.log10(tau_min), np.log10(tau_max))
    )  # log-uniform tau
    t = copy.deepcopy(_BASE)
    t["zred"]["init"] = z
    t["agebins"]["init"] = I["agebins"](z=z)
    t["logmass"]["init"] = float(rng.uniform(9.5, 11.8))  # quiescent are massive
    t["logzsol"]["init"] = float(rng.uniform(logzmin, logzmax))
    t["dust_index"]["init"] = float(rng.uniform(-1.0, 0.4))
    t["dust_ratio"]["init"] = float(np.clip(rng.normal(1.0, 0.3), 0, 2))
    t["dust2"]["init"] = float(np.clip(rng.normal(0.3, 1.0), 0, 4))
    # PSD: modest sigma_reg so ACF scatter does not leak the quiescent baseline to SF
    t["sigma_reg"]["init"] = float(10 ** rng.uniform(np.log10(0.1), np.log10(1.0)))
    th = I["uage"](z)
    t["tau_eq"]["init"] = float(rng.uniform(0.01, th))
    t["tau_in"]["init"] = float(rng.uniform(0.01, th))
    t["sigma_dyn"]["init"] = float(10 ** rng.uniform(np.log10(0.001), np.log10(0.5)))
    t["tau_dyn"]["init"] = float(np.clip(rng.normal(0.01, 0.02), 0.005, 0.2))
    # gas params (mid-range; quiescent have weak lines, exact values ~irrelevant to DN4000)
    for k, v in [
        ("gas_logz", -0.5),
        ("gas_logu", -3.0),
        ("gas_lognH", 2.0),
        ("gas_logno", 0.0),
        ("gas_logco", 0.0),
    ]:
        if k in t:
            t[k]["init"] = v
    t = I["adjust"](t)
    # quiescent baseline mu + ACF scatter: adjust() built prior = MVN(mean=0, Sigma=ACF),
    # so mu + prior.sample() ~ MVN(mu, ACF) using prospect's own sampler.
    mu = I["decl"](z, tau, tform_frac=tform_frac)
    t["logsfr_ratios"]["init"] = (
        mu + np.asarray(t["logsfr_ratios"]["prior"].sample()).ravel()
    )
    m = I["HyperSpecModel"](configuration=t)
    preds, _ = m.predict(m.theta, [_OBS], sps=_sps())
    spec = _Rmat().dot(preds[0])
    return dn4000_narrow(np.asarray(I["DESI_WAV"]), np.asarray(spec), z)


# module-level (per worker) caches set in initializer
_BASE = None
_OBS = None


def _init():
    global _BASE, _OBS
    _BASE = _base_template()
    _OBS = _obs()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--tau-min", type=float, default=0.4)
    ap.add_argument("--tau-max", type=float, default=3.0)
    ap.add_argument("--tform-frac", type=float, default=0.95)
    ap.add_argument("--logzsol-min", type=float, default=-1.0)
    ap.add_argument("--logzsol-max", type=float, default=0.19)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target", default="results/quiescent_target_dn4000.npz")
    ap.add_argument("--out", default="results/quiescent_mock_dn4000.npz")
    args = ap.parse_args()

    from scipy.stats import ks_2samp

    tgt = np.load(args.target)["dn4000"]

    jobs = [
        (
            i,
            args.tau_min,
            args.tau_max,
            args.tform_frac,
            args.logzsol_min,
            args.logzsol_max,
        )
        for i in range(args.n)
    ]
    ctx = mp.get_context("spawn")
    dn = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=ctx, initializer=_init
    ) as ex:
        for k, r in enumerate(ex.map(_one_quiescent, jobs, chunksize=20)):
            dn.append(r)
            if (k + 1) % 200 == 0:
                print(f"[{k + 1}/{args.n}]", flush=True)
    dn = np.asarray(dn, float)
    dn = dn[np.isfinite(dn)]

    ks, p = ks_2samp(dn, tgt)
    print(
        f"\nmock DN4000 p5/50/95 = {np.percentile(dn, [5, 50, 95]).round(3)}  (n={dn.size})"
    )
    print(
        f"target    p5/50/95 = {np.percentile(tgt, [5, 50, 95]).round(3)}  (n={tgt.size})"
    )
    print(
        f"frac mock>2.0 = {100 * np.mean(dn > 2.0):.1f}%   target = {100 * np.mean(tgt > 2.0):.1f}%"
    )
    print(f"KS(mock,target) D={ks:.3f}  p={p:.2e}   (lower D = better match)")
    np.savez(
        args.out,
        dn4000_mock=dn.astype(np.float32),
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tform_frac=args.tform_frac,
        ks_D=ks,
        ks_p=p,
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axx = plt.subplots(figsize=(7, 4.5))
        axx.hist(
            tgt,
            bins=60,
            range=(1.0, 2.3),
            density=True,
            alpha=0.55,
            label=f"observed target ({tgt.size})",
        )
        axx.hist(
            dn,
            bins=60,
            range=(1.0, 2.3),
            density=True,
            alpha=0.55,
            label=f"mock τ∈[{args.tau_min},{args.tau_max}] ({dn.size})",
        )
        axx.set_xlabel("DN4000 (narrow)")
        axx.set_ylabel("pdf")
        axx.set_title(f"Forward-matched quiescent prior — KS D={ks:.3f}")
        axx.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out.replace(".npz", ".png"), dpi=130, bbox_inches="tight")
        print("saved", args.out.replace(".npz", ".png"))
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
