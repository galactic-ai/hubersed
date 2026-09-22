"""Does the alpha-tilted SFH prior actually FIT the three outliers better than mean-zero?

The mock fix (get_stochastic_priors.py + make_cue_model_sed.py) only changed the TRAINING SET.
The FIT carries the identical defect: prospect/models/templates.py:140, inside
adjust_stochastic_params ->
    mean = np.zeros(ncomp - 1)
    rprior = priors.MultiVariateNormal(mean=mean, Sigma=sfr_ratio_covar)
So the fit's SFH prior cannot express a declining SFH either. This script rebuilds THE SAME
MultiVariateNormal with mean = alpha*dlog and profiles chi2 over alpha.

THREE ARMS (identical galaxies / obs / free params / optimizer -- ONLY the logsfr_ratios prior differs):
  alpha = 0.0        the current mean-zero stochastic prior          <- the defect
  alpha profiled     the tilted stochastic prior                     <- the fix
  free SFH           TopHat(+/-5), from results/freesfh_nopoly_fit.pkl <- achievable ceiling

READING THE RESULT:
  chi2(alpha_best) ~= chi2(free)  -> the tilt closes the gap; the mean-zero baseline WAS the defect.
  chi2(alpha_best) ~= chi2(0)     -> the tilt does NOT help; the ACF COVARIANCE binds, not the mean,
                                     and the mock fix addresses a real defect that is not THE defect.
  alpha_best per galaxy is itself the interesting number: quiescent 42580 should want alpha>0
  (declining); the EELGs should want alpha<0 (rising). If 42580 rails at the grid top, the
  U(-1,2.5) mock range is too narrow at the old end.

CAVEAT: MAP on a degenerate ridge is a point estimate, not a posterior -- established earlier in
this project. Use it to compare ARMS (same degeneracy both sides), not to trust individual bins.

Run from hubersed root (venv):
    python tmp/alpha_tilt_map.py quick    # alpha in {0, 1.5}, 1 seed  -- plumbing, ~minutes
    python tmp/alpha_tilt_map.py          # full profile
"""

import sys, pickle, warnings, copy
import numpy as np

sys.path.insert(0, "bin/prospector")
from hubersed.prospector import parameter_file as P
from hubersed.fitting import config as FC
from hubersed.fitting import chi2 as MC
from prospect.models.sedmodel import SpecModel
from prospect.models.templates import TemplateLibrary
from prospect.models import priors as PRIORS
from prospect.models import hyperparam_transforms
from prospect.fitting import lnprobfn
from hubersed.conversion import flambda_to_maggies, ivar_flambda_to_ivar_maggies
from hubersed.prospector.derived_quantities import compute_logssfr

from scipy.stats import multivariate_normal as _scipy_mvn


class WorkingMVN(PRIORS.MultiVariateNormal):
    """prospect's MultiVariateNormal (priors.py:269) defines sample/unit_transform/range/bounds
    but NO __call__. It therefore inherits Prior.__call__, which evaluates

        scipy.stats.norm.logpdf(x, loc=self.loc, scale=self.scale)

    with `distribution = scipy.stats.norm` (SCALAR normal) and `scale = Sigma`, the full (9,9)
    COVARIANCE MATRIX. That broadcasts (9,) against (9,9) -> returns a 9x9 matrix, NaN wherever
    Sigma < 0 (the negative off-diagonal covariances) -> lnprobfn raises
    "truth value of an array with more than one element is ambiguous".

    This never bites nested sampling: dynesty uses unit_transform (line 300), which is correctly
    implemented (cholesky(Sigma) @ norm.ppf(x)) and never touches the logpdf. It is fatal for
    MAP/emcee, which DO call the logpdf. => the stochastic ExReg prior has never been usable for
    MAP in this project; every MAP script in tmp/ uses TopHat instead.

    Verified upstream bug, not a local misuse. Override with the true MVN log-density (a scalar).
    """

    def __call__(self, x, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        xv = np.asarray(x, dtype=float).reshape(-1)
        try:
            return float(
                _scipy_mvn(
                    mean=np.asarray(self.loc, float).reshape(-1),
                    cov=np.asarray(self.scale, float),
                    allow_singular=True,
                ).logpdf(xv)
            )
        except Exception:
            return -np.inf


QUICK = "quick" in sys.argv
ALPHA_GRID = [0.0, 1.5] if QUICK else [-1.0, 0.0, 1.0, 2.0]
# Budget MUST match tmp/freesfh_nopoly_fit.py (NSEEDS=8, MAXFEV=30000), otherwise the stochastic
# arm is compared against a free-SFH chi2 that got more optimization and any "gap" is partly just
# me under-optimizing one side. Coarse alpha grid is the price of a fair budget.
NSEEDS = 1 if QUICK else 8
MAXFEV = 3000 if QUICK else 30000
EXTREME_EELG_TID = 39633149675702814
OUT = "results/alpha_tilt_map.pkl"

MC._fsps()

# ---- galaxies: identical loading to tmp/freesfh_nopoly_fit.py ----
d = pickle.load(open("results/mapfit_cont_line_examples.pkl", "rb"))
gals = []
for key, label in [
    ("continuum-only", "42580 quiescent (control)"),
    ("emission-line-only", "94183 EELG (moderate, EW~342)"),
]:
    r = d["results"][key]
    gals.append(
        dict(
            label=label,
            tid=int(r["id"]),
            z=float(r["z"]),
            flux=np.asarray(r["flux"], float),
            unc=np.asarray(r["unc"], float),
            mask=np.asarray(r["mask"], bool),
        )
    )
gidx = int(MC.tids_to_indices(np.array([EXTREME_EELG_TID], dtype=np.int64))[0])
spec, ivar, zt, tid2 = MC.load_by_index(gidx)
sm = flambda_to_maggies(P.WAVE_OBS, spec)
iv = ivar_flambda_to_ivar_maggies(P.WAVE_OBS, ivar)
sig = 1.0 / np.sqrt(np.where(iv > 0, iv, np.inf))
mk = (sig > 0) & np.isfinite(sig) & np.isfinite(sm)
gals.append(
    dict(
        label="702814 EELG (extreme, EW~640)",
        tid=int(tid2),
        z=float(zt),
        flux=sm,
        unc=sig,
        mask=mk,
    )
)

# free-SFH reference (the achievable ceiling), if present
REF = {}
try:
    fr = pickle.load(open("results/freesfh_nopoly_fit.pkl", "rb"))["results"]
    REF = {k: v["chi2_red"] for k, v in fr.items()}
    print("free-SFH reference chi2_red:", {k: round(v, 2) for k, v in REF.items()})
except Exception as e:
    print("(no free-SFH reference:", e, ")")

res_lsf = MC._lsf_sigma_kms()
sps = MC._cue()
sps.ssp.params["tpagb_norm_type"] = 2
sps.ssp.params["add_agb_dust_model"] = True


def build_model(z, alpha):
    """Stochastic ExReg SFH with the prior mean TILTED by alpha. Everything else is byte-identical
    to tmp/freesfh_nopoly_fit.py so the arms differ ONLY in the logsfr_ratios prior."""
    cmodel, ctemplate = FC.build_continuum_model(
        z
    )  # already ran adjust_stochastic_params
    ft = copy.deepcopy(ctemplate)
    ft.update(copy.deepcopy(TemplateLibrary["cue_stellar_nebular"]))
    ft["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}
    ft["use_stellar_ionizing"]["init"] = True
    # --- rebuild adjust_stochastic_params' prior, but with mean = alpha*dlog (templates.py:140) ---
    ab = np.asarray(ft["agebins"]["init"], float)
    mid = (10.0**ab).mean(axis=1)
    dlog = np.log10(mid[:-1] / mid[1:])  # same convention as the mock generator
    psd = [
        ft["sigma_reg"]["init"],
        ft["tau_eq"]["init"],
        ft["tau_in"]["init"],
        ft["sigma_dyn"]["init"],
        ft["tau_dyn"]["init"],
    ]
    sfr_covar = hyperparam_transforms.get_sfr_covar(psd, agebins=ab)
    cov = hyperparam_transforms.sfr_covar_to_sfr_ratio_covar(sfr_covar)
    mean = alpha * dlog
    ft["logsfr_ratios"]["isfree"] = True
    ft["logsfr_ratios"]["N"] = len(mean)
    ft["logsfr_ratios"]["init"] = mean
    ft["logsfr_ratios"]["prior"] = WorkingMVN(
        mean=mean, Sigma=cov
    )  # see class docstring: upstream __call__ is broken
    FREE = [
        "logsfr_ratios",
        "logmass",
        "logzsol",
        "dust2",
        "dust_ratio",
        "dust_index",
        "sigma_smooth",
        "gas_logz",
        "gas_logu",
        "gas_lognH",
        "gas_logno",
        "gas_logco",
        "eline_sigma",
    ]
    for k in list(ft.keys()):
        if isinstance(ft[k], dict) and "isfree" in ft[k]:
            ft[k]["isfree"] = k in FREE
    ft["eline_sigma"] = {
        "N": 1,
        "isfree": True,
        "init": 80.0,
        "units": "km/s",
        "prior": PRIORS.TopHat(mini=20.0, maxi=250.0),
    }
    return SpecModel(ft)


def make_obs(g):
    return P.build_obs(
        spec=g["flux"], unc=g["unc"], mask=g["mask"], resolution=res_lsf
    )  # NO PolyOptCal


def line_ratio(flux, sp, wr, mask, L0):
    core = (wr >= L0 - 9) & (wr <= L0 + 9) & mask
    lf = (wr >= L0 - 30) & (wr <= L0 - 15) & mask
    rt = (wr >= L0 + 15) & (wr <= L0 + 30) & mask
    cl = np.median(flux[lf | rt]) if (lf | rt).sum() else 0.0
    fd = np.nansum((flux - cl)[core])
    fm = np.nansum((sp - cl)[core])
    return fd / fm if fm > 0 else np.nan


results = {}
for g in gals:
    print(f"\n{'=' * 74}\n{g['label']}   TID {g['tid']}  z={g['z']:.4f}")
    prof = []
    for alpha in ALPHA_GRID:
        fmodel = build_model(g["z"], alpha)
        obs = make_obs(g)
        nfree = len(fmodel.free_params)
        assert "logsfr_ratios" in fmodel.free_params

        def neg(th):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    lp = lnprobfn(
                        th, model=fmodel, observations=obs, sps=sps, nested=False
                    )
                    return -lp if np.isfinite(lp) else 1e18
                except Exception:
                    return 1e18

        # _map_optimize returns None if EVERY seed fails its validity gate (neg >= 1e17).
        # That means the init itself is rejected -> diagnose which prior is killing it rather
        # than dying on `bf.x` with an opaque AttributeError.
        th0 = fmodel.theta.copy()
        n0 = neg(th0)
        if not np.isfinite(n0) or n0 >= 1e17:
            print(
                f"  alpha={alpha:+.2f}  *** init REJECTED (neg={n0:.3g}) -- diagnosing:"
            )
            try:
                print(f"      prior_product(theta_init) = {fmodel.prior_product(th0)}")
            except Exception as e:
                print(f"      prior_product RAISED: {type(e).__name__}: {e}")
            for k in fmodel.free_params:
                v = th0[fmodel.theta_index[k]]
                pr = fmodel.config_dict[k].get("prior", None)
                try:
                    lnp = pr(v) if pr is not None else "no prior"
                except Exception as e:
                    lnp = f"RAISED {type(e).__name__}: {e}"
                bad = (
                    ""
                    if (isinstance(lnp, (int, float)) and np.isfinite(lnp))
                    else "   <-- CULPRIT"
                )
                print(
                    f"      {k:16s} val={np.round(np.atleast_1d(v), 3)}  "
                    f"prior={type(pr).__name__}  lnp={lnp}{bad}"
                )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lnp_full = lnprobfn(
                        th0, model=fmodel, observations=obs, sps=sps, nested=False
                    )
                print(f"      lnprobfn(theta_init) = {lnp_full}")
            except Exception as e:
                print(f"      lnprobfn RAISED: {type(e).__name__}: {e}")
            raise SystemExit(
                "init rejected -- fix the model build before profiling alpha"
            )
        bf = MC._map_optimize(neg, th0, n_seeds=NSEEDS, maxfev=MAXFEV)
        if bf is None:
            print(f"  alpha={alpha:+.2f}  *** all seeds failed to converge; skipping")
            continue
        th = bf.x
        preds, _ = fmodel.predict(th, observations=obs, sps=sps)
        sp = np.asarray(preds[0], float)
        m = obs[0].mask
        chi2 = float(np.nansum(((obs[0].flux[m] - sp[m]) / obs[0].uncertainty[m]) ** 2))
        cr = chi2 / (int(m.sum()) - nfree)
        thd = {k: np.asarray(th[i], float) for k, i in fmodel.theta_index.items()}
        wr = P.WAVE_OBS / (1 + g["z"])
        rHb, rHa, rO3 = (
            line_ratio(g["flux"], sp, wr, g["mask"], L) for L in (4861, 6563, 5007)
        )
        try:
            lss = float(compute_logssfr(fmodel, th))
        except Exception:
            lss = np.nan
        lmv = float(np.atleast_1d(thd["logmass"])[0])
        print(
            f"  alpha={alpha:+.2f}  chi2_red={cr:7.3f}  logmass={lmv:5.2f}  logsSFR={lss:+6.2f}"
            f"  Hb {rHb:.2f} Ha {rHa:.2f} O3 {rO3:.2f}"
        )
        prof.append(
            dict(
                alpha=alpha,
                chi2_red=cr,
                logmass=lmv,
                logssfr=lss,
                dHb=rHb,
                dHa=rHa,
                dO3=rO3,
                theta_dict=thd,
                model=sp,
            )
        )
    crs = np.array([p["chi2_red"] for p in prof])
    i = int(np.nanargmin(crs))
    a0 = next((p for p in prof if p["alpha"] == 0.0), None)
    ref = REF.get(g["label"], np.nan)
    print(f"\n  --> best alpha = {prof[i]['alpha']:+.2f}   chi2_red = {crs[i]:.3f}")
    if a0 is not None:
        print(
            f"      mean-zero (alpha=0) chi2_red = {a0['chi2_red']:.3f}   -> improvement {a0['chi2_red'] - crs[i]:+.3f}"
        )
    print(
        f"      free-SFH ceiling    chi2_red = {ref if ref == ref else float('nan'):.3f}"
    )
    if a0 is not None and ref == ref:
        gap0 = a0["chi2_red"] - ref
        gapb = crs[i] - ref
        closed = 100 * (1 - gapb / gap0) if abs(gap0) > 1e-9 else float("nan")
        print(
            f"      gap to free-SFH: mean-zero {gap0:+.3f} -> tilted {gapb:+.3f}   ({closed:.0f}% of the gap closed)"
        )
    if len(ALPHA_GRID) > 2:  # a 2-point quick grid makes "railing" meaningless
        if prof[i]["alpha"] == max(ALPHA_GRID):
            print(
                "      *** alpha RAILED at the grid top -> U(-1,2.5) mock range may be too narrow"
            )
        if prof[i]["alpha"] == min(ALPHA_GRID):
            print(
                "      *** alpha RAILED at the grid bottom -> mock range too narrow at the rising end"
            )
    results[g["label"]] = dict(
        tid=g["tid"],
        z=g["z"],
        profile=prof,
        best_alpha=prof[i]["alpha"],
        chi2_best=float(crs[i]),
        chi2_alpha0=(a0["chi2_red"] if a0 else np.nan),
        chi2_freesfh=ref,
        flux=g["flux"],
        unc=g["unc"],
        mask=g["mask"],
    )

pickle.dump(
    dict(results=results, alpha_grid=ALPHA_GRID, nseeds=NSEEDS), open(OUT, "wb")
)
print(f"\nsaved {OUT}")
