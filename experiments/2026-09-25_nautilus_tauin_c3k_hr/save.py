"""Save the plot inputs of the MILES and C3K runs so later experiments can compare against them
without rebuilding the one-component model.
"""

# %%
import numpy as np
from fit import load_data, loglike, make_model, make_obs
from nautilus import Sampler

from hubersed.fitting.map_fits import chi2_parts, get_sps, sfh_from_theta
from hubersed.paths import PATHS

TID = 39627770174637084
OUT = PATHS["RESULTS"] / "2026-09-25_nautilus_tauin_c3k_hr"
CKPT = {
    "MILES": PATHS["RESULTS"] / "2026-09-24_nautilus_tauin" / f"{TID}_miles_seed0.h5",
    "C3K": OUT / f"{TID}_c3k_miles_seed0.h5",
}

z, flux, unc, good = load_data(TID, "miles")
model = make_model(z)
labels = model.theta_labels()
phys = [lab for lab in labels if not lab.startswith("logsfr_ratios")]
prior_range = np.array([model.config_dict[lab]["prior"].range for lab in phys], float)
cue = get_sps(zero_library_resolution=False)["cue"]
gen = np.random.default_rng(0)

# %%
for run, path in CKPT.items():
    # Only reads the checkpoint. The likelihood is never called.
    s = Sampler(
        model.prior_transform,
        lambda x: 0.0,
        n_dim=model.ndim,
        n_live=1000,
        filepath=str(path),
        resume=True,
    )
    assert s.explored, path
    points, log_w, log_l = s.posterior()  # about 2 min each
    w = np.exp(log_w)
    best = points[np.argmax(log_l)]
    draws = [sfh_from_theta(model, points[i]) for i in gen.choice(len(w), size=2000, p=w)]
    out = dict(
        points=points,
        w=w,
        labels=np.array(labels),
        best=best,
        max_lnl=log_l.max(),
        n_eff=s.n_eff,
        n_like=s.n_like,
        prior_labels=np.array(phys),
        prior_range=prior_range,
        sfh_edges=draws[0]["edges_gyr"],
        sfh_ssfr=np.array([d["ssfr"] for d in draws]),
        sfh_cmf=np.array([d["cmf"] for d in draws]),
        z=z,
        good=good,
    )
    if run == "C3K":
        lnl = loglike(best, TID, "miles")
        assert np.isclose(lnl, log_l.max(), atol=0.1), (lnl, log_l.max())
        out["sp_best"], _ = chi2_parts(
            model, best, make_obs(flux, unc, good), cue, np.zeros_like(good)
        )
    np.savez(OUT / f"{TID}_{run}_plotdata.npz", **out)
    print(f"{run}: {len(points)} points, N_eff {s.n_eff:.0f}, max lnL {log_l.max():.1f}, saved")

# %%
