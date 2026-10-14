"""Build the noiseless mocks of the broad-line injection test and check the injector.

The checks, all at the truth point unless stated, are written to validation.txt.

1. InjectedLineModel with our own line structure (S0) reproduces ResolutionMatrixModel to
   rtol 1e-6, at the truth point and at five jittered points, for the shared and the free
   forbidden broad fraction.
2. The no-broad fit model (F-none) reduces correctly. With eline_fbroad = 0 it equals the shared
   model at eline_fbroad = 0 and the injector with one Gaussian per line, and it does not depend
   on eline_sigma_broad or eline_vbroad.
3. Line fluxes. For each scenario the line spectrum on the fine grid is integrated over
   frequency, line by line, and compared with the intended flux (Cue flux times the sum of the
   component weights). For S1 and S2 the intended flux is the Cue flux. For S3 the forbidden
   lines of [N II], [S II], [O I] and [O III] change by design.
4. The EmFit numbers in injection.py match the csv.

It also gives the chi2 distance of each noiseless mock from the S0 mock on the fitted pixels.

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-13_nautilus_resolution_matrix
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-13_broadline_injection/mocks.py``

OUTPUT: mocks.npz and validation.txt in results/2026-10-13_broadline_injection/.
"""

import warnings

import numpy as np
from injection import (
    OUT,
    SCENARIOS,
    InjectedLineModel,
    check_emfit,
    jitter_start,
    line_group,
    make_fit_params,
    s3_offset,
    setup,
    theta_for,
)
from model_setup import get_cue

from hubersed.sps.resolution_matrix import LIGHTSPEED, line_profile
from hubersed.sps.resolution_model import ResolutionMatrixModel

MAGGIE_TO_1E17 = 3631e-23 / 1e-17  # maggies * Hz to 1e-17 erg/s/cm2
RTOL = 1e-6


def predict(model, theta, obs, cue):
    """Return the predicted spectrum of ``model`` at ``theta``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict(theta, observations=obs, sps=cue)[0][0]


def max_rel(a, b, sel):
    """Return the largest |a - b| / |b| over ``sel``."""
    return float(np.max(np.abs(a[sel] - b[sel]) / np.abs(b[sel])))


def line_integrals(model, fine):
    """Integrate each injected line's components over frequency on the fine grid.

    Returns
    -------
    got : np.ndarray
        Flux of each line on the fine grid, in 1e-17 erg/s/cm2.
    want : np.ndarray
        Cue flux times the sum of the component weights, in the same units.
    inside : np.ndarray of bool
        Lines whose components lie fully inside the fine grid and that were added.
    """
    comps = model.line_components()
    elum = model._eline_lum * model.flux_norm() / (1 + model._zred)
    step = (fine[-1] - fine[0]) / (fine.size - 1)
    n = elum.size
    got, want = np.zeros(n), np.zeros(n)
    inside = np.zeros(n, dtype=bool)
    use = model._use_eline & model._fix_eline
    for i in np.where(use)[0]:
        mu = model._ewave_obs[i]
        full = True
        any_added = False
        for weight, sigma, shift in comps[i]:
            m = mu * (1 + shift / 299792.458)
            half = 8.0 * m * sigma / 299792.458
            full &= (m - half > fine[0]) and (m + half < fine[-1])
            sl, g = line_profile(fine, m, sigma)
            if g.size:
                any_added = True
                got[i] += weight * elum[i] * np.sum(g * LIGHTSPEED / fine[sl] ** 2) * step
            want[i] += weight * elum[i]
        inside[i] = full and any_added
    return got * MAGGIE_TO_1E17, want * MAGGIE_TO_1E17, inside


def main():
    """Run the checks, write validation.txt and save the noiseless mocks."""
    s = setup()
    obs, cue, good = s["obs"], get_cue(), s["good"]
    out = ["Injector validation, broad-line injection test of 39627770174637084", ""]
    out.append(check_emfit())

    p_sh = make_fit_params(s["z"], "shared", s["hypers"])
    ref = ResolutionMatrixModel(p_sh)
    t_sh = theta_for(ref, s["truth"])
    assert np.allclose(t_sh, [s["truth"][k] for k in ref.theta_labels()], rtol=0, atol=0)
    m_ref = predict(ref, t_sh, obs, cue)
    fine = obs[0].resampled()[1]

    # 1. S0 injector against the model, at the truth and at jittered points
    out += ["", "1. InjectedLineModel(S0) vs ResolutionMatrixModel, max |rel diff|"]
    inj = InjectedLineModel(p_sh, "S0")
    worst = 0.0
    for k in range(6):
        th = t_sh if k == 0 else jitter_start(ref, t_sh, seed=500 + k, frac=0.05)
        a, b = predict(inj, th, obs, cue), predict(ref, th, obs, cue)
        allpix = np.isfinite(b) & (b != 0)
        r_all, r_fit = max_rel(a, b, allpix), max_rel(a, b, good)
        worst = max(worst, r_all)
        tag = "truth" if k == 0 else f"jitter seed {500 + k}"
        out.append(f"   shared, {tag}: all pixels {r_all:.2e}, fitted pixels {r_fit:.2e}")
        assert np.allclose(a, b, rtol=RTOL, atol=0), tag
    p_fr = make_fit_params(s["z"], "free", s["hypers"])
    ref_fr, inj_fr = ResolutionMatrixModel(p_fr), InjectedLineModel(p_fr, "S0")
    for fforb in (0.05, 0.4):
        th = theta_for(ref_fr, s["truth"], fill={"eline_fbroad_forb": fforb})
        a, b = predict(inj_fr, th, obs, cue), predict(ref_fr, th, obs, cue)
        r_all = max_rel(a, b, np.isfinite(b) & (b != 0))
        worst = max(worst, r_all)
        out.append(f"   free, truth with eline_fbroad_forb = {fforb}: all pixels {r_all:.2e}")
        assert np.allclose(a, b, rtol=RTOL, atol=0)
    out.append(f"   PASS: worst {worst:.2e} < rtol {RTOL:g}")

    # 2. F-none reduction
    out += ["", "2. F-none reduction (eline_fbroad = 0)"]
    p_no = make_fit_params(s["z"], "none", s["hypers"])
    m_no = ResolutionMatrixModel(p_no)
    t_no = theta_for(m_no, s["truth"])
    a = predict(m_no, t_no, obs, cue)
    th0 = t_sh.copy()
    th0[ref.theta_index["eline_fbroad"]] = 0.0
    b = predict(ref, th0, obs, cue)
    c = predict(inj, th0, obs, cue)
    one = [len(x) for x in inj.line_components()]
    assert max(one) == 1, "with f_b = 0 the injector must build one Gaussian per line"
    p_no2 = make_fit_params(s["z"], "none", s["hypers"])
    p_no2["eline_sigma_broad"] = dict(p_no2["eline_sigma_broad"], init=400.0)
    p_no2["eline_vbroad"] = dict(p_no2["eline_vbroad"], init=150.0)
    d = predict(ResolutionMatrixModel(p_no2), t_no, obs, cue)
    allpix = np.isfinite(b) & (b != 0)
    for name, x in (
        ("shared model at f_b = 0", b),
        ("one-Gaussian injector", c),
        ("F-none with sigma_b 400, v_b 150", d),
    ):
        r = max_rel(a, x, allpix)
        out.append(f"   F-none vs {name}: max |rel diff| {r:.2e}")
        assert np.allclose(a, x, rtol=RTOL, atol=0), name
    out.append(f"   F-none free parameters: {m_no.theta_labels()}")
    out.append("   PASS")

    # 3. noiseless mocks and line fluxes
    out += ["", "3. Line fluxes on the fine grid (1e-17 erg/s/cm2), by scenario"]
    mocks, mult = {"S0": m_ref}, {}
    names = np.char.strip(np.asarray(ref.emline_info["name"]).astype(str))
    for sc in SCENARIOS:
        model = InjectedLineModel(p_sh, sc)
        mocks[sc] = predict(model, t_sh, obs, cue)
        if sc == "S0":
            assert np.allclose(mocks[sc], m_ref, rtol=RTOL, atol=0)
        got, want, inside = line_integrals(model, fine)
        mult[sc] = model.flux_multipliers()
        cue_flux = model._eline_lum * model.flux_norm() / (1 + model._zred) * MAGGIE_TO_1E17
        sel = inside & (want > 0)
        dev = np.abs(got[sel] / want[sel] - 1)
        out.append(
            f"   {sc}: {sel.sum()} lines fully inside the fine grid; max |integral / intended - 1| "
            f"{dev.max():.2e}; summed lines {got[sel].sum():.3f} vs intended {want[sel].sum():.3f} "
            f"vs Cue {cue_flux[sel].sum():.3f}"
        )
        lo, hi = obs[0].reach()
        for i in np.where((want > 0) & (got > 0) & ~inside)[0]:
            out.append(
                f"      partly off the fine grid: {names[i]} at {model._ewave_obs[i]:.1f} A "
                f"observed (fitted pixels reach {lo:.1f}-{hi:.1f} A), "
                f"{got[i] / want[i]:.4f} of its flux on the grid"
            )
        changed = np.where(sel & ~np.isclose(mult[sc], 1.0, rtol=1e-12, atol=0))[0]
        if sc in ("S0", "S1", "S2"):
            assert changed.size == 0, f"{sc} must keep every Cue flux"
            assert dev.max() < 1e-10
        for i in changed:
            out.append(
                f"      {names[i]:18s} total / Cue = {mult[sc][i]:.4f} (offset {s3_offset(names[i]):+.3f} dex"
                f" on the broad part), Cue {cue_flux[i]:.2f} -> {got[i]:.2f}"
            )
        resolved = model._resolved_lines
        out.append(
            f"      lines after R' summed over all pixels, relative to S0: "
            f"{resolved.sum() / mocks_lines_s0(ref, t_sh, obs, cue):.5f}"
        )
        diff = (mocks[sc] - m_ref)[good] / s["unc"][good]
        out.append(f"      chi2 of the noiseless mock against the S0 mock: {np.sum(diff**2):.1f}")
    out += ["", "S2 broad fraction by line (lines with Cue flux > 1e-17 inside the fit range):"]
    m2 = InjectedLineModel(p_sh, "S2")
    predict(m2, t_sh, obs, cue)
    flux2 = m2._eline_lum * m2.flux_norm() / (1 + m2._zred) * MAGGIE_TO_1E17
    lo, hi = obs[0].reach()
    for i, comps in enumerate(m2.line_components()):
        if flux2[i] > 1 and lo < m2._ewave_obs[i] < hi:
            fb = comps[1][0] if len(comps) > 1 else 0.0
            out.append(f"   {names[i]:18s} {line_group(names[i]):14s} f_b = {fb:.4f}")
    np.savez(
        OUT / "mocks.npz",
        scenarios=np.array(SCENARIOS),
        mocks=np.array([mocks[sc] for sc in SCENARIOS]),
        multipliers=np.array([mult[sc] for sc in SCENARIOS]),
        line_names=names,
        unc=s["unc"],
        good=good,
        truth_labels=np.array(ref.theta_labels()),
        truth_theta=t_sh,
    )
    text = "\n".join(out) + "\n"
    print(text)
    (OUT / "validation.txt").write_text(text)


_S0_LINES = {}


def mocks_lines_s0(ref, theta, obs, cue):
    """Return the summed S0 line spectrum after R' over all pixels, computed once."""
    if "v" not in _S0_LINES:
        predict(ref, theta, obs, cue)
        _S0_LINES["v"] = float(ref._resolved_lines.sum())
    return _S0_LINES["v"]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
