"""Evaluate the ER max-L model of 39627770174637084 with this target's own DESI resolution matrix.

QUESTION: does the chi2 of the ER max-L point change, and do the line-core residuals at
[OIII]5007, [OIII]4959, Hbeta and Halpha go away, when the model is blurred with this target's
resolution matrix instead of the median Gaussian LSF? No refit.

HYPOTHESIS: resolution_check found this target's matrix columns about 10-20% narrower than the
median at its strong lines, close to the profile_check scales of the blue lines. If the matrix is
right, the blue-line cores improve with it. Halpha (profile_check s_C1s 1.08) should not.

The model is built at the max-L point as in fit.py. Six versions of the spectrum are compared.

- current, the fit's own path. Prospect smooths the continuum by sqrt(inst^2 - lib^2) with the
  median sigma and adds the lines as Gaussians of width sqrt(intrinsic^2 + inst^2) sampled at the
  pixel centres.
- R, the target's merged matrix from resolution_matrix.npz. The continuum is smoothed by the
  stellar velocity dispersion with the library resolution taken out in quadrature,
  sqrt(sigma_smooth^2 - lib^2), and averaged over each pixel. The lines have their intrinsic
  widths only and are integrated over each pixel. The matrix is applied to the sum.
- R_nolibfix, as R but with the full sigma_smooth, so the library resolution stays in the
  continuum. This measures the leftover cost of applying a matrix to a library spectrum.
- R_colnorm, as R with each column of the matrix rescaled to sum to one. The rows of the DESI
  matrix sum to one, but its columns at this target's strong lines sum to 0.83-0.92, so R removes
  flux from unresolved lines. R_colnorm keeps the shape of the matrix and conserves flux.
- R_fine, the raw matrix resampled row by row onto a 0.05 A model grid with a cubic spline (Guy et
  al. 2023, appendix on resampling the resolution matrix; resolution_model.py). The lines and the
  continuum go through R' on the fine grid without rebinning first. R and R_colnorm average the
  model over each 0.8 A pixel before the matrix, which that appendix calls incorrect for
  unresolved lines.
- sigma_t, the fit's own path with this target's per-pixel Gaussian sigma (a Gaussian fit to each
  column of the merged matrix) in place of the median.

The chi2 is reported in total, per arm, on the line pixels and the continuum pixels, and in the
four profile_check windows with the chi of the core pixels. Each model's line is also fitted with
profile_check's fits B and C1s, to give the scale s of the median sigma that the model line itself
has, next to the data values in profile_check.txt.

A flux table gives, for the lines of the four windows, the fraction of each model line's flux that
R and R_fine keep, and the observed line flux over the current model's line flux, with the background taken
from the current model. With an emline file of the DESI pipeline (desispec.emlinefit, a Gaussian
on a constant continuum fitted to the coadd flux without the resolution matrix) the script also
lists its fluxes next to the observed fluxes summed here.

INPUTS: the ``_pred.npz`` of predict.py, resolution_matrix.npz of resolution_check.py, and
optionally the emline file of healpix 26004 (sv3 bright) as the first argument. SEED: none.

COMMAND: from the repository root on the Mac with the C3K_ER build, as profile_check.py,
``PYTHONPATH=~/.claude/jobs/16ec6e43/tmp/er_path:experiments/2026-10-09_nautilus_c3k_er_broadline
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-09_nautilus_c3k_er_broadline/eval_resolution.py [emline file]``. Building the
Cue source takes a few minutes.

OUTPUT: eval_resolution.txt and eval_resolution.png in
results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from fit import process_state
from profile_check import CORE, FB, OUT, REST_MAX, SIG_FIX, SPLIT, STEM, TID, WINDOWS, fit_window
from resolution_check import PROFILE_S, band_profiles, fit_width
from resolution_model import apply_segments, fine_grid, intrinsic_line_segments, resampled_matrix
from scipy.special import erf

from hubersed.plotting.style import use_apj_style
from hubersed.sps.parameter_file import build_obs

C_KMS = 299792.458
C_AA = 2.998e18  # speed of light in Angstrom/s, as prospect's lightspeed
ARMS = {"b": (3600.0, 5780.0), "r": (5780.0, 7570.0), "z": (7570.0, 9825.0)}


def edges_of(wave):
    """Return the pixel edges of a wavelength grid, at the midpoints between centres."""
    mid = 0.5 * (wave[1:] + wave[:-1])
    return np.concatenate([[2 * wave[0] - mid[0]], mid, [2 * wave[-1] - mid[-1]]])


def pixel_average(wave, flux, edges):
    """Average a finely sampled spectrum over each pixel, from its cumulative trapezoid integral."""
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(wave))])
    return np.diff(np.interp(edges, wave, cum)) / np.diff(edges)


def apply_banded(data, x):
    """Return R @ x for a banded DESI matrix, with R[j - offset, j] = data[k, j]."""
    ndiag, n = data.shape
    out = np.zeros(n)
    j = np.arange(n)
    for k in range(ndiag):
        i = j - (ndiag // 2 - k)
        ok = (i >= 0) & (i < n)
        np.add.at(out, i[ok], data[k, ok] * x[ok])
    return out


def unit_line(edges, mu, sigma_kms):
    """Return a unit-flux Gaussian line in f_nu (per Hz), averaged over each pixel.

    The Gaussian is in velocity about ``mu``, as prospect's get_eline_gaussians, so it is a
    Gaussian in wavelength with sigma mu * sigma_kms / c.
    """
    sl = mu * sigma_kms / C_KMS
    centre = 0.5 * (edges[1:] + edges[:-1])
    flam = 0.5 * np.diff(erf((edges - mu) / (np.sqrt(2) * sl))) / np.diff(edges)
    return flam * centre**2 / C_AA


def intrinsic_lines(model, edges):
    """Return each added line at its intrinsic width, pixel-integrated, in maggies.

    Returns
    -------
    inds : np.ndarray
        Indices of the added lines in the model's line list.
    lines : np.ndarray
        Shape (n_line, n_pix).
    """
    inds = np.where(model._fix_eline & model._valid_eline)[0]
    prof, _, _, inst2 = model._profile_widths()
    narrow = np.sqrt(np.clip(prof**2 - inst2, 0, None))
    fb = model._broad_fraction()
    s_b = model._p("eline_sigma_broad", 150.0)
    v_b = model._p("eline_vbroad", 0.0)
    elum = model._eline_lum * model.flux_norm() / (1 + model._zred)
    lines = np.zeros((inds.size, edges.size - 1))
    for n, i in enumerate(inds):
        mu = model._ewave_obs[i]
        g = (1 - fb[i]) * unit_line(edges, mu, narrow[i])
        if fb[i] > 0:
            g += fb[i] * unit_line(edges, mu * (1 + v_b / C_KMS), s_b)
        lines[n] = elum[i] * g
    return inds, lines


def continuum(model, sigma):
    """Return the model continuum smoothed by a velocity dispersion ``sigma``, on the library grid."""
    keep = model.params["sigma_smooth"]
    try:
        model.params["sigma_smooth"] = np.atleast_1d(sigma)
        spec = model.losvd_smoothing(model._wave, model._norm_spec)
        spec = model.add_dla(model._wave, spec)
        spec = model.add_damping_wing(model._wave, spec)
    finally:
        model.params["sigma_smooth"] = keep
    return spec


def column_sigma(wave, data):
    """Return the Gaussian sigma in km/s of each column of a banded matrix, NaN where the fit fails."""
    x, cols, _ = band_profiles(data)
    sig = np.array([fit_width(x, y)[1] for y in cols])
    return sig * np.gradient(wave) / wave * C_KMS


def chi2(flux, unc, model_spec, m):
    """Return the chi2 of ``model_spec`` on the pixels ``m``."""
    return float(np.sum(((flux - model_spec)[m] / unc[m]) ** 2))


def main(emline_file=None):
    use_apj_style()
    d = np.load(OUT / f"{STEM}_pred.npz")
    rm = np.load(OUT / str(TID) / "resolution_matrix.npz")
    wave, flux, unc = d["wave"], d["flux"], d["unc"]
    good = d["good"].astype(bool)
    z = float(d["z"])
    assert np.allclose(rm["wave"], wave)
    edges = edges_of(wave)

    # current path, as the fit
    model, obs, sps = process_state(TID, FB, SPLIT, REST_MAX)
    best = d["best"]
    pred = model.predict(best, observations=obs, sps=sps)[0][0]
    assert np.allclose(pred, d["sp_best"], rtol=1e-3), "the model does not reproduce sp_best"
    assert np.all(np.asarray(model._speccal) == 1.0)
    assert not np.any(model._fit_eline & model._valid_eline)
    spec_obs = obs[0]
    obs_wave = model.observed_wave(model._wave, do_wavecal=False)
    cont_cur = spec_obs.instrumental_smoothing(
        obs_wave, model._smooth_spec, libres=model._library_resolution
    )
    emask = model._fix_eline_pixelmask
    lines_cur = np.zeros((model._fix_eline_spec.shape[1], wave.size))
    lines_cur[:, emask] = model._fix_eline_spec.T
    assert np.allclose(cont_cur + lines_cur.sum(0), pred, rtol=1e-6)

    # library sampling and resolution in the fitted range
    sel = (obs_wave > wave[0]) & (obs_wave < wave[-1])
    dlib = np.median(np.diff(obs_wave[sel]))
    lib = float(np.median(model._library_resolution[sel]))
    sig_star = float(np.atleast_1d(model.params["sigma_smooth"])[0])
    sig_fix = np.sqrt(sig_star**2 - lib**2)

    # target matrix
    res = rm["res"]
    inds, lines_int = intrinsic_lines(model, edges)
    assert np.array_equal(inds, np.where(model._fix_eline & model._valid_eline)[0])
    lines_R = np.array([apply_banded(res, s) for s in lines_int])
    cont_R = apply_banded(res, pixel_average(obs_wave, continuum(model, sig_fix), edges))
    cont_R0 = apply_banded(res, pixel_average(obs_wave, continuum(model, sig_star), edges))
    colsum = res.sum(axis=0)
    res_cn = res / np.where(colsum > 0, colsum, 1.0)
    lines_cn = np.array([apply_banded(res_cn, s) for s in lines_int])
    cont_cn = apply_banded(res_cn, pixel_average(obs_wave, continuum(model, sig_fix), edges))
    fine = fine_grid(rm["wave"], res.shape[0])
    rp = resampled_matrix(rm["wave"], res, fine)
    cont_Rf = rp @ np.interp(fine, obs_wave, continuum(model, sig_fix))
    inds_f, segs = intrinsic_line_segments(model, fine)
    assert np.array_equal(inds_f, inds)
    lines_Rf = np.array([apply_segments(rp, c) for c in segs])
    elum = model._eline_lum[inds] * model.flux_norm() / (1 + model._zred)
    flux_check = lines_int @ (C_AA / edges[:-1] - C_AA / edges[1:]) / elum

    # current path with this target's per-pixel column sigma
    sig_t = column_sigma(wave, res)
    sig_med = np.asarray(spec_obs.resolution, dtype=float)
    bad = ~np.isfinite(sig_t) | (sig_t < 13.0)
    sig_t[bad] = sig_med[bad]
    obs_t = build_obs(spec=flux, unc=unc, mask=good, resolution=sig_t, wavelength=wave)
    pred_t = model.predict(best, observations=obs_t, sps=sps)[0][0]
    lines_t = np.zeros_like(lines_cur)
    lines_t[:, model._fix_eline_pixelmask] = model._fix_eline_spec.T
    model.predict(best, observations=obs, sps=sps)  # restore the current caches

    specs = {
        "current": (pred, lines_cur),
        "R": (cont_R + lines_R.sum(0), lines_R),
        "R_nolibfix": (cont_R0 + lines_R.sum(0), lines_R),
        "R_colnorm": (cont_cn + lines_cn.sum(0), lines_cn),
        "R_fine": (cont_Rf + lines_Rf.sum(0), lines_Rf),
        "sigma_t": (pred_t, lines_t),
    }

    # report
    rest = wave / (1 + z)
    linepix = good & emask
    contpix = good & ~emask
    out = []
    out.append(f"ER max-L of {TID} with this target's resolution matrix. No refit.")
    out.append(
        f"Library sampling {dlib:.3f} A (obs frame, median in 3600-9824 A), DESI pixel "
        f"{np.median(np.diff(wave)):.3f} A. Library sigma {lib:.2f} km/s, sigma_smooth "
        f"{sig_star:.2f} km/s, with the library taken out {sig_fix:.2f} km/s."
    )
    out.append(
        "Line flux check, each intrinsic line summed over pixels in frequency over its Cue flux: "
        f"min {flux_check.min():.4f}, max {flux_check.max():.4f}."
    )
    out.append(
        f"sigma_t: {bad.sum()} pixels with a failed column fit or sigma below 13 km/s take the median."
    )
    out.append("")
    regions = {"all": good, "lines": linepix, "continuum": contpix}
    for a, (lo, hi) in ARMS.items():
        regions[f"arm {a}"] = good & (wave >= lo) & (wave < hi)
    for key, ((a, b), _) in WINDOWS.items():
        regions[key] = good & (rest >= a) & (rest < b)
    out.append(f"{'chi2 on':>12} {'npix':>5} " + " ".join(f"{k:>11}" for k in specs))
    for name, m in regions.items():
        vals = " ".join(f"{chi2(flux, unc, s, m):11.1f}" for s, _ in specs.values())
        out.append(f"{name:>12} {m.sum():5d} {vals}")
    out.append("")

    # line fluxes
    names = np.char.strip(np.asarray(model.emline_info["name"]).astype(str))[inds]
    dnu = C_AA / edges[:-1] - C_AA / edges[1:]
    kfac = np.median(
        flux[good] / (rm["flux"][good] * wave[good] ** 2)
    )  # maggies per 1e-17 f_lambda
    to_flam = 1.0 / (kfac * wave**2)
    dlam = np.diff(edges)
    out.append(
        "Line fluxes. keep_R and keep_Rf are the fractions of each model line's flux that R and "
        "R_fine keep (sum over all pixels in frequency). obs/cur is the data minus the current background over the current "
        "line, on the fitted pixels of the window. F_obs and F_cur are those two in 1e-17 erg/s/cm2."
    )
    out.append(
        f"{'window':>11} {'keep_R':>7} {'keep_Rf':>7} {'obs/cur':>8} {'F_obs':>9} {'F_cur':>9}"
    )
    keep, keepf, fobs = {}, {}, {}
    for key, ((a, b), lname) in WINDOWS.items():
        j = int(np.where(names == lname)[0][0])
        m = good & (rest >= a) & (rest < b)
        keep[key] = float(np.sum(lines_R[j] * dnu) / np.sum(lines_int[j] * dnu))
        keepf[key] = float(np.sum(lines_Rf[j] * dnu) / np.sum(lines_int[j] * dnu))
        y = flux - (pred - lines_cur[j])
        ratio = float(np.sum((y * dnu)[m]) / np.sum((lines_cur[j] * dnu)[m]))
        fobs[key] = float(np.sum((y * to_flam * dlam)[m]))
        fcur = float(np.sum((lines_cur[j] * to_flam * dlam)[m]))
        out.append(
            f"{key:>11} {keep[key]:7.4f} {keepf[key]:7.4f} {ratio:8.4f} {fobs[key]:9.1f} {fcur:9.1f}"
        )
    out.append(
        f"Hbeta/Halpha after R over before: {keep['Hbeta'] / keep['Halpha']:.4f}, after R_fine "
        f"{keepf['Hbeta'] / keepf['Halpha']:.4f}. [OIII]5007/4959 after R_fine over before: "
        f"{keepf['[OIII]5007'] / keepf['[OIII]4959']:.4f} "
        "(profile_check notes: data Hbeta/Halpha 4.7% below the current model)."
    )
    if emline_file is not None:
        with fits.open(emline_file) as h:
            e = h[1].data[h[1].data["TARGETID"] == TID][0]
        pairs = {
            "OIII (4959+5007)": ("OIII", fobs["[OIII]4959"] + fobs["[OIII]5007"]),
            "HALPHA": ("HALPHA", fobs["Halpha"]),
            "HBETA": ("HBETA", fobs["Hbeta"]),
        }
        out.append(f"emline file {emline_file}")
        out.append(f"{'line':>17} {'emline FLUX':>12} {'+-':>8} {'F_obs here':>11}")
        for label, (col, fo) in pairs.items():
            fe, iv = float(e[f"{col}_FLUX"]), float(e[f"{col}_FLUX_IVAR"])
            err = 1 / np.sqrt(iv) if iv > 0 else np.nan
            out.append(f"{label:>17} {fe:12.1f} {err:8.2g} {fo:11.1f}")
    out.append("")

    # line scale s of each model's own line, and core chi
    inst_med = np.sqrt(model._profile_widths()[3])[inds]
    out.append(
        "Scale s of the median sigma that each model's own line has, from profile_check fits B "
        f"and C1s (intrinsic {SIG_FIX:g} km/s). Data values from profile_check.txt."
    )
    out.append(
        f"{'window':>11} {'data s_B':>8} {'data s_C1s':>10} "
        + " ".join(f"{k + ' s_B':>13} {k + ' s_C1s':>14}" for k in specs)
    )
    core_rows = []
    panels = {}
    for key, ((a, b), lname) in WINDOWS.items():
        j = int(np.where(names == lname)[0][0])
        mu0, inst = float(model._ewave_obs[inds[j]]), float(inst_med[j])
        m = good & (rest >= a) & (rest < b)
        w = wave[m]
        row = f"{key:>11} {PROFILE_S[key][0]:8.3f} {PROFILE_S[key][1]:10.3f}"
        dv = (w / mu0 - 1) * C_KMS
        core = np.abs(dv) < CORE
        core_rows.append(f"{key}  dv [km/s] " + " ".join(f"{v:+6.0f}" for v in dv[core]))
        panels[key] = dict(dv=dv, data=flux[m], unc=unc[m], models={})
        for k, (s, ls) in specs.items():
            y = ls[j][m]
            f0 = np.trapezoid(ls[j], wave)
            fw = fit_window(w, y, np.full(y.size, 1e-3 * np.abs(y).max()), wave, mu0, inst, f0)
            row += f" {fw['B'][0][2]:13.3f} {fw['C1s'][0][4]:14.3f}"
            c = ((flux - s)[m] / unc[m])[core]
            core_rows.append(f"  chi {k:>10} " + " ".join(f"{v:+6.1f}" for v in c))
            panels[key]["models"][k] = s[m]
        out.append(row)
    out.append("")
    out.append(f"Chi of the pixels within {CORE:g} km/s of each line centre")
    out.extend(core_rows)
    text = "\n".join(out) + "\n"
    print(text)
    dest = OUT / str(TID)
    (dest / "eval_resolution.txt").write_text(text)

    colors = {
        "current": "#b2182b",
        "R": "#2166ac",
        "R_nolibfix": "#92c5de",
        "R_colnorm": "#e08214",
        "R_fine": "#542788",
        "sigma_t": "#1b7837",
    }
    fig, ax = plt.subplots(
        2,
        len(panels),
        figsize=(4 * len(panels), 5.5),
        gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05},
        sharex="col",
    )
    for i, (key, p) in enumerate(panels.items()):
        ax[0, i].errorbar(p["dv"], p["data"], p["unc"], fmt="o", ms=3, color="0.3", lw=0.8)
        for k, s in p["models"].items():
            c2 = float(np.sum(((p["data"] - s) / p["unc"]) ** 2))
            ax[0, i].plot(p["dv"], s, color=colors[k], lw=1.0, label=rf"{k}, $\chi^2$ = {c2:.0f}")
            ax[1, i].plot(p["dv"], (p["data"] - s) / p["unc"], color=colors[k], lw=1.0, marker=".")
        ax[0, i].set_title(key)
        ax[0, i].legend(frameon=False, fontsize=7)
        for h in (-2, 2):
            ax[1, i].axhline(h, color="0.5", ls="--", lw=0.6)
        ax[1, i].axhline(0, color="0.3", lw=0.6)
        ax[1, i].set_xlim(-400, 400)
        ax[1, i].set_xlabel(r"velocity from max-L narrow centre [km s$^{-1}$]")
    ax[0, 0].set_ylabel("flux [maggies]")
    ax[1, 0].set_ylabel(r"$\chi$")
    fig.savefig(dest / "eval_resolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
