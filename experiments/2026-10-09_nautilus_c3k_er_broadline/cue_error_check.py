"""Cue emulator error and training range at the ER max-L point of 37084.

QUESTION: are the line flux misfits of the ER fit within the Cue emulator's own error, and is
the max-L point inside the emulator's training range?

The emulator error comes from cuejax/data/nn_stats_v0.pkl. SN_quantile has 7 rows of a test-set
S/N per line, where the noise is the emulator error (cue demo notebook). The package does not say
which quantile levels the 7 rows are. prospect uses 1 / SN_quantile[1] as the line error, and
cuejax/data/eline_frac_error.dat holds the same numbers. This script prints 1 / SN for every row.

The young CSP ionizing parameters are found as prospect.sources.nebssp_basis.get_spectrum does
it. After a predict at max-L the young and old CSPs are read back with ssp._csp_young_old and
passed to cuejax.utils.fit_4loglinear_ionparam with ssp.wavelengths. The raw fit, before the clip,
is rebuilt from the returned power-law coefficients.

The training range is read from the input normalization (mean and std) of the Cue networks. For a
parameter drawn uniformly the range is mean -/+ sqrt(3) std. For n_H, which the networks take in
linear units, the log-uniform range with the same mean and std is solved for.

INPUTS: the ER _pred.npz (max-L point), line_flux_check.txt of line_flux_check.py, and the cuejax
package data. SEED: none (deterministic).

COMMAND: from the repository root on the Mac with the C3K_ER build,
``PYTHONPATH=<site with the C3K_ER fsps>:experiments/2026-10-09_nautilus_c3k_er_broadline
SPS_HOME=~/Astronomy_Research/fsps uv run --no-sync python
experiments/2026-10-09_nautilus_c3k_er_broadline/cue_error_check.py``.

OUTPUT: results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/cue_error_check.txt.
"""

import importlib.resources
import pickle

import numpy as np
from cuejax.utils import Ltotal, fit_4loglinear_ionparam
from fit import process_state
from prospect.sources.fake_fsps import frac_line_err
from prospect.sources.nebssp_basis import cue_keys
from scipy.optimize import fsolve

from hubersed.paths import PATHS

TID = 39627770174637084
OUT = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline"
FIG = OUT / str(TID)
STEM = f"{TID}_er_broad_shared_split45_rest9000_seed0"
# rest-frame vacuum wavelengths of the Cue lines in each line_flux_check window
WINDOWS = {
    "[OII]3727": [3727.1, 3729.9],
    "[NeIII]3869": [3869.9],
    "Hdelta": [4102.9],
    "Hgamma": [4341.7],
    "[OIII]4363": [4364.4],
    "Hbeta": [4862.7],
    "[OIII]4959": [4960.3],
    "[OIII]5007": [5008.2],
    "HeI5876": [5877.3],
    "[OI]6300": [6302.0],
    "[NII]6548": [6549.9],
    "Halpha": [6564.6],
    "[NII]6584": [6585.3],
    "[SII]6716": [6718.3],
    "[SII]6731": [6732.7],
}
# output clip of fit_4loglinear_ionparam (cuejax/utils.py)
CLIP = {
    "ionspec_index1": (1, 42),
    "ionspec_index2": (-0.3, 30),
    "ionspec_index3": (-1, 14),
    "ionspec_index4": (-1.7, 8),
    "ionspec_logLratio1": (-1, 10.1),
    "ionspec_logLratio2": (-0.5, 1.9),
    "ionspec_logLratio3": (-0.4, 2.2),
}
ANN_ORDER = [
    "ionspec_index1",
    "ionspec_index2",
    "ionspec_index3",
    "ionspec_index4",
    "ionspec_logLratio1",
    "ionspec_logLratio2",
    "ionspec_logLratio3",
    "gas_logq",
    "gas_nH",
    "gas_logz",
    "gas_logno",
    "gas_logco",
]


def load_data(name):
    """Load one pickle from the cuejax package data."""
    with importlib.resources.files("cuejax.data").joinpath(name).open("rb") as f:
        return pickle.load(f)


def read_misfits():
    """Return R/F of each window from line_flux_check.txt."""
    rf = {}
    for row in (FIG / "line_flux_check.txt").read_text().splitlines():
        parts = row.split()
        if parts and parts[0] in WINDOWS:
            rf[parts[0]] = float(parts[3])
    return rf


def loguniform_range(mean, std, guess):
    """Return the range in log10 x of a log-uniform x with this mean and std of x."""

    def moments(p):
        a, b = p
        m = (10**b - 10**a) / (np.log(10) * (b - a))
        m2 = (10 ** (2 * b) - 10 ** (2 * a)) / (2 * np.log(10) * (b - a))
        return np.array([m - mean, np.sqrt(m2 - m * m) - std])

    return fsolve(moments, guess)


def raw_fit(wave, spec):
    """Run fit_4loglinear_ionparam and rebuild the unclipped indices and log L ratios."""
    fit = fit_4loglinear_ionparam(wave, spec)
    coeff = fit["powerlaw_params"]
    lr = np.diff(np.squeeze(Ltotal(param=coeff.reshape(1, 4, 2))))
    raw = {f"ionspec_index{i + 1}": coeff[i, 0] for i in range(4)}
    raw.update({f"ionspec_logLratio{i + 1}": lr[i] for i in range(3)})
    return fit, raw


def main():
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    # 1. structure of nn_stats_v0.pkl
    st = load_data("nn_stats_v0.pkl")
    say("nn_stats_v0.pkl")
    for k, v in st.items():
        a = np.asarray(v)
        say(f"   {k:14s} {type(v).__name__:8s} shape {a.shape} dtype {a.dtype}")
    wav = np.asarray(st["wav"])
    snq = np.asarray(st["SN_quantile"])
    efe = np.loadtxt(importlib.resources.files("cuejax.data").joinpath("eline_frac_error.dat"))
    say(f"eline_frac_error.dat == 1/SN_quantile[1], same order: {np.allclose(efe, 1 / snq[1])}")
    srt = np.argsort(wav)
    say(
        "prospect frac_line_err == 1/SN_quantile[1][argsort(wav)]: "
        f"{np.allclose(frac_line_err, 1 / snq[1][srt])}"
    )
    say(f"SN_quantile rows increase for every line: {bool(np.all(np.diff(snq, axis=0) >= 0))}")

    # Cue line list: cue_emlines_info.dat is vacuum and sorted, nn_stats wav is air and unsorted
    info = importlib.resources.files("cuejax.data").joinpath("cue_emlines_info.dat")
    vac = np.genfromtxt(info, delimiter=",", usecols=0)
    ratio = vac / wav[srt]
    say(f"vacuum / air of sorted lines: {ratio.min():.6f} to {ratio.max():.6f}")

    # 2. misfit against emulator error
    rf = read_misfits()
    model, obs, sps = process_state(TID, "shared", 45.0, 9000.0)
    pred = np.load(OUT / f"{STEM}_pred.npz")
    labels = [str(x) for x in pred["labels"]]
    assert labels == list(model.theta_labels())
    best = pred["best"]
    spec, _ = model.predict(best, observations=obs, sps=sps)
    good = pred["good"]
    dev = np.abs(spec[0][good] / pred["sp_best"][good] - 1).max()
    say(f"\npredict at best vs saved sp_best, max fractional difference on fitted pixels {dev:.2e}")
    ewave, elum = sps.get_galaxy_elines()
    assert np.allclose(ewave, vac)

    say("\nfractional emulator error 1/SN per SN_quantile row q0..q6 (row 1 is prospect's error)")
    say("R/F: (data - model) / model line flux from line_flux_check.txt")
    say("ratio1 = |R/F| / (1/SN q1), ratio0 = |R/F| / (1/SN q0)")
    say(
        f"{'window':>12s} {'cue lines (air)':>22s} {'R/F':>7s} "
        + " ".join(f"{'q' + str(k):>6s}" for k in range(7))
        + f" {'1/SNstd':>7s} {'ratio1':>6s} {'ratio0':>6s}"
    )
    for name, waves in WINDOWS.items():
        idx = [srt[np.argmin(np.abs(vac - w))] for w in waves]
        jdx = [int(np.argmin(np.abs(vac - w))) for w in waves]
        assert all(abs(vac[j] - w) < 1.0 for j, w in zip(jdx, waves, strict=True))
        # doublet: flux-weighted mean of the line errors (fully correlated errors)
        wgt = elum[jdx] / elum[jdx].sum()
        fe = np.array([wgt @ (1 / snq[k][idx]) for k in range(7)])
        fe_std = wgt @ (1 / np.asarray(st["SN_from_std"])[idx])
        lines = "+".join(f"{wav[i]:.2f}" for i in idx)
        say(
            f"{name:>12s} {lines:>22s} {rf[name]:+7.3f} "
            + " ".join(f"{x:6.3f}" for x in fe)
            + f" {fe_std:7.3f} {abs(rf[name]) / fe[1]:6.1f} {abs(rf[name]) / fe[0]:6.1f}"
        )

    # 3. does the emulator error enter the likelihood
    say("\nline-error switches in the model (prospect SpecModel.predict_init, parse_elines)")
    for k in ("marginalize_elines", "use_eline_nn_unc", "eline_prior_width"):
        say(f"   {k}: {model.params.get(k, 'absent')}")
    # the line covariance, which holds frac_line_err, is only read by fit_mle_elines, and
    # predict_spec calls that only when some pixel belongs to a fitted (marginalized) line
    say(f"   lines marginalized: {bool(model._fit_eline.any())}")
    say(f"   pixels of marginalized lines: {int(model._fit_eline_pixelmask.sum())}")

    # 4. young and old CSP ionizing parameters, as nebssp_basis.get_spectrum computes them
    wave = sps.ssp.wavelengths
    young, old = sps.ssp._csp_young_old
    fit_y, raw_y = raw_fit(wave, young)
    fit_o, raw_o = raw_fit(wave, old)
    same = all(np.isclose(sps.params[k], fit_o[k]) for k in CLIP)
    say(f"\nold CSP refit equals the ionspec values left in sps.params by predict: {same}")

    p = {k: float(np.squeeze(sps.params[k])) for k in cue_keys if k.startswith("gas_")}
    gas_logq = (
        p["gas_logu"] + np.log10(4 * np.pi) + 2 * 19 + p["gas_lognH"] + np.log10(2.99792458e10)
    )

    cont = load_data("fspse_cue_cont_ann.pkl")["normalizations"]
    anns = load_data("fspse_cue_eline_anns.pkl")
    lnorm = {g: anns[g]["normalizations"] for g in anns}
    say("\nnetwork input normalization (mean, std); line nets O3, Ne3, He1, O2, H1, N")
    for i, k in enumerate(ANN_ORDER):
        row = (
            f"   {k:19s} cont {float(cont['input_mu'][i]):9.4f} {float(cont['input_std'][i]):9.4f}"
        )
        for g in ("O3", "Ne3", "He1", "O2", "H1", "N"):
            row += (
                f" | {float(lnorm[g]['input_mu'][i]):8.4f} {float(lnorm[g]['input_std'][i]):8.4f}"
            )
        say(row)

    mu = np.asarray(cont["input_mu"], dtype=float)
    sd = np.asarray(cont["input_std"], dtype=float)
    lo, hi = mu - np.sqrt(3) * sd, mu + np.sqrt(3) * sd
    nh_lo, nh_hi = loguniform_range(mu[8], sd[8], (1.0, 4.0))
    say(
        f"\nn_H log-uniform range from the continuum net moments: log n_H {nh_lo:.3f} to {nh_hi:.3f}"
    )
    lmu = np.asarray(lnorm["O3"]["input_mu"], dtype=float)
    lsd = np.asarray(lnorm["O3"]["input_std"], dtype=float)
    no_lo, no_hi = loguniform_range(lmu[10], lsd[10], (-1.0, 0.7))
    say(
        f"line nets, gas_logno slot: uniform mean -/+ sqrt(3) std {lmu[10] - np.sqrt(3) * lsd[10]:.3f}"
        f" to {lmu[10] + np.sqrt(3) * lsd[10]:.3f}; if the slot were linear N/O drawn log-uniform,"
        f" log N/O {no_lo:.3f} to {no_hi:.3f}"
    )

    say("\nCue inputs at max-L. young and old: fit_4loglinear_ionparam output (raw before clip).")
    say(
        "range: continuum net, mean -/+ sqrt(3) std (log-uniform for n_H); pos = (x - lo)/(hi - lo)"
    )
    say(
        f"{'input':19s} {'young':>9s} {'raw':>9s} {'old':>9s} {'raw':>9s} {'clip':>13s} "
        f"{'range':>17s} {'pos young':>9s} {'flag':>s}"
    )
    vals = {k: (fit_y[k], raw_y[k], fit_o[k], raw_o[k]) for k in CLIP}
    for k in ("gas_logz", "gas_logno", "gas_logco"):
        vals[k] = (p[k],) * 4
    vals["gas_logq"] = (gas_logq,) * 4
    vals["gas_lognH"] = (p["gas_lognH"],) * 4
    for i, k in enumerate(ANN_ORDER):
        key = "gas_lognH" if k == "gas_nH" else k
        a, b = (nh_lo, nh_hi) if k == "gas_nH" else (lo[i], hi[i])
        y, ry, o, ro = vals[key]
        c = CLIP.get(k)
        flag = []
        if c is not None and (np.isclose(y, c[0]) or np.isclose(y, c[1])):
            flag.append("young at clip")
        if c is not None and (np.isclose(o, c[0]) or np.isclose(o, c[1])):
            flag.append("old at clip")
        pos = (y - a) / (b - a)
        if pos < 0 or pos > 1:
            flag.append("outside range")
        elif min(pos, 1 - pos) < 0.05:
            flag.append("within 5% of edge")
        cs = f"{c[0]:g},{c[1]:g}" if c else "-"
        say(
            f"{key:19s} {y:9.4f} {ry:9.4f} {o:9.4f} {ro:9.4f} {cs:>13s} "
            f"{a:8.3f},{b:8.3f} {pos:9.3f} {'; '.join(flag)}"
        )
    say(f"gas_logu {p['gas_logu']:.4f} (gas_logq above is computed from it with R = 1e19 cm)")
    say(f"gas_logqion young {fit_y['gas_logqion']:.4f}, old {fit_o['gas_logqion']:.4f}")

    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "cue_error_check.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
