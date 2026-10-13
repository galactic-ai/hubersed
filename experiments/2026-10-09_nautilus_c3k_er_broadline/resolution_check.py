"""Compare the DESI resolution matrix of 39627770174637084 with the median LSF the fit uses.

QUESTION: profile_check found the blue-arm lines of the ER max-L model about 10-15% too broad
(s_C1s 0.84-0.91) and Halpha too narrow (s_C1s 1.08). Is that because this target's own
resolution differs from the median curve, because the median curve's estimator misreads the
matrix rows, or neither?

HYPOTHESIS: if this target's matrix gives the same sigma as the median curve at the lines, the
matrix is not the cause and the misfit lies in how the model draws the lines.

The fit uses one sigma curve in km/s on the merged grid, C/(2.355 R_median) from
hubersed.sps.lsf.desi_resolution, as map_fits.LSF. R_median was built from the median of about 100
SV3 bright targets, each pixel's sigma taken from the ratio of the first off-diagonal to the
diagonal of the matrix (resolution_to_sigma_kms).

For each arm of this target the script computes

- sig_ratio, the same off-diagonal ratio estimator on this target's matrix;
- sig_col and sig_row, a Gaussian fit to every stored diagonal of each column and each row;
- sig_med, the median curve the fit uses, interpolated to the arm.

At each line it also applies this target's matrix to a line with intrinsic sigma 10 km/s and fits
a Gaussian at the pixel centres to the result, as profile_check fits B. The input line is
integrated over each pixel, because a 10 km/s line is narrower than a pixel and sampling it at the
pixel centres depends on where it falls. Whether the DESI matrix expects pixel averages is not
checked here. The line centre is moved across one pixel in eight steps, which also covers the
redshift uncertainty, and the table gives the mean and range. s_app is the instrumental sigma
this implies, sqrt(sig_tot^2 - 10^2), divided by sig_med. It is the quantity profile_check fitted
as s_B and s_C1s.

The banded storage is read with the desispec convention, offsets running from ndiag//2 down to
-(ndiag//2), so that R[j - offset, j] is data[k, j]. This convention is from memory. Columns and
rows give the same width if it is wrong, but the sign of the centroid shift flips.

INPUTS: the DR1 coadd of healpix 26004 (sv3 bright), given on the command line, and the median
curve tracked in hubersed/sps/desi_lsf_calibration.npz. The profile_check s values are copied
from profile_check.txt. Redshift 0.025319 is the SV3 ZCAT_PRIMARY row of zall-pix-iron-bgs.fits.
SEED: none.

COMMAND: on an ls6 compute node from the repository root,
``.venv/bin/python experiments/2026-10-09_nautilus_c3k_er_broadline/resolution_check.py
/work/11006/nikhilgaruda/ls6/research/spender/DATA/26004/coadd-sv3-bright-26004.fits``.
No FSPS or Cue needed.

OUTPUT: resolution_check.txt, resolution_check.png and resolution_matrix.npz in
results/2026-10-09_nautilus_c3k_er_broadline/39627770174637084/. The npz holds this target's
banded matrix, flux, ivar and mask per arm, and the matrix and flux merged over the arms. The
merged matrix follows desispec.coaddition.coadd_cameras, each column weighted by the unmasked ivar
of its pixel. The merged flux follows spender.data.desi.prepare_spectra, weighted by the masked
ivar, without the skyline mask or the normalisation. Spender's own merged matrix divides by the
summed ivar of each whole arm, so it is not used.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.special import erf

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style
from hubersed.sps.lsf import (
    C_KMS,
    DESI_WAV,
    desi_resolution,
    extract_resolution,
    resolution_to_sigma_kms,
)

TID = 39627770174637084
Z = 0.025319
SIG_INT = 10.0  # intrinsic sigma of the test line in km/s, as SIG_FIX in profile_check
OUT = PATHS["RESULTS"] / "2026-10-09_nautilus_c3k_er_broadline" / str(TID)

# rest-frame vacuum wavelengths from hubersed.sps.utils.load_lines
LINES = {
    "[OII]3729": 3729.875,
    "[NeIII]3869": 3869.857,
    "Hdelta": 4102.892,
    "Hgamma": 4341.684,
    "Hbeta": 4862.683,
    "[OIII]4959": 4960.295,
    "[OIII]5007": 5008.240,
    "Halpha": 6564.614,
    "[NII]6585": 6585.269,
    "[SII]6718": 6718.294,
    "[SII]6733": 6732.678,
}

# s_B and s_C1s from profile_check.txt (ER max-L, 2026-10-12)
PROFILE_S = {
    "[OIII]5007": (1.136, 0.885),
    "[OIII]4959": (1.100, 0.842),
    "Halpha": (1.351, 1.078),
    "Hbeta": (1.121, 0.907),
}


def gauss(x, a, mu, s):
    """Return a Gaussian with amplitude a, centre mu and sigma s."""
    return a * np.exp(-0.5 * ((x - mu) / s) ** 2)


def fit_width(x, y):
    """Fit a Gaussian to a profile and return its centre and sigma, NaN if the fit fails."""
    if not np.all(np.isfinite(y)) or y.max() <= 0:
        return np.nan, np.nan
    try:
        p, _ = curve_fit(gauss, x, y, p0=(y.max(), 0.0, 1.5), maxfev=2000)
    except RuntimeError:
        return np.nan, np.nan
    return p[1], abs(p[2])


def band_profiles(data):
    """Return the column and row profiles of a banded matrix.

    Parameters
    ----------
    data : np.ndarray
        Banded matrix, shape ``(ndiag, n)``.

    Returns
    -------
    x : np.ndarray
        Pixel displacement of each profile point.
    cols, rows : np.ndarray
        Shape ``(n, ndiag)``. NaN where a row runs off the arm edge.
    """
    ndiag, n = data.shape
    c = ndiag // 2
    offsets = c - np.arange(ndiag)
    # column j holds R[j - offset, j], so the output pixel sits at -offset from j
    x = -offsets.astype(float)
    cols = data.T.copy()
    # row i holds R[i, i + offset] = data[k, i + offset], at input displacement +offset
    rows = np.full((n, ndiag), np.nan)
    for k, o in enumerate(offsets):
        j = np.arange(n) + o
        ok = (j >= 0) & (j < n)
        rows[ok, k] = data[k, j[ok]]
    return x, cols, rows[:, ::-1]


def full_matrix(data):
    """Return the dense matrix of a banded resolution matrix, with R[j - offset, j] = data[k, j]."""
    ndiag, n = data.shape
    c = ndiag // 2
    R = np.zeros((n, n))
    for k in range(ndiag):
        o = c - k
        j = np.arange(n)
        i = j - o
        ok = (i >= 0) & (i < n)
        R[i[ok], j[ok]] = data[k, j[ok]]
    return R


def applied_width(wave, R, lam0, sig_kms):
    """Apply R to a pixel-integrated unit line and return the fitted total sigma in km/s.

    The line centre steps across one pixel around lam0. Returns the sigma at each step.
    """
    sig_a = lam0 * sig_kms / C_KMS
    edges = np.concatenate(
        [
            [1.5 * wave[0] - 0.5 * wave[1]],
            0.5 * (wave[1:] + wave[:-1]),
            [1.5 * wave[-1] - 0.5 * wave[-2]],
        ]
    )
    i0 = np.argmin(abs(wave - lam0))
    dpix = wave[i0 + 1] - wave[i0]
    win = slice(max(i0 - 15, 0), i0 + 16)
    out = []
    for shift in np.arange(-0.5, 0.5, 0.125) * dpix:
        lc = lam0 + shift
        line = 0.5 * np.diff(erf((edges - lc) / (np.sqrt(2) * sig_a))) / np.diff(edges)
        y = (R @ line)[win]
        x = wave[win] - lc
        try:
            p, _ = curve_fit(gauss, x, y, p0=(y.max(), 0.0, 1.0), maxfev=2000)
            out.append(abs(p[2]) / lc * C_KMS)
        except RuntimeError:
            out.append(np.nan)
    return np.array(out)


def save_matrix(coadd, path):
    """Save this target's per-arm and merged matrix, flux, ivar and mask to an npz file."""
    out = {"targetid": TID, "coadd": str(coadd)}
    with fits.open(coadd) as h:
        idx = np.where(h["FIBERMAP"].data["TARGETID"] == TID)[0][0]
        for b in ("b", "r", "z"):
            B = b.upper()
            out[f"{b}_wave"] = np.asarray(h[f"{B}_WAVELENGTH"].data, dtype=float)
            out[f"{b}_flux"] = np.asarray(h[f"{B}_FLUX"].data[idx], dtype=float)
            out[f"{b}_ivar"] = np.asarray(h[f"{B}_IVAR"].data[idx], dtype=float)
            out[f"{b}_mask"] = np.asarray(h[f"{B}_MASK"].data[idx])
            out[f"{b}_res"] = np.asarray(h[f"{B}_RESOLUTION"].data[idx], dtype=float)
    wave = DESI_WAV
    ndiag = max(out[f"{b}_res"].shape[0] for b in "brz")
    ivar_un = np.zeros(wave.size)
    ivar = np.zeros(wave.size)
    flux = np.zeros(wave.size)
    rdata = np.zeros((ndiag, wave.size))
    for b in "brz":
        imin = np.argmin(abs(out[f"{b}_wave"][0] - wave))
        w = np.arange(imin, imin + out[f"{b}_wave"].size)
        assert np.allclose(out[f"{b}_wave"], wave[w], atol=1e-4)
        iv, good = out[f"{b}_ivar"], out[f"{b}_mask"] == 0
        nd = out[f"{b}_res"].shape[0]
        ivar_un[w] += iv
        ivar[w] += iv * good
        flux[w] += iv * good * out[f"{b}_flux"]
        rdata[(ndiag - nd) // 2 : (ndiag + nd) // 2, w] += iv * out[f"{b}_res"]
    ok = ivar > 0
    flux[ok] /= ivar[ok]
    ok = ivar_un > 0
    rdata[:, ok] /= ivar_un[ok]
    out.update(wave=wave, res=rdata, flux=flux, ivar=ivar)
    np.savez(path, **out)


def main(coadd):
    """Read the matrix, compute the widths and write the table and figure."""
    use_apj_style()
    OUT.mkdir(parents=True, exist_ok=True)
    save_matrix(coadd, OUT / "resolution_matrix.npz")
    arms = extract_resolution(coadd, TID)
    wave_obs = DESI_WAV.astype(np.float32)
    sig_med_full = (C_KMS / (2.355 * desi_resolution(wave_obs))).astype(np.float64)

    res = {}
    for b in sorted(arms, key=lambda b: arms[b][0][0]):
        wave, data = arms[b]
        wave = np.asarray(wave, dtype=float)
        data = np.asarray(data, dtype=float)
        dpix = np.gradient(wave)
        to_kms = dpix / wave * C_KMS
        x, cols, rows = band_profiles(data)
        cfit = np.array([fit_width(x, y) for y in cols])
        rfit = np.array([fit_width(x, y) for y in rows])
        res[b] = dict(
            wave=wave,
            data=data,
            x=x,
            cols=cols,
            rows=rows,
            sig_ratio=resolution_to_sigma_kms(wave, data),
            sig_col=cfit[:, 1] * to_kms,
            sig_row=rfit[:, 1] * to_kms,
            mu_col=cfit[:, 0],
            sig_med=np.interp(wave, wave_obs, sig_med_full),
        )

    lines = []
    lines.append(
        f"DESI resolution matrix of {TID} (coadd {coadd}) against the median LSF of the fit."
    )
    lines.append(
        "Sigmas in km/s. sig_med is map_fits.LSF. sig_ratio is the off-diagonal ratio estimator on"
    )
    lines.append(
        "this target. sig_col and sig_row are Gaussian fits to all diagonals of the column and row."
    )
    lines.append(
        "mu_col is the column centroid in pixels. tot is the fitted total sigma of a pixel-integrated"
    )
    lines.append(
        f"{SIG_INT:g} km/s line after this target's matrix, mean over 8 sub-pixel centres, with"
    )
    lines.append(
        "s_app = sqrt(tot^2 - 10^2) / sig_med, mean and min-max over the centres. s_B and s_C1s are"
    )
    lines.append("from profile_check.txt.")
    lines.append("")
    lines.append(
        "Per arm, median and 16-84% of ratios to sig_med over the arm (sky spikes included)."
    )
    for b, r in res.items():
        parts = [f"  {b}: {r['wave'][0]:.0f}-{r['wave'][-1]:.0f} A, ndiag {r['data'].shape[0]}"]
        for k in ("sig_ratio", "sig_col", "sig_row"):
            q = np.nanpercentile(r[k] / r["sig_med"], [16, 50, 84])
            parts.append(f"{k}/med {q[1]:.3f} [{q[0]:.3f},{q[2]:.3f}]")
        lines.append(", ".join(parts))
    lines.append("")
    hdr = (
        f"{'line':>12} {'arm':>3} {'lam_obs':>8} {'sig_med':>7} {'sig_ratio':>9} {'sig_col':>7} "
        f"{'sig_row':>7} {'mu_col':>6} {'tot':>6} {'s_app':>6} {'s_app range':>13} "
        f"{'s_B':>6} {'s_C1s':>6}"
    )
    lines.append(hdr)
    dense = {}
    for name, rest in LINES.items():
        lam = rest * (1 + Z)
        inarm = [b for b, r in res.items() if r["wave"][0] + 5 < lam < r["wave"][-1] - 5]
        for b in inarm:
            r = res[b]
            i = np.argmin(abs(r["wave"] - lam))
            if b not in dense:
                dense[b] = full_matrix(r["data"])
            tot = applied_width(r["wave"], dense[b], lam, SIG_INT)
            med = r["sig_med"][i]
            s_app = np.sqrt(tot**2 - SIG_INT**2) / med
            sb, sc = PROFILE_S.get(name, (np.nan, np.nan))
            rng = f"{np.nanmin(s_app):.3f}-{np.nanmax(s_app):.3f}"
            lines.append(
                f"{name:>12} {b:>3} {lam:8.1f} {med:7.2f} {r['sig_ratio'][i]:9.2f} {r['sig_col'][i]:7.2f} "
                f"{r['sig_row'][i]:7.2f} {r['mu_col'][i]:+6.3f} {np.nanmean(tot):6.2f} "
                f"{np.nanmean(s_app):6.3f} {rng:>13} {sb:6.3f} {sc:6.3f}"
            )
    text = "\n".join(lines)
    print(text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "resolution_check.txt").write_text(text + "\n")

    fig, ax = plt.subplots(3, 1, figsize=(7.5, 9))
    colors = {"b": "C0", "r": "C3", "z": "C2"}
    for b, r in res.items():
        c = colors.get(b, "k")
        ax[0].plot(r["wave"], r["sig_med"], color="k", lw=0.8)
        ax[0].plot(r["wave"], r["sig_ratio"], color=c, lw=0.5, alpha=0.7)
        ax[0].plot(r["wave"], r["sig_col"], color=c, lw=0.5, ls="--")
        ax[1].plot(r["wave"], r["sig_ratio"] / r["sig_med"], color=c, lw=0.5, alpha=0.7)
        ax[1].plot(r["wave"], r["sig_col"] / r["sig_med"], color=c, lw=0.5, ls="--")
    for name in PROFILE_S:
        lam = LINES[name] * (1 + Z)
        for a in ax[:2]:
            a.axvline(lam, color="0.6", lw=0.5, ls=":")
    ax[0].set_ylabel(r"$\sigma$ [km/s]")
    ax[0].set_ylim(15, 80)
    ax[1].set_ylabel(r"$\sigma$ / median")
    ax[1].set_ylim(0.8, 1.2)
    ax[1].axhline(1, color="k", lw=0.5)
    ax[1].set_xlabel(r"$\lambda_{\rm obs}$ [$\AA$]")
    ax[0].plot([], [], color="k", label="median (fit)")
    ax[0].plot([], [], color="0.4", label="this target, ratio")
    ax[0].plot([], [], color="0.4", ls="--", label="this target, Gaussian fit to column")
    ax[0].legend(fontsize=7)
    for name in PROFILE_S:
        lam = LINES[name] * (1 + Z)
        for b, r in res.items():
            if r["wave"][0] + 5 < lam < r["wave"][-1] - 5:
                i = np.argmin(abs(r["wave"] - lam))
                y = r["cols"][i] / np.nansum(r["cols"][i])
                (h,) = ax[2].plot(r["x"], y, "o", ms=3, label=f"{name} ({b})")
                p0 = fit_width(r["x"], y)
                xx = np.linspace(r["x"][0], r["x"][-1], 200)
                yy = gauss(xx, 1.0, p0[0], p0[1])
                yy *= np.nanmax(y) / yy.max()
                ax[2].plot(xx, yy, color=h.get_color(), lw=0.7)
    ax[2].set_yscale("log")
    ax[2].set_ylim(1e-5, 1)
    ax[2].set_xlabel("pixel offset")
    ax[2].set_ylabel("column of R (normalised)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "resolution_check.png", dpi=150)
    print(f"wrote {OUT / 'resolution_check.txt'} and resolution_check.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("coadd", help="DR1 coadd FITS file holding the target")
    main(ap.parse_args().coadd)
