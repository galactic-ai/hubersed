"""
Plot MAP fit diagnostics from a map_chi2 *_full.pkl.

Per galaxy, makes:
  (1) MAP model vs DESI data + residual panel        -> map_fit_<id>.png
  (2) SFH: SFR vs lookback time + mass assembly       -> map_sfh_<id>.png
  (3) SFR/Mtot (Gyr^-1) vs cosmic time                -> map_ssfr_cosmictime_<id>.png
and prints a continuum-vs-line chi^2 split + the MAP theta_dict rails.

Usage:
  python tmp/plot_map_fits.py results/map_chi2_cue_outliers_lsf_full.pkl            # all results
  python tmp/plot_map_fits.py results/..._full.pkl --id 39627896930701693           # one by TARGETID
  python tmp/plot_map_fits.py results/..._full.pkl --outdir results/mapfit_plots
Needs only numpy, astropy, matplotlib (no FSPS).
"""

import os, pickle, argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

LINES = {
    "[OII]": 3727,
    "Hδ": 4102,
    "Hγ": 4340,
    "Hβ": 4861,
    "[OIII]": 5007,
    "[OI]": 6300,
    "Hα": 6563,
    "[NII]": 6584,
    "[SII]": 6716,
}


def make_stochastic_agebins(z):
    """Same 10-bin agebins used in fit_config / make_cue_model_sed."""
    t = cosmo.age(z).to_value(u.Gyr)
    ab = np.zeros((10, 2))
    ab[0] = [0.001, 0.005]
    ab[1] = [0.005, 0.01]
    e = np.geomspace(0.01, 0.95 * t, 9)
    for i in range(2, 10):
        ab[i] = [e[i - 2], e[i - 1]]
    return np.log10(ab * 1e9)


def logsfr_ratios_to_masses(logmass, logsfr_ratios, agebins):
    """prospect.models.transforms.logsfr_ratios_to_masses (j=0 = most recent)."""
    nb = agebins.shape[0]
    sr = 10 ** np.clip(logsfr_ratios, -10, 10)
    dt = 10 ** agebins[:, 1] - 10 ** agebins[:, 0]
    coeffs = np.array(
        [
            (1.0 / np.prod(sr[:i])) * (np.prod(dt[1 : i + 1]) / np.prod(dt[:i]))
            for i in range(nb)
        ]
    )
    return 10**logmass / coeffs.sum() * coeffs


def chi2_split(r):
    """Continuum vs emission-line chi^2 split for one result dict."""
    wave = np.asarray(r["wave"]) if "wave" in r else None
    return None  # wave is stored at top level; handled in main


def plot_modeldata(r, wave, outdir):
    z = r["z"]
    wr = wave / (1 + z)
    flux = np.asarray(r["flux"])
    unc = np.asarray(r["unc"])
    mod = np.asarray(r["model"])
    m = np.asarray(r["mask"])
    resid = np.full_like(flux, np.nan)
    ok = m & (unc > 0)
    resid[ok] = (flux[ok] - mod[ok]) / unc[ok]
    fl = flux.copy()
    fl[~m] = np.nan
    mo = mod.copy()
    mo[~m] = np.nan

    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    a1.plot(wr, fl, lw=0.6, color="0.4", label="DESI data")
    a1.plot(wr, mo, lw=0.8, color="C3", label="MAP model")
    a1.set_ylabel("flux (maggies)")
    a1.legend(loc="upper right", fontsize=9)
    a1.set_title(
        f"idx {r.get('gidx', '?')}  TID {r['id']}  z={z:.3f}  χ²_red={r['chi2_red']:.2f}"
    )
    ymax = np.nanpercentile(fl, 99.5)
    a1.set_ylim(0, ymax * 1.15)
    for nm, w0 in LINES.items():
        a1.axvline(w0, ls=":", color="C0", lw=0.5, alpha=0.6)
        a1.text(
            w0,
            ymax * 1.08,
            nm,
            fontsize=6,
            rotation=90,
            va="top",
            ha="center",
            color="C0",
        )
    a2.axhspan(-1, 1, color="0.85")
    a2.axhline(0, color="k", lw=0.5)
    a2.plot(wr, resid, lw=0.5, color="C3")
    a2.set_ylim(-12, 12)
    a2.set_ylabel("(data-model)/σ")
    a2.set_xlabel("rest wavelength (Å)")
    a2.set_xlim(np.nanmin(wr[m]), np.nanmax(wr[m]))
    fig.tight_layout()
    p = os.path.join(outdir, f"map_fit_{r['id']}.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # continuum vs line chi2 split
    line_win = np.zeros_like(wave, bool)
    for w0 in LINES.values():
        line_win |= np.abs(wr - w0) < 7.0
    inl = m & line_win
    inc = m & (~line_win)
    c_all = np.nansum(resid[m] ** 2)
    c_l = np.nansum(resid[inl] ** 2)
    c_c = np.nansum(resid[inc] ** 2)
    print(
        f"  chi2: line {100 * c_l / c_all:.0f}% ({c_l / max(inl.sum(), 1):.2f}/pix)  "
        f"cont {100 * c_c / c_all:.0f}% ({c_c / max(inc.sum(), 1):.2f}/pix)"
    )
    return p


def _sfh(r):
    z = r["z"]
    td = r["theta_dict"]
    lr = np.atleast_1d(np.asarray(td["logsfr_ratios"], float))
    lm = float(np.atleast_1d(td["logmass"])[0])
    ab = make_stochastic_agebins(z)
    m = logsfr_ratios_to_masses(lm, lr, ab)
    dt = 10 ** ab[:, 1] - 10 ** ab[:, 0]
    sfr = m / dt
    lb = 10**ab / 1e9
    mid = 0.5 * (lb[:, 0] + lb[:, 1])
    return z, lm, ab, m, sfr, lb, mid


def plot_sfh(r, outdir):
    z, lm, ab, m, sfr, lb, mid = _sfh(r)
    mwa = np.sum(m * mid) / np.sum(m)
    dt = 10 ** ab[:, 1] - 10 ** ab[:, 0]
    w100 = np.clip(np.minimum(lb[:, 1], 0.1) - lb[:, 0], 0, None) / (
        lb[:, 1] - lb[:, 0]
    )
    ssfr = np.sum(sfr * w100 * dt) / 1e8 / 10**lm
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a1.step(mid, sfr, where="mid", color="C3", lw=1.5)
    a1.scatter(mid, sfr, color="C3", s=20, zorder=5)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.invert_xaxis()
    a1.set_xlabel("lookback time (Gyr)")
    a1.set_ylabel("SFR (M$_\\odot$/yr)")
    a1.set_title(
        f"MAP SFH  TID {r['id']}  z={z:.3f}  logM={lm:.1f}\nmwa={mwa:.1f} Gyr  log sSFR$_{{100}}$={np.log10(ssfr):.1f}"
    )
    o = np.argsort(-mid)
    cum = np.cumsum(m[o]) / m.sum()
    a2.step(mid[o], cum, where="mid", color="C0", lw=1.5)
    a2.scatter(mid[o], cum, color="C0", s=20)
    a2.set_xscale("log")
    a2.invert_xaxis()
    a2.set_ylim(0, 1.05)
    a2.grid(alpha=0.3)
    a2.set_xlabel("lookback time (Gyr)")
    a2.set_ylabel("cumulative mass frac (old→young)")
    a2.set_title("Mass assembly")
    fig.tight_layout()
    p = os.path.join(outdir, f"map_sfh_{r['id']}.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_ssfr_cosmictime(r, outdir, floor=1e-4):
    z, lm, ab, m, sfr, lb, mid = _sfh(r)
    t_obs = cosmo.age(z).to_value(u.Gyr)
    ssfr_gyr = np.clip(
        sfr / 10**lm * 1e9, floor, None
    )  # 1/Gyr, floored so quench renders
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(ab.shape[0]):
        t_lo = t_obs - lb[i, 1]
        t_hi = t_obs - lb[i, 0]
        ax.plot([t_lo, t_hi], [ssfr_gyr[i], ssfr_gyr[i]], color="C3", lw=2)
        if i < ab.shape[0] - 1:
            tj = t_obs - lb[i, 0]
            ax.plot(
                [tj, tj], [ssfr_gyr[i], ssfr_gyr[i + 1]], color="C3", lw=2, alpha=0.6
            )
    ax.set_yscale("log")
    ax.set_xlabel("cosmic time (Gyr)")
    ax.set_ylabel("SFR / M$_{tot}$  (Gyr$^{-1}$)")
    ax.set_title(
        f"MAP SFH  TID {r['id']}  z={z:.3f}  logM={lm:.1f}  (t_obs={t_obs:.1f} Gyr)"
    )
    ax.axvline(t_obs, ls=":", color="k", lw=1)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = os.path.join(outdir, f"map_ssfr_cosmictime_{r['id']}.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("--id", type=int, default=None, help="single TARGETID")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    d = pickle.load(open(args.pkl, "rb"))
    wave = np.asarray(d["wave"])
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.pkl))
    os.makedirs(outdir, exist_ok=True)
    res = [r for r in d["results"] if r.get("status") == "ok"]
    if args.id is not None:
        res = [r for r in res if int(r["id"]) == args.id]
    for r in res:
        print(f"TID {r['id']}  z={r['z']:.3f}  chi2_red={r['chi2_red']:.2f}")
        plot_modeldata(r, wave, outdir)
        plot_sfh(r, outdir)
        plot_ssfr_cosmictime(r, outdir)
        td = r.get("theta_dict", {})
        rails = {
            k: np.round(np.atleast_1d(np.asarray(v)), 3).tolist() for k, v in td.items()
        }
        print("  theta:", rails)
    print(f"\nsaved {3 * len(res)} plots to {outdir}")


if __name__ == "__main__":
    main()
