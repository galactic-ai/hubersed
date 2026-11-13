"""Do the single exposures of gallery105 figures 05 and 67 agree with their coadds?

Adapted from 2026-11-05_exposures_39627770174637084/analysis.py to two targets.
QUESTION: 05 (39627770317246630): its SV3 bright coadd (2 exposures) has negative flux below rest
3400 A, a step near rest 4850 A (observed ~5845 A, near the b/r overlap), Dn4000 5.1 and D(g-r) 0.69,
and its main bright coadd is not flagged (repeats.csv, cleared). Is one SV3 exposure bad, with bright
sky, poor seeing, low efftime or a FIBERSTATUS flag? 67 (39633136522363591): its only coadd (SV3
bright, 3 exposures) has HdA 7.0 +- 1.1 and HgA 6.6 +- 1.1 at Dn4000 1.53 (90th and 99th percentile
of the parent at that break, 2026-11-13_hda_parent_67). Is the Balmer absorption in every exposure?
HYPOTHESIS: none registered. Descriptive, no threshold.
INPUTS: results/2026-11-13_exposures_gallery105/extract_<nn>.npz (extract.py, DR1 iron healpix files)
and conditions.csv (conditions.py). Arms are joined ivar-weighted as spender does. Excess is the
median of exposure minus reference coadd flux (1e-17 erg/s/cm2/A) over good pixels in each rest
window, at the SV3 redrock redshift. Reference: the main coadd for 05, the SV3 coadd for 67 (its
only one). For 67, HdA and HgA of each exposure and coadd as gallery105 seed.py (lick_index copied
unchanged).
SEED: 0 (index noise draws).
COMMAND: ``uv run --no-sync python experiments/2026-11-13_exposures_gallery105/analysis.py``.
RESULT: 2026-11-13, exposures_05.txt and exposures_67.txt (job 3715509). 05: two of four SV3 exposures
have BADPOSITION and are not in the coadd; both used (tile 261, fibre 4277, 2021-04-30, sky g 19.7-19.8)
show blue -3.3 and red +0.6 against the main coadd, so the defect is common to that night, not one
exposure. 67: HdA 6.1, 4.9, 9.0 and HgA 6.3, 4.0, 7.0 in its three exposures (errors 1.6-3.0), excess
against the coadd below 0.35 in every window, so the Balmer absorption is in every exposure.
Assumptions: per-exposure flux calibration; main coadd as the reference for 05.
FIGURES: results/2026-11-13_exposures_gallery105/ exposures_<nn>.txt, full_<nn>.png,
zoom_<nn>_<window>.png.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from spender.data.desi import DESI

from hubersed.paths import PATHS
from hubersed.plotting.style import use_apj_style
from hubersed.sps.utils import _airtovac

OUT = PATHS["RESULTS"] / "2026-11-13_exposures_gallery105"
# Redshifts are the SV3 bright redrock values (SPARCL, sparcl_spectra.npz of 2026-11-10).
TARGETS = {
    "05": dict(tid=39627770317246630, z=0.2050000, ref="main",
               windows={"blue": (3000, 3400), "step": (4750, 4950), "red": (6000, 7000)}),
    "67": dict(tid=39633136522363591, z=0.4781099, ref="sv3",
               windows={"Hdelta": (4060, 4170), "Hgamma": (4280, 4420), "break": (3850, 4050)}),
}  # fmt: skip
N_FLUX, N_RESID = 9, 51  # medfilt widths of the full-spectrum figure, as 11-02
YLABEL = r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]"
BANDS = {
    "hda": ((4083.50, 4122.25), (4041.60, 4079.75), (4128.50, 4161.00)),
    "hga": ((4319.75, 4363.50), (4283.50, 4319.75), (4367.25, 4419.75)),
}
MIN_GOOD, N_MC, SEED = 0.25, 200, 0
COADD_STYLE = {"main": dict(color="k", lw=1.0), "sv3": dict(color="0.45", lw=1.0, ls="--")}


def merge_arms(d, prefix, k=None):
    """Return flux, ivar and mask on the DESI grid, arms joined as spender does.

    Copy of merge_arms in 2026-11-02_exposure_checks/analysis.py. ``prefix`` picks the exposure
    (``sv3`` or ``main``, row ``k``) or coadd (``coadd_sv3`` or ``coadd_main``) arrays.
    """
    wave = DESI._wave_obs.numpy()
    flux, ivar = np.zeros(wave.size), np.zeros(wave.size)
    mask = np.zeros(wave.size, int)
    ncam = np.zeros(wave.size)
    win = {}
    for b in "BRZ":
        imin = np.argmin(np.abs(d[f"{b}_wave"][0] - wave))
        win[b] = np.arange(imin, imin + len(d[f"{b}_wave"]))
        assert np.abs(d[f"{b}_wave"] - wave[win[b]]).max() < 1e-4, "arm grid differs from DESI grid"
        ncam[win[b]] += 1
    for b in "BRZ":
        fl, iv, mk = (d[f"{prefix}_{b}_{x}"] for x in ("flux", "ivar", "mask"))
        if k is not None:
            fl, iv, mk = fl[k], iv[k], mk[k]
        j = win[b]
        good = mk == 0
        ivar[j] += iv * good
        flux[j] += iv * good * fl
        one, many = ncam[j] == 1, ncam[j] > 1
        mask[j[one]] = mk[one]
        mask[j[many]] = mask[j[many]] & mk[many]
    ok = ivar > 0
    flux[ok] /= ivar[ok]
    return flux, ivar, mask


def lick_index(rest, flux, ivar, bands, rng):
    """Return a Lick index, its noise-draw std and the smallest good-pixel fraction of its bands."""
    vac = [(_airtovac(lo), _airtovac(hi)) for lo, hi in bands]
    sel = [(rest >= lo) & (rest <= hi) for lo, hi in vac]
    good = min((ivar[s] > 0).mean() if s.any() else 0.0 for s in sel)
    if good < MIN_GOOD:
        return np.nan, np.nan, good
    sel = [s & (ivar > 0) for s in sel]
    (f_lo, f_hi), (b_lo, b_hi), (r_lo, r_hi) = vac
    wb, wr = (b_lo + b_hi) / 2, (r_lo + r_hi) / 2
    sig = 1 / np.sqrt(np.where(ivar > 0, ivar, np.inf))

    def index(fl):
        cb, cr = (fl[s].mean() for s in sel[1:])
        cont = cb + (cr - cb) * (rest[sel[0]] - wb) / (wr - wb)
        return float((f_hi - f_lo) * (1 - np.mean(fl[sel[0]] / cont)))

    draws = [index(flux + rng.standard_normal(flux.size) * sig) for _ in range(N_MC)]
    return index(flux), float(np.std(draws)), good


def run(nn, cfg, cond, rng):
    """Write the table and figures of one target."""
    d = np.load(OUT / f"extract_{nn}.npz")
    surveys = [str(s) for s in d["surveys"]]
    wave = DESI._wave_obs.numpy()
    rest = wave / (1 + cfg["z"])
    coadds = {}
    for s in surveys:
        f, iv, m = merge_arms(d, f"coadd_{s}")
        coadds[s] = dict(flux=f, ivar=iv, good=(m == 0) & (iv > 0),
                         numexp=int(d[f"coadd_{s}_numexp"]), fstat=int(d[f"coadd_{s}_fiberstatus"]))  # fmt: skip
    ref = coadds[cfg["ref"]]
    exps = []
    for s in surveys:
        for k, expid in enumerate(d[f"{s}_EXPID"]):
            f, iv, m = merge_arms(d, s, k)
            c = cond[int(expid)]
            exps.append(dict(
                survey=s, expid=int(expid), night=int(d[f"{s}_NIGHT"][k]),
                tile=int(d[f"{s}_TILEID"][k]), fiber=int(d[f"{s}_FIBER"][k]),
                fstat=str(d[f"{s}_fiberstatus_names"][k]) or "-", sky=float(c["sky_mag_g_spec"]),
                seeing=float(c["seeing_gfa"]), efft=float(c["efftime_spec"]), flux=f, ivar=iv,
                good=(m == 0) & (iv > 0),
            ))  # fmt: skip

    def excess(flux, good):
        """Return the median of flux minus the reference coadd in each rest window."""
        both = good & ref["good"]
        return {w: float(np.median((flux - ref["flux"])[both & (rest >= lo) & (rest < hi)]))
                for w, (lo, hi) in cfg["windows"].items()}  # fmt: skip

    def indices(flux, ivar):
        """Return HdA and HgA with errors as text."""
        out = []
        for k, b in BANDS.items():
            v, e, _ = lick_index(rest, flux, np.where(np.isfinite(ivar), ivar, 0), b, rng)
            out.append(f"{k} {v:5.2f} +- {e:4.2f}")
        return "  ".join(out)

    windows = cfg["windows"]
    head = "  ".join(f"{w:>10}" for w in windows)
    lines = [
        f"{cfg['tid']} (figure {nn}), rest frame at z {cfg['z']}. Reference coadd: {cfg['ref']}. "
        "Per exposure: night, tile, fibre, FIBERSTATUS, sky g (mag, brighter = smaller), seeing "
        "(arcsec), EFFTIME (s), excess = median exposure - reference (1e-17 erg/s/cm2/A) in rest "
        "windows " + ", ".join(f"{w} {lo}-{hi} A" for w, (lo, hi) in windows.items()) + ".",
        "",
        f"{'survey':6} {'EXPID':>6} {'night':>8} {'tile':>5} {'fibre':>5} {'FIBERSTATUS':>12} "
        f"{'sky g':>5} {'seeing':>6} {'efftime':>7}  {head}",
    ]
    for e in exps:
        ex = excess(e["flux"], e["good"])
        lines.append(
            f"{e['survey']:6} {e['expid']:6d} {e['night']:8d} {e['tile']:5d} {e['fiber']:5d} "
            f"{e['fstat']:>12} {e['sky']:5.2f} {e['seeing']:6.2f} {e['efft']:7.0f}  "
            + "  ".join(f"{ex[w]:+10.3f}" for w in windows)
        )
    for s in surveys:
        if s != cfg["ref"]:
            ex = excess(coadds[s]["flux"], coadds[s]["good"])
            lines.append(
                f"{f'{s} coadd - reference':>70}  " + "  ".join(f"{ex[w]:+10.3f}" for w in windows)
            )
    lines.append("")
    for s in surveys:
        lines.append(f"{s} coadd: COADD_NUMEXP {coadds[s]['numexp']}, "
                     f"COADD_FIBERSTATUS {coadds[s]['fstat']}")  # fmt: skip
    if nn == "67":
        lines += ["", "Balmer indices (rest A, absorption positive):"]
        for e in exps:
            lines.append(f"  {e['survey']} {e['expid']}: {indices(e['flux'], e['ivar'])}")
        for s in surveys:
            lines.append(f"  {s} coadd: {indices(coadds[s]['flux'], coadds[s]['ivar'])}")
    text = "\n".join(lines)
    (OUT / f"exposures_{nn}.txt").write_text(text + "\n")
    print(text + "\n")

    n = {s: sum(e["survey"] == s for e in exps) for s in surveys}
    cmaps = {"sv3": plt.cm.Blues, "main": plt.cm.Oranges}
    seen = {s: 0 for s in surveys}
    for e in exps:
        s = e["survey"]
        e["colour"] = cmaps[s](0.4 + 0.5 * seen[s] / max(n[s] - 1, 1))
        seen[s] += 1

    def panel(ax, sel, smooth_flux, smooth_resid):
        """Draw exposures and coadds on ax[0] and exposure minus reference on ax[1]."""
        for e in exps:
            g = e["good"] & sel
            ax[0].plot(rest[g], medfilt(e["flux"][g], smooth_flux), color=e["colour"], lw=0.7,
                       label=f"{e['survey']} {e['expid']}")  # fmt: skip
            both = g & ref["good"]
            ax[1].plot(rest[both], medfilt((e["flux"] - ref["flux"])[both], smooth_resid),
                       color=e["colour"], lw=0.7)  # fmt: skip
        for s in surveys:
            g = coadds[s]["good"] & sel
            ax[0].plot(rest[g], medfilt(coadds[s]["flux"][g], smooth_flux), label=f"{s} coadd",
                       **COADD_STYLE[s])  # fmt: skip
        ax[1].axhline(0, color="0.3", lw=0.6)
        ax[0].set_ylabel(YLABEL, fontsize=13)
        ax[1].set_xlabel(r"rest wavelength [\AA]", fontsize=13)
        ax[0].legend(frameon=False, fontsize=9, loc="upper right")

    def new_fig():
        """Return a two-row figure with a flux and a residual panel."""
        return plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})  # fmt: skip

    fig, ax = new_fig()
    panel(ax, np.ones(wave.size, bool), N_FLUX, N_RESID)
    ax[1].set_ylabel(f"minus {cfg['ref']} coadd, {N_RESID} px", fontsize=12)
    ax[0].set_title(f"{cfg['tid']}   figure {nn}   z {cfg['z']:.4f}", fontsize=13)
    fig.savefig(OUT / f"full_{nn}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    for w, (lo, hi) in windows.items():
        fig, ax = new_fig()
        panel(ax, (rest >= lo) & (rest < hi), 1, 1)
        ax[1].set_ylabel(f"minus {cfg['ref']} coadd", fontsize=12)
        ax[0].set_title(f"{cfg['tid']}   figure {nn}   {w}   unsmoothed", fontsize=13)
        fig.savefig(OUT / f"zoom_{nn}_{w}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


def main():
    """Run every target."""
    cond = {int(r["expid"]): r for r in csv.DictReader(open(OUT / "conditions.csv"))}
    rng = np.random.default_rng(SEED)
    use_apj_style()
    for nn, cfg in TARGETS.items():
        run(nn, cfg, cond, rng)


if __name__ == "__main__":
    main()
