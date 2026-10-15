"""Direct-method (T_e) oxygen abundance from the line-flux table, following Scholte+26 choices.

Recipe (Scholte et al. 2026, arXiv:2601.02463, main.tex:197-202, unless noted):

- PyNeb with Scholte's atomic data (tables/pyneb_data.tex): O+ A FFT04 and collision strengths
  Kal09, O++ A FFT04 and SSB14, S+ RGJ19 and TZ10, H+ SH95 case B.
- Dust: A_V from Halpha/Hbeta and Hgamma/Hbeta against case B at 1e4 K and 100 cm^-3, combined by
  inverse variance and floored at 0. Curve: the Wild+11 line curve with mu ``--mu`` (default 0.4),
  or CCM89 R_V 3.1 with ``--curve ccm89`` (Scholte's choice). Here we differ from Scholte, who fit
  A_V jointly in the PyNeb model.
- n_e and T(O+) together from [SII]6731/6716 (EmFit) and [OII]3727+/7325+ (all four 2P-2D lines),
  with PyNeb getCrossTemDen. Here we differ: Scholte use [OII]3726/3729 for n_e, which the
  FastSpecFit split cannot give at our resolution.
- T(O++) from [OIII]4363/5007 at that n_e where 4363 and 5007 have S/N > 3. Otherwise the inverted
  T_low = 0.7 T_high + 3000 K (Scholte's fiducial), or ``--trel scholte`` for their fit T(O+) =
  0.648 T(O++) + 3270 K (main.tex:264-266).
- O/H = O+/H+ from [OII]3727/Hbeta at T(O+) plus O++/H+ from [OIII]5007/Hbeta at T(O++), no ICF.
- Errors: Monte Carlo over the flux errors, N_MC draws (seed SEED). Here we differ: Scholte sample a
  forward model with UltraNest. A galaxy is kept if its 12+log(O/H) 16-84 width is < 0.6 dex.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/direct.py
[--targetid T] [--curve wild|ccm89] [--mu 0.4] [--trel campbell|scholte]`` from the repository
root. Without --targetid it runs every galaxy of lineflux_table.fits.

OUTPUT: direct_<curve>_<trel>.fits (all galaxies) in results/2026-10-14_cue_lineflux_calibration/,
or a printout for one --targetid.
"""

import argparse
import warnings

import numpy as np
import pyneb as pn
from astropy.table import Table

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
N_MC = 500
SEED = 0
SNR_4363 = 3.0
WIDTH_MAX = 0.6
WAVE = {
    "HA": 6562.8,
    "HB": 4861.3,
    "HG": 4340.5,
    "OII3727": 3727.4,
    "OII7325": 7325.0,
    "OIII5007": 5006.8,
    "OIII4363": 4363.2,
}
DRAWN = ("HA", "HB", "HG", "OII3727", "OII7325", "OIII5007", "OIII4363", "SII6716", "SII6731")

pn.atomicData.setDataFile("o_ii_atom_FFT04.dat")
pn.atomicData.setDataFile("o_ii_coll_Kal09.dat")
pn.atomicData.setDataFile("o_iii_atom_FFT04.dat")
pn.atomicData.setDataFile("o_iii_coll_SSB14.dat")
pn.atomicData.setDataFile("s_ii_atom_RGJ19.dat")
pn.atomicData.setDataFile("s_ii_coll_TZ10.dat")
O2, O3, H1 = pn.Atom("O", 2), pn.Atom("O", 3), pn.RecAtom("H", 1)
DIAGS = pn.Diagnostics()
for d in ("[OII] 3727+/7325+b", "[SII] 6731/6716", "[OIII] 4363/5007"):
    DIAGS.addDiag(d)


def curve(wave, kind, mu):
    """Return A_lambda/A_V of the chosen line curve."""
    if kind == "ccm89":
        rc = pn.RedCorr(law="CCM89", R_V=3.1)
        return np.asarray(rc.X(np.atleast_1d(wave)), float) / 3.1
    x = np.asarray(wave, float) / 5500.0
    return (1.0 - mu) * x**-1.3 + mu * x**-0.7


def av_balmer(f, e, k):
    """Return A_V per draw from Ha/Hb and Hg/Hb, inverse-variance combined, floored at 0."""
    hb = H1.getEmissivity(1e4, 100.0, label="4_2")
    est, wts = [], []
    for line, label in (("HA", "3_2"), ("HG", "5_2")):
        intr = H1.getEmissivity(1e4, 100.0, label=label) / hb
        with np.errstate(divide="ignore", invalid="ignore"):
            av = 2.5 * np.log10(f[line] / f["HB"] / intr) / (k["HB"] - k[line])
        var = (2.5 / np.log(10) / (k["HB"] - k[line])) ** 2 * (
            (e[line] / np.nanmedian(f[line])) ** 2 + (e["HB"] / np.nanmedian(f["HB"])) ** 2
        )
        est.append(av)
        wts.append(np.full_like(av, 1.0 / var))
    est, wts = np.array(est), np.array(wts)
    good = np.isfinite(est)
    av = np.nansum(np.where(good, est * wts, 0), axis=0) / np.sum(np.where(good, wts, 0), axis=0)
    return np.clip(av, 0.0, None)


def t_high_from_low(t_low, trel):
    """Invert the low/high zone temperature relation."""
    if trel == "scholte":
        return (t_low - 3270.0) / 0.648
    return (t_low - 3000.0) / 0.7


def one(row, kind, mu, trel, rng):
    """Return the Monte Carlo draws of n_e, T(O+), T(O++), A_V and 12+log(O/H) for one galaxy."""
    f = {x: rng.normal(float(row[f"F_{x}"]), float(row[f"E_{x}"]), N_MC) for x in DRAWN}
    e = {x: float(row[f"E_{x}"]) for x in DRAWN}
    k = {x: curve(WAVE[x], kind, mu) for x in WAVE}
    av = av_balmer(f, e, k)
    c = {x: f[x] * 10 ** (0.4 * av * k[x]) for x in WAVE}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_low, n_e = DIAGS.getCrossTemDen(
            "[OII] 3727+/7325+b",
            "[SII] 6731/6716",
            c["OII3727"] / c["OII7325"],
            f["SII6731"] / f["SII6716"],
        )
        snr = float(row["F_OIII4363"]) / float(row["E_OIII4363"])
        use_4363 = np.isfinite(snr) and snr > SNR_4363
        if use_4363:
            t_high = O3.getTemDen(c["OIII4363"] / c["OIII5007"], den=n_e, to_eval="L(4363)/L(5007)")
        else:
            t_high = t_high_from_low(t_low, trel)
        op = O2.getIonAbundance(
            100 * c["OII3727"] / c["HB"], tem=t_low, den=n_e, to_eval="L(3726)+L(3729)", Hbeta=100
        )
        opp = O3.getIonAbundance(100 * c["OIII5007"] / c["HB"], tem=t_high, den=n_e, wave=5007)
    oh = 12 + np.log10(op + opp)
    return {"n_e": n_e, "t_low": t_low, "t_high": t_high, "av": av, "oh": oh}, use_4363


def summary(d):
    """Return the 16, 50 and 84 percentiles of each entry, ignoring NaN, and the NaN fraction."""
    return {k: (np.nanpercentile(v, [16, 50, 84]), np.mean(~np.isfinite(v))) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targetid", type=int)
    ap.add_argument("--curve", choices=("wild", "ccm89"), default="wild")
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--trel", choices=("campbell", "scholte"), default="campbell")
    a = ap.parse_args()
    t = Table.read(OUT / "lineflux_table.fits")
    rng = np.random.default_rng(SEED)
    if a.targetid is not None:
        row = t[t["TARGETID"] == a.targetid]
        assert len(row) == 1, f"{a.targetid} not in lineflux_table.fits"
        d, use_4363 = one(row[0], a.curve, a.mu, a.trel, rng)
        print(f"{a.targetid}: curve {a.curve}, T(O++) from {'4363' if use_4363 else a.trel}")
        for key, (q, bad) in summary(d).items():
            print(f"{key:>7} {q[1]:10.3f} [{q[0]:10.3f}, {q[2]:10.3f}]  NaN fraction {bad:.3f}")
        return
    rows = []
    for row in t:
        d, use_4363 = one(row, a.curve, a.mu, a.trel, rng)
        s = summary(d)
        rows.append(
            [int(row["TARGETID"]), use_4363] + [x for key in d for x in (*s[key][0], s[key][1])]
        )
    names = ["TARGETID", "T_FROM_4363"]
    names += [
        f"{key.upper()}_{p}"
        for key in ("n_e", "t_low", "t_high", "av", "oh")
        for p in ("P16", "P50", "P84", "NAN")
    ]
    out = Table(rows=rows, names=names)
    out["KEEP"] = (out["OH_P84"] - out["OH_P16"] < WIDTH_MAX) & (out["OH_NAN"] < 0.5)
    name = f"direct_{a.curve}{'' if a.curve == 'ccm89' else f'{a.mu:.1f}'}_{a.trel}.fits"
    out.write(OUT / name, overwrite=True)
    print(f"wrote {name}: {len(out)} galaxies, kept {int(out['KEEP'].sum())}")


if __name__ == "__main__":
    main()
