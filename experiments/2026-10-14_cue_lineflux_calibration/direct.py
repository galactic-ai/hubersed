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
  FastSpecFit split cannot give at our resolution. n_e is bounded to Scholte's prior range 10-1000
  cm^-3 (tables/pyneb_table.tex:10): a draw outside it, or with an [SII] ratio beyond the ratio at
  the bound, gets n_e at the bound and T(O+) from [OII] alone at that n_e. NE_BOUND_FRAC is the
  fraction of such draws.
- T(O++) from [OIII]4363/5007 at that n_e where 4363 and 5007 have S/N > 3. Otherwise the inverted
  T_low = 0.7 T_high + 3000 K (Scholte's fiducial), or ``--trel scholte`` for their fit T(O+) =
  0.648 T(O++) + 3270 K (main.tex:264-266).
- O/H = O+/H+ from [OII]3727/Hbeta at T(O+) plus O++/H+ from [OIII]5007/Hbeta at T(O++), no ICF.
- Errors: Monte Carlo over the flux errors, N_MC draws with the seed (SEED, TARGETID), so a galaxy
  gets the same draws alone or in the full run. Here we differ: Scholte sample a forward model with
  UltraNest. A galaxy is kept if its 12+log(O/H) 16-84 width is < 0.6 dex.

COMMAND: ``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/direct.py
[--targetid T] [--curve wild|ccm89] [--mu 0.4] [--trel campbell|scholte]`` from the repository
root. Without --targetid it runs every galaxy of lineflux_table.fits over ``--workers`` processes
(default 4), writes the output every SAVE_EVERY galaxies, and on a rerun skips galaxies already in
the output file (meta COMPLETE is True once all are done). ``--ids FILE`` runs only the TARGETIDs
listed in FILE and adds its stem to the output name.

OUTPUT: direct_<curve>_<trel>_r2[_<ids stem>].fits in results/2026-10-14_cue_lineflux_calibration/,
or a printout for one --targetid.
"""

import argparse
import multiprocessing as mp
import warnings
from pathlib import Path

import numpy as np
import pyneb as pn
from astropy.table import Table

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
N_MC = 500
SEED = 0
SAVE_EVERY = 50
RECIPE = 2
NE_MIN, NE_MAX = 10.0, 1000.0
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
O2, O3, S2, H1 = pn.Atom("O", 2), pn.Atom("O", 3), pn.Atom("S", 2), pn.RecAtom("H", 1)
OII_TEM = "(L(3726)+L(3729))/(I(4,2)+I(4,3)+I(5,2)+I(5,3))"
SII_DEN = "L(6731)/L(6716)"
SII_LO, SII_HI = (
    float(S2.getEmissivity(1e4, n, wave=6731) / S2.getEmissivity(1e4, n, wave=6716))
    for n in (NE_MIN, NE_MAX)
)
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


def one(row, kind, mu, trel):
    """Return the Monte Carlo draws of n_e, T(O+), T(O++), A_V and 12+log(O/H) for one galaxy.

    ``row`` maps TARGETID and F_<line>, E_<line> of DRAWN to numbers.
    """
    rng = np.random.default_rng([SEED, int(row["TARGETID"])])
    f = {x: rng.normal(row[f"F_{x}"], row[f"E_{x}"], N_MC) for x in DRAWN}
    e = {x: row[f"E_{x}"] for x in DRAWN}
    k = {x: curve(WAVE[x], kind, mu) for x in WAVE}
    av = av_balmer(f, e, k)
    c = {x: f[x] * 10 ** (0.4 * av * k[x]) for x in WAVE}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_o2 = c["OII3727"] / c["OII7325"]
        r_s2 = f["SII6731"] / f["SII6716"]
        t_low, n_e = DIAGS.getCrossTemDen("[OII] 3727+/7325+b", "[SII] 6731/6716", r_o2, r_s2)
        t_low, n_e = np.array(t_low, float), np.array(n_e, float)
        bad = ~np.isfinite(n_e)
        bound = np.zeros(N_MC, bool)
        for ne, sel in (
            (NE_MIN, (bad & (r_s2 < SII_LO)) | (n_e < NE_MIN)),
            (NE_MAX, (bad & (r_s2 > SII_HI)) | (n_e > NE_MAX)),
        ):
            if sel.any():
                n_e[sel] = ne
                t_low[sel] = O2.getTemDen(r_o2[sel], den=ne, to_eval=OII_TEM)
                bound |= sel
        snr = row["F_OIII4363"] / row["E_OIII4363"]
        use_4363 = bool(np.isfinite(snr) and snr > SNR_4363)
        if use_4363:
            t_high = O3.getTemDen(c["OIII4363"] / c["OIII5007"], den=n_e, to_eval="L(4363)/L(5007)")
        else:
            t_high = t_high_from_low(t_low, trel)
        op = O2.getIonAbundance(
            100 * c["OII3727"] / c["HB"], tem=t_low, den=n_e, to_eval="L(3726)+L(3729)", Hbeta=100
        )
        opp = O3.getIonAbundance(100 * c["OIII5007"] / c["HB"], tem=t_high, den=n_e, wave=5007)
        oh = 12 + np.log10(op + opp)
    d = {"n_e": n_e, "t_low": t_low, "t_high": t_high, "av": av, "oh": oh}
    return d, use_4363, float(bound.mean())


def summary(d):
    """Return the 16, 50 and 84 percentiles of each entry, ignoring NaN, and the NaN fraction."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            k: (np.nanpercentile(v, [16, 50, 84]), np.mean(~np.isfinite(v))) for k, v in d.items()
        }


def as_dict(row):
    """Return TARGETID and the DRAWN fluxes and errors of a table row as plain numbers, NaN if masked."""
    out = {"TARGETID": int(row["TARGETID"])}
    for x in DRAWN:
        for p in ("F", "E"):
            out[f"{p}_{x}"] = float(np.ma.filled(np.ma.asarray(row[f"{p}_{x}"], float), np.nan))
    return out


def work(args):
    """Return one output row: TARGETID, T_FROM_4363 and the summary numbers."""
    row, kind, mu, trel = args
    d, use_4363, frac = one(row, kind, mu, trel)
    s = summary(d)
    nums = [float(x) for key in d for x in (*s[key][0], s[key][1])]
    return [row["TARGETID"], use_4363, *nums, frac]


NAMES = (
    ["TARGETID", "T_FROM_4363"]
    + [
        f"{key.upper()}_{p}"
        for key in ("n_e", "t_low", "t_high", "av", "oh")
        for p in ("P16", "P50", "P84", "NAN")
    ]
    + ["NE_BOUND_FRAC"]
)


def write(rows, path, complete):
    """Write the output rows with the KEEP flag and the run settings."""
    out = Table(rows=rows, names=NAMES)
    out["KEEP"] = (out["OH_P84"] - out["OH_P16"] < WIDTH_MAX) & (out["OH_NAN"] < 0.5)
    out.meta.update({"N_MC": N_MC, "SEED": SEED, "RECIPE": RECIPE, "COMPLETE": complete})
    out.write(path, overwrite=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targetid", type=int)
    ap.add_argument("--curve", choices=("wild", "ccm89"), default="wild")
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--trel", choices=("campbell", "scholte"), default="campbell")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ids")
    a = ap.parse_args()
    t = Table.read(OUT / "lineflux_table.fits")
    tag = ""
    if a.ids:
        ids = np.loadtxt(a.ids, dtype=np.int64, ndmin=1)
        t = t[np.isin(t["TARGETID"], ids)]
        tag = f"_{Path(a.ids).stem}"
    if a.targetid is not None:
        row = t[t["TARGETID"] == a.targetid]
        assert len(row) == 1, f"{a.targetid} not in lineflux_table.fits"
        d, use_4363, frac = one(as_dict(row[0]), a.curve, a.mu, a.trel)
        print(
            f"{a.targetid}: curve {a.curve}, T(O++) from {'4363' if use_4363 else a.trel}, "
            f"n_e at a bound in {frac:.3f} of draws"
        )
        for key, (q, bad) in summary(d).items():
            print(f"{key:>7} {q[1]:10.3f} [{q[0]:10.3f}, {q[2]:10.3f}]  NaN fraction {bad:.3f}")
        return
    curve_tag = a.curve if a.curve == "ccm89" else f"{a.curve}{a.mu:.1f}"
    name = f"direct_{curve_tag}_{a.trel}_r{RECIPE}{tag}.fits"
    path = OUT / name
    rows = []
    if path.exists():
        old = Table.read(path)
        same = [old.meta.get(k) for k in ("N_MC", "SEED", "RECIPE")] == [N_MC, SEED, RECIPE]
        assert same, "settings changed"
        rows = [[old[c][i] for c in NAMES] for i in range(len(old))]
    done = {int(r[0]) for r in rows}
    todo = [as_dict(r) for r in t if int(r["TARGETID"]) not in done]
    print(f"{name}: {len(done)} done, {len(todo)} to do, {a.workers} workers")
    with mp.get_context("spawn").Pool(a.workers) as pool:
        jobs = ((r, a.curve, a.mu, a.trel) for r in todo)
        for n, res in enumerate(pool.imap_unordered(work, jobs, chunksize=2), 1):
            rows.append(res)
            if n % SAVE_EVERY == 0:
                write(rows, path, False)
                print(f"  {len(rows)} of {len(t)} written", flush=True)
    out = write(rows, path, len(rows) == len(t))
    print(f"wrote {name}: {len(out)} galaxies, kept {int(out['KEEP'].sum())}")


if __name__ == "__main__":
    main()
