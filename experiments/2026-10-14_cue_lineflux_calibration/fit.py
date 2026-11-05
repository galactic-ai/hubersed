"""Nautilus fit of Cue to the line fluxes of one galaxy, with the auroral lines held out or fitted.

What changed from the full-spectrum fits (2026-10-13_nautilus_resolution_matrix): no continuum and
no spectrum. Cue is fitted to the line-flux table of data.py. The ionizing spectrum is fixed to the
young CSP of a 10 Myr constant-SFR population at stellar Z ``--zstar`` (ionspec.npz of
ionspec.py). Dust is a screen with the Wild+11 line curve A_lambda/A_V = (1 - mu)(lambda/5500)^-1.3
+ mu(lambda/5500)^-0.7 (arXiv:1008.3160, spitz_emlines_astro.tex:1273-1280), mu = ``--mu``.

Free parameters and uniform priors: gas_logu (-4, -1), gas_lognH (1, 4), gas_logz (-2.2, 0.5),
gas_logno and gas_logco (-1, log10 5.4), as the pipeline's cue_stellar_nebular template; av (0, 3);
log_fhb, the log10 intrinsic Hbeta flux, from log10 of the observed Hbeta -0.3 to +1.6.

Likelihood: Gaussian on each fitted line, with variance the flux error squared plus Cue's emulation
error (line_frac_error times the model flux) squared, and its log term, since the variance depends
on the model. Summed lines ([OII]3727, [OII]7325) add the emulation errors of their members in
quadrature. Lines with a NaN flux or a non-positive error are skipped.

``--lines strong`` fits HB, OIII5007, NII6584, HA, SII6716, SII6731, OII3727, NEIII3869, HG and
OI6300. ``--lines all`` adds OIII4363, SIII6312 and OII7325. ``--lines semi`` adds SIII9069 and
HEI5876 to the strong lines (Korhonen Cuestas+26 arXiv:2609.22488 Table 2, line list 2; they also
drop OI6300); only these fits save the two extra lines. ``--fluxes fsf`` uses the FastSpecFit
Gaussian fluxes of the six EmFit lines instead of the EmFit totals. ``--drop`` leaves the named
lines out of the fit; they are still predicted.

``--prior nicholls`` multiplies the flat priors by Gaussians in gas_logno and gas_logco around the
Nicholls+17 relations (arXiv:1612.03546, abundances.tex:301-304, total abundances, so on Cue's
total [O/H] scale): gas_logno = log10(10^-0.852 + 10^gas_logz), sigma NO_SIGMA 0.3 dex (typical for
individual objects, Andrews & Martini 2013, arXiv:1211.3418, ms.tex:1550), and gas_logco =
log10(10^-0.43 + 10^(gas_logz + 0.02)), sigma CO_SIGMA 0.17 dex (Berg+19, arXiv:1901.08160, below
12+log(O/H) 8.0). It enters as an extra term of the sampled log density; log_l_data saves the data
log likelihood alone at every posterior point. ``--zmin`` raises the lower bound of the
gas_logz prior, for example to force the upper metallicity branch. ``--table`` reads another table in the
lineflux_table.fits format, such as mocks/mock_table.fits, and writes to a fits/ folder beside it.

COMMAND: from the repository root,
``uv run --no-sync python experiments/2026-10-14_cue_lineflux_calibration/fit.py TARGETID
--lines strong`` (defaults: --mu 0.4 --zstar -0.41 --fluxes emfit --n-live 2000). No FSPS needed.

OUTPUT: fits/<TARGETID>_<lines>_<fluxes>_mu<mu>_zs<zstar>[_pnicholls][_zmin<zmin>].npz in
results/2026-10-14_cue_lineflux_calibration/ (or beside --table): the posterior points, log weights, log likelihoods,
log evidence, N_eff, likelihood calls, run time, settings, and the model flux of every line at 200
posterior draws.
"""

import argparse
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from astropy.table import Table
from cuejax.emulator import Emulator, fast_line_prediction
from nautilus import Prior, Sampler

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-10-14_cue_lineflux_calibration"
LOG54 = float(np.log10(5.4))
NO_SIGMA = 0.3
CO_SIGMA = 0.17
CHUNK = 1024
N_DRAW = 200
SEED = 0
# table line name, Cue line names summed
LINES = {
    "HB": ("H 1 4861.32A",),
    "OIII5007": ("O 3 5006.84A",),
    "NII6584": ("N 2 6583.45A",),
    "HA": ("H 1 6562.80A",),
    "SII6716": ("S 2 6716.44A",),
    "SII6731": ("S 2 6730.82A",),
    "OII3727": ("O 2 3726.03A", "O 2 3728.81A"),
    "NEIII3869": ("Ne 3 3868.76A",),
    "HG": ("H 1 4340.46A",),
    "OI6300": ("O 1 6300.30A",),
    "OIII4363": ("O 3 4363.21A",),
    "SIII6312": ("S 3 6312.06A",),
    "OII7325": ("O 2 7323.00A", "O 2 7332.00A"),
}
# semi-strong lines, fitted only with --lines semi
SEMI = {
    "SIII9069": ("S 3 9068.62A",),
    "HEI5876": ("He 1 5875.66A",),
}
AURORAL = ("OIII4363", "SIII6312", "OII7325")
EMFIT_LINES = ("HB", "OIII5007", "NII6584", "HA", "SII6716", "SII6731")
GAS = (
    ("gas_logu", -4.0, -1.0),
    ("gas_lognH", 1.0, 4.0),
    ("gas_logz", -2.2, 0.5),
    ("gas_logno", -1.0, LOG54),
    ("gas_logco", -1.0, LOG54),
)


def nicholls_no(z):
    """Return gas_logno on the Nicholls+17 nitrogen relation at gas_logz z."""
    return np.log10(10**-0.852 + 10.0**z)


def nicholls_co(z):
    """Return gas_logco on the Nicholls+17 carbon relation at gas_logz z."""
    return np.log10(10**-0.43 + 10.0 ** (z + 0.02))


def wild_curve(wave, mu):
    """Return A_lambda/A_V of the Wild+11 two-component line curve."""
    x = np.asarray(wave, float) / 5500.0
    return (1.0 - mu) * x**-1.3 + mu * x**-0.7


class LineModel:
    """Cue line fluxes of the table lines, attenuated, for batches of parameter vectors."""

    def __init__(self, ion, mu):
        self.emul = Emulator()
        names = [str(x).strip() for x in np.atleast_1d(self.emul.line_names)]
        wave = np.asarray(self.emul.line_wavelengths, float)
        frac = np.asarray(self.emul.line_frac_error, float)
        idx = {n: i for i, n in enumerate(names)}
        self.members = {k: np.array([idx[n] for n in v]) for k, v in (LINES | SEMI).items()}
        self.k = {k: wild_curve(wave[m], mu) for k, m in self.members.items()}
        self.frac = {k: frac[m] for k, m in self.members.items()}
        self.ion = np.asarray(ion, float)

    def cue(self, gas):
        """Return Cue line luminosities for an (N, 5) array of gas parameters, in fixed chunks."""
        n = len(gas)
        theta = np.hstack([np.tile(self.ion, (n, 1)), gas])
        out = []
        for s in range(0, n, CHUNK):
            block = theta[s : s + CHUNK]
            pad = CHUNK - len(block)
            if pad:
                block = np.vstack([block, np.tile(block[-1], (pad, 1))])
            lum = np.asarray(fast_line_prediction(jnp.asarray(block), self.emul)[0], float)
            out.append(lum[: CHUNK - pad] if pad else lum)
        return np.vstack(out)

    def fluxes(self, x):
        """Return model fluxes and emulation errors, each a dict of (N,) arrays, for an (N, 7) x."""
        lum = self.cue(x[:, :5])
        av, log_fhb = x[:, 5], x[:, 6]
        scale = 10.0**log_fhb / lum[:, self.members["HB"][0]]
        flux, err = {}, {}
        for k, m in self.members.items():
            part = lum[:, m] * 10.0 ** (-0.4 * av[:, None] * self.k[k][None, :]) * scale[:, None]
            flux[k] = part.sum(axis=1)
            err[k] = np.sqrt(((part * self.frac[k][None, :]) ** 2).sum(axis=1))
        return flux, err


def load_row(tid, fluxes, table=OUT / "lineflux_table.fits"):
    """Return the observed fluxes and errors of one galaxy as dicts over LINES and SEMI.

    A line whose columns the table lacks (older tables have no SEMI lines) gets NaN.
    """
    t = Table.read(table)
    row = t[t["TARGETID"] == tid]
    assert len(row) == 1, f"{tid} not in {table}"
    row = row[0]
    pre = {k: ("FSF_" if fluxes == "fsf" and k in EMFIT_LINES else "") for k in LINES | SEMI}

    def get(col):
        return float(row[col]) if col in t.colnames else np.nan

    obs = {k: get(f"{pre[k]}F_{k}") for k in LINES | SEMI}
    err = {k: get(f"{pre[k]}E_{k}") for k in LINES | SEMI}
    return obs, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targetid", type=int)
    ap.add_argument("--lines", choices=("strong", "all", "semi"), default="strong")
    ap.add_argument("--fluxes", choices=("emfit", "fsf"), default="emfit")
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--zstar", type=float, default=-0.41)
    ap.add_argument("--n-live", type=int, default=2000)
    ap.add_argument("--drop", nargs="+", choices=list(LINES | SEMI), default=[])
    ap.add_argument("--prior", choices=("flat", "nicholls"), default="flat")
    ap.add_argument("--table", default=str(OUT / "lineflux_table.fits"))
    ap.add_argument("--zmin", type=float)
    a = ap.parse_args()

    grid = np.load(OUT / "ionspec.npz")
    i = np.flatnonzero(np.isclose(grid["zstar"], a.zstar))
    assert i.size == 1, f"zstar {a.zstar} not in ionspec.npz {grid['zstar']}"
    model = LineModel(grid["ionspec"][i[0]], a.mu)
    obs, err = load_row(a.targetid, a.fluxes, a.table)
    lines = LINES | SEMI if a.lines == "semi" else LINES
    want = [k for k in lines if (a.lines == "all" or k not in AURORAL) and k not in a.drop]
    used = [k for k in want if np.isfinite(obs[k]) and err[k] > 0]
    print(f"{a.targetid}: fitting {used}; skipped {sorted(set(want) - set(used))}")
    o = np.array([obs[k] for k in used])
    e2 = np.array([err[k] for k in used]) ** 2

    def loglike_data(x):
        flux, emu = model.fluxes(np.atleast_2d(x))
        m = np.stack([flux[k] for k in used], axis=1)
        var = e2[None, :] + np.stack([emu[k] for k in used], axis=1) ** 2
        return -0.5 * np.sum((o[None, :] - m) ** 2 / var + np.log(2 * np.pi * var), axis=1)

    def log_prior_extra(x):
        x = np.atleast_2d(x)
        if a.prior == "flat":
            return np.zeros(len(x))
        z, no, co = x[:, 2], x[:, 3], x[:, 4]
        return -0.5 * (
            ((no - nicholls_no(z)) / NO_SIGMA) ** 2 + ((co - nicholls_co(z)) / CO_SIGMA) ** 2
        )

    def loglike(x):
        return loglike_data(x) + log_prior_extra(x)

    prior = Prior()
    for name, lo, hi in GAS:
        if name == "gas_logz" and a.zmin is not None:
            assert lo < a.zmin < hi, f"zmin {a.zmin} outside the gas_logz range"
            lo = a.zmin
        prior.add_parameter(name, dist=(lo, hi))
    prior.add_parameter("av", dist=(0.0, 3.0))
    lhb = np.log10(obs["HB"])
    prior.add_parameter("log_fhb", dist=(lhb - 0.3, lhb + 1.6))

    t0 = time.time()
    sampler = Sampler(prior, loglike, n_live=a.n_live, vectorized=True, pass_dict=False, seed=SEED)
    sampler.run(verbose=False)
    seconds = time.time() - t0
    points, log_w, log_l = sampler.posterior()
    log_l_data = np.concatenate(
        [loglike_data(points[s : s + 20 * CHUNK]) for s in range(0, len(points), 20 * CHUNK)]
    )
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    draws = points[np.random.default_rng(SEED).choice(len(w), size=N_DRAW, p=w)]
    flux, _ = model.fluxes(draws)

    labels = [p for p, _, _ in GAS] + ["av", "log_fhb"]
    print(
        f"{seconds:.1f} s, {sampler.n_like} likelihood calls, N_eff {sampler.n_eff:.0f}, "
        f"log Z {sampler.log_z:.2f}"
    )
    for j, lab in enumerate(labels):
        q = np.interp(
            [0.16, 0.5, 0.84], np.cumsum(w[np.argsort(points[:, j])]), np.sort(points[:, j])
        )
        print(f"{lab:>10} {q[1]:8.3f} [{q[0]:8.3f}, {q[2]:8.3f}]")
    print(f"{'line':>10} {'data':>9} {'model/data median [16, 84]':>30} fitted")
    for k in lines:
        if np.isfinite(obs[k]) and obs[k] != 0:
            r = np.percentile(flux[k] / obs[k], [16, 50, 84])
            print(
                f"{k:>10} {obs[k]:9.2f} {r[1]:10.3f} [{r[0]:6.3f}, {r[2]:6.3f}] "
                f"{'yes' if k in used else 'no'}"
            )

    fits_dir = Path(a.table).parent / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)
    drop = "".join(f"_no{k}" for k in a.drop)
    pri = "" if a.prior == "flat" else f"_p{a.prior}"
    pri += "" if a.zmin is None else f"_zmin{a.zmin:+.2f}"
    name = f"{a.targetid}_{a.lines}{drop}_{a.fluxes}_mu{a.mu:.1f}_zs{a.zstar:+.2f}{pri}.npz"
    np.savez(
        fits_dir / name,
        labels=np.array(labels),
        points=points,
        log_w=log_w,
        log_l=log_l,
        log_l_data=log_l_data,
        prior=a.prior,
        log_z=sampler.log_z,
        n_eff=sampler.n_eff,
        n_like=sampler.n_like,
        seconds=seconds,
        used=np.array(used),
        lines=np.array(list(lines)),
        obs=np.array([obs[k] for k in lines]),
        err=np.array([err[k] for k in lines]),
        draw_flux=np.stack([flux[k] for k in lines], axis=1),
        mu=a.mu,
        zstar=a.zstar,
        fluxes=a.fluxes,
    )


if __name__ == "__main__":
    main()
