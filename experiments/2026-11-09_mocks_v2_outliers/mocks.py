"""Mock DESI spectra from the ladder step 5/6 model, for the outlier rerun.

Changed from the old mocks (src/hubersed/mocks/seds.py, run of 2026-08-25): C3K ER with afe free,
TopHat(-0.2, 0.6) as 2026-10-08_continuum_outlier, and the ladder's model and priors (make_params
of 2026-11-07_emline_ladder_39627770174637084/fit.py, with its SFH hyperpriors and tau_in = t_H(z)).
Redshift is U(0.01, 0.6). Each mock takes the DR1
coadd R of a random target of the 2026-11-09_r_matrix_survey. The free line columns are drawn from
the fit prior, EW uniform from 0 to W per column, through MockLineModel. The observation mask is
the ladder's rest 3320-9000 A, where the library resolution is constant (prospect refuses a
varying one); pixels outside it get prospect's own smoothed continuum and lines (smoke test 11-09:
finite, continuous at the edge). Plan:
~/.claude/plans/mocks_v2_outlier_rerun_plan.md.

Actions. ``rbank`` merges the R of the 1,000 survey targets into rbank.npz (stampede3, reads the
coadds on /work) and fills each target's all-zero columns (its ivar = 0 pixels) with the median
column of the others; without the fill 30% of the pilot mocks failed prospect's row-sum check.
``run --start --stop`` writes mocks start to stop to one h5 part, with the fork pool of the
ladder (SSP build, speclib free, fork before JAX). Mock i draws all its numbers from
default_rng([SEED, i]), so a part does not depend on the pool size or the order of the tasks.

COMMAND: on stampede3, ``sbatch experiments/2026-11-09_mocks_v2_outliers/mocks.slurm`` (see there).

OUTPUT: results/2026-11-09_mocks_v2_outliers/rbank.npz and mocks_<start>_<stop>.h5 with fluxes
(maggies, DESI grid), wavelength, priors/redshifts, priors/theta, priors/column_ew, priors/rbank_index,
failed, and the theta and column labels as attributes.
"""

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

from hubersed.paths import PATHS

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = PATHS["RESULTS"] / "2026-11-09_mocks_v2_outliers"
SURVEY = EXP / "2026-11-09_r_matrix_survey"
LADDER = EXP / "2026-11-07_emline_ladder_39627770174637084" / "fit.py"
SEED = 20261109
N_TOTAL = 500_000
Z_RANGE = (0.01, 0.6)
_W = {}  # per-process cache: ladder module, Cue source, floor, columns, R bank


def load_module(name, path):
    """Return a module loaded from ``path``."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ladder():
    """Return the ladder fit.py module, loaded once per process."""
    if "ladder" not in _W:
        _W["ladder"] = load_module("ladder_fit", LADDER)
    return _W["ladder"]


def rbank():
    """Merge the DR1 coadd R of the survey targets into rbank.npz."""
    from astropy.io import fits

    from hubersed.sps.lsf import DESI_WAV

    survey = load_module("survey", SURVEY / "survey.py")
    tids, hps = np.loadtxt(survey.TARGETS, dtype=np.int64, unpack=True)
    wave = np.asarray(DESI_WAV, float)
    res = None
    for hp in np.unique(hps):
        with fits.open(survey.DATA / str(hp) / f"coadd-sv3-bright-{hp}.fits", memmap=True) as h:
            ftid = np.asarray(h["FIBERMAP"].data["TARGETID"])
            for i in np.where(hps == hp)[0]:
                r = survey.merged_matrix(h, np.where(ftid == tids[i])[0][0], wave)
                if res is None:
                    res = np.zeros((tids.size, *r.shape), np.float32)
                res[i] = r
    res, filled = fill_empty_columns(res)
    rows = np.stack([row_sums(r) for r in res])
    inner = slice(res.shape[1], -res.shape[1])  # rows near the grid ends run off the matrix
    assert rows[:, inner].min() >= 0.5, f"row sum {rows[:, inner].min():.3f} after the fill"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "rbank.npz", targetid=tids, wave=wave, res=res, filled=filled)
    print(
        f"rbank: {res.shape} from {np.unique(hps).size} healpix; filled {filled.sum()} columns in "
        f"{filled.any(1).sum()} targets; min inner row sum {rows[:, inner].min():.3f}"
    )


def fill_empty_columns(res):
    """Fill each target's all-zero columns with the median column of the other targets.

    A column is all zero where the target's ivar is zero in every arm (merged_matrix divides by
    the summed ivar only where it is positive). Those pixels say nothing about the instrument,
    so they take the element-wise median over the targets whose column is not empty.

    Returns
    -------
    res : np.ndarray
        The filled bank, shape (n_targets, ndiag, n_pix).
    filled : np.ndarray
        Bool, shape (n_targets, n_pix), True where a column was filled.
    """
    empty = ~np.any(res != 0, axis=1)
    med = np.nanmedian(np.where(empty[:, None, :], np.nan, res), axis=0)
    res = np.where(empty[:, None, :], med[None], res).astype(np.float32)
    return res, empty


def row_sums(data):
    """Return the row sums of a banded matrix (desispec storage, R[j - o, j] = data[k, j])."""
    ndiag, n = data.shape
    out = np.zeros(n)
    for k, o in enumerate(ndiag // 2 - np.arange(ndiag)):
        j = np.arange(n)
        i = j - o
        ok = (i >= 0) & (i < n)
        np.add.at(out, i[ok], data[k, j[ok]])
    return out


def mock_model_class():
    """Return HyperSpecModel with the free columns drawn from their prior instead of marginalised.

    _marginalise_columns follows the prospect fork's code for the column templates and the
    amplitude bound, then sets each amplitude to u * upper with u ~ U(0, 1) from the mock's
    generator, which is the uniform EW prior from 0 to W. The drawn EWs (u * W) are kept.
    """
    from prospect.models.sedmodel import HyperSpecModel, lightspeed
    from scipy.sparse import csc_matrix

    class MockLineModel(HyperSpecModel):
        """HyperSpecModel that draws its free line columns from the prior."""

        mock_rng = None

        def _marginalise_columns(self, obs, rprime, columns, spec, response):
            idx = np.array([c[0] for c in columns])
            centre = np.array([c[3] for c in columns])
            ew_max = np.array([self.line_columns[j].ew_max for j in idx], dtype=float)
            a = 1 + float(np.atleast_1d(self._zred)[0])
            cont = np.interp(centre, obs.wavelength, self._resolved_continuum)
            upper = np.clip(ew_max * a * cont * lightspeed / centre**2, 0, None)
            rows = np.concatenate([c[1] for c in columns])
            cols = np.repeat(np.arange(len(columns)), [c[1].size for c in columns])
            fine = csc_matrix(
                (np.concatenate([c[2] for c in columns]), (rows, cols)),
                shape=(rprime.shape[1], len(columns)),
            )
            u = self.mock_rng.uniform(size=len(columns))
            self.mock_ew = np.full(len(self.line_columns), np.nan)
            self.mock_ew[idx] = u * ew_max
            self._line_marginal = None
            return (rprime @ fine).toarray() @ (u * upper)

    return MockLineModel


def setup_worker():
    """Build Cue, the sigma_smooth floor, the column list and the model class in this process."""
    if "cue" in _W:
        return
    import json

    from prospect.models.sedmodel import HyperSpecModel

    from hubersed.sps.config import library_sigma_floor
    from hubersed.sps.line_columns import build_columns

    lad = ladder()
    cue = lad.build_cue_sps()
    floor = library_sigma_floor(cue, lad.REST_MIN, lad.REST_MAX)
    params = lad.make_params(0.1, cue, lines=True, floor=floor)
    names = HyperSpecModel(params).emline_info["name"]
    columns = build_columns(names, json.loads(lad.LIMITS.read_text()))
    bank = np.load(OUT / "rbank.npz")
    _W.update(
        cue=cue,
        floor=floor,
        columns=columns,
        model_cls=mock_model_class(),
        res=bank["res"],
        wave=bank["wave"],
    )


def _afe_prior():
    """Return the [alpha/Fe] prior, TopHat(-0.2, 0.6), the C3K ER grid as in 2026-10-08.

    With afe free, logzsol is [Fe/H] (Nikhil 11-10: the mocks cover both the line and the
    continuum models).
    """
    from prospect.models.priors import TopHat

    return TopHat(mini=-0.2, maxi=0.6)


def draw_params(i):
    """Return the parameter template of mock i and its generator, all free values drawn.

    z ~ U(0.01, 0.6); make_params(z) sets zred, the age bins, tau_in and the hyperpriors. Every
    free parameter but logsfr_ratios is drawn through its prior's unit_transform. logsfr_ratios is
    then drawn from the multivariate normal that adjust_stochastic_params builds from the drawn
    hyperparameters.
    """
    from prospect.models.templates import adjust_stochastic_params

    rng = np.random.default_rng([SEED, i])
    z = rng.uniform(*Z_RANGE)
    params = ladder().make_params(z, _W["cue"], lines=True, floor=_W["floor"])
    params["afe"] = {"N": 1, "isfree": True, "init": 0.0, "prior": _afe_prior()}
    for name in sorted(params):
        p = params[name]
        if p.get("isfree") and name != "logsfr_ratios":
            x = np.asarray(p["prior"].unit_transform(rng.uniform(size=p["N"])), float)
            p["init"] = x if p["N"] > 1 else float(x.reshape(-1)[0])
    params = adjust_stochastic_params(params)
    pr = params["logsfr_ratios"]["prior"]
    params["logsfr_ratios"]["init"] = rng.multivariate_normal(pr.loc, pr.scale)
    return z, params, rng


def one_mock(i, r):
    """Return (flux in maggies, theta, column EWs, ok) of mock i with R bank row r."""
    from prospect.observation import ResolutionMatrixSpectrum

    z, params, rng = draw_params(i)
    model = _W["model_cls"](params, line_columns=_W["columns"])
    model.mock_rng = rng
    wave = _W["wave"]
    lad = ladder()
    # the ladder's rest window, where the library resolution is constant; outside it prospect
    # gives its own smoothed continuum, which no fit uses
    rest = wave / (1 + z)
    obs = ResolutionMatrixSpectrum(
        wavelength=wave,
        resolution_matrix=np.asarray(_W["res"][r], float),
        flux=np.zeros(wave.size),
        uncertainty=np.ones(wave.size),
        mask=(rest >= lad.REST_MIN) & (rest <= lad.REST_MAX),
    )
    obs.rectify()
    try:
        spec = model.predict(model.theta, [obs], sps=_W["cue"])[0][0]
        ew = getattr(model, "mock_ew", np.full(len(_W["columns"]), np.nan))
        return np.asarray(spec, np.float32), model.theta.copy(), ew, True
    except Exception as err:  # a failed draw is recorded, not redrawn
        print(f"mock {i} failed: {err!r}", flush=True)
        return np.full(wave.size, np.nan, np.float32), model.theta.copy(), None, False


def task(args):
    """Return the mocks of one block, as (indices, fluxes, thetas, EWs, ok, z, r)."""
    setup_worker()
    idx, rows = args
    out = [one_mock(i, r) for i, r in zip(idx, rows, strict=True)]
    z = [np.random.default_rng([SEED, i]).uniform(*Z_RANGE) for i in idx]
    return idx, rows, out, z


def labels():
    """Return the theta labels and column names of the model (in this process)."""
    setup_worker()
    _, params, _ = draw_params(0)
    model = _W["model_cls"](params, line_columns=_W["columns"])
    return model.theta_labels(), [c.name for c in _W["columns"]]


def run(start, stop, pool_size, block, gib_worker):
    """Write mocks start to stop to mocks_<start>_<stop>.h5."""
    import h5py

    rows = np.random.default_rng(SEED).integers(0, 1000, size=N_TOTAL)[start:stop]
    idx = np.arange(start, stop)
    tasks = [(idx[k : k + block], rows[k : k + block]) for k in range(0, idx.size, block)]
    lad = ladder()
    path = OUT / f"mocks_{start}_{stop}.h5"
    t0 = time.time()
    if np.isnan(lad.commit_room()):  # macOS smoke tests: no commit limit, plain fork pool
        import multiprocessing as mp

        pool_ctx = mp.get_context("fork").Pool(pool_size or 2)
    else:
        pool_ctx = lad.node_pool(pool_size, gib_worker)
    with pool_ctx as pool:
        theta_labels, col_names = pool.apply(labels)
        n, nw = stop - start, 7781
        with h5py.File(path, "w") as hf:
            hf.attrs.update(seed=SEED, start=start, stop=stop, n_total=N_TOTAL)
            hf.attrs["theta_labels"] = theta_labels
            hf.attrs["column_names"] = col_names
            flux = hf.create_dataset("fluxes", (n, nw), np.float32, compression="gzip")
            theta = hf.create_dataset("priors/theta", (n, len(theta_labels)), np.float64)
            ew = hf.create_dataset("priors/column_ew", (n, len(col_names)), np.float64)
            zs = hf.create_dataset("priors/redshifts", (n,), np.float64)
            rb = hf.create_dataset("priors/rbank_index", (n,), np.int32)
            failed = hf.create_dataset("failed", (n,), bool)
            done = 0
            for ids, rr, out, z in pool.imap_unordered(task, tasks):
                k = ids - start
                for j, (f, th, e, good) in enumerate(out):
                    flux[k[j]] = f
                    theta[k[j]] = th
                    ew[k[j]] = np.nan if e is None else e
                    failed[k[j]] = not good
                zs[k] = z
                rb[k] = rr
                done += len(ids)
                print(f"{done}/{n} in {time.time() - t0:.0f} s", flush=True)
            hf.create_dataset("wavelength", data=_wave_from_bank())
    print(f"wrote {path} in {time.time() - t0:.0f} s")


def _wave_from_bank():
    """Return the DESI grid stored in rbank.npz."""
    return np.load(OUT / "rbank.npz")["wave"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("action", choices=["rbank", "run"])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=1000)
    p.add_argument("--pool", type=int, default=0, help="0 sizes it from the commit room")
    p.add_argument("--block", type=int, default=20, help="mocks per task")
    # skx pilot 3699130: 121 GB committed with 18 workers on a 15 GB baseline, ~5.6 GB each
    p.add_argument(
        "--gib-worker", type=float, default=6.5, help="commit per worker, sizes the pool"
    )
    a = p.parse_args()
    if a.action == "rbank":
        rbank()
    else:
        run(a.start, a.stop, a.pool, a.block, a.gib_worker)
