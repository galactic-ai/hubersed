"""Screen continuum flow outliers for observed-frame and rest-frame causes.

Each candidate is compared with a median template of unflagged DESI galaxies matched in
redshift, Dn4000 and stellar mass. The residual is summed in observed-frame zones (arms,
telluric bands, sky lines, masked pixels) and in rest-frame index bands. Each zone is then
replaced by the template plus noise, the spectrum is re-encoded and rescored by the flows,
and the change in log p shows which zone drives the flag.

The encoder input is the chunk spectrum as stored, which spender already divided by its
median over rest 5300-5850 A. Templates are built from the stored encoder inputs in the
cont10 latent file, so they share that normalisation. The flow scores standardised latents,
(latent - scaler_mean) / scaler_scale, with the scaler saved next to the flow state.

A self-test runs first. It re-encodes 11 chunk spectra and requires the stored log p to be
reproduced within 0.1 for every flow that has an encoder on disk. The script stops if it
fails. A flow without an encoder checkpoint is skipped and every call is then provisional.

Classification rule, fixed before the run. For a zone, frac is the mean log p gain from
replacing it, divided by the gap from the candidate's log p to the threshold. frac of 1
means the galaxy crosses back above the threshold. The groups are OBS_ALL (all
observed-frame zones), REST_ALL (the stellar index bands, without Halpha), HALPHA (rest
6548-6578 A, where emission and absorption both sit) and TILT (the candidate divided by its
quadratic ratio to the template). A group with frac of at least 0.7 is a hit. One hit gives
its class. Two or more give the class of the largest when it leads the next by at least 0.3,
and diffuse otherwise. No hit gives diffuse. If the full template does not cross the
threshold the template is not a valid inlier and the class is "unreliable template". A call
is firm only when both flows ran and agree.

Control null, added after the first look at one candidate and before the full table. Three
unflagged analogues per candidate get the same zone replacements against a template of the
other analogues. In the null-adjusted class (class_null, call_null) a group counts only when
the candidate's gain also exceeds the 95th percentile of the control gains for that group.
This guards against a zone that raises log p for any galaxy, for example because the template
is smoother than real data there. Halpha was moved out of REST_ALL into its own group at the
same time, after the first run showed Halpha mean chi2 of 27 to 114 in some candidates, so that
an emission mismatch is not counted as a stellar feature.

Run from the repository root with
``uv run --no-sync python experiments/2026-10-08_continuum_outlier/screen_candidates.py``.
Outputs go to results/2026-10-08_continuum_outlier/screen/. SEED is 0.
"""

import csv
import pickle
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
from astropy.io import fits
from spender import load_model
from spender.data import desi

from hubersed.detect.flow_model import build_flow
from hubersed.io.desi import tids_to_indices
from hubersed.paths import PATHS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
NDRAW = 8
DATA = PATHS["DATA"]
RES = PATHS["RESULTS"]
OUT = RES / "2026-10-08_continuum_outlier" / "screen"
CAND_FILE = (
    Path.home()
    / "Documents/UberSED/knowledge/_evidence/2026-10-08_continuum_outlier/candidates.txt"
)
SCORE_DIR = RES / "wide_flow_corrected"
FLOW_DIR = RES / "noised_cue_meanzero_wide_flow"
LATENT_H5 = DATA / "latents" / "spender_spec_cont10latent_snr3.h5"
VAC = DATA / "fastspec-iron-sv3-bright.fits"
FLOWS = {
    "cont10": DATA / "checkpoints" / "spender_desi_cont_10latent_zmax.pt",
    "cont15": DATA / "checkpoints" / "spender_desi_cont_15latent_zmax.pt",
}
WAVE = desi.DESI._wave_obs.numpy()
SKY_SPENDER = desi.DESI._skyline_mask.numpy()

# Observed-frame zones in A.
OBS_ZONES = {
    "zarm": (7470.0, 9800.0),
    "Aband": (7590.0, 7700.0),
    "Bband": (6860.0, 6960.0),
    "br_overlap": (5600.0, 5930.0),
    "rz_overlap": (7470.0, 7720.0),
}
# Rest-frame central bandpasses in air A. Lick and HdeltaA from $SPS_HOME/data/allindices.dat.
# CaHK and Halpha are local choices.
REST_ZONES = {
    "CaHK": (3925.0, 3980.0),
    "HdeltaA": (4083.5, 4122.25),
    "CN1": (4142.125, 4177.125),
    "G4300": (4281.375, 4316.375),
    "Hbeta": (4847.875, 4876.625),
    "Mgb": (5160.125, 5192.625),
    "Fe5270": (5245.65, 5285.65),
    "Fe5335": (5312.125, 5352.125),
    "NaD": (5876.875, 5909.375),
    "Halpha": (6548.0, 6578.0),
}
AIR_TO_VAC = 1.00028
WIN = {"z": 0.02, "dn": 0.08, "m": 0.2}
NMIN, NMAX, NCTRL = 30, 100, 3
HIT, LEAD = 0.7, 0.3


def load_scorers():
    """Load the flows, their thresholds, stored scores and encoders.

    Returns
    -------
    dict
        Keyed by flow name. Each value holds ``flow``, ``mu``, ``sc``, ``thr``, ``lp``
        (stored log p by TARGETID), ``out`` (flagged TARGETIDs) and ``enc`` (the spender
        model, or None when its checkpoint is not on disk).
    """
    inst = desi.DESI()
    res = {}
    for name, ckpt in FLOWS.items():
        o = torch.load(
            SCORE_DIR / f"desi_outliers_flow_nsf_{name}latent_snr3.pt", weights_only=False
        )
        st = torch.load(
            FLOW_DIR / f"flow_nsf_{name}latent.pt", weights_only=False, map_location="cpu"
        )
        assert st["encoder"] == o["encoder"] == ckpt.name, (st["encoder"], o["encoder"])
        nde = build_flow(
            st["method"], st["dim"], st["hidden"], st["num_transforms"], st["num_bins"]
        )
        nde.load_state_dict(st["state_dict"])
        nde.eval()
        enc = None
        if ckpt.exists():
            enc = load_model(str(ckpt), inst, map_location="cpu", weights_only=False).float().eval()
        tid = np.asarray(o["desi_target_ids"], np.int64)
        lp = np.asarray(o["log_p_desi"], np.float64)
        res[name] = {
            "flow": nde,
            "mu": st["scaler_mean"],
            "sc": st["scaler_scale"],
            "thr": float(o["threshold"]),
            "lp": dict(zip(tid.tolist(), lp, strict=True)),
            "out": set(int(x) for x in o["outlier_target_ids"]),
            "enc": enc,
        }
    return res


def score(s, specs, batch=256):
    """Encode normalised spectra and return their log p under one flow.

    Parameters
    ----------
    s : dict
        One entry of ``load_scorers``. Its encoder must not be None.
    specs : np.ndarray
        Encoder inputs, shape (N, 7781).
    batch : int
        Spectra per encoder call.

    Returns
    -------
    np.ndarray
        Log p of each spectrum.
    """
    out = []
    with torch.no_grad():
        for i in range(0, len(specs), batch):
            lat = s["enc"].encode(torch.from_numpy(np.asarray(specs[i : i + batch], np.float32)))
            x = ((lat.numpy() - s["mu"]) / s["sc"]).astype(np.float32)
            out.append(s["flow"].log_prob(torch.from_numpy(x)).numpy())
    return np.concatenate(out).astype(np.float64)


def load_chunk_rows(tids):
    """Read the stored encoder input, weights and redshift of each TARGETID from the chunks.

    Parameters
    ----------
    tids : list of int
        TARGETIDs.

    Returns
    -------
    dict
        TARGETID to (spec, w, z) as float64 arrays and a float.
    """
    gi = tids_to_indices(np.asarray(tids, np.int64))
    out = {}
    for c in np.unique(gi // 1024):
        with open(DATA / "desi_spectra" / f"DESIchunk1024_{c}.pkl", "rb") as f:
            s, w, z, tt, *_ = pickle.load(f)
        for t, g in zip(tids, gi, strict=True):
            if g // 1024 == c:
                r = g % 1024
                assert int(tt[r]) == t
                out[t] = (
                    s[r].numpy().astype(np.float64),
                    w[r].numpy().astype(np.float64),
                    float(z[r]),
                )
    return out


def self_test(scorers, h5tid):
    """Re-encode 11 chunk spectra and compare with the stored log p.

    The known case 39627757533007793 is always included, with five flagged and five
    unflagged cont10 galaxies drawn with SEED.

    Parameters
    ----------
    scorers : dict
        Output of ``load_scorers``.
    h5tid : np.ndarray
        TARGETIDs of the scored DESI sample.

    Returns
    -------
    list of dict
        One row per galaxy and flow with the stored and recomputed log p.

    Raises
    ------
    SystemExit
        If any available flow misses by 0.1 or more.
    """
    rng = np.random.default_rng(SEED)
    s10 = scorers["cont10"]
    lp10 = np.array([s10["lp"][t] for t in h5tid])
    flag = lp10 <= s10["thr"]
    pick = [39627757533007793]
    pick += h5tid[rng.choice(np.flatnonzero(flag), 5, replace=False)].tolist()
    pick += h5tid[rng.choice(np.flatnonzero(~flag), 5, replace=False)].tolist()
    rows_ = load_chunk_rows(pick)
    specs = np.stack([rows_[t][0] for t in pick])
    rows = []
    bad = False
    for name, s in scorers.items():
        if s["enc"] is None:
            rows += [
                {
                    "targetid": t,
                    "flow": name,
                    "stored": s["lp"][t],
                    "recomputed": np.nan,
                    "diff": np.nan,
                    "status": "no encoder",
                }
                for t in pick
            ]
            continue
        lp = score(s, specs)
        for t, v in zip(pick, lp, strict=True):
            d = v - s["lp"][t]
            bad |= abs(d) >= 0.1
            rows.append(
                {
                    "targetid": t,
                    "flow": name,
                    "stored": s["lp"][t],
                    "recomputed": v,
                    "diff": d,
                    "status": "ok" if abs(d) < 0.1 else "FAIL",
                }
            )
    write_csv(OUT / "selftest.csv", rows)
    if bad:
        raise SystemExit("self-test failed, see selftest.csv")
    return rows


def write_csv(path, rows):
    """Write a list of dicts to CSV. The header is every key in order of first appearance."""
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_candidates():
    """Return the candidate TARGETIDs and their S/N per observed A from candidates.txt."""
    rows = np.genfromtxt(CAND_FILE, names=True, dtype=None, encoding=None)
    return [int(t) for t in rows["TARGETID"]], dict(
        zip((int(t) for t in rows["TARGETID"]), rows["SN_per_obsA"].astype(float), strict=True)
    )


def load_vac(tids):
    """Return LOGMSTAR and DN4000 from the FastSpecFit VAC for the given TARGETIDs.

    Rows with nonpositive DN4000_IVAR or nonfinite values get NaN.
    """
    with fits.open(VAC, memmap=True) as h:
        d = h["FASTSPEC"].data
        t = np.asarray(d["TARGETID"], np.int64)
        m = np.asarray(d["LOGMSTAR"], np.float64)
        dn = np.asarray(d["DN4000"], np.float64)
        ok = np.asarray(d["DN4000_IVAR"]) > 0
    dn[~ok] = np.nan
    idx = dict(zip(t.tolist(), range(len(t)), strict=True))
    j = np.array([idx.get(int(x), -1) for x in tids])
    mm = np.where(j >= 0, m[j], np.nan)
    dd = np.where(j >= 0, dn[j], np.nan)
    return mm, dd


def pick_analogues(i, pool, z, m, dn):
    """Choose analogues of sample row i from the unflagged pool.

    Analogues are within WIN in z, Dn4000 and LOGMSTAR. If fewer than NMIN match, the window
    grows by 1.5 and then 2. If more than NMAX match, the NMAX nearest in scaled distance are
    kept.

    Returns
    -------
    sel : np.ndarray
        Sample rows of the analogues, nearest first.
    grow : float
        Window factor used.
    """
    dz = (z[pool] - z[i]) / WIN["z"]
    dd = (dn[pool] - dn[i]) / WIN["dn"]
    dm = (m[pool] - m[i]) / WIN["m"]
    for grow in (1.0, 1.5, 2.0):
        ok = (np.abs(dz) <= grow) & (np.abs(dd) <= grow) & (np.abs(dm) <= grow)
        if ok.sum() >= NMIN:
            break
    r = np.sqrt(dz**2 + dd**2 + dm**2)
    sel = np.flatnonzero(ok)
    sel = sel[np.argsort(r[sel])][:NMAX]
    return pool[sel], grow


def read_h5_specs(h5, rows):
    """Read stored encoder inputs for sample rows, in the order given."""
    order = np.argsort(rows)
    srt = rows[order]
    data = np.asarray(h5["specs"][srt.tolist()], np.float64)
    out = np.empty_like(data)
    out[order] = data
    return out


def build_template(specs, zs, zc):
    """Median of analogue spectra moved to the candidate redshift on the observed grid.

    Parameters
    ----------
    specs : np.ndarray
        Analogue encoder inputs, shape (N, 7781).
    zs : np.ndarray
        Analogue redshifts.
    zc : float
        Candidate redshift.

    Returns
    -------
    t : np.ndarray
        Median template, NaN where no analogue covers the pixel.
    sig : np.ndarray
        Error of the median, 1.2533 times the MAD scatter over sqrt(n).
    """
    stack = np.full(specs.shape, np.nan)
    for k, (f, za) in enumerate(zip(specs, zs, strict=True)):
        f = np.where(f == 0, np.nan, f)
        stack[k] = np.interp(WAVE * (1 + za) / (1 + zc), WAVE, f, left=np.nan, right=np.nan)
    n = np.isfinite(stack).sum(0)
    with np.errstate(all="ignore"):
        t = np.nanmedian(stack, 0)
        mad = 1.4826 * np.nanmedian(np.abs(stack - t), 0)
        sig = 1.2533 * mad / np.sqrt(n)
    t[n < 5] = np.nan
    return t, sig


def zones_for(z, w):
    """Return boolean pixel masks of every zone for a galaxy at redshift z with weights w.

    The masked zone is w = 0 outside the spender sky mask. The encoder still sees those pixels.
    """
    sky = np.genfromtxt(
        Path(desi.__file__).parent / "sky-lines.txt",
        names=["wavelength", "intensity", "name", "status"],
        dtype=None,
        encoding=None,
    )
    lam = sky["wavelength"][sky["intensity"] > 0.5] * 10.0
    near = np.abs(WAVE[:, None] - lam[None, :]).min(1) < 2.0
    zn = {k: (WAVE >= a) & (WAVE <= b) for k, (a, b) in OBS_ZONES.items()}
    zn["sky05"] = near
    zn["sky_spender"] = SKY_SPENDER.copy()
    zn["masked"] = (w == 0) & ~SKY_SPENDER
    obs = np.zeros_like(WAVE, bool)
    for k in ("Aband", "Bband", "br_overlap", "rz_overlap", "sky05", "sky_spender", "masked"):
        obs |= zn[k]
    rest = {}
    for k, (a, b) in REST_ZONES.items():
        f = (1 + z) * AIR_TO_VAC
        rest[k] = (WAVE >= a * f) & (WAVE <= b * f)
    rall = np.zeros_like(WAVE, bool)
    for k, v in rest.items():
        if k != "Halpha":
            rall |= v
    zn.update(rest)
    zn["OBS_ALL"] = obs
    zn["REST_ALL"] = rall
    return zn


def residuals(c, w, t, sig):
    """Scale the template to the candidate and form the normalised residual.

    Returns
    -------
    a : float
        Weighted least squares scale of the template.
    chi : np.ndarray
        (c - a t) / sqrt(sigma_c^2 + a^2 sig^2). sigma_c is interpolated over w = 0 pixels.
    good : np.ndarray
        Pixels with w > 0 and a finite template.
    sc : np.ndarray
        Candidate noise with the interpolation filled in.
    poly : np.ndarray
        Quadratic fit to c / (a t) over the grid.
    """
    fin = np.isfinite(t)
    good = (w > 0) & fin
    a = np.sum(w[good] * c[good] * t[good]) / np.sum(w[good] * t[good] ** 2)
    sc = np.full_like(c, np.nan)
    sc[w > 0] = 1 / np.sqrt(w[w > 0])
    ok = w > 0
    sc[~ok] = np.interp(WAVE[~ok], WAVE[ok], sc[ok])
    with np.errstate(all="ignore"):
        chi = (c - a * t) / np.sqrt(sc**2 + (a * sig) ** 2)
    x = (WAVE - WAVE.mean()) / (np.ptp(WAVE) / 2)
    ratio = c[good] / (a * t[good])
    wt = np.sqrt(w[good]) * a * t[good]
    coef = np.polyfit(x[good], ratio, 2, w=wt)
    return a, chi, good, sc, np.polyval(coef, x)


def zone_stats(chi, good, w, zn):
    """Chi2 share and mean chi2 of each zone.

    The two zones dominated by w = 0 pixels (sky_spender and masked) use the interpolated
    noise and are reported against the good-pixel total plus their own chi2.
    """
    fin = np.isfinite(chi)
    tot = np.sum(chi[good] ** 2)
    out = {}
    for k, m in zn.items():
        if k in ("sky_spender", "masked"):
            sel = m & fin
            c2 = np.sum(chi[sel] ** 2)
            share = c2 / (tot + np.sum(chi[sel & ~good] ** 2))
        else:
            sel = m & good
            c2 = np.sum(chi[sel] ** 2)
            share = c2 / tot
        n = int(sel.sum())
        out[k] = {
            "npix": n,
            "pix_share": n / max(int(good.sum()), 1),
            "chi2_share": share if n else np.nan,
            "mean_chi2": c2 / n if n else np.nan,
            "mean_resid": float(np.mean(chi[sel])) if n else np.nan,
        }
    out["_total"] = {
        "npix": int(good.sum()),
        "pix_share": 1.0,
        "chi2_share": 1.0,
        "mean_chi2": tot / good.sum(),
        "mean_resid": float(np.mean(chi[good])),
    }
    return out


def artifact_stats(c, t_s, sc):
    """Count zero-flux pixels and find the largest sky-pixel deviation from the template.

    Parameters
    ----------
    c : np.ndarray
        Encoder input.
    t_s : np.ndarray
        Scaled template.
    sc : np.ndarray
        Noise with interpolation over w = 0 pixels.

    Returns
    -------
    dict
        ``n_zero_flux`` and ``sky_spike``, the largest abs(c - t_s) / sc over spender sky pixels.
    """
    m = SKY_SPENDER & np.isfinite(t_s)
    return {
        "n_zero_flux": int((c == 0).sum()),
        "sky_spike": float(np.max(np.abs(c[m] - t_s[m]) / sc[m])),
    }


def occlusion_inputs(c, t_s, sc, poly, zn, rng, ndraw=NDRAW, keep_only=True):
    """Build the occluded encoder inputs for one candidate.

    Parameters
    ----------
    ndraw : int
        Noise draws for the template pixels.
    keep_only : bool
        Also build the complement inputs, candidate inside the zone and template outside.

    Returns
    -------
    specs : np.ndarray
        All inputs, stacked.
    keys : list of (str, str)
        (zone, mode) of each input. Modes are ``replace``, ``keep_only``, ``full``,
        ``full_nonoise`` and ``detilt``.
    """
    fin = np.isfinite(t_s)
    specs, keys = [], []
    for _ in range(ndraw):
        noisy = np.where(fin, t_s + sc * rng.standard_normal(c.size), c)
        for k, m in zn.items():
            if not m.any():
                continue
            x = c.copy()
            x[m] = noisy[m]
            specs.append(x)
            keys.append((k, "replace"))
            if not keep_only:
                continue
            x = noisy.copy()
            x[m] = c[m]
            specs.append(x)
            keys.append((k, "keep_only"))
        specs.append(noisy)
        keys.append(("ALL", "full"))
    specs.append(np.where(fin, t_s, c))
    keys.append(("ALL", "full_nonoise"))
    specs.append(c / poly)
    keys.append(("TILT", "detilt"))
    return np.stack(specs), keys


def classify(fr):
    """Apply the fixed rule to the group fracs of one flow and return the class name."""
    groups = {
        "observed-frame driven": fr.get("OBS_ALL", np.nan),
        "rest-feature driven": fr.get("REST_ALL", np.nan),
        "tilt/continuum driven": fr.get("TILT", np.nan),
        "rest-feature driven (Halpha)": fr.get("Halpha", np.nan),
    }
    if not fr.get("ALL", 0) >= 1:
        return "unreliable template"
    hits = sorted(((v, k) for k, v in groups.items() if v >= HIT), reverse=True)
    if not hits:
        return "diffuse"
    if len(hits) == 1 or hits[0][0] - hits[1][0] >= LEAD:
        return hits[0][1]
    return "diffuse"


def classify_null(fr, dl, null95):
    """Apply the fixed rule, counting a group only if its gain also beats the control null.

    Parameters
    ----------
    fr : dict
        Zone to frac for one flow.
    dl : dict
        Zone to delta log p for one flow.
    null95 : dict
        Zone to the 95th percentile of delta log p over the control galaxies.

    Returns
    -------
    str
        Class name.
    """
    fr2 = dict(fr)
    for k in ("OBS_ALL", "REST_ALL", "TILT", "Halpha"):
        if not dl.get(k, -np.inf) > null95.get(k, np.inf):
            fr2[k] = -np.inf
    return classify(fr2)


def control_occlusion(cc, ttc, tsc, cw, cz, scorers, active, rng):
    """Delta log p of every zone replacement for one unflagged control galaxy.

    Returns
    -------
    dict
        Flow name to dict of zone to delta log p, with four noise draws.
    """
    a, _, _, csc, cpoly = residuals(cc, cw, ttc, tsc)
    czn = zones_for(cz, cw)
    specs, keys = occlusion_inputs(cc, a * ttc, csc, cpoly, czn, rng, ndraw=4, keep_only=False)
    out = {}
    for nm in active:
        s = scorers[nm]
        lp0 = score(s, cc[None])[0]
        lp = score(s, specs)
        acc = {}
        for (k, mode), v in zip(keys, lp, strict=True):
            if mode in ("replace", "detilt"):
                acc.setdefault(k, []).append(v)
        out[nm] = {k: float(np.mean(v)) - lp0 for k, v in acc.items()}
    return out


def plot_candidate(tid, z, c, w, t_s, chi, good, zn, occ, thr0, path):
    """Save the residual and occlusion figure of one candidate.

    Parameters
    ----------
    occ : dict
        Flow name to dict of zone to mean delta log p (replace mode).
    thr0 : dict
        Flow name to the gap from the candidate's log p to the threshold.
    """
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1.2, 1, 1.4]})
    b = 8
    nb = WAVE.size // b
    wb = WAVE[: nb * b].reshape(nb, b).mean(1)

    def binned(y, m):
        y = np.where(m, y, np.nan)[: nb * b].reshape(nb, b)
        with np.errstate(all="ignore"):
            return np.nanmean(y, 1)

    ax[0].plot(wb, binned(c, w > 0), color="#222222", lw=0.7, label="candidate")
    ax[0].plot(wb, binned(t_s, np.isfinite(t_s)), color="#d55e00", lw=0.9, label="scaled template")
    ax[0].set_ylabel("normalised flux")
    ax[0].legend(loc="upper right", frameon=False)
    ax[0].set_title(f"TARGETID {tid}  z = {z:.3f}")
    ax[1].plot(wb, binned(chi, good), color="#222222", lw=0.6)
    ax[1].axhline(0, color="#999999", lw=0.5)
    ax[1].set_ylabel("residual / sigma (8 pix mean)")
    for a_ in ax[:2]:
        for k in ("Aband", "Bband", "br_overlap", "rz_overlap"):
            lo, hi = OBS_ZONES[k]
            a_.axvspan(lo, hi, color="#e69f00", alpha=0.18, lw=0)
        for lo, hi in REST_ZONES.values():
            f = (1 + z) * AIR_TO_VAC
            a_.axvspan(lo * f, hi * f, color="#0072b2", alpha=0.18, lw=0)
        for x in (5600, 5930, 7470, 7720):
            a_.axvline(x, color="#999999", lw=0.5, ls=":")
        a_.set_xlim(WAVE[0], WAVE[-1])
    for k, (lo, hi) in REST_ZONES.items():
        f = (1 + z) * AIR_TO_VAC
        if lo * f < WAVE[-1]:
            ax[1].text(
                (lo + hi) / 2 * f,
                0.97,
                k,
                transform=ax[1].get_xaxis_transform(),
                fontsize=6,
                ha="center",
                va="top",
                color="#0072b2",
                rotation=90,
            )
    ax[1].set_xlabel("observed wavelength [A]  (orange observed-frame zones, blue rest bands)")
    keys = [k for k in occ["cont10"] if k not in ("ALL",)]
    y = np.arange(len(keys))
    names = list(occ)
    h = 0.8 / len(names)
    colors = {"cont10": "#0072b2", "cont15": "#d55e00"}
    for j, nm in enumerate(names):
        ax[2].barh(
            y + j * h,
            [occ[nm].get(k, np.nan) for k in keys],
            height=h,
            color=colors[nm],
            label=f"{nm} (gap {thr0[nm]:.1f})",
        )
        ax[2].axvline(thr0[nm], color=colors[nm], ls="--", lw=1)
    ax[2].set_yticks(y + h * (len(names) - 1) / 2, keys, fontsize=7)
    ax[2].axvline(0, color="#999999", lw=0.5)
    ax[2].set_xlabel("delta log p after replacing the zone (dashed line is the gap to threshold)")
    ax[2].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    """Run the self-test, the residual screen and the occlusion test, then write outputs."""
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    scorers = load_scorers()
    active = [k for k, s in scorers.items() if s["enc"] is not None]
    h5 = h5py.File(LATENT_H5, "r")
    h5tid = np.asarray(h5["target_ids"], np.int64)
    zs = np.asarray(h5["zs"], np.float64)
    st = self_test(scorers, h5tid)
    print("self-test", [(r["flow"], r["status"]) for r in st][:3], "...")

    cands, sn = read_candidates()
    m, dn = load_vac(h5tid)
    row = dict(zip(h5tid.tolist(), range(len(h5tid)), strict=True))
    flagged = np.zeros(len(h5tid), bool)
    for s in scorers.values():
        flagged |= np.array([s["lp"][t] <= s["thr"] for t in h5tid])
    pool = np.flatnonzero(~flagged & np.isfinite(m) & np.isfinite(dn))

    cand_rows = load_chunk_rows(cands)
    picks = {}
    ctrl_tids = []
    for t in cands:
        sel, grow = pick_analogues(row[t], pool, zs, m, dn)
        picks[t] = (sel, grow)
        ctrl_tids += h5tid[sel[:NCTRL]].tolist()
    ctrl_rows = load_chunk_rows(ctrl_tids)

    summary, long_rows, ctrl_long = [], [], []
    for t in cands:
        c, w, z = cand_rows[t]
        sel, grow = picks[t]
        aspec = read_h5_specs(h5, sel)
        tmpl, sig = build_template(aspec, zs[sel], z)
        a, chi, good, sc, poly = residuals(c, w, tmpl, sig)
        zn = zones_for(z, w)
        stats = zone_stats(chi, good, w, zn)
        t_s = a * tmpl
        tilt_amp = float(np.ptp(poly[np.isfinite(tmpl)]))

        # Controls: the NCTRL nearest analogues against a template of the rest.
        for k in range(NCTRL):
            ct = int(h5tid[sel[k]])
            cc, cw, cz = ctrl_rows[ct]
            ttc, tsc = build_template(aspec[NCTRL:], zs[sel[NCTRL:]], cz)
            ca, cchi, cgood, csc, cpoly = residuals(cc, cw, ttc, tsc)
            czn = zones_for(cz, cw)
            for ak, av in artifact_stats(cc, ca * ttc, csc).items():
                ctrl_long.append(
                    {"candidate": t, "control": ct, "zone": ak, "npix": 0, "mean_chi2": av}
                )
            cocc = control_occlusion(cc, ttc, tsc, cw, cz, scorers, active, rng)
            for zk, v in zone_stats(cchi, cgood, cw, czn).items():
                r = {"candidate": t, "control": ct, "zone": zk, **v}
                for nm in active:
                    r[f"dlogp_{nm}"] = cocc[nm].get(zk, np.nan)
                ctrl_long.append(r)
            ctrl_long.append(
                {
                    "candidate": t,
                    "control": ct,
                    "zone": "tilt_amp",
                    "npix": 0,
                    "pix_share": np.nan,
                    "chi2_share": np.nan,
                    "mean_chi2": float(np.ptp(cpoly[np.isfinite(ttc)])),
                    **{f"dlogp_{nm}": cocc[nm]["TILT"] for nm in active},
                }
            )

        specs, keys = occlusion_inputs(c, t_s, sc, poly, zn, rng)
        occ, frac, gap, lp0s, after = {}, {}, {}, {}, {}
        for nm in active:
            s = scorers[nm]
            lp0 = s["lp"][t]
            lp = score(s, specs)
            g = s["thr"] - lp0
            acc = {}
            for (k, mode), v in zip(keys, lp, strict=True):
                acc.setdefault((k, mode), []).append(v)
            mean = {km: float(np.mean(v)) for km, v in acc.items()}
            occ[nm] = {k: mean[(k, "replace")] - lp0 for k, mode in mean if mode == "replace"}
            occ[nm]["TILT"] = mean[("TILT", "detilt")] - lp0
            occ[nm]["ALL"] = mean[("ALL", "full")] - lp0
            frac[nm] = {k: v / g for k, v in occ[nm].items()}
            gap[nm], lp0s[nm], after[nm] = g, lp0, mean
        cls = {nm: classify(frac[nm]) for nm in active}
        firm = len(active) == 2 and len(set(cls.values())) == 1
        call = cls["cont10"] if "cont10" in cls else "none"
        if len(active) == 2 and not firm:
            call = "disagree"

        for k in zn:
            r = {"targetid": t, "zone": k, **stats[k]}
            if k in REST_ZONES:
                r["overlap_obs_all"] = float((zn[k] & zn["OBS_ALL"]).sum() / max(zn[k].sum(), 1))
            else:
                r["overlap_obs_all"] = np.nan
            for nm in active:
                if (k, "replace") in after[nm]:
                    r[f"dlogp_{nm}"] = occ[nm][k]
                    r[f"frac_{nm}"] = frac[nm][k]
                    r[f"cross_{nm}"] = after[nm][(k, "replace")] > scorers[nm]["thr"]
                    r[f"keep_only_lp_{nm}"] = after[nm][(k, "keep_only")]
                    r[f"keep_only_flagged_{nm}"] = after[nm][(k, "keep_only")] <= scorers[nm]["thr"]
            long_rows.append(r)
        best = {
            nm: max(
                (
                    (v, k)
                    for k, v in frac[nm].items()
                    if k not in ("ALL", "OBS_ALL", "REST_ALL", "zarm")
                ),
                default=(np.nan, ""),
            )
            for nm in active
        }
        srow = {
            "targetid": t,
            "z": z,
            "sn_per_obsA": sn[t],
            "logmstar": m[row[t]],
            "dn4000": dn[row[t]],
            "n_analogues": len(sel),
            "window_factor": grow,
            "template_scale": a,
            "chi2_total_mean": stats["_total"]["mean_chi2"],
            "tilt_amp": tilt_amp,
            **artifact_stats(c, t_s, sc),
        }
        for nm in scorers:
            srow[f"lp_{nm}"] = scorers[nm]["lp"][t]
            srow[f"thr_{nm}"] = scorers[nm]["thr"]
        for nm in active:
            srow[f"frac_full_{nm}"] = frac[nm]["ALL"]
            srow[f"frac_full_nonoise_{nm}"] = (after[nm][("ALL", "full_nonoise")] - lp0s[nm]) / gap[
                nm
            ]
            srow[f"frac_obs_{nm}"] = frac[nm]["OBS_ALL"]
            srow[f"frac_rest_{nm}"] = frac[nm]["REST_ALL"]
            srow[f"frac_tilt_{nm}"] = frac[nm]["TILT"]
            srow[f"frac_halpha_{nm}"] = frac[nm].get("Halpha", np.nan)
            srow[f"frac_zarm_{nm}"] = frac[nm]["zarm"]
            srow[f"best_zone_{nm}"] = best[nm][1]
            srow[f"best_zone_frac_{nm}"] = best[nm][0]
            srow[f"class_{nm}"] = cls[nm]
        srow["call"] = call
        srow["firm"] = firm
        srow["_occ"] = occ
        srow["_frac"] = frac
        summary.append(srow)
        plot_candidate(t, z, c, w, t_s, chi, good, zn, occ, gap, OUT / f"screen_{t}.png")
        print(
            t,
            srow["n_analogues"],
            {nm: round(frac[nm]["OBS_ALL"], 2) for nm in active},
            {nm: round(frac[nm]["REST_ALL"], 2) for nm in active},
            call,
        )

    # Control null: 95th percentile of delta log p per zone over all control galaxies.
    null95 = {}
    for nm in active:
        by = {}
        for r in ctrl_long:
            key = "TILT" if r["zone"] == "tilt_amp" else r["zone"]
            by.setdefault(key, []).append(r.get(f"dlogp_{nm}", np.nan))
        null95[nm] = {k: float(np.nanpercentile(v, 95)) for k, v in by.items()}
    for r in long_rows:
        for nm in active:
            r[f"null95_dlogp_{nm}"] = null95[nm].get(r["zone"], np.nan)
    for sr in summary:
        occ, frac = sr.pop("_occ"), sr.pop("_frac")
        cls = {nm: classify_null(frac[nm], occ[nm], null95[nm]) for nm in active}
        for nm in active:
            for g in ("OBS_ALL", "REST_ALL", "TILT", "Halpha"):
                sr[f"dlogp_{g}_{nm}"] = occ[nm].get(g, np.nan)
                sr[f"null95_{g}_{nm}"] = null95[nm].get(g, np.nan)
            sr[f"class_null_{nm}"] = cls[nm]
        if len(active) == 2:
            sr["call_null"] = cls["cont10"] if len(set(cls.values())) == 1 else "disagree"
        else:
            sr["call_null"] = cls.get("cont10", "none")

    write_csv(OUT / "screen_summary.csv", summary)
    write_csv(OUT / "screen_zones.csv", long_rows)
    write_csv(OUT / "control_zones.csv", ctrl_long)
    print("flows run:", active, " skipped:", [k for k in scorers if k not in active])


if __name__ == "__main__":
    main()
