"""Do high-S/N cont-flow outliers stay flagged in independent DR1 repeats more than matched controls?

QUESTION: for SV3 bright targets above S/N 8, is a cont10 or cont15 outlier flagged again in its
deepest independent DR1 coadd more often than a control matched in S/N and z?
HYPOTHESIS: R = P(flag in repeat | SV3 outlier) / P(flag in repeat | control), S/N-reweighted. R < 2
means exposure-specific systematics, R > 10 the galaxy or DR1-wide systematics, between is mixed.
INPUTS: results/2026-11-13_repeat_persistence/{targets.npz, sparcl_spectra.npz} (sample.py,
fetch.py), results/2026-11-09_mocks_v2_outliers/flows, data/checkpoints.
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-13_repeat_persistence/analysis.py``.
RESULT: 2026-11-14, persistence.txt. R = 12.6 +- 2.0 (cont10) and 50.8 +- 17.8 (cont15), both above
10. Flagged again cont10 290/562, cont15 96/126.
Assumptions: repeats share SV3's pipeline and calibration; deepest repeat only; no type match.
FIGURES: none (persistence.txt, persistence.csv).
"""

import numpy as np
import pandas as pd
import torch
from spender import load_model
from spender.data.desi import DESI

from hubersed.detect.flow_model import build_flow
from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-11-13_repeat_persistence"
FLOWS = RES / "2026-11-09_mocks_v2_outliers" / "flows"
TAGS = ("cont10", "cont15")
SNR_BINS = (3, 5, 8, 15, np.inf)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # encoder only


def prepare(flux, ivar, mask, z):
    """Return spender's normalised spectrum and weights from one arm-merged DESI spectrum."""
    ivar = np.asarray(ivar, float).copy()
    bad = (np.asarray(mask) != 0) | (ivar <= 1e-6)
    ivar[bad] = 0
    spec = torch.from_numpy(np.asarray(flux, float).astype(np.float32))
    w = torch.from_numpy(ivar.astype(np.float32))
    w[DESI._skyline_mask] = 0
    rest = DESI._wave_obs / (1 + torch.tensor(np.float32(z)))
    sel = (w > 0) & (rest > 5300) & (rest < 5850)
    norm = torch.median(spec[sel]) if sel.count_nonzero() > 0 else torch.tensor(0.0)
    if not torch.isfinite(norm):
        norm = torch.tensor(0.0)
    else:
        spec /= norm
    w *= norm**2
    return spec.numpy(), w.numpy()


def score(tag, spec):
    """Return log p of every spectrum, the run log p by target, and the threshold of ``tag``."""
    d = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu")
    nde = build_flow(d["method"], d["dim"], d["hidden"], d["num_transforms"], d["num_bins"])
    nde.load_state_dict(d["state_dict"])
    nde.eval()
    enc = load_model(str(PATHS["DATA"] / "checkpoints" / d["encoder"]), DESI(),
                     map_location="cpu", weights_only=False).float().eval().to(DEVICE)  # fmt: skip
    lp = []
    with torch.no_grad():
        for k in range(0, len(spec), 512):
            lat = enc.encode(torch.from_numpy(spec[k : k + 512]).to(DEVICE)).cpu().numpy()
            x = ((lat - d["scaler_mean"]) / d["scaler_scale"]).astype(np.float32)
            lp.append(nde.log_prob(torch.from_numpy(x)).numpy())
            print(f"{tag}: scored {min(k + 512, len(spec))} of {len(spec)}", flush=True)
    o = torch.load(FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt", weights_only=False,
                   map_location="cpu")  # fmt: skip
    run = dict(zip(np.asarray(o["desi_target_ids"]).tolist(), np.asarray(o["log_p_desi"], float),
                   strict=True))  # fmt: skip
    return np.concatenate(lp), run, float(o["threshold"])


def ratio(a, na, b, nb):
    """Return a/na over b/nb and its propagated error."""
    if na == 0 or nb == 0 or a == 0 or b == 0:
        return np.nan, np.nan
    pa, pb = a / na, b / nb
    r = pa / pb
    return r, r * np.sqrt((1 - pa) / a + (1 - pb) / b)


def main():
    """Score every spectrum and write the persistence table."""
    sp = np.load(OUT / "sparcl_spectra.npz", allow_pickle=True)
    assert np.allclose(sp["wavelength"], DESI._wave_obs.numpy())
    n = len(sp["targetid"])
    tid = np.asarray(sp["targetid"], np.int64)
    key = [(str(s), str(p)) for s, p in zip(sp["survey"], sp["program"], strict=True)]
    z = np.asarray(sp["redshift"], float)
    flux, ivar, mask = sp["flux"], sp["ivar"], sp["mask"]  # read each npz member once
    spec = np.zeros((n, flux.shape[1]), np.float32)
    snr = np.zeros(n)
    for i in range(n):
        spec[i], w = prepare(flux[i], ivar[i], mask[i], z[i])
        s = spec[i] * np.sqrt(w)
        snr[i] = np.median(s[w > 0]) if (w > 0).any() else 0.0
    print(f"prepared {n} spectra", flush=True)
    sv3 = {int(tid[i]): i for i in range(n) if key[i] == ("sv3", "bright")}
    tg = np.load(OUT / "targets.npz")
    control = set(tg["control"].tolist())
    lines = [f"{n} SPARCL spectra of {len(set(tid.tolist()))} targets; SV3 bright {len(sv3)}"]
    rows = []
    for tag in TAGS:
        lp, run, thr = score(tag, spec)
        dd = [abs(lp[i] - run[t]) for t, i in sv3.items() if t in run]
        lines.append(f"{tag}: SPARCL SV3 bright vs run |d log p| median {np.median(dd):.3f}, "
                     f"p95 {np.percentile(dd, 95):.3f}")  # fmt: skip
        for t in sorted(set(tg["outlier"].tolist()) | control):
            i0 = sv3.get(t)
            if i0 is None or t not in run:
                continue
            group = "outlier" if run[t] <= thr else "control" if t in control else None
            if group is None:
                continue
            reps = [i for i in np.where(tid == t)[0] if key[i] != ("sv3", "bright")]
            use = [i for i in reps if snr[i] > 3 and abs(z[i] - z[i0]) <= 0.01 * (1 + z[i0])]
            best = max(use, key=lambda i: snr[i]) if use else None
            rows.append({"flow": tag, "target_id": t, "group": group, "sv3_snr": snr[i0],
                         "n_repeats": len(reps), "n_usable": len(use),
                         "rep_survey": "" if best is None else "/".join(key[best]),
                         "rep_snr": np.nan if best is None else snr[best],
                         "rep_flag": None if best is None else bool(lp[best] <= thr),
                         "rep_dlogp": np.nan if best is None else lp[best] - run[t]})  # fmt: skip
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "persistence.csv", index=False)
    labels = [f"{a}-{b}" for a, b in zip(SNR_BINS[:-1], SNR_BINS[1:], strict=True)]
    for tag in TAGS:
        g = tab[tab.flow == tag]
        lines.append(f"\n{tag}: outliers {np.sum(g.group == 'outlier')}, controls "
                     f"{np.sum(g.group == 'control')}; with a usable repeat: outliers "
                     f"{np.sum((g.group == 'outlier') & (g.n_usable > 0))}, controls "
                     f"{np.sum((g.group == 'control') & (g.n_usable > 0))}")  # fmt: skip
        u = g[g.n_usable > 0].copy()
        u["bin"] = pd.cut(u.rep_snr, SNR_BINS, labels=labels)
        lines.append(
            "  repeat surveys: "
            + ", ".join(f"{k} {v}" for k, v in u.rep_survey.value_counts().items())
        )
        num, den, var = 0.0, 0.0, 0.0
        for lab in labels:
            o, c = (
                u[(u.bin == lab) & (u.group == "outlier")],
                u[(u.bin == lab) & (u.group == "control")],
            )
            a, b = int(o.rep_flag.sum()), int(c.rep_flag.sum())
            r, e = ratio(a, len(o), b, len(c))
            lines.append(f"  repeat S/N {lab:>7}: outliers {a}/{len(o)} flagged, controls {b}/{len(c)}"
                         f" flagged, R {r:.1f} +- {e:.1f}")  # fmt: skip
            if len(c) and len(o):
                pc = b / len(c)
                num, den = num + a, den + len(o) * pc
                var += len(o) ** 2 * pc * (1 - pc) / len(c)
        po = u[u.group == "outlier"]
        r = num / den if den else np.nan
        e = r * np.sqrt(1 / max(num, 1) + var / den**2) if den else np.nan
        lines.append(f"  all bins: outliers {int(po.rep_flag.sum())}/{len(po)} flagged; R (control "
                     f"reweighted to the outlier repeat S/N bins) {r:.1f} +- {e:.1f}")  # fmt: skip
        lines.append(f"  median repeat - SV3 log p: outliers {np.nanmedian(po.rep_dlogp):.2f}, "
                     f"controls {np.nanmedian(u[u.group == 'control'].rep_dlogp):.2f}")  # fmt: skip
    text = "\n".join(lines)
    (OUT / "persistence.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
