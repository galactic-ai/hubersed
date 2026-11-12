"""Do the screened outliers stay outliers in their other DESI DR1 spectra?

QUESTION: For each screened old and new outlier (screened.npz), do its other DR1 coadds (main, SV1,
SV3 dark ...) also pass the flag rule of its branch (full: 6 & 10 & 15, cont: cont10 & cont15)
with the same flows? An outlier set by one bad exposure, fibre or sky should not repeat; one set by
the galaxy should.
HYPOTHESIS: none stated before the run.
METHOD: as experiments/2026-11-01_repeats92/analysis.py: each SPARCL spectrum goes through spender's
steps (mask or ivar <= 1e-6 to zero weight, sky-line mask, median at rest 5300-5850 A, own
redshift), is encoded with each tag's encoder and scored with its flow (old flows for old lists,
new for new). A repeat counts if its median per-pixel S/N > 3 (the parent cut) and its redshift
is within 0.01 (1 + z) of the SV3 bright one. Per target: robust (flagged in every usable repeat),
cleared (in none), mixed, or no usable repeat. Check: SPARCL SV3 bright spectra must reproduce the
run log p.
INPUTS: results/2026-11-10_data_quality_screen/{screened.npz, sparcl_spectra.npz}; flows of
results/noised_cue_meanzero_wide_flow (old) and results/2026-11-09_mocks_v2_outliers/flows (new);
encoders in data/checkpoints; gallery92 labels.
SEED: none.
COMMAND: ``uv run python experiments/2026-11-10_data_quality_screen/repeats.py``.
RESULT: SPARCL SV3 bright reproduces the run (|d log p| median <= 0.002). New: full 88 -> robust
16, cleared 5, no usable repeat 67; cont 81 -> robust 15, cleared 15, mixed 2, none 49. Old: full
223 -> robust 51, cleared 11, mixed 1, none 160; cont 67 -> robust 8, cleared 10, mixed 1, none
48. gallery92 (old cont): all 6 bad_sky/bad_data with a usable repeat clear; AGN 7 robust, 2
cleared. Assumptions: a usable repeat needs median S/N > 3 and z within 0.01 (1 + z); main-survey
coadds are shallower, so a faint real feature can clear. New outliers not in the old screened
list (100): near-miss (some old flow of the branch flagged it) 67 -> robust 12, cleared 7, mixed 2,
none 46; genuinely new 33 -> robust 2, cleared 5, none 26.
FIGURES: none (repeats.txt, repeats.csv).
"""

# %%
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from spender import load_model
from spender.data.desi import DESI

from hubersed.detect.flow_model import build_flow
from hubersed.paths import PATHS

RES = PATHS["RESULTS"]
OUT = RES / "2026-11-10_data_quality_screen"
FLOWS = {
    "old": RES / "noised_cue_meanzero_wide_flow",
    "new": RES / "2026-11-09_mocks_v2_outliers" / "flows",
}
BRANCH = {"full": ("6", "10", "15"), "cont": ("cont10", "cont15")}
lines = []


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


# %% spectra
sp = np.load(OUT / "sparcl_spectra.npz", allow_pickle=True)
assert np.allclose(sp["wavelength"], DESI._wave_obs.numpy())
n = len(sp["targetid"])
tid = np.asarray(sp["targetid"], np.int64)
key = [(str(s), str(p)) for s, p in zip(sp["survey"], sp["program"], strict=True)]
spec = np.zeros((n, sp["flux"].shape[1]), np.float32)
snr = np.zeros(n)
for i in range(n):
    spec[i], w = prepare(sp["flux"][i], sp["ivar"][i], sp["mask"][i], sp["redshift"][i])
    s = spec[i] * np.sqrt(w)
    snr[i] = np.median(s[w > 0]) if (w > 0).any() else 0.0
z = np.asarray(sp["redshift"], float)
sv3 = {int(tid[i]): i for i in range(n) if key[i] == ("sv3", "bright")}

# %% scores: every spectrum under every flow; flagged = log p <= the run threshold
inst = DESI()
flag, check = {}, []
for fs, folder in FLOWS.items():
    for tags in BRANCH.values():
        for tag in tags:
            d = torch.load(
                folder / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu"
            )
            nde = build_flow(d["method"], d["dim"], d["hidden"], d["num_transforms"], d["num_bins"])
            nde.load_state_dict(d["state_dict"])
            nde.eval()
            enc = load_model(
                str(PATHS["DATA"] / "checkpoints" / d["encoder"]),
                inst,
                map_location="cpu",
                weights_only=False,
            )
            enc = enc.float().eval()
            with torch.no_grad():
                lat = enc.encode(torch.from_numpy(spec)).numpy()
                x = torch.from_numpy(
                    ((lat - d["scaler_mean"]) / d["scaler_scale"]).astype(np.float32)
                )
                lp = nde.log_prob(x).numpy()
            o = torch.load(
                folder / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt",
                weights_only=False,
                map_location="cpu",
            )
            run = dict(
                zip(
                    np.asarray(o["desi_target_ids"]).tolist(),
                    np.asarray(o["log_p_desi"], float),
                    strict=True,
                )
            )
            dd = [abs(lp[i] - run[t]) for t, i in sv3.items() if t in run]
            check.append(
                f"{fs} {tag}: SPARCL SV3 bright vs run |d log p| median {np.median(dd):.3f}, p95 {np.percentile(dd, 95):.3f}"
            )
            flag[(fs, tag)] = lp <= float(o["threshold"])
lines += check

# %% per target: repeats under its branch rule
s = np.load(OUT / "screened.npz")
lab = {}
for dd in glob.glob(str(RES / "2026-11-01_gallery92") + "/*/"):
    c = os.path.basename(dd.rstrip("/"))
    for f in glob.glob(dd + "*.png"):
        m = re.search(r"(\d{15,})", os.path.basename(f))
        if m and c != "cutouts":
            lab[int(m[1])] = c
rows = []
for fs in FLOWS:
    for br, tags in BRANCH.items():
        for t in s[f"{fs}_{br}"].tolist():
            i0 = sv3.get(t)
            z0 = z[i0] if i0 is not None else np.nan
            reps = [i for i in np.where(tid == t)[0] if key[i] != ("sv3", "bright")]
            use = [i for i in reps if snr[i] > 3 and abs(z[i] - z0) <= 0.01 * (1 + z0)]
            hit = [all(flag[(fs, tg)][i] for tg in tags) for i in use]
            state = (
                "no repeat"
                if not use
                else "robust"
                if all(hit)
                else "cleared"
                if not any(hit)
                else "mixed"
            )
            rows.append({"flows": fs, "branch": br, "target_id": t, "n_repeats": len(reps), "n_usable": len(use),
                         "n_flagged": sum(hit), "state": state, "gallery92": lab.get(t, "")})  # fmt: skip
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "repeats.csv", index=False)
for (fs, br), g in tab.groupby(["flows", "branch"]):
    lines.append(
        f"{fs} {br}: {len(g)} targets, "
        + ", ".join(f"{k} {v}" for k, v in g["state"].value_counts().items())
    )
g = tab[tab["gallery92"] != ""]
lines.append("gallery92 label x state (old cont):")
lines.append(
    pd.crosstab(g[(g.flows == "old") & (g.branch == "cont")]["gallery92"], g["state"]).to_string()
)

# %% new outliers not in the old screened list: near-miss (some old flow of the branch flagged it)
# or genuinely new (none did), against the repeat state
old_fl = {}
for tags in BRANCH.values():
    for tg in tags:
        o = torch.load(
            FLOWS["old"] / f"desi_outliers_flow_nsf_{tg}latent_snr3.pt",
            weights_only=False,
            map_location="cpu",
        )
        old_fl[tg] = {int(x) for x in o["outlier_target_ids"]}
old_all = set(s["old_full"].tolist()) | set(s["old_cont"].tolist())
nw = tab[(tab.flows == "new") & ~tab.target_id.isin(old_all)].copy()
nw["old_flows"] = [
    sum(t in old_fl[tg] for tg in BRANCH[b]) for t, b in zip(nw.target_id, nw.branch, strict=True)
]
nw["kind"] = np.where(nw["old_flows"] > 0, "near-miss", "genuinely new")
lines.append("new outliers not in the old screened list, by old-flow coverage and repeat state:")
lines.append(pd.crosstab([nw.branch, nw.kind], nw.state, margins=True).to_string())

text = "\n".join(lines)
print(text)
(OUT / "repeats.txt").write_text(text + "\n")
