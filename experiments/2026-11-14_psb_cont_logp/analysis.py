"""Are PSB-like mock SFHs thin in the cont flows' mock density?

QUESTION: The gallery105 PSBs 67 and 71 are flagged only by cont10 and cont15, just past threshold,
and sit in the bulk of flows 6, 10, 15. Do PSB-like mocks have low log p in the cont flows (thin
coverage under the prior, like the emitters), or do they sit in the cont bulk?
HYPOTHESIS: none registered. Plan ~/.claude/plans/psb_cont_logp_2026-11-14.md. PSB-like median log
p at least 2 below the rest and share below threshold at least 5x the rest means thin; within 1 of
the rest means covered; quenched-not-PSB as low as PSB-like means old continua in general.
INPUTS: results/2026-11-13_mock_psb_coverage/sfh_psb.npz (sfh.py on stampede3, finish job 3719051),
the saved flows and mock latents of results/2026-11-09_mocks_v2_outliers/flows, and
data/prospector_model/mocks_v2_500000.h5 (priors/redshifts for the row check).
METHOD: PSB-like = nssfr100 < 0.2, ratio > 1, f15 > 0.4, not failed (sfh.py); quenched = nssfr100 <
0.2 and not PSB-like; rest = the others. All mocks of the S/N > 3 latent set scored with the saved
flow (training and held-out; held-out tenth also shown). Per group it gives count, median log p
and share below the saved threshold, at all z and z 0.45-0.60. Gallery105 67, 71, 88, 99
(hda_parent_67) log p and flag.
SEED: none (the split uses the flows' seed 0).
COMMAND: ``uv run --no-sync python experiments/2026-11-14_psb_cont_logp/analysis.py``.
RESULT: 2026-11-14, psb_logp.txt. Checks pass. PSB-like mocks (628; 210 in the S/N > 3 set, 38 at
z 0.45-0.60) sit in the bulk of every flow, none below threshold: cont10 median log p -4.80 vs rest
-4.65, cont15 6.49 vs 7.90 (threshold -26.3); same at z 0.45-0.60. Quenched-not-PSB also in the bulk.
So PSB SFHs are not thin in the cont flows; the cont flags of gallery105 67, 71, 88, 99 (all just
past both cont thresholds, -18.8 to -20.0 and -26.6 to -31.8) need another cause. Assumptions:
SFH-based PSB label (Wild/Pacifici cuts, mass formed); small counts; flow log p in its tail.
FIGURES: none (psb_logp.txt).
"""

import h5py
import numpy as np
import torch

from hubersed.detect.flow_model import build_flow
from hubersed.io.latents import load_latents
from hubersed.paths import PATHS

RES, ROOT, DATA = PATHS["RESULTS"], PATHS["ROOT"], PATHS["DATA"]
OUT = RES / "2026-11-14_psb_cont_logp"
FLOWS = RES / "2026-11-09_mocks_v2_outliers" / "flows"
SFH = RES / "2026-11-13_mock_psb_coverage" / "sfh_psb.npz"
MOCKS = DATA / "prospector_model" / "mocks_v2_500000.h5"
TAGS = ("cont10", "cont15", "6", "10", "15")
GALLERY = {
    "67": 39633136522363591,
    "71": 39633149579232417,
    "88": 39633407818335390,
    "99": 39633421919584782,
}
SSFR_CUT, F15_MIN = 0.2, 0.4


def log_prob(g, x):
    """Return log p of raw latents x under the saved flow checkpoint g."""
    m = build_flow(g["method"], g["dim"], g["hidden"], g["num_transforms"], g["num_bins"])
    m.load_state_dict(g["state_dict"])
    m.eval()
    xs = ((x - g["scaler_mean"]) / g["scaler_scale"]).astype(np.float32)
    with torch.no_grad():
        return np.concatenate(
            [
                m.log_prob(torch.from_numpy(xs[i : i + 65536])).numpy()
                for i in range(0, len(xs), 65536)
            ]
        )


def main():
    """Run the row check and write psb_logp.txt."""
    OUT.mkdir(parents=True, exist_ok=True)
    s = np.load(SFH)
    good = ~s["failed"].astype(bool)
    quench = s["nssfr100"] < SSFR_CUT
    psb = quench & (s["ratio"] > 1) & (s["f15"] > F15_MIN) & good
    group = {"PSB-like": psb, "quenched": quench & ~psb & good, "rest": ~quench & good}
    with h5py.File(MOCKS, "r") as h:
        z500 = h["priors/redshifts"][:]
    dz = np.max(np.abs(s["z"] - z500))
    lines = [
        f"sfh_psb.npz vs 500k file: max |z diff| {dz:.2e}; "
        + ", ".join(f"{k} {v.sum()}" for k, v in group.items())
    ]
    if dz > 1e-6:
        raise SystemExit(lines[0] + "\ncheck failed, stopping")
    for tag in TAGS:
        g = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu")
        r = torch.load(FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt", weights_only=False)
        thr = float(r["threshold"])
        mock, tid, _ = load_latents(ROOT / r["mock_file"])
        with h5py.File(ROOT / r["mock_file"], "r") as h:
            zl = h["zs"][:]
        dzl = np.max(np.abs(zl - z500[tid]))
        lines.append(
            f"\n{tag}latent: threshold {thr:.2f}; latent z vs 500k[target_id] max |diff| {dzl:.2e}"
        )
        if dzl > 1e-5:
            raise SystemExit("\n".join(lines) + "\ncheck failed, stopping")
        lp = log_prob(g, mock)
        held = np.zeros(len(mock), bool)
        held[np.random.default_rng(0).permutation(len(mock))[: len(mock) // 10]] = True
        zmid = (zl >= 0.45) & (zl <= 0.60)
        for zlab, zs in (("all z", np.ones(len(mock), bool)), ("z 0.45-0.60", zmid)):
            for slab, ss in (("all scored", np.ones(len(mock), bool)), ("held-out", held)):
                lines.append(f"  {zlab}, {slab}:")
                for name, m in group.items():
                    sel = m[tid] & zs & ss
                    if sel.sum() == 0:
                        lines.append(f"    {name:9} 0")
                        continue
                    lines.append(
                        f"    {name:9} {sel.sum():6d}  median log p {np.median(lp[sel]):7.2f}  "
                        f"below threshold {100 * np.mean(lp[sel] <= thr):6.2f}%"
                    )
        lpd = dict(
            zip(
                np.asarray(r["desi_target_ids"]).tolist(),
                np.asarray(r["log_p_desi"], float),
                strict=True,
            )
        )
        lines.append(
            "  gallery105: "
            + "; ".join(
                f"{k} {lpd[v]:.1f}{' flagged' if lpd[v] <= thr else ''}"
                if v in lpd
                else f"{k} absent"
                for k, v in GALLERY.items()
            )
        )
        print(f"{tag} done", flush=True)
    text = "\n".join(lines)
    (OUT / "psb_logp.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
