"""Does the outlier flag of 11 and 62 depend on depth? SV3 bright coadd degraded to the main spectrum's noise.

QUESTION: 39627776201853253 (11) and 39633275186053602 (62) are flagged in their SV3 bright coadd but
not in any single exposure or in their main bright spectrum rescored at the SV3 redshift. Does the
flag go away when the SV3 coadd gets the main spectrum's noise?
HYPOTHESIS: none registered. If the flag needs the coadd's depth, most degraded realisations are
not flagged.
INPUTS: results/2026-11-01_repeats92/sparcl_spectra.npz (SV3 bright and main bright coadds); encoders
and seed 0 flows as 2026-11-01_repeats92/analysis.py; prepare() of analysis.py in this folder.
Gaussian noise of variance max(1/ivar_main - 1/ivar_sv3, 0) is added to the SV3 flux on pixels good
in both, which gives it the main spectrum's per-pixel noise there; the degraded spectrum keeps the
SV3 mask and redshift. The main spectrum is also scored at the SV3 redshift for comparison.
SEED: 0, N_REAL realisations per target.
COMMAND: ``uv run --no-sync python experiments/2026-11-02_exposure_checks/degrade.py``.
RESULT: 2026-11-02, degrade.txt. Degrading removes the flag. 11 (added variance 3.8x the SV3
variance): cont10 -24.93 to a median -4.55, cont15 -54.38 to -8.41, flagged in 1/50 realisations.
62 (added 2.0x): cont10 -15.77 to -3.39 (0/50 flagged), cont15 -41.29 to -15.75 (8/50). The main
spectra at the SV3 redshift score -3.43/+3.76 and -0.32/-8.36, inside those ranges.
Assumptions: Gaussian, uncorrelated added noise; main ivar taken as correct; seed 0 flows; the SV3
coadd keeps its own mask and its known DR1 ivar underestimate.
FIGURES: results/2026-11-02_exposure_checks/degrade.txt.
"""

from pathlib import Path

import numpy as np
import torch
from spender import load_model
from spender.data.desi import DESI

from hubersed.paths import PATHS

SRC = Path(__file__).with_name("analysis.py").read_text()
NS = {"__file__": str(Path(__file__).with_name("analysis.py"))}
exec(SRC.split("# %% targets")[0], NS)  # constants and helper functions only
prepare, load_flow = NS["prepare"], NS["load_flow"]
FLOWS, OUT, TAGS = NS["FLOWS"], NS["OUT"], NS["TAGS"]

TARGETS = {"11": 39627776201853253, "62": 39633275186053602}
N_REAL = 50


def main():
    """Score the SV3 coadd, the main spectrum and N_REAL degraded SV3 coadds of each target."""
    sp = np.load(
        PATHS["RESULTS"] / "2026-11-01_repeats92" / "sparcl_spectra.npz", allow_pickle=True
    )
    rng = np.random.default_rng(0)
    batches = {}
    for nn, t in TARGETS.items():
        idx = [i for i in range(len(sp["targetid"])) if int(sp["targetid"][i]) == t]
        pick = {(str(sp["survey"][i]), str(sp["program"][i])): i for i in idx}
        a, b = pick[("sv3", "bright")], pick[("main", "bright")]
        z = float(sp["redshift"][a])
        fa, iva, ma = (np.asarray(sp[k][a], float) for k in ("flux", "ivar", "mask"))
        ivb, mb = np.asarray(sp["ivar"][b], float), np.asarray(sp["mask"][b])
        both = (ma == 0) & (iva > 0) & (mb == 0) & (ivb > 0)
        var_add = np.where(
            both, np.clip(1 / np.where(both, ivb, 1) - 1 / np.where(both, iva, 1), 0, None), 0
        )
        specs = [prepare(fa, iva, ma, z)[0], prepare(sp["flux"][b], ivb, mb, z)[0]]
        for _ in range(N_REAL):
            noisy = fa + rng.normal(0.0, np.sqrt(var_add))
            iv_new = np.where(both, ivb, iva)
            specs.append(prepare(noisy, iv_new, ma, z)[0])
        batches[nn] = (t, z, np.array(specs), float(np.median(var_add[both] * iva[both])))
    inst = DESI()
    res = {nn: {} for nn in TARGETS}
    for tag in TAGS:
        nde, mean, scale, enc = load_flow(tag)
        model = load_model(str(PATHS["DATA"] / "checkpoints" / enc), inst, map_location="cpu",
                           weights_only=False).float().eval()  # fmt: skip
        o = torch.load(FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt", weights_only=False,
                       map_location="cpu")  # fmt: skip
        thr = float(o["threshold"])
        for nn, (_, _, x, _) in batches.items():
            with torch.no_grad():
                lat = model.encode(torch.from_numpy(x.astype(np.float32))).numpy()
                lp = nde.log_prob(
                    torch.from_numpy(((lat - mean) / scale).astype(np.float32))
                ).numpy()
            res[nn][tag] = (lp, thr)
    lines = [f"SV3 bright coadd degraded to the main bright noise, {N_REAL} realisations (seed 0). "
             "added = median added variance over the SV3 variance on shared pixels.", ""]  # fmt: skip
    for nn, (t, z, _, ratio) in batches.items():
        lines.append(f"== {nn} {t} z {z:.4f}, added {ratio:.2f}")
        for tag in TAGS:
            lp, thr = res[nn][tag]
            deg = lp[2:]
            lines.append(
                f"  {tag}: threshold {thr:.2f}; SV3 coadd {lp[0]:.2f}; main at SV3 z {lp[1]:.2f}; "
                f"degraded median {np.median(deg):.2f} [16-84: {np.percentile(deg, 16):.2f}, "
                f"{np.percentile(deg, 84):.2f}], flagged {int(np.sum(deg <= thr))}/{deg.size}"
            )
    text = "\n".join(lines)
    (OUT / "degrade.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
