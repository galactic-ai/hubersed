"""Is the v2 mock density in latent space even across the prior, and where is it thin?

QUESTION: Model only, no DESI. With parameters drawn independently and uniformly in rank, how
uneven is the mock density in the latent space of flows 6, 10 and 15, which parameter ranges give
low density (spectra spread out there), and how dense are mocks in the emission-line emitters' region?
HYPOTHESIS: none registered. Plan ~/.claude/plans/logp_hist_and_effective_volume_2026-11-14.md.
INPUTS: results/2026-11-09_mocks_v2_outliers/flows (flow_nsf_<tag>latent.pt, desi_outliers_flow_nsf_
<tag>latent_snr3.pt for the mock file and threshold); data/prospector_model/mocks_v2_500000.h5
(priors/theta, theta_labels, priors/redshifts, failed).
METHOD: held-out mocks (first tenth of the default_rng(0) permutation) scored with the saved flow.
Check first: mock target_id is the row of the 500k file (make_prospector_noisy_sed.py), so the
latent-file z must equal priors/redshifts at that row and no row may be failed. Unevenness: weights
w = 1/p, Kish ESS/N = (sum w)^2 / (N sum w^2), and the largest single weight share. Per parameter,
rank u among not failed mocks (as params.py) in deciles: median log p and share below the saved
threshold. Region: u < 0.3 in dust2, gas_logno, logzsol and u > 0.6 in gas_logu, gas_logz; count
among not failed mocks against N x 0.3^3 x 0.4^2, and held-out log p inside vs outside it.
SEED: none (the split uses the flows' seed 0).
COMMAND: ``uv run --no-sync python experiments/2026-11-14_mock_effective_volume/analysis.py``.
RESULT: 2026-11-14, effvol.txt. Check passes (z match 3e-8, no failed rows). Region holds 2,204
mocks against 2,160 if independent; its held-out mocks (74) have median log p -4.7/-4.0/-4.7
against -2.3/-1.1/-1.8 for the rest (flows 6/10/15), about 10-20x lower density. Largest decile
gradients: eline_sigma (lowest decile median log p 4-6.5 below the top), then r_broad_forb, logmass,
logzsol, r_broad, dust2, all lower at low u. ESS/N of 1/p weights is 6e-5 (one mock holds 95-100% of
the weight), so raw 1/p weights are unusable. Assumptions: flow log p as the density, poor in its
own tail; 74 region mocks; u mixes redshifts for z-dependent priors.
FIGURES: none (effvol.txt).
"""

import h5py
import numpy as np
import torch
from scipy.special import logsumexp

from hubersed.detect.flow_model import build_flow
from hubersed.io.latents import load_latents
from hubersed.paths import PATHS

RES, ROOT, DATA = PATHS["RESULTS"], PATHS["ROOT"], PATHS["DATA"]
OUT = RES / "2026-11-14_mock_effective_volume"
FLOWS = RES / "2026-11-09_mocks_v2_outliers" / "flows"
MOCKS = DATA / "prospector_model" / "mocks_v2_500000.h5"
TAGS = ("6", "10", "15")
LOW, HIGH = ("dust2", "gas_logno", "logzsol"), ("gas_logu", "gas_logz")
ULOW, UHIGH = 0.3, 0.6


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


def ranks(theta, ok):
    """Return the rank u in (0, 1) of each parameter among the not failed rows, NaN elsewhere."""
    u = np.full(theta.shape, np.nan)
    for j in range(theta.shape[1]):
        r = np.argsort(np.argsort(theta[ok, j], kind="stable"), kind="stable")
        u[ok, j] = (r + 0.5) / ok.sum()
    return u


def main():
    """Run the check and write effvol.txt."""
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(MOCKS, "r") as h:
        labels = [s.decode() if isinstance(s, bytes) else str(s) for s in h.attrs["theta_labels"]]
        theta, z500 = h["priors/theta"][:], h["priors/redshifts"][:]
        ok = ~h["failed"][:].astype(bool)
    u = ranks(theta, ok)
    col = {lab: j for j, lab in enumerate(labels)}
    region = np.ones(len(u), bool)
    for lab in LOW:
        region &= u[:, col[lab]] < ULOW
    for lab in HIGH:
        region &= u[:, col[lab]] > UHIGH
    expect = ok.sum() * ULOW ** len(LOW) * (1 - UHIGH) ** len(HIGH)
    lines = [
        f"{ok.sum()} not failed mocks of {len(ok)}",
        f"region {', '.join(LOW)} u < {ULOW}; {', '.join(HIGH)} u > {UHIGH}: {region[ok].sum()} mocks, "
        f"{expect:.0f} expected if independent",
    ]
    for tag in TAGS:
        g = torch.load(FLOWS / f"flow_nsf_{tag}latent.pt", weights_only=False, map_location="cpu")
        r = torch.load(FLOWS / f"desi_outliers_flow_nsf_{tag}latent_snr3.pt", weights_only=False)
        thr = float(r["threshold"])
        mock, tid, _ = load_latents(ROOT / r["mock_file"])
        with h5py.File(ROOT / r["mock_file"], "r") as h:
            zl = h["zs"][:]
        dz = np.max(np.abs(zl - z500[tid]))
        lines.append(
            f"\n{tag}latent check: max |z latent - z 500k[target_id]| {dz:.2e}; failed rows {np.sum(~ok[tid])}"
        )
        if dz > 1e-5 or np.any(~ok[tid]):
            raise SystemExit("\n".join(lines) + "\ncheck failed, stopping")
        val = np.random.default_rng(0).permutation(len(mock))[: len(mock) // 10]
        lp, rows = log_prob(g, mock[val]), tid[val]
        lw = -lp.astype(float)
        ess = np.exp(2 * logsumexp(lw) - logsumexp(2 * lw)) / lw.size
        top = np.exp(lw.max() - logsumexp(lw))
        lines.append(
            f"  held-out {lp.size}, threshold {thr:.2f}; w = 1/p: ESS/N {ess:.2e}, largest weight share "
            f"{top:.3f} (log p {lp.min():.1f})"
        )
        uu = u[rows]
        prof = []
        for j, lab in enumerate(labels):
            d = np.clip((uu[:, j] * 10).astype(int), 0, 9)
            med = np.array([np.median(lp[d == k]) for k in range(10)])
            low = np.array([np.mean(lp[d == k] <= thr) for k in range(10)])
            prof.append((med.max() - med.min(), lab, med, low))
        lines.append("  per parameter, u deciles 1-10 (sorted by range of median log p):")
        for rng_, lab, med, low in sorted(prof, key=lambda p: -p[0]):
            lines.append(
                f"    {lab:16} range {rng_:5.2f}  median log p {' '.join(f'{v:6.2f}' for v in med)}"
            )
            lines.append(
                f"    {'':16}             % below thr  {' '.join(f'{100 * v:6.2f}' for v in low)}"
            )
        inr = region[rows]
        for lab, s in (("region", inr), ("rest", ~inr)):
            lines.append(
                f"  {lab}: {s.sum()} held-out mocks, median log p {np.median(lp[s]):.2f}, "
                f"below threshold {100 * np.mean(lp[s] <= thr):.2f}%"
            )
        print(f"{tag} done", flush=True)
    text = "\n".join(lines)
    (OUT / "effvol.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
