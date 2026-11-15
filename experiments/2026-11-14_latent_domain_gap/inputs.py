"""Compare the encoder inputs of DESI and the v2 mocks as stored next to their latents.

QUESTION: Do the spectra fed to the encoders differ between DESI and the mocks in ways unrelated to
galaxy physics (length, masked zeros, noise at the spender sky-line mask)?
INPUTS: data/latents/spender_spec_cont10latent_snr3.h5 and data/latents/mocks_v2/
prospector_noise_spec_cuev2_cont10latent_snr3.h5 (specs: the normalised spectra the encoder saw).
METHOD: first 3000 spectra of each. Exact-zero pixels per spectrum; median |pixel-to-pixel
difference| (a noise proxy) at the spender sky-line mask pixels and elsewhere. Code paths read:
spender data/desi.py prepare_spectra l.520-560 (bitmasked pixels keep flux 0, sky-line pixels keep
their flux with weight 0); scripts/make_prospector_noisy_sed.py l.104 (last pixel dropped), l.123-136
(sky-line pixels get S/N NaN, so sigma 0 and no noise).
SEED: none.
COMMAND: ``uv run --no-sync python experiments/2026-11-14_latent_domain_gap/inputs.py``.
RESULT: 2026-11-14, inputs.txt. The sky-line mask is 1941 of 7781 pixels (25%). Mock inputs have
7780 pixels, DESI 7781. At sky-mask pixels the mocks are noiseless (median |diff| 0.001 vs 0.064
elsewhere); DESI is noisy there (0.127; 0.170 elsewhere). 20.6% of DESI spectra have exact-zero
(bitmasked) pixels (p90 13), mocks 1.9%. Whether these drive the separation is not tested.
Assumptions: cont10 inputs only; first 3000 of each; |diff| as a noise proxy.
FIGURES: none (inputs.txt).
"""

import h5py
import numpy as np
from spender.data.desi import DESI

from hubersed.paths import PATHS

OUT = PATHS["RESULTS"] / "2026-11-14_latent_domain_gap"
FILES = {
    "DESI": PATHS["DATA"] / "latents" / "spender_spec_cont10latent_snr3.h5",
    "mocks": PATHS["DATA"]
    / "latents"
    / "mocks_v2"
    / "prospector_noise_spec_cuev2_cont10latent_snr3.h5",
}
NSPEC = 3000


def main():
    """Write the input comparison."""
    OUT.mkdir(parents=True, exist_ok=True)
    sky = DESI._skyline_mask.numpy()
    lines = [f"spender sky-line mask: {sky.sum()} of {sky.size} pixels"]
    for name, path in FILES.items():
        with h5py.File(path, "r") as h:
            s = h["specs"][:NSPEC].astype(float)
        n = s.shape[1]
        zero = (s == 0).sum(1)
        d = np.abs(np.diff(s, axis=1))
        sk = sky[1:n]
        lines.append(
            f"{name}: {n} pixels; spectra with an exact-zero pixel {np.mean(zero > 0):.3f}, zero pixels "
            f"per spectrum p50 {np.median(zero):.0f} p90 {np.percentile(zero, 90):.0f}; median |diff| at "
            f"sky-mask pixels {np.median(d[:, sk]):.3f}, elsewhere {np.median(d[:, ~sk]):.3f}"
        )
    text = "\n".join(lines)
    (OUT / "inputs.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
