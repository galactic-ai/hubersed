"""Draw the stochastic-SFH prior sample used for generating mock SEDs.

Run as ``uv run python scripts/get_stochastic_priors.py``. Writes one npz of parameter
arrays to ``data/prospector_model/`` unless ``--out`` is given. The draws come from
``hubersed.mocks.priors.draw_priors``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from hubersed.mocks.priors import draw_priors
from hubersed.paths import PATHS


def parse_args(argv=None):
    """Parse the command line options."""
    p = argparse.ArgumentParser(
        description="Draw the stochastic-SFH prior sample used for generating mock SEDs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-n", "--sample-size", type=int, default=500_000, help="Number of samples to draw"
    )
    p.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument(
        "--cue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cue (Li+24) nebular model with free N/O, C/O, nH",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output file path (default: DATA_PATH/prospector_model/stochastic_priors_sample[_cue]_{n}.npz)",
    )
    p.add_argument(
        "-f", "--force", action="store_true", help="Overwrite existing output file if it exists"
    )

    return p.parse_args(argv)


def main(argv=None):
    """Draw the prior sample and save it as an npz.

    Returns
    -------
    int
        Exit status.

    Raises
    ------
    SystemExit
        If the output file exists and ``--force`` is not given.
    """
    args = parse_args(argv)
    n = args.sample_size

    data_path = PATHS["DATA"] / "prospector_model"
    data_path.mkdir(parents=True, exist_ok=True)

    out = args.out or data_path / (
        f"stochastic_priors_sample_cue_{n}.npz" if args.cue else f"stochastic_priors_sample_{n}.npz"
    )
    if out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing file {out}. Use --force to overwrite.")

    arrays = draw_priors(n, args.seed, args.cue)
    np.savez(out, **arrays)
    print(f"saved {out}  (n={n}, cue={args.cue}, seed={args.seed}, keys={len(arrays)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
