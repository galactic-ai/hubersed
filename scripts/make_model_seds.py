"""Make mock DESI spectra from the prior sample written by get_stochastic_priors.

Run as ``uv run python scripts/make_model_seds.py``. Reads the priors from and writes the h5
to ``data/prospector_model/`` unless ``-n`` or ``--out`` say otherwise. The work is done by
``hubersed.mocks.seds.write_mock_seds``.
"""

import argparse
import glob
import sys
from pathlib import Path

from hubersed.mocks.seds import write_mock_seds
from hubersed.paths import PATHS

DATA_PATH = PATHS["DATA"] / "prospector_model"


def priors_path(nebular, sample_size):
    """Return the prior npz path for this nebular model and sample size."""
    stem = "stochastic_priors_sample_cue" if nebular == "cue" else "stochastic_priors_sample"
    return DATA_PATH / f"{stem}_{sample_size}.npz"


def out_path(nebular, sample_size):
    """Return the default output h5 path for this nebular model and sample size."""
    stem = (
        "prospector_stochastic_model_seds_cue"
        if nebular == "cue"
        else "prospector_stochastic_model_seds"
    )
    return DATA_PATH / f"{stem}_{sample_size}.h5"


def resolve_sample_size(nebular):
    """Return the largest sample size with a prior npz on disk, compared as numbers.

    Raises
    ------
    SystemExit
        If there is no prior npz for this nebular model.
    """
    pat = str(priors_path(nebular, "*"))
    sizes = []
    for f in glob.glob(pat):
        tail = Path(f).name.removesuffix(".npz").split("_")[-1]
        if tail.isdigit():
            sizes.append(int(tail))
    if not sizes:
        raise SystemExit(f"no priors npz matching {pat}; pass -n explicitly")
    return max(sizes)


def parse_args(argv=None):
    """Parse the command line options."""
    p = argparse.ArgumentParser(
        description="Generate mock DESI spectra from the stochastic-SFH prior sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nebular", choices=["cue", "fsps"], default="cue")
    p.add_argument(
        "-n",
        "--sample-size",
        type=int,
        default=None,
        help="number of spectra; default = largest N with a priors npz on disk",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for the logsfr_ratios draw. Keyed per-index, so it is "
        "independent of --workers and completion order. Recorded in h5 attrs.",
    )
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument(
        "-f", "--force", action="store_true", help="overwrite the output h5 if it exists"
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=500)
    return p.parse_args(argv)


def main(argv=None):
    """Compute every mock and write the h5 file.

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
    nebular = args.nebular
    n = args.sample_size if args.sample_size is not None else resolve_sample_size(nebular)

    output_file = args.out or out_path(nebular, n)
    if output_file.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite {output_file}\n"
            f"  {output_file.stat().st_size / 1e9:.1f} GB\n"
            f"  pass --force if you mean it"
        )

    write_mock_seds(
        priors_path(nebular, n), output_file, nebular, args.seed, args.workers, args.chunk_size
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
