"""Combine the 6, 10 and 15 latent flow outlier files in DIR into the 3-of-3 core.

Run as ``uv run python scripts/combine_flow_outliers.py DIR``. Reads
``desi_outliers_flow_nsf_<tag>_snr3.pt`` for each tag from DIR and writes
``ood_core_3nsf_em_lines.pt`` there. The voting is ``hubersed.detect.outliers.core``.
"""

import argparse
import json
from pathlib import Path

import torch

from hubersed.detect.outliers import core


def main(argv=None):
    """Write the 3-of-3 core of the three line-flow outlier files in DIR.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. By default they come from sys.argv.
    """
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("dir", type=Path, help="folder holding the flow outlier files")
    d = p.parse_args(argv).dir
    paths = [d / f"desi_outliers_flow_nsf_{t}_snr3.pt" for t in ("6latent", "10latent", "15latent")]
    tids, metas, counts = core(paths)
    print(json.dumps(counts, indent=2))

    for m in metas:
        print(
            f"{m['tag']}: {m['n']} flagged, c2st={m['c2st']:.3f}, threshold={m['threshold']:.3f}, mock_file={m['mock_file']}"
        )

    torch.save(
        {
            "core_target_ids": torch.tensor(sorted(tids)),
            "k": 3,
            "encoders": ["6latent", "10latent", "15latent"],
            "source_dir": str(d),
            "per_encoder_c2st": counts["c2st"],
        },
        d / "ood_core_3nsf_em_lines.pt",
    )


if __name__ == "__main__":
    main()
