"""Write one DESI spectrum as an alf input file.

Run ``uv run python scripts/make_alf_input.py --tid TARGETID -o $ALF_HOME/indata/NAME.dat``
and then ``mpirun -np 8 $ALF_HOME/bin/alf.exe NAME``. The work is done by
``hubersed.alf.alf_input.write_alf_input``.
"""

import argparse
import sys

from hubersed.alf.alf_input import DEFAULT_INTERVALS, DEFAULT_MASK, write_alf_input


def main(argv=None):
    """Write the alf input file for one TARGETID.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. By default they come from sys.argv.

    Returns
    -------
    int
        Exit status.
    """
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--tid", type=int, default=39633140817331167)
    p.add_argument("-o", "--out", required=True)
    p.add_argument(
        "--intervals",
        default=DEFAULT_INTERVALS,
        help="interval edges in microns, as l1,l2,l1,l2. The default covers "
        "4000-6400 and 8000-8800 A. It leaves out 6400-8000 A, where the continuum "
        "polynomial over-fits TiO, as Beverage et al. 2025 do. At most 10 intervals.",
    )
    p.add_argument(
        "--mask",
        default=DEFAULT_MASK,
        help="rest-frame Angstrom edges to set to wgt=0, as l1,l2,l1,l2. "
        "The default is Na D, alf's NaD index band 5876.875-5909.375 A "
        "(air) padded about 2 A for vacuum. Na D picks up interstellar absorption "
        "(Conroy, Graves and van Dokkum 2014) and Beverage et al. 2025 mask it. "
        "Pass '' to disable.",
    )
    a = p.parse_args(argv)
    write_alf_input(a.tid, a.out, a.intervals, a.mask)
    return 0


if __name__ == "__main__":
    sys.exit(main())
