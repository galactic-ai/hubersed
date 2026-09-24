"""Read alf chains with the walker count from the .sum header, not an assumed one."""

import numpy as np

from hubersed.alf.alf_output import LABELS, _lib_corr, convergence, read_header

HEADER = """#   Elapsed Time:   1.27 hr
#    fit_type  = 0
#   Nwalkers   =   1024
#   facc:  0.034
0.0 1.0 2.0
"""


def frozen_chain(nwalkers, nsteps=6):
    """Chain in alf's row order where each walker keeps its own value at every step."""
    walker_value = np.arange(nwalkers, dtype=float)
    W = np.broadcast_to(walker_value[None, :, None], (nsteps, nwalkers, len(LABELS)))
    rows = W.reshape(nsteps * nwalkers, len(LABELS))
    return {k: rows[:, i] for i, k in enumerate(LABELS)}


def test_read_header_gets_nwalkers(tmp_path):
    """Header lines with an equals sign are read, other lines are skipped."""
    (tmp_path / "run.sum").write_text(HEADER)
    header = read_header(tmp_path / "run")
    assert header["Nwalkers"] == "1024"
    assert header["fit_type"] == "0"
    assert "facc" not in "".join(header)


def test_frozen_walkers_do_not_move():
    """With the right walker count, walkers that never move report zero moves."""
    assert convergence(frozen_chain(8), nwalkers=8)["moved"] == 0.0


def test_wrong_walker_count_mixes_walkers():
    """Reading 8 walkers as 4 compares different walkers and reports moves."""
    assert convergence(frozen_chain(8), nwalkers=4)["moved"] > 0.0


def test_library_correction_extrapolates_like_read_alf():
    """Outside the table the correction continues the end slope, as alf's read_alf.py does."""
    zh = np.array([0.4, -2.0])
    np.testing.assert_allclose(_lib_corr("Mg", zh)[0], 0.03)  # 0.04 - 0.2 * 0.05
    np.testing.assert_allclose(_lib_corr("a", zh)[1], 0.8)  # 0.6 + 0.4 * 0.5
    assert _lib_corr("C", zh) == 0.0
