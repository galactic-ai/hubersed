"""Tests of the free line column ions."""

import pytest


def test_ion_lines():
    """Balmer lines are found by prefix, and a missing ion line raises."""
    from hubersed.sps.line_columns import IONS, ion_lines

    names = ["Ba-beta 4861", "Ba-alpha 6563"]
    names += [n for spec in IONS.values() if spec["lines"] for n in spec["lines"]]
    idx = ion_lines(names)
    assert idx["H"].tolist() == [0, 1]
    assert [names[i] for i in idx["[OIII]"]] == ["[O III] 4959", "[O III] 5007"]
    with pytest.raises(ValueError, match=r"\[O I\] 6363"):
        ion_lines([n for n in names if n != "[O I] 6363"])
