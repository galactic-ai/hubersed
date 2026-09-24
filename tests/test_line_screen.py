"""read_gaia_screen refuses a Gaia CSV without the on-source columns."""

import pytest

from hubersed.detect.line_screen import read_gaia_screen


def test_full_screen_is_keyed_by_target_id(tmp_path):
    """A file with onsource_star and sep_arcsec comes back keyed by integer target id."""
    path = tmp_path / "gaia.csv"
    path.write_text("target_id,sep_arcsec,onsource_star,flagged\n7,0.2,True,True\n")
    rows = read_gaia_screen(path)
    assert list(rows) == [7]
    assert rows[7]["onsource_star"] == "True"


def test_flag_only_screen_raises(tmp_path):
    """The 2026-09-04 layout, target_id and flagged only, is refused by name."""
    path = tmp_path / "gaia.csv"
    path.write_text("target_id,flagged\n7,True\n")
    with pytest.raises(ValueError, match="onsource_star, sep_arcsec"):
        read_gaia_screen(path)
