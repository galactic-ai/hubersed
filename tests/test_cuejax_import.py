"""prospect's Cue basis keeps its cuejax functions under the pytest warning filters."""

import pytest

pytestmark = pytest.mark.fsps


def test_nebssp_basis_has_cuejax_functions():
    """The jaxopt import warning must not make nebssp_basis silently drop cuejax."""
    pytest.importorskip("fsps")
    import prospect.sources.nebssp_basis as nebssp

    assert hasattr(nebssp, "fit_4loglinear_ionparam")
    assert hasattr(nebssp, "Emulator")
