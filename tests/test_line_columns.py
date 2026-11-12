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


def _names():
    """Return Balmer and every IONS line name."""
    from hubersed.sps.line_columns import IONS

    names = ["Ba-beta 4861", "Ba-alpha 6563"]
    return names + [n for spec in IONS.values() if spec["lines"] for n in spec["lines"]]


@pytest.mark.fsps
def test_build_columns():
    """Nine weak columns replace Cue lines, six broad ones add to Cue with their own widths."""
    from hubersed.sps.line_columns import build_columns

    keys = ["[SII]", "[OI]", "[ArIII]7135", "[OIII]4363", "[OII]7325", "[NII]5755", "[NeIII]",
            "[SIII]6312", "[SIII]9069"]  # fmt: skip
    broad = ["H", "[OII]", "[OIII]", "[NII]", "[SII]", "[OI]"]
    limits = {k: float(i + 1) for i, k in enumerate(keys + [f"{b} broad" for b in broad])}
    cols = build_columns(_names(), limits)
    assert [c.name for c in cols] == keys + [f"{b} broad" for b in broad]
    assert [c.replace for c in cols] == [True] * 9 + [False] * 6
    assert {c.sigma for c in cols[:9]} == {"eline_sigma"} and {c.velocity for c in cols[:9]} == {
        None
    }
    assert cols[9].sigma == "eline_sigma_broad" and cols[9].lines == (
        "Ba-beta 4861",
        "Ba-alpha 6563",
    )
    assert {c.sigma for c in cols[10:]} == {"eline_sigma_broad_forb"}
    assert {c.velocity for c in cols[9:]} == {"eline_vbroad"}
    assert [c.ew_max for c in cols] == [limits[c.name] for c in cols]


@pytest.mark.fsps
def test_add_line_params():
    """Narrow width 10-250 km/s, broad widths 250-800 km/s, broad offset within +-300 km/s."""
    from prospect.models import priors

    from hubersed.sps.line_columns import add_line_params

    params = add_line_params({"eline_sigma": dict(N=1, isfree=True, init=100.0,
                                                  prior=priors.TopHat(mini=10.0, maxi=250.0))})  # fmt: skip
    assert params["eline_sigma"]["prior"].range == (10.0, 250.0)
    assert params["eline_sigma"]["init"] == 130.0
    for k in ("eline_sigma_broad", "eline_sigma_broad_forb"):
        assert params[k]["isfree"] and params[k]["prior"].range == (250.0, 800.0)
    assert params["eline_vbroad"]["prior"].range == (-300.0, 300.0)
