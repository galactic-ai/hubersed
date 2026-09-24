"""The FSPS and Cue full models start eline_sigma at the same 100 km/s."""

import pytest

pytestmark = pytest.mark.fsps


@pytest.mark.parametrize("kind", ["fsps", "cue"])
def test_eline_sigma_starts_at_100(kind):
    """Both full models start eline_sigma at 100 km/s, the prospect template value."""
    pytest.importorskip("fsps")
    from hubersed.sps.config import build_continuum_model, build_full_cue_model, build_full_model

    cont_model, cont_template = build_continuum_model(0.1)
    build = build_full_model if kind == "fsps" else build_full_cue_model
    template = build(cont_template, cont_model.theta, cont_model)[1]
    assert template["eline_sigma"]["init"] == 100.0
