"""theta is read by name via theta_index: logsfr_ratios fills 9 slots, so positional labels shift."""

import numpy as np
import pytest

pytestmark = pytest.mark.fsps
Z = 0.1


@pytest.fixture(scope="module", params=["fsps", "cue"])
def model(request):
    """Full MAP model with FSPS or Cue nebular emission, seeded from the continuum init."""
    pytest.importorskip("fsps")
    from hubersed.fitting.config import (
        build_continuum_model,
        build_full_cue_model,
        build_full_model,
    )

    cont_model, cont_template = build_continuum_model(Z)
    build = build_full_model if request.param == "fsps" else build_full_cue_model
    return build(cont_template, cont_model.theta, cont_model, Z)[0]


def test_logsfr_ratios_fills_nine_slots(model):
    """logsfr_ratios is one name but 9 theta entries, so theta is 8 longer than free_params."""
    ind = model.theta_index["logsfr_ratios"]
    assert ind.stop - ind.start == 9
    assert len(model.theta) == len(model.free_params) + 8


def test_positional_labels_misread_gas_logu(model):
    """Zipping free_params with theta reads gas_logu from the wrong slot; theta_index does not."""
    theta = np.arange(len(model.theta), dtype=float)  # value = slot number
    by_name = theta[model.theta_index["gas_logu"]][0]
    by_position = dict(zip(model.free_params, theta, strict=False))["gas_logu"]
    assert by_position == by_name - 8


def test_theta_labels_match_theta(model):
    """theta_labels() has one label per theta entry, so it is the safe positional list."""
    labels = model.theta_labels()
    assert len(labels) == len(model.theta)
    assert labels[model.theta_index["gas_logu"].start] == "gas_logu"
