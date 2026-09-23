"""MapFitResult refuses labels that do not match theta by name."""

import numpy as np
import pytest

from hubersed.fitting.result import MapFitResult

THETA = {
    "logzsol": np.array([-0.3]),
    "logsfr_ratios": np.arange(9.0),
    "gas_logu": np.array([-2.5]),
}
LABELS = ("logzsol", *(f"logsfr_ratios_{i}" for i in range(1, 10)), "gas_logu")


def test_vector_follows_labels():
    """vector() puts every entry of theta in labels order."""
    res = MapFitResult(1, 0.1, THETA, LABELS)
    assert res.vector().tolist() == [-0.3, *range(9), -2.5]


def test_parameter_names_as_labels_raise():
    """Names without the _1.._9 entries, like free_params, are refused."""
    with pytest.raises(ValueError, match="record has"):
        MapFitResult(1, 0.1, THETA, ("logzsol", "logsfr_ratios", "gas_logu"))


def test_record_with_other_theta_raises():
    """from_record refuses a record whose theta vector differs from theta_dict."""
    rec = {"target_id": 1, "z": 0.1, "theta_dict": THETA, "labels": list(LABELS)}
    rec["theta"] = np.r_[-0.3, np.arange(9.0), -2.4]
    with pytest.raises(ValueError, match="does not match theta"):
        MapFitResult.from_record(rec)


@pytest.mark.fsps
@pytest.mark.parametrize("kind", ["fsps", "cue"])
def test_real_model_round_trip(kind):
    """A theta dict made from theta_index matches theta_labels() for both full models."""
    pytest.importorskip("fsps")
    from hubersed.sps.config import build_continuum_model, build_full_cue_model, build_full_model

    cont_model, cont_template = build_continuum_model(0.1)
    build = build_full_model if kind == "fsps" else build_full_cue_model
    model = build(cont_template, cont_model.theta, cont_model)[0]
    theta = {k: np.atleast_1d(model.theta[v]) for k, v in model.theta_index.items()}
    res = MapFitResult(1, 0.1, theta, tuple(model.theta_labels()))
    np.testing.assert_array_equal(res.vector(), model.theta)
