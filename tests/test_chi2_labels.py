"""map_chi2_one saves one label per theta entry, so labels and theta line up by position."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.fsps


@pytest.mark.parametrize("use_cue", [False, True])
def test_labels_match_theta(monkeypatch, use_cue):
    """Each named value in theta_dict sits at its labels' positions in theta.

    The spectrum, the optimizer, the stellar sources and the model prediction are
    replaced, so only the record is exercised.
    """
    pytest.importorskip("fsps")
    from prospect.models.sedmodel import HyperSpecModel

    from hubersed.fitting import chi2

    n = len(chi2.WAVE_OBS)
    fake = (np.full(n, 5.0, np.float32), np.full(n, 4.0, np.float32), 0.1, 123)
    sps = SimpleNamespace(ssp=SimpleNamespace(emline_wavelengths=np.array([4862.7, 6564.6])))
    monkeypatch.setattr(chi2, "load_by_index", lambda gidx: fake)
    monkeypatch.setattr(chi2, "_fsps", lambda: sps)
    monkeypatch.setattr(chi2, "_cue", lambda: sps)
    monkeypatch.setattr(chi2, "_map_optimize", lambda neg, th, **kw: SimpleNamespace(x=th))
    monkeypatch.setattr(HyperSpecModel, "predict", lambda self, th, **kw: ([np.ones(n)], None))

    rec = chi2.map_chi2_one(0, use_cue=use_cue)
    assert rec["status"] == "ok"
    labels, theta = rec["theta_labels"], rec["theta"]
    assert len(labels) == len(theta)
    for name, value in rec["theta_dict"].items():
        at = [i for i, lab in enumerate(labels) if lab == name or lab.startswith(name + "_")]
        at = [i for i in at if labels[i] == name or labels[i][len(name) + 1 :].isdigit()]
        np.testing.assert_array_equal(theta[at], value)
