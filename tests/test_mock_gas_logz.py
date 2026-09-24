"""Each mock uses its drawn gas_logz, not the stellar logzsol, for FSPS and Cue nebular models."""

import numpy as np
import pytest

pytestmark = pytest.mark.fsps


@pytest.mark.parametrize("nebular", ["fsps", "cue"])
def test_model_gas_logz_is_the_drawn_value(tmp_path, monkeypatch, nebular):
    """The model built for mock i has gas_logz equal to the prior sample, not logzsol."""
    pytest.importorskip("fsps")
    from prospect.models.sedmodel import HyperSpecModel

    from hubersed.mocks import seds as mms
    from hubersed.mocks.priors import draw_priors

    monkeypatch.setattr(mms, "_setup_process", lambda: None)  # keep this process untouched
    priors_file = tmp_path / "priors.npz"
    np.savez(priors_file, **draw_priors(3, seed=0, cue=nebular == "cue"))
    mms._init_worker(priors_file, nebular, seed=0)

    for i in range(3):
        parset, _ = mms.build_parset_for_index(i)
        model = HyperSpecModel(configuration=parset)
        # predict applies depends_on through set_parameters (prospect parameters.py:146-158)
        model.set_parameters(model.theta)
        gas = float(np.atleast_1d(model.params["gas_logz"])[0])
        drawn = float(mms._S["priors"]["gas_metallicities"][i])
        stellar = float(mms._S["priors"]["stellar_metallicities"][i])
        assert gas == pytest.approx(drawn)
        assert gas != pytest.approx(stellar)
