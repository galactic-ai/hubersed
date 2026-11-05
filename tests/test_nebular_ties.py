"""The soft C/O tie reproduces Nicholls et al. (2017) and stays on Cue's grid."""

import numpy as np
import pytest

pytestmark = pytest.mark.fsps  # nebular_ties and prospect read SPS_HOME files on import


def test_nicholls_fiducial_point():
    """At 12+log(O/H) = 8.76 Eq. 3 gives log(C/O) = -0.337 (Nicholls+17 Table 1: 8.423 - 8.760)."""
    from hubersed.sps import nebular_ties

    gas_logz = 8.76 - 12 - nebular_ties.CUE_LOGOH_SUN
    log_co = nebular_ties.co_from_oh(gas_logz=gas_logz)[0] + nebular_ties.CUE_LOGCO_SUN
    assert abs(log_co - (8.423 - 8.760)) < 2e-3


def test_offset_and_clip():
    """The offset adds in dex, and the result is clipped to Cue's C/O grid."""
    from hubersed.sps import nebular_ties

    a = nebular_ties.co_from_oh(gas_logz=-0.8, gas_dlogco=0.0)[0]
    b = nebular_ties.co_from_oh(gas_logz=-0.8, gas_dlogco=0.1)[0]
    assert np.isclose(b - a, 0.1)
    assert nebular_ties.co_from_oh(gas_logz=-0.8, gas_dlogco=5.0)[0] == np.log10(5.4)
    assert nebular_ties.co_from_oh(gas_logz=-0.8, gas_dlogco=-5.0)[0] == -1.0


def test_add_co_tie():
    """gas_logco becomes fixed and derived, and gas_dlogco free with the scatter prior."""
    from prospect.models import priors

    from hubersed.sps import nebular_ties

    params = {
        "gas_logz": dict(N=1, isfree=True, init=0.0),
        "gas_logco": dict(N=1, isfree=True, init=0.0, prior=priors.TopHat(mini=-1.0, maxi=0.7)),
    }
    nebular_ties.add_co_tie(params, scatter=0.2)
    assert params["gas_logco"]["isfree"] is False
    assert params["gas_logco"]["depends_on"] is nebular_ties.co_from_oh
    assert "prior" not in params["gas_logco"]
    assert params["gas_dlogco"]["isfree"] and params["gas_dlogco"]["prior"].params["sigma"] == 0.2
