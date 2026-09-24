"""FSPS gas priors match the FSPS nebular grid, which FSPS clamps instead of extrapolating."""

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.fsps


def _grid_axes(path):
    """Read the log Z and log U values of an FSPS nebular .lines file.

    After the header and the wavelength line, the file alternates a line of
    ``logZ age logU`` with a line of line fluxes.
    """
    rows = path.read_text().splitlines()[2::2]
    params = np.array([[float(x) for x in r.split()] for r in rows])
    return np.unique(params[:, 0]), np.unique(params[:, 2])


@pytest.fixture(scope="module", params=["ZAU_ND_mist.lines", "ZAU_WD_mist.lines"])
def axes(request):
    """Return the log Z and log U axes of one FSPS nebular grid."""
    if "SPS_HOME" not in os.environ:
        pytest.skip("SPS_HOME not set")
    return _grid_axes(Path(os.environ["SPS_HOME"]) / "nebular" / request.param)


def test_fit_priors_span_the_grid(axes):
    """The FSPS full model's gas_logz and gas_logu priors are exactly the grid range."""
    pytest.importorskip("fsps")
    from hubersed.sps.config import build_continuum_model, build_full_model

    logz, logu = axes
    cont_model, cont_template = build_continuum_model(0.1)
    model = build_full_model(cont_template, cont_model.theta, cont_model)[0]
    for name, grid in (("gas_logz", logz), ("gas_logu", logu)):
        lo, hi = model.config_dict[name]["prior"].range
        assert (lo, hi) == pytest.approx((grid.min(), grid.max()))


def test_fsps_mock_draws_stay_on_the_grid(axes, tmp_path):
    """draw_priors draws FSPS gas_logz inside the grid, filling most of it."""
    from hubersed.mocks.priors import draw_priors

    logz, _ = axes
    z = draw_priors(2000, seed=0, cue=False)["gas_metallicities"]
    assert logz.min() <= z.min() and z.max() <= logz.max()
    assert z.max() - z.min() > 0.95 * (logz.max() - logz.min())
