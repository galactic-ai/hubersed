"""Round trip of the SSP slot cache: a loaded slot gives the same spectrum as a built one."""

import numpy as np
import pytest

pytestmark = pytest.mark.fsps


def _population():
    """Return a population fixed at one metallicity index, so one slot is built."""
    import fsps

    return fsps.StellarPopulation(zcontinuous=0, zmet=3, sfh=0, imf_type=1)


def test_round_trip(tmp_path):
    """Save two slots, clear the cache, load them and compare the spectrum exactly.

    zmet 4 is saved first, so the load rebuilds it and copies in zmet 3, the slot the spectrum uses.
    """
    from hubersed.sps.ssp_cache import load_ssp_cache, save_ssp_cache

    sp = _population()
    _, built = sp.get_spectrum(tage=1.0)
    path = save_ssp_cache(sp, tmp_path / "cache.npz", zmet=[4, 3], afeindx=[1])
    sp.params["imf_type"] = 2  # another SSP input: FSPS clears its cache
    sp.get_spectrum(tage=1.0)
    sp.params["imf_type"] = 1
    assert load_ssp_cache(sp, path) == 2
    _, loaded = sp.get_spectrum(tage=1.0)
    assert np.array_equal(built, loaded)


def test_refuses_other_inputs(tmp_path):
    """A cache built with other SSP parameters is not loaded."""
    from hubersed.sps.ssp_cache import load_ssp_cache, save_ssp_cache

    sp = _population()
    sp.get_spectrum(tage=1.0)
    path = save_ssp_cache(sp, tmp_path / "cache.npz", zmet=[3], afeindx=[1])
    sp.params["imf_type"] = 2
    with pytest.raises(ValueError, match="other SSP parameters"):
        load_ssp_cache(sp, path)


def test_loaded_slot_is_used(tmp_path):
    """The spectrum comes from the copied slot, not a rebuild: doubling it doubles the spectrum."""
    from hubersed.sps.ssp_cache import load_ssp_cache, save_ssp_cache

    sp = _population()
    _, built = sp.get_spectrum(tage=1.0)
    path = save_ssp_cache(sp, tmp_path / "cache.npz", zmet=[4, 3], afeindx=[1])
    with np.load(path) as f:
        parts = {k: f[k] for k in f.files}
    parts["spec"][1] *= 2.0  # slot zmet 3, copied in by the load
    np.savez(path, **parts)
    sp.params["imf_type"] = 2
    sp.get_spectrum(tage=1.0)
    sp.params["imf_type"] = 1
    load_ssp_cache(sp, path)
    _, loaded = sp.get_spectrum(tage=1.0)
    assert np.allclose(loaded, 2.0 * built, rtol=1e-12, atol=0)
