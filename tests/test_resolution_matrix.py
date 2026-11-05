"""Resampled DESI resolution matrix R', line profiles and LOSVD smoothing on synthetic inputs."""

import numpy as np
import pytest
from scipy.sparse import dia_matrix

from hubersed.sps import resolution_matrix as rm

NDIAG = 11
DPIX = 0.8


def synthetic_banded(n=300, ndiag=NDIAG, seed=0):
    """Return a grid and a banded matrix with skewed rows of varying width and row sum.

    Row i is a skewed Gaussian of width 0.55-0.95 pixels, scaled to a row sum of 0.95-1.05, so
    the rows are not symmetric, not normalised to one, and differ from their columns.
    """
    rng = np.random.default_rng(seed)
    wave = 5000.0 + DPIX * np.arange(n)
    c = ndiag // 2
    d = np.arange(-c, c + 1)
    width = 0.55 + 0.4 * rng.uniform(size=n)
    skew = 0.3 * rng.uniform(-1, 1, size=n)
    norm = 0.95 + 0.1 * rng.uniform(size=n)
    dense = np.zeros((n, n))
    for i in range(n):
        k = np.exp(-0.5 * (d / width[i]) ** 2) * (1 + skew[i] * np.tanh(d))
        j = i + d
        ok = (j >= 0) & (j < n)
        dense[i, j[ok]] = norm[i] * k[ok] / k.sum()
    data = np.zeros((ndiag, n))
    offsets = rm.banded_offsets(ndiag)
    for kk, off in enumerate(offsets):
        j = np.arange(n)
        i = j - off
        ok = (i >= 0) & (i < n)
        data[kk, j[ok]] = dense[i[ok], j[ok]]
    return wave, data, dense


def test_banded_convention_matches_definition_and_scipy():
    """R[j - offset_k, j] = data[k, j] with offsets ndiag//2 down to -(ndiag//2), as scipy dia."""
    wave, data, dense = synthetic_banded()
    offsets = NDIAG // 2 - np.arange(NDIAG)
    assert np.array_equal(rm.banded_offsets(NDIAG), offsets)
    scipy_dense = dia_matrix((data, offsets), shape=dense.shape).toarray()
    assert np.array_equal(scipy_dense, dense)
    assert np.array_equal(rm.banded_to_csr(data).toarray(), dense)
    # data[:, j] holds column j, from row j + ndiag//2 at k = 0 down to row j - ndiag//2
    j = 100
    assert np.array_equal(data[:, j], dense[j - offsets, j])
    with pytest.raises(ValueError, match="odd"):
        rm.banded_offsets(10)


def test_rprime_keeps_row_sums():
    """The rows of R' sum to the rows of R, which are not normalised to one here."""
    wave, data, dense = synthetic_banded()
    fine = rm.fine_grid(wave, NDIAG)
    rp = rm.resampled_matrix(wave, data, fine)
    assert rp.shape == (wave.size, fine.size)
    assert np.allclose(np.asarray(rp.sum(axis=1)).ravel(), dense.sum(axis=1), rtol=0, atol=1e-5)


def test_rprime_equals_r_on_smooth_input():
    """On a smooth model R' gives what R gives on the model sampled at the pixels."""
    wave, data, dense = synthetic_banded()
    fine = rm.fine_grid(wave, NDIAG)
    rp = rm.resampled_matrix(wave, data, fine)

    def model(w):
        return 1.0 + 0.3 * np.sin(2 * np.pi * (w - 5000.0) / 150.0) + 1e-4 * (w - 5000.0)

    out, ref = rp @ model(fine), dense @ model(wave)
    assert np.max(np.abs(out - ref)) < 1e-5 * np.max(np.abs(ref))


def test_unresolved_line_on_a_pixel_gives_its_column():
    """A delta function at pixel j, on the fine grid, maps to column j of R times step / 0.8 A."""
    wave, data, dense = synthetic_banded()
    fine = rm.fine_grid(wave, NDIAG)
    step = fine[1] - fine[0]
    rp = rm.resampled_matrix(wave, data, fine)
    for j in (0, 3, 150, wave.size - 1):
        f = int(np.argmin(np.abs(fine - wave[j])))
        assert abs(fine[f] - wave[j]) < 1e-9
        m = np.zeros(fine.size)
        m[f] = 1.0
        # the fine points sit within 1e-9 A of the pixels, which the spline slope turns into 1e-9
        assert np.allclose(rp @ m * DPIX / step, dense[:, j], rtol=0, atol=1e-8)


def test_rprime_requires_uniform_float64_grid():
    """A float32 or uneven grid is refused; extraction_grid rebuilds the float64 DESI grid."""
    wave, data, _ = synthetic_banded()
    fine = rm.fine_grid(wave, NDIAG)
    with pytest.raises(ValueError, match="not uniform"):
        rm.resampled_matrix(wave.astype(np.float32), data, fine)
    uneven = wave.copy()
    uneven[50] += 0.1 * DPIX
    with pytest.raises(ValueError, match="not uniform"):
        rm.resampled_matrix(uneven, data, fine)
    with pytest.raises(ValueError, match="not uniform"):
        rm.extraction_grid(uneven)
    with pytest.raises(ValueError, match="cover"):
        rm.resampled_matrix(wave, data, fine[20:])

    desi = np.linspace(3600.0, 9824.0, 7781)
    w32 = desi.astype(np.float32)
    with pytest.raises(ValueError, match="not uniform"):
        rm.uniform_step(w32)
    grid = rm.extraction_grid(w32)
    assert grid.dtype == np.float64
    assert np.array_equal(grid, desi)


def raw_profile(w, mu, sigma):
    """Return prospect's unnormalised Gaussian line profile per Hz at ``w``."""
    dv = rm.C_KMS * (w / mu - 1)
    g = np.exp(-0.5 * (dv / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return g * rm.C_KMS * w**2 / (rm.LIGHTSPEED * mu)


@pytest.mark.parametrize("sigma", [10.0, 45.0, 800.0])
def test_line_profile_has_unit_flux_in_frequency(sigma):
    """The profile sums to one over frequency, is centred on mu, and has width mu sigma / c."""
    fine = np.arange(4000.0, 7000.0, rm.FINE_STEP)
    mu = 5008.24
    sl, g = rm.line_profile(fine, mu, sigma)
    w = fine[sl]
    dnu = rm.LIGHTSPEED / w**2 * (fine[-1] - fine[0]) / (fine.size - 1)
    assert np.isclose(np.sum(g * dnu), 1.0, rtol=0, atol=1e-12)
    # the scale only fixes the discrete sum; it is one to the sampling accuracy
    assert np.allclose(g, raw_profile(w, mu, sigma), rtol=1e-9, atol=0)
    mean = np.sum(w * g * dnu)
    assert abs(mean - mu) < 1e-6 * mu * sigma / rm.C_KMS + 1e-9
    var = np.sum((w - mu) ** 2 * g * dnu)
    assert np.isclose(np.sqrt(var), mu * sigma / rm.C_KMS, rtol=1e-6)


def test_line_profile_at_grid_edge_and_too_narrow():
    """A line cut by the grid keeps its inside values; a line narrower than the step is refused."""
    fine = np.arange(4000.0, 7000.0, rm.FINE_STEP)
    cut = fine[: np.searchsorted(fine, 5010.0)]
    mu = 5008.24
    sl_full, g_full = rm.line_profile(fine, mu, 100.0)
    sl_cut, g_cut = rm.line_profile(cut, mu, 100.0)
    assert sl_cut.start == sl_full.start and sl_cut.stop == cut.size
    assert np.allclose(g_cut, g_full[: sl_cut.stop - sl_cut.start], rtol=1e-12, atol=0)
    sl, g = rm.line_profile(fine, 3000.0, 100.0)
    assert g.size == 0
    with pytest.raises(ValueError, match="narrower"):
        rm.line_profile(fine, mu, 1.0)
    assert rm.line_profile(fine, 1000.0, 1.0)[1].size == 0


def test_losvd_kernel_removes_library_in_quadrature():
    """The kernel is sqrt(sigma^2 - lib^2), and zero when sigma_smooth <= lib."""
    assert np.isclose(rm.losvd_kernel(30.0, 12.7), np.sqrt(30.0**2 - 12.7**2))
    assert rm.losvd_kernel(10.0, 12.7) == 0.0
    assert rm.losvd_kernel(12.7, 12.7) == 0.0
    assert rm.losvd_kernel(12.7, 0.0) == 12.7


def log_grid(lo=4000.0, hi=6000.0, dv=10.0):
    """Return a grid uniform in log wavelength with step ``dv`` km/s."""
    return np.exp(np.arange(np.log(lo), np.log(hi), dv / rm.C_KMS))


def test_smooth_log_uniform_without_kernel_keeps_samples():
    """With no kernel the samples come back unchanged, on a grid four times finer."""
    rng = np.random.default_rng(1)
    w = log_grid()
    s = 1.0 + 0.1 * rng.normal(size=w.size)
    for n in (w.size, w.size - 1):
        wo, so = rm.smooth_log_uniform(w[:n], s[:n], 0.0, upsample=4)
        assert wo.size == 4 * (n - 1) + 1
        assert np.allclose(wo[::4], w[:n], rtol=1e-12, atol=0)
        assert np.allclose(so[::4], s[:n], rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="log"):
        rm.smooth_log_uniform(np.linspace(4000.0, 6000.0, 500), s[:500], 10.0)


def test_smooth_log_uniform_adds_widths_in_quadrature():
    """A Gaussian absorption line of width s0 smoothed by sk has width sqrt(s0^2 + sk^2)."""
    w = log_grid()
    v = rm.C_KMS * np.log(w / 5000.0)
    s0, sk = 15.0, 25.0
    depth = 0.5 * np.exp(-0.5 * (v / s0) ** 2)
    wo, so = rm.smooth_log_uniform(w, 1.0 - depth, sk, upsample=4)
    vo = rm.C_KMS * np.log(wo / 5000.0)
    d = 1.0 - so
    dv = vo[1] - vo[0]
    assert np.isclose(np.sum(d) * dv, np.sum(depth) * (v[1] - v[0]), rtol=1e-9)
    width = np.sqrt(np.sum(d * vo**2) / np.sum(d))
    assert np.isclose(width, np.hypot(s0, sk), rtol=1e-4)
    # the ends are far from the line, where the continuum stays at one
    assert np.allclose(so[:100], 1.0, atol=1e-9) and np.allclose(so[-100:], 1.0, atol=1e-9)


def test_smooth_continuum_uses_native_samples_when_log_uniform():
    """On a log-uniform window the library samples are used; elsewhere sedpy resamples."""
    lo = np.linspace(1000.0, 3300.0, 200, endpoint=False)
    w = np.concatenate([lo, log_grid(3300.0, 10000.0)])
    s = 1.0 + 0.2 * np.sin(w / 37.0)
    wo, so = rm.smooth_continuum(w, s, 0.0, 4000.0, 9000.0)
    assert wo[0] <= 4000.0 and wo[-1] >= 9000.0
    i0 = np.searchsorted(w, wo[0])
    assert np.allclose(so[::4], s[i0 : i0 + so[::4].size], rtol=0, atol=1e-12)
    wo, so = rm.smooth_continuum(w, s, 20.0, 1500.0, 9000.0)
    assert wo[0] <= 1500.0 and wo[-1] >= 9000.0
    assert np.array_equal(wo, w[(w >= wo[0]) & (w <= wo[-1])])


def test_load_resolution_npz(tmp_path):
    """The merged and per-arm keys load as float64 and are checked."""
    wave, data, _ = synthetic_banded()
    n = wave.size
    path = tmp_path / "res.npz"
    np.savez(
        path,
        wave=wave,
        res=data,
        flux=np.ones(n),
        ivar=np.ones(n),
        b_wave=wave[:100],
        b_res=data[:, :100].astype(np.float32),
        b_flux=np.ones(100),
        b_ivar=np.ones(100),
        b_mask=np.zeros(100, dtype=np.uint32),
    )
    d = rm.load_resolution_npz(path)
    assert d["res"].shape == (NDIAG, n) and all(v.dtype == np.float64 for v in d.values())
    b = rm.load_resolution_npz(path, arm="b")
    assert b["res"].shape == (NDIAG, 100) and b["res"].dtype == np.float64
    with pytest.raises(ValueError, match="arm"):
        rm.load_resolution_npz(path, arm="x")
    bad = tmp_path / "bad.npz"
    np.savez(bad, wave=wave, res=data[:10], flux=np.ones(n), ivar=np.ones(n))
    with pytest.raises(ValueError, match="odd"):
        rm.load_resolution_npz(bad)
