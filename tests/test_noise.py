"""Tests for pens.noise and its integration with EnsembleTS.random_paths."""
import numpy as np
import pytest
import pens
from pens.noise import ColoredNoise, FractionalGaussianNoise


# ---------------------------------------------------------------------------
# Unit tests for noise generators
# ---------------------------------------------------------------------------

class TestColoredNoise:
    def test_output_shape(self):
        cn = ColoredNoise(beta=1.0, t=100)
        s = cn.sample(99)
        assert s.shape == (100,)

    def test_white_noise(self):
        cn = ColoredNoise(beta=0, t=50)
        s = cn.sample(49)
        assert s.shape == (50,)

    def test_even_n(self):
        cn = ColoredNoise(beta=1.5, t=200)
        s = cn.sample(100)
        assert s.shape == (101,)

    def test_odd_n(self):
        cn = ColoredNoise(beta=1.0, t=101)
        s = cn.sample(100)  # n+1 = 101, odd
        assert s.shape == (101,)

    def test_bad_beta_type(self):
        with pytest.raises(TypeError):
            ColoredNoise(beta="red")

    def test_bad_t(self):
        with pytest.raises(ValueError):
            ColoredNoise(t=-1)

    def test_bad_n(self):
        cn = ColoredNoise()
        with pytest.raises(ValueError):
            cn.sample(0)


class TestFractionalGaussianNoise:
    @pytest.mark.parametrize("algorithm", ["daviesharte", "hosking"])
    def test_output_shape(self, algorithm):
        fgn = FractionalGaussianNoise(hurst=0.7, t=100)
        s = fgn.sample(100, algorithm=algorithm)
        assert s.shape == (100,)

    def test_hurst_half_daviesharte(self):
        fgn = FractionalGaussianNoise(hurst=0.5, t=100)
        s = fgn.sample(100, algorithm="daviesharte")
        assert s.shape == (100,)

    def test_hurst_half_hosking(self):
        fgn = FractionalGaussianNoise(hurst=0.5, t=100)
        s = fgn.sample(100, algorithm="hosking")
        assert s.shape == (100,)

    def test_bad_hurst_type(self):
        with pytest.raises(TypeError):
            FractionalGaussianNoise(hurst=1)  # int, not float

    def test_hurst_out_of_range(self):
        with pytest.raises(ValueError):
            FractionalGaussianNoise(hurst=1.0)

    def test_bad_algorithm(self):
        fgn = FractionalGaussianNoise(hurst=0.7, t=10)
        with pytest.raises(ValueError):
            fgn.sample(10, algorithm="unknown")

    def test_caching_is_consistent(self):
        """Two calls with the same hurst/n should use the cached eigenvalues."""
        fgn = FractionalGaussianNoise(hurst=0.8, t=50)
        s1 = fgn.sample(50, algorithm="daviesharte")
        s2 = fgn.sample(50, algorithm="daviesharte")
        # Results differ (random), but both have correct shape
        assert s1.shape == s2.shape == (50,)


# ---------------------------------------------------------------------------
# Integration tests via EnsembleTS.random_paths
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_ens():
    """A small synthetic EnsembleTS that mirrors the notebook pattern."""
    rng = np.random.default_rng(42)
    N, nEns = 100, 20
    time = np.arange(850, 850 + N)
    value = rng.standard_normal((N, nEns))
    return pens.EnsembleTS(time=time, value=value)


class TestRandomPathsNoise:
    def test_power_law(self, synthetic_ens):
        beta = 1.0
        paths = synthetic_ens.random_paths(model='power-law', param=beta, p=10, seed=0)
        assert paths.value.shape == (synthetic_ens.nt, 10)

    def test_fgn(self, synthetic_ens):
        hurst = 0.7
        paths = synthetic_ens.random_paths(model='fGn', param=hurst, p=10, seed=0)
        assert paths.value.shape == (synthetic_ens.nt, 10)

    def test_power_law_preserves_std(self, synthetic_ens):
        """Resampled ensemble std should track the original (statistically)."""
        beta = 1.0
        paths = synthetic_ens.random_paths(model='power-law', param=beta, p=200, seed=1)
        orig_std = synthetic_ens.get_std().value.mean()
        new_std = paths.get_std().value.mean()
        assert abs(new_std - orig_std) / orig_std < 0.15

    def test_fgn_preserves_std(self, synthetic_ens):
        hurst = 0.7
        paths = synthetic_ens.random_paths(model='fGn', param=hurst, p=200, seed=2)
        orig_std = synthetic_ens.get_std().value.mean()
        new_std = paths.get_std().value.mean()
        assert abs(new_std - orig_std) / orig_std < 0.15

    def test_power_law_reproducible(self, synthetic_ens):
        paths1 = synthetic_ens.random_paths(model='power-law', param=1.0, p=5, seed=99)
        paths2 = synthetic_ens.random_paths(model='power-law', param=1.0, p=5, seed=99)
        np.testing.assert_array_equal(paths1.value, paths2.value)

    def test_fgn_reproducible(self, synthetic_ens):
        paths1 = synthetic_ens.random_paths(model='fGn', param=0.7, p=5, seed=99)
        paths2 = synthetic_ens.random_paths(model='fGn', param=0.7, p=5, seed=99)
        np.testing.assert_array_equal(paths1.value, paths2.value)
