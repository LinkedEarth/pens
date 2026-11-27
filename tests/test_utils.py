"""Tests for pens.utils.means_and_trends_ensemble (block slicing, see issue #10)."""
import numpy as np
import pytest
from pens.utils import means_and_trends_ensemble


def _ramp(n_years, n_ens=2):
    years = np.arange(1991, 1991 + n_years)
    var = np.tile(np.arange(float(n_years))[:, None], (1, n_ens))  # value = index
    return var, years


def test_blocks_use_full_segment_length():
    var, years = _ramp(40)
    means, trends, tm, idxs = means_and_trends_ensemble(var, 10, 10, years)
    assert idxs.tolist() == [[0, 9], [10, 19], [20, 29], [30, 39]]
    np.testing.assert_allclose(means[:, 0], [4.5, 14.5, 24.5, 34.5])
    np.testing.assert_allclose(tm, [1995.5, 2005.5, 2015.5, 2025.5])
    assert years[idxs[-1, 1]] == years[-1]  # most recent block ends on the last year


def test_skip_leading_years_when_not_divisible():
    var, years = _ramp(43)
    _, _, _, idxs = means_and_trends_ensemble(var, 10, 10, years)
    assert idxs.tolist() == [[3, 12], [13, 22], [23, 32], [33, 42]]


def test_overlapping_blocks():
    var, years = _ramp(30)
    _, _, _, idxs = means_and_trends_ensemble(var, 10, 5, years)
    assert idxs.tolist() == [[0, 9], [5, 14], [10, 19], [15, 24], [20, 29]]


def test_three_full_blocks_accepted():
    var, years = _ramp(30)
    means, _, _, idxs = means_and_trends_ensemble(var, 10, 10, years)
    assert means.shape == (3, 2)
    assert idxs.tolist() == [[0, 9], [10, 19], [20, 29]]


def test_fewer_than_two_blocks_rejected():
    var, years = _ramp(19)
    with pytest.raises(ValueError):
        means_and_trends_ensemble(var, 10, 10, years)


def test_trends_exact_on_raw_block_data():
    rng = np.random.default_rng(0)
    years = np.arange(1991, 2031)
    var = rng.standard_normal((40, 3))
    _, trends, _, idxs = means_and_trends_ensemble(var, 10, 10, years)
    for k, (i0, i1) in enumerate(idxs):
        for m in range(var.shape[1]):
            expected = np.polyfit(years[i0:i1 + 1], var[i0:i1 + 1, m], 1)[0]
            np.testing.assert_allclose(trends[k, m], expected)


def test_trends_are_block_local():
    # changing years outside a block must not change that block's trend
    rng = np.random.default_rng(1)
    years = np.arange(1991, 2031)
    var = rng.standard_normal((40, 2))
    _, trends, _, _ = means_and_trends_ensemble(var, 10, 10, years)
    var2 = var.copy()
    var2[:30] += 5 * rng.standard_normal((30, 2))  # perturb all but the last block
    _, trends2, _, _ = means_and_trends_ensemble(var2, 10, 10, years)
    np.testing.assert_allclose(trends2[-1], trends[-1])
