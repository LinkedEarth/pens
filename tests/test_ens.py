"""Tests for EnsembleTS trend/mean KDE plots."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pens.ens import EnsembleTS


@pytest.mark.parametrize('segment_length, step, expected', [
    (50, 50, '50y'),    # 1026 years -> 20 blocks; nt // nblocks would give 51
    (100, 100, '100y'), # 10 blocks; nt // nblocks would give 102
    (50, 25, '50y'),    # overlapping blocks
])
@pytest.mark.parametrize('tag', ['mean', 'trend'])
def test_plot_trend_mean_kde_block_length_label(segment_length, step, expected, tag):
    rng = np.random.default_rng(0)
    time = np.arange(1000, 2026)  # 1026 years, not a multiple of the block length
    ens = EnsembleTS(time=time, value=rng.standard_normal((time.size, 5)))
    ens = ens.get_means_and_trends(segment_length=segment_length, step=step)
    fig, ax = ens.plot_trend_mean_kde(tag=tag)
    label = ax.get_legend().get_texts()[0].get_text()
    plt.close(fig)
    assert label.startswith(expected + ',')


@pytest.mark.parametrize('tag', ['mean', 'trend'])
def test_plot_block_kde_runs(tag):
    rng = np.random.default_rng(0)
    time = np.arange(1000, 2026)
    ens = EnsembleTS(time=time, value=rng.standard_normal((time.size, 5)))
    ens = ens.get_means_and_trends(segment_length=50, step=50)
    fig, ax = ens.plot_block_kde(tag=tag)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close(fig)
    assert labels[1] == '1976–2025'
