"""Peak finding, selection, and downsampling."""

from __future__ import annotations

import numpy as np

from cuperiod.core.peaks import (
    local_maxima,
    peak_preserving_downsample,
    select_top_peaks,
)


def test_local_maxima_interior_only() -> None:
    score = np.array([0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 0.0])
    idx = local_maxima(score)
    assert list(idx) == [1, 3, 5]


def test_select_top_peaks_separation() -> None:
    freq = np.linspace(0.1, 1.0, 100)
    score = np.zeros(100)
    score[10] = 5.0
    score[11] = 4.9  # adjacent to the first peak -> suppressed by separation
    score[80] = 4.0
    cand = local_maxima(score)
    chosen = select_top_peaks(freq, score, cand, n_peaks=2, min_separation=0.1)
    assert 10 in chosen
    assert 80 in chosen
    assert 11 not in chosen


def test_downsample_preserves_global_max() -> None:
    freq = np.linspace(0, 1, 10_000)
    power = np.random.default_rng(0).random(10_000)
    power[1234] = 99.0
    f_ds, p_ds = peak_preserving_downsample(freq, power, 500)
    assert p_ds.size == 500
    assert p_ds.max() == np.float32(99.0)


def test_downsample_noop_when_small() -> None:
    freq = np.arange(10.0)
    power = np.arange(10.0)
    f_ds, p_ds = peak_preserving_downsample(freq, power, 500)
    assert f_ds.size == 10
