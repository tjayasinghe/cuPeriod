"""Peak finding, selection, and downsampling."""

from __future__ import annotations

import numpy as np
import pytest

from cuperiod.core.peaks import (
    is_alias_related,
    local_maxima,
    peak_preserving_downsample,
    select_diverse_peaks,
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


# --- CORE-B-6: non-finite neighbours --------------------------------------------------


def test_local_maxima_sees_past_non_finite_neighbours() -> None:
    score = np.array([1.0, 2.0, 1.0, 9.0, np.nan, 1.0, 3.0, 1.0, np.inf, 0.5, 4.0, 1.0])
    idx = local_maxima(score)
    assert 3 in idx  # 9.0 beside a NaN
    assert 10 in idx  # 4.0 two samples after a +inf
    assert 4 not in idx and 8 not in idx  # non-finite samples are never maxima


def test_local_maxima_all_finite_path_unchanged() -> None:
    rng = np.random.default_rng(3)
    score = np.round(rng.random(5000), 2)  # many ties
    expected = np.flatnonzero(
        (score[1:-1] > score[:-2]) & (score[1:-1] >= score[2:])
    ) + 1
    assert np.array_equal(local_maxima(score), expected)


# --- CORE-B-11: non-positive peak counts --------------------------------------------


def test_selectors_return_nothing_for_non_positive_counts() -> None:
    freq = np.linspace(0.1, 1.0, 100)
    score = np.zeros(100)
    score[50] = 1.0
    cand = local_maxima(score)
    for n in (0, -3):
        top = select_top_peaks(freq, score, cand, n_peaks=n, min_separation=0.1)
        div = select_diverse_peaks(freq, score, cand, n, 0.01, 8)
        assert top.size == 0 and top.dtype == np.int64
        assert div.size == 0 and div.dtype == np.int64


# --- CORE-B-1: separation and alias tolerance are separate --------------------------


def test_is_alias_related_uses_its_own_separation_for_the_direct_test() -> None:
    f1 = 1.0 / 2.47
    # 0.15 c/d apart: inside a 0.2 duplicate separation, outside a 0.01 tolerance.
    assert is_alias_related(f1 + 0.15, f1, 0.2, 8)
    assert not is_alias_related(f1 + 0.15, f1, 0.01, 8)
    assert is_alias_related(f1 + 0.15, f1, 0.01, 8, min_separation=0.2)
    # Harmonic / subharmonic / daily-alias tests keep the (small) tolerance.
    assert is_alias_related(2 * f1 + 0.005, f1, 0.01, 8, min_separation=0.2)
    assert not is_alias_related(f1 / 2 + 0.05, f1, 0.01, 8, min_separation=0.02)
    assert is_alias_related(f1 + 1.0, f1, 0.001, 8)  # solar-day alias
    assert is_alias_related(f1 + 1.0027379, f1, 0.001, 8)  # sidereal-day alias


# --- CORE-B-2 / PERF-METHODS-11: sense-aware vectorized downsampling ----------------


def _reference_downsample(
    frequency: np.ndarray, power: np.ndarray, n_points: int, sense: str
) -> tuple[np.ndarray, np.ndarray]:
    """The original per-bin loop, made NaN-safe and sense-aware."""
    n = frequency.size
    if n <= n_points:
        return frequency.astype(np.float32), power.astype(np.float32)
    score = power if sense == "max" else -power
    score = np.where(np.isnan(score), -np.inf, score)
    edges = np.linspace(0, n, n_points + 1).astype(np.int64)
    fo = np.empty(n_points, dtype=np.float32)
    po = np.empty(n_points, dtype=np.float32)
    for i in range(n_points):
        lo, hi = int(edges[i]), int(edges[i + 1])
        j = lo + int(np.argmax(score[lo:hi]))
        fo[i] = frequency[j]
        po[i] = power[j]
    return fo, po


def test_downsample_matches_reference_loop_on_random_spectra() -> None:
    rng = np.random.default_rng(0)
    for case in range(300):
        n = int(rng.integers(2, 4000))
        n_points = int(rng.integers(1, n + 2))
        power = np.round(rng.random(n), 1)  # plenty of in-bin ties
        power[rng.random(n) < 0.02] = np.nan
        power[rng.random(n) < 0.01] = np.inf
        power[rng.random(n) < 0.01] = -np.inf
        if case % 7 == 0:
            power[: n // 3] = np.nan  # whole all-NaN bins
        freq = np.sort(rng.random(n)) if case % 2 else rng.random(n)
        if case % 5 == 0:
            power = power.astype(np.float32)
        for sense in ("max", "min"):
            got = peak_preserving_downsample(freq, power, n_points, sense=sense)
            ref = _reference_downsample(freq, power, n_points, sense)
            np.testing.assert_array_equal(got[0], ref[0])
            np.testing.assert_array_equal(got[1], ref[1])


def test_downsample_large_spectrum_uses_loop_and_agrees() -> None:
    rng = np.random.default_rng(1)
    n = 300_000
    power = rng.random(n)
    power[123_456] = 50.0
    freq = np.linspace(0.0, 10.0, n)
    for sense in ("max", "min"):
        got = peak_preserving_downsample(freq, power, 100, sense=sense)
        ref = _reference_downsample(freq, power, 100, sense)
        np.testing.assert_array_equal(got[1], ref[1])


def test_downsample_non_positive_n_points_is_empty() -> None:
    freq = np.linspace(0, 1, 50)
    for n_points in (0, -2):
        f_ds, p_ds = peak_preserving_downsample(freq, freq, n_points)
        assert f_ds.size == 0 and p_ds.size == 0
        assert f_ds.dtype == np.float32


def test_downsample_rejects_unknown_sense() -> None:
    freq = np.linspace(0, 1, 50)
    with pytest.raises(ValueError, match="sense"):
        peak_preserving_downsample(freq, freq, 10, sense="maximum")  # type: ignore[arg-type]
