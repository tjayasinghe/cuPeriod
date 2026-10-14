"""Tests for slow-trend removal (`cuperiod.running_median` / `cuperiod.detrend`).

The behavioural tests of record (a known trend tracked, outliers ignored,
non-finite points, level restoration) live with the period-profile tests that
motivated detrending; these pin the kernels themselves: the sliding-window
median against the direct per-point ``np.median`` it replaced, the binned
kernel's occupied-bin bookkeeping, and ``min_points`` honoured by both methods.
"""

from __future__ import annotations

import numpy as np
import pytest

import cuperiod as cup
from cuperiod.core._typing import FloatArray

# --- references: the v1.3.0 kernels, kept verbatim -----------------------------


def _reference_median(t: FloatArray, y: FloatArray, width: float, min_points: int):
    """Direct per-point running median (the v1.3.0 implementation)."""
    out = np.empty(t.size, dtype=np.float64)
    good = np.isfinite(t) & np.isfinite(y)
    if not np.any(good):
        out.fill(np.nan)
        return out
    fallback = float(np.median(y[good]))
    out.fill(fallback)
    order = np.argsort(t[good], kind="stable")
    idx = np.flatnonzero(good)[order]
    ts, ys = t[idx], y[idx]
    half = 0.5 * width
    lo = np.searchsorted(ts, ts - half, side="left")
    hi = np.searchsorted(ts, ts + half, side="right")
    trend = np.full(ts.size, fallback, dtype=np.float64)
    for i in range(ts.size):
        a, b = int(lo[i]), int(hi[i])
        if b - a >= min_points:
            trend[i] = float(np.median(ys[a:b]))
    out[idx] = trend
    return out


def _reference_binned(t: FloatArray, y: FloatArray, width: float):
    """Every occupied bin as a node (the v1.3.0 kernel, i.e. min_points=1)."""
    good = np.isfinite(t) & np.isfinite(y)
    out = np.full(t.size, float(np.median(y[good])), dtype=np.float64)
    order = np.argsort(t[good], kind="stable")
    idx = np.flatnonzero(good)[order]
    ts, ys = t[idx], y[idx]
    step = 0.5 * width
    bin_of = np.floor((ts - ts[0]) / step).astype(np.int64)
    n_bins = int(bin_of[-1]) + 1
    edges = np.searchsorted(bin_of, np.arange(n_bins + 1))
    centres, levels = [], []
    for k in range(n_bins):
        a, b = int(edges[k]), int(edges[k + 1])
        if b > a:
            centres.append(float(np.median(ts[a:b])))
            levels.append(float(np.median(ys[a:b])))
    if len(centres) == 1:
        out[idx] = levels[0]
    else:
        out[idx] = np.interp(ts, centres, levels)
    return out


def _case(kind: str, seed: int, n: int = 400) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 300.0, n)
    y = 13.0 + 0.3 * np.sin(t / 40.0) + rng.normal(0.0, 0.05, n)
    if kind == "sorted":
        t = np.sort(t)
    elif kind == "ties":
        t = np.round(t / 3.0) * 3.0  # many identical epochs
        y = np.round(y, 1)  # many identical values
    elif kind == "nonfinite":
        t[rng.choice(n, 10, replace=False)] = np.nan
        y[rng.choice(n, 10, replace=False)] = np.nan
        t[rng.choice(n, 3, replace=False)] = np.inf
        y[rng.choice(n, 3, replace=False)] = -np.inf
    elif kind == "gapped":
        t = np.sort(np.concatenate([t[: n // 2] * 0.2, 900.0 + t[n // 2 :] * 0.2]))
    return t, y


# --- running median: sliding window ---------------------------------------------


@pytest.mark.parametrize("kind", ["sorted", "shuffled", "ties", "nonfinite", "gapped"])
@pytest.mark.parametrize("width", [0.05, 2.0, 40.0, 1e9])
@pytest.mark.parametrize("min_points", [1, 3, 50])
def test_sliding_median_matches_direct_median(kind, width, min_points):
    t, y = _case(kind, seed=len(kind) + int(width))
    ref = _reference_median(t, y, width, min_points)
    got = cup.running_median(t, y, width, min_points=min_points)
    assert np.array_equal(got, ref, equal_nan=True)


def test_sliding_median_extreme_magnitudes():
    t = np.arange(64.0)
    y = np.where(np.arange(64) % 2 == 0, 1e308, -1e308)
    ref = _reference_median(t, y, 6.0, 3)
    assert np.array_equal(cup.running_median(t, y, 6.0), ref, equal_nan=True)


# --- binned trend ----------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sorted", "shuffled", "ties", "nonfinite", "gapped"])
@pytest.mark.parametrize("width", [0.5, 10.0, 60.0, 1e9])
def test_binned_matches_reference_on_occupied_bins(kind, width):
    t, y = _case(kind, seed=3 + len(kind))
    ref = _reference_binned(t, y, width)
    got = cup.running_median(t, y, width, method="binned", min_points=1)
    assert np.array_equal(got, ref, equal_nan=True)


@pytest.mark.parametrize("stray", [0.0, 1e15, 1e300, -1e300])
def test_binned_work_scales_with_points_not_span(stray):
    # Bins used to be allocated and visited over the whole span: one stray epoch
    # (a 0.0 "missing" sentinel among JDs) meant millions of empty bins, 1e15
    # asked for hundreds of TiB, and 1e300 overflowed the int64 bin index.
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(2450000.0, 2453000.0, 2000))
    y = 13.0 + 0.2 * np.sin(t / 300.0) + rng.normal(0.0, 0.02, t.size)
    t[0] = stray
    trend = cup.running_median(t, y, 0.05 if stray == 0.0 else 40.0, method="binned")
    assert trend.shape == t.shape
    assert np.all(np.isfinite(trend))


def test_binned_respects_min_points():
    # A bin holding one observation used to become its own interpolation node:
    # the trend passed exactly through the point and erased its excursion.
    rng = np.random.default_rng(4)
    s1 = np.sort(rng.uniform(0.0, 150.0, 400))
    s2 = np.sort(rng.uniform(365.0, 515.0, 400))
    t = np.concatenate([s1, [257.0], s2])
    y = 13.2 + 0.05 * np.sin(2 * np.pi * t / 1.3) + rng.normal(0.0, 0.01, t.size)
    iso = 400
    y[iso] = 12.65
    trend = cup.running_median(t, y, 40.0, method="binned", min_points=3)
    assert abs(trend[iso] - y[iso]) > 0.4
    flat = cup.detrend(t, y, 40.0, method="binned", min_points=3)
    assert flat[iso] < 12.8  # the excursion survives detrending
    # min_points=1 keeps the old every-bin behaviour.
    loose = cup.running_median(t, y, 40.0, method="binned", min_points=1)
    assert loose[iso] == pytest.approx(y[iso])


@pytest.mark.parametrize("method", ["median", "binned"])
def test_min_points_above_n_returns_the_global_median(method):
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0.0, 100.0, 300))
    y = rng.normal(10.0, 0.5, t.size)
    trend = cup.running_median(t, y, 10.0, method=method, min_points=10**6)
    assert np.all(trend == np.median(y))


def test_binned_narrower_than_cadence_keeps_the_signal():
    # Every width/2 bin holds one point: with min_points honoured nothing
    # qualifies, so the trend is the global median instead of the data itself.
    t = np.arange(40.0)
    y = np.sin(t)
    flat = cup.detrend(t, y, 2.0, method="binned")
    assert np.std(flat) == pytest.approx(np.std(y))


def test_masked_input_counts_as_missing():
    # np.asarray() of a masked array silently used the hidden fill values.
    base_t, base_y = np.arange(50.0), np.sin(np.arange(50.0))
    t = np.ma.masked_array(base_t, mask=np.arange(50) == 7)
    y = np.ma.masked_array(base_y, mask=np.arange(50) == 11)
    trend = cup.running_median(t, y, 10.0)
    keep = np.setdiff1d(np.arange(50), [7, 11])
    ref = cup.running_median(base_t[keep], base_y[keep], 10.0)
    assert np.array_equal(trend[keep], ref)
    assert np.isnan(cup.detrend(t, y, 10.0)[11])
