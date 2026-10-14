"""Slow-trend removal: :func:`running_median` and :func:`detrend`.

Real survey photometry of a period-changing star usually carries a brightness
trend far slower than the period and far larger than the variability: a
spotted star's activity cycle, an inspiralling binary brightening as it
contracts, a blended neighbour, or plain calibration drift. A periodogram sees
that trend as power at very low frequency, and the sampling window scatters it
straight back onto the 1 c/d alias comb — which is where it competes with the
signal. On the OGLE photometry of V1309 Sco the 0.8 mag secular brightening
does exactly this, and an undetrended search returns a one-day alias rather
than the orbital period.

The fix is a high-pass filter with a width chosen to sit in the gap between the
two scales: several times the period, and a small fraction of the trend's own
timescale. A *median* rather than a mean, because eclipses and outliers are
one-sided and would drag a running mean into the very feature being measured.

Detrending is deliberately not automatic anywhere in the period-change tools.
The width is a physical judgement about a particular star, a too-narrow window
eats the signal it was meant to preserve, and silently reshaping a light curve
before measuring its period is not something a library should do unasked.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from typing import Any, Literal

import numpy as np

from cuperiod.core._coerce import as_float_array
from cuperiod.core._typing import FloatArray, IntArray

#: Filter kernels understood by :func:`running_median`.
TrendMethod = Literal["median", "binned"]


def running_median(
    time: Any,
    value: Any,
    width: float,
    *,
    method: TrendMethod = "median",
    min_points: int = 3,
) -> FloatArray:
    """The slow trend of ``value``: a moving median of full width ``width``.

    Parameters
    ----------
    time : array-like
        Observation epochs (days). Need not be sorted.
    value : array-like
        Brightness (magnitudes or flux).
    width : float
        Full window width in days. Choose it well above the period being
        studied and well below the trend's own timescale; a factor of ~20 above
        the period is a reasonable starting point.
    method : {"median", "binned"}, default "median"
        ``"median"`` evaluates an exact running median centred on every point,
        sliding one sorted window across the time-ordered data: ``O(N log k)``
        comparisons plus ``O(k)`` memory moves per point for ``k`` points per
        window, the right choice up to some hundreds of thousands of points.
        ``"binned"`` takes the median of fixed ``width/2`` bins and linearly
        interpolates between bin centres — ``O(N log N)`` whatever the span or
        the gaps, and the pragmatic choice for very long or very dense light
        curves, at the price of a coarser, slightly phase-shifted filter.
    min_points : int, default 3
        Windows holding fewer points than this do not set the trend from one
        or two observations. For ``"median"`` such a point falls back to the
        global median; for ``"binned"`` such a bin is not used as an
        interpolation node, so its points take the trend interpolated from the
        populated bins around them (the global median when no bin qualifies).

    Returns
    -------
    numpy.ndarray
        The trend, in the input's order and units. Non-finite (or masked)
        inputs contribute nothing to the filter and receive the global median.

    Examples
    --------
    >>> trend = running_median(t, mag, 40.0)          # doctest: +SKIP
    >>> flat = mag - trend                            # doctest: +SKIP
    """
    t = as_float_array(time, name="time", time=True).ravel()
    y = as_float_array(value, name="value").ravel()
    if t.size != y.size:
        raise ValueError("time and value must have the same length")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError(f"width must be positive and finite, got {width}")
    # A NaN threshold failed every `count >= min_points` test and silently returned
    # the global median everywhere.
    if not (
        np.isfinite(min_points) and float(min_points).is_integer() and min_points >= 1
    ):
        raise ValueError(f"min_points must be an integer >= 1, got {min_points}")
    if method not in ("median", "binned"):
        raise ValueError(f"method must be 'median' or 'binned', got {method!r}")

    out = np.empty(t.size, dtype=np.float64)
    good = np.isfinite(t) & np.isfinite(y)
    if not np.any(good):
        out.fill(np.nan)
        return out
    fallback = float(np.median(y[good]))
    out.fill(fallback)
    if not np.any(good):  # pragma: no cover - guarded above
        return out

    order = np.argsort(t[good], kind="stable")
    idx = np.flatnonzero(good)[order]
    ts, ys = t[idx], y[idx]

    if method == "binned":
        out[idx] = _binned_trend(ts, ys, width, fallback, min_points)
        return out

    half = 0.5 * width
    lo = np.searchsorted(ts, ts - half, side="left")
    hi = np.searchsorted(ts, ts + half, side="right")
    out[idx] = _sliding_median(ys, lo, hi, fallback, min_points)
    return out


def _sliding_median(
    ys: FloatArray,
    lo: IntArray,
    hi: IntArray,
    fallback: float,
    min_points: int,
) -> FloatArray:
    """Median of ``ys[lo[i]:hi[i]]`` for every ``i``, from one sorted window.

    Over time-sorted points both window edges only move right, so the window's
    values are kept in one sorted list: each point is inserted once as the
    leading edge passes it and removed once as the trailing edge does, and the
    median is read by index. That replaces a fresh ``np.median`` (call overhead
    plus an ``O(k)`` partition) per point. The values are those ``np.median``
    returns — the middle element, or the mean of the two middle ones — so the
    trend is unchanged; only the sign of a zero median can differ.
    """
    vals = ys.tolist()
    starts = lo.tolist()
    stops = hi.tolist()
    trend = [fallback] * len(vals)
    window: list[float] = []
    a = b = 0
    for i, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        # Insert before deleting: point i lies in its own window, so start < stop
        # and every value about to leave has already been inserted.
        while b < stop:
            insort(window, vals[b])
            b += 1
        while a < start:
            del window[bisect_left(window, vals[a])]
            a += 1
        k = stop - start
        if k >= min_points:
            mid = k >> 1
            trend[i] = (
                window[mid] if k & 1 else (window[mid - 1] + window[mid]) / 2.0
            )
    return np.asarray(trend, dtype=np.float64)


def _binned_trend(
    ts: FloatArray,
    ys: FloatArray,
    width: float,
    fallback: float,
    min_points: int,
) -> FloatArray:
    """Bin medians on a ``width/2`` grid, linearly interpolated back.

    Only occupied bins are visited, found from the sorted bin index itself, so
    the work is ``O(N)`` after the sort however long the span or wide the gaps:
    a stray epoch (a ``0.0`` "missing" sentinel among JDs) no longer means
    millions of empty bins. The index stays a float, so a far-out epoch cannot
    overflow it. Bins holding fewer than ``min_points`` points are not used as
    nodes, so one or two observations never set the trend on their own.
    """
    step = 0.5 * width
    bin_of = np.floor((ts - float(ts[0])) / step)
    starts = np.flatnonzero(np.r_[True, bin_of[1:] != bin_of[:-1]])
    stops = np.r_[starts[1:], ts.size]
    populated = (stops - starts) >= min_points
    starts, stops = starts[populated], stops[populated]
    if starts.size == 0:
        return np.full(ts.size, fallback, dtype=np.float64)
    centres = np.array(
        [np.median(ts[a:b]) for a, b in zip(starts, stops, strict=True)],
        dtype=np.float64,
    )
    levels = np.array(
        [np.median(ys[a:b]) for a, b in zip(starts, stops, strict=True)],
        dtype=np.float64,
    )
    if levels.size == 1:
        return np.full(ts.size, levels[0], dtype=np.float64)
    return np.asarray(np.interp(ts, centres, levels), dtype=np.float64)


def detrend(
    time: Any,
    value: Any,
    width: float,
    *,
    method: TrendMethod = "median",
    min_points: int = 3,
    restore_level: bool = True,
) -> FloatArray:
    """Remove the slow trend, keeping the variability.

    Parameters
    ----------
    time, value, width, method, min_points
        As :func:`running_median`.
    restore_level : bool, default True
        Add the global median back, so the result keeps the original zero point
        and stays plottable on a magnitude axis. Set ``False`` for a
        mean-subtracted residual.

    Returns
    -------
    numpy.ndarray
        ``value`` minus its trend, in the input's order and units.

    Examples
    --------
    >>> flat = detrend(t, mag, 40.0)                  # doctest: +SKIP
    >>> res = cup.pdot_search((t, flat, err), period=1.43)   # doctest: +SKIP
    """
    y = as_float_array(value, name="value").ravel()
    trend = running_median(
        time, value, width, method=method, min_points=min_points
    )
    out = y - trend
    if restore_level:
        good = np.isfinite(y)
        if np.any(good):
            out = out + float(np.median(y[good]))
    return np.asarray(out, dtype=np.float64)


__all__ = ["TrendMethod", "detrend", "running_median"]
