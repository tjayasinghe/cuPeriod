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

from typing import Any, Literal

import numpy as np

from cuperiod.core._typing import FloatArray

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
        ``"median"`` evaluates an exact running median centred on every point.
        It costs ``O(N * k)`` for ``k`` points per window, which is the right
        choice up to some tens of thousands of points. ``"binned"`` takes the
        median of fixed ``width/2`` bins and linearly interpolates between bin
        centres — ``O(N log N)``, and the pragmatic choice for very long or
        very dense light curves, at the price of a coarser, slightly
        phase-shifted filter.
    min_points : int, default 3
        Windows holding fewer points than this fall back to the global median
        rather than returning a value driven by one or two observations.

    Returns
    -------
    numpy.ndarray
        The trend, in the input's order and units. Non-finite inputs contribute
        nothing to the filter and receive the global median.

    Examples
    --------
    >>> trend = running_median(t, mag, 40.0)          # doctest: +SKIP
    >>> flat = mag - trend                            # doctest: +SKIP
    """
    t = np.asarray(time, dtype=np.float64).ravel()
    y = np.asarray(value, dtype=np.float64).ravel()
    if t.size != y.size:
        raise ValueError("time and value must have the same length")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError(f"width must be positive and finite, got {width}")
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
        out[idx] = _binned_trend(ts, ys, width, fallback)
        return out

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


def _binned_trend(
    ts: FloatArray, ys: FloatArray, width: float, fallback: float
) -> FloatArray:
    """Bin medians on a ``width/2`` grid, linearly interpolated back."""
    step = 0.5 * width
    start = float(ts[0])
    bin_of = np.floor((ts - start) / step).astype(np.int64)
    n_bins = int(bin_of[-1]) + 1
    centres = np.empty(n_bins, dtype=np.float64)
    levels = np.empty(n_bins, dtype=np.float64)
    edges = np.searchsorted(bin_of, np.arange(n_bins + 1))
    filled = 0
    for k in range(n_bins):
        a, b = int(edges[k]), int(edges[k + 1])
        if b > a:
            centres[filled] = float(np.median(ts[a:b]))
            levels[filled] = float(np.median(ys[a:b]))
            filled += 1
    if filled == 0:  # pragma: no cover - ts is non-empty, so a bin is filled
        return np.full(ts.size, fallback, dtype=np.float64)
    if filled == 1:
        return np.full(ts.size, levels[0], dtype=np.float64)
    return np.asarray(
        np.interp(ts, centres[:filled], levels[:filled]), dtype=np.float64
    )


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
    y = np.asarray(value, dtype=np.float64).ravel()
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
