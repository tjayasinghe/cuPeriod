"""Phase-folding helpers (pure numpy, no Qt).

cuPeriod has no public fold utility, so the GUI keeps a tiny one here. Folding a light
curve at a trial period is what the live phased view does: as the user picks a different
peak or drags the line, only these O(n) functions re-run — never the periodogram. Kept
Qt-free so they unit-test headlessly.
"""

from __future__ import annotations

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.result import Peak


def earliest_time(time: FloatArray) -> float:
    """The earliest *finite* sample time, or 0.0 when there is none.

    Light curves keep their non-finite rows (every method drops them itself), and
    survey products do write them: SPOC/Kepler FITS files put NaN in TIME for gap
    cadences. A plain ``min`` would then return NaN and turn every phase NaN.
    """
    finite = time[np.isfinite(time)]
    return float(finite.min()) if finite.size else 0.0


def fold_phase(
    time: FloatArray,
    period: float,
    t0: float | None = None,
    *,
    two_cycles: bool = False,
) -> FloatArray:
    """Phase in ``[0, 1)`` (or ``[0, 2)`` if ``two_cycles``) for a period in days.

    ``t0`` sets phase zero; when omitted the earliest finite sample is used (a
    non-finite time keeps a NaN phase of its own). Raises on a non-positive or
    non-finite period.
    """
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be a positive, finite number of days")
    epoch = earliest_time(time) if t0 is None else float(t0)
    phase = np.remainder((time - epoch) / period, 1.0)
    if two_cycles:
        return np.concatenate([phase, phase + 1.0])
    return phase


def fold_series(
    time: FloatArray,
    value: FloatArray,
    period: float,
    t0: float | None = None,
    *,
    two_cycles: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Fold ``(time, value)`` at ``period``; returns aligned ``(phase, value)``."""
    phase = fold_phase(time, period, t0, two_cycles=two_cycles)
    values = np.concatenate([value, value]) if two_cycles else value
    return phase, values


def epoch_for_peak(peak: Peak | None, time: FloatArray) -> float:
    """Fold epoch: ``peak.extra['t0']`` if finite (BLS/TLS), else the earliest finite
    time (0.0 when no time is finite)."""
    if peak is not None:
        t0 = peak.extra.get("t0", float("nan"))
        if np.isfinite(t0):
            return float(t0)
    return earliest_time(time)


__all__ = ["earliest_time", "epoch_for_peak", "fold_phase", "fold_series"]
