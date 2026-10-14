"""Exception hierarchy for cuPeriod.

All package-specific errors derive from :class:`CuPeriodError` so callers can catch
the whole family with a single ``except``. Each concrete error also derives from the
closest built-in (``ValueError``/``RuntimeError``) so existing error-handling that
keys on the standard types keeps working.
"""

from __future__ import annotations

import sys


def _user_stacklevel() -> int:
    """``stacklevel`` that attributes a warning to the nearest caller outside cuPeriod.

    For a ``warnings.warn`` issued in the function that calls this. One warning can
    be reached through call chains of different depths (a table read from
    ``LightCurve.from_dataframe``, ``to_input`` or ``periodogram``; batch keys from a
    list or a glob); a fixed level pointed some of them at a library line, where
    Python's default filter shows the warning once for every caller there is.
    """
    level = 1
    frame = sys._getframe(1)  # the function that will call warnings.warn
    while frame.f_back is not None and (
        str(frame.f_globals.get("__name__", "")).partition(".")[0] == "cuperiod"
    ):
        frame = frame.f_back
        level += 1
    return level


class CuPeriodError(Exception):
    """Base class for every error raised by cuPeriod."""


class ColumnResolutionError(CuPeriodError, ValueError):
    """A required light-curve column could not be resolved unambiguously.

    Raised by :class:`cuperiod.core.columns.ColumnMap` resolution when a role
    (time/value/error/band) has no matching column, or when more than one
    candidate matches and no explicit override was supplied.
    """


class BackendUnavailableError(CuPeriodError, RuntimeError):
    """A requested compute backend is not importable or has no usable device.

    Typically raised when ``backend="gpu"`` is requested but the ``[gpu]`` extra
    (cupy / cufinufft) is not installed or no CUDA device is present.
    """


class UnknownMethodError(CuPeriodError, ValueError):
    """The requested periodogram method name is not in the registry."""


class InsufficientDataError(CuPeriodError, ValueError):
    """A light curve has too few finite points or no usable time baseline.

    Also raised (via :func:`cuperiod.core.lightcurve.raise_if_unusable`) when an
    error column holds no usable weight -- every error zero, negative or
    non-finite -- so that cause is named instead of the empty baseline it leaves.
    """


__all__ = [
    "BackendUnavailableError",
    "ColumnResolutionError",
    "CuPeriodError",
    "InsufficientDataError",
    "UnknownMethodError",
]
