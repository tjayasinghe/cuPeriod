"""Exception hierarchy for cuPeriod.

All package-specific errors derive from :class:`CuPeriodError` so callers can catch
the whole family with a single ``except``. Each concrete error also derives from the
closest built-in (``ValueError``/``RuntimeError``) so existing error-handling that
keys on the standard types keeps working.
"""

from __future__ import annotations


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
    """A light curve has too few finite points or no usable time baseline."""


__all__ = [
    "BackendUnavailableError",
    "ColumnResolutionError",
    "CuPeriodError",
    "InsufficientDataError",
    "UnknownMethodError",
]
