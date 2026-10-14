"""One float64 coercion for every array that enters cuPeriod.

Light curves arrive in many containers -- plain lists and ndarrays, numpy masked
arrays, astropy ``Column``/``MaskedColumn``/``Masked``/``Quantity``/``Time``, pandas
``Series`` (including the nullable ``Int64``/``Float64``/``boolean`` and Arrow-backed
dtypes), pyarrow arrays -- and a bare ``np.asarray(x, dtype=float64)`` gets several of
them silently wrong: it returns the raw data *under* a mask (survey sentinels such as
99.99, or the 0.0 astropy stores under a blank CSV cell, then enter the fit as real
measurements), it cannot fill an integer masked column with NaN, it turns
``datetime64`` into integer ticks of whatever resolution pandas inferred, and it
cannot read an astropy ``Time`` at all. :func:`as_float_array` is the single place
that knows every container: masked or missing entries become NaN (so
:meth:`~cuperiod.core.lightcurve.LightCurve.finite` drops them), and calendar times
are either converted to days (``time=True``) or refused with a clear error.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cuperiod.core._typing import FloatArray

#: The MJD origin. Calendar times without a numeric day scale (``datetime64``, astropy
#: ``Time`` in ISO/Unix/... formats) are reported as MJD, a convention every survey
#: archive understands and one that keeps ~1 microsecond float64 resolution today.
_MJD_EPOCH = np.datetime64("1858-11-17T00:00:00", "s")
_ONE_DAY = np.timedelta64(1, "D")
_FLOAT64 = np.dtype(np.float64)  # native byte order; numpy interns builtin dtypes

#: astropy ``Time`` formats whose ``.value`` is already in days. Their values are kept
#: as given (a lightkurve BTJD stays BTJD); every other format is reported as MJD.
_DAY_FORMATS = frozenset({"jd", "mjd", "btjd", "bkjd"})


def as_float_array(
    values: object,
    *,
    name: str = "values",
    time: bool = False,
    ndim: int | None = None,
) -> FloatArray:
    """Coerce any array-like to a C-contiguous float64 ndarray, missing data as NaN.

    Parameters
    ----------
    values : array-like
        A list, ndarray, numpy masked array, astropy ``Column``/``MaskedColumn``/
        ``Masked``/``Quantity``/``Time``/``TimeDelta``, pandas ``Series``/``Index``/
        extension array, or pyarrow ``Array``/``ChunkedArray``. Masked, null and NA
        entries become NaN.
    name : str, default ``"values"``
        What the array is, used in error messages (e.g. ``"time column 'date'"``).
    time : bool, default False
        Whether ``values`` holds observation times, which may then be calendar
        values: ``datetime64`` (and tz-aware pandas datetimes, taken in UTC) become
        MJD days, ``timedelta64`` and ``TimeDelta`` become days, an astropy ``Time``
        keeps its value in the day-valued formats (jd, mjd, btjd, bkjd) and is
        otherwise reported as MJD. No time-scale conversion is made: ``datetime64``
        is taken as UTC without leap seconds, a ``Time`` stays in its own scale. A
        unit on a ``Quantity`` or ``Column`` is not applied, whatever ``time`` is:
        its number is used as given (days, for a time).
    ndim : int, optional
        Required number of dimensions; ``None`` accepts any shape.

    Returns
    -------
    numpy.ndarray
        float64 and C-contiguous. A float64 C-contiguous plain ndarray is returned
        as-is (no copy).

    Raises
    ------
    TypeError
        Calendar values (``datetime64``/``timedelta64``/``Time``) when ``time`` is
        false: they are not brightnesses, and silently casting them to integer ticks
        is how a pandas datetime column used to become "days" of nanoseconds.
    ValueError
        Values that cannot be read as numbers, or the wrong ``ndim``.
    """
    if type(values) is np.ndarray:
        if (
            values.dtype is _FLOAT64
            and values.flags.c_contiguous
            and (ndim is None or values.ndim == ndim)
        ):
            # Already what is asked for: every LightCurve transform (finite, as_flux,
            # ...) re-validates its arrays, so this path must stay ~free.
            return values
        arr: np.ndarray = values
    else:
        arr = _unwrap(values, name, time)
    if arr.dtype.kind in "Mm":
        arr = _calendar_to_days(arr, name, time)
    if arr.dtype != _FLOAT64:  # also turns big-endian FITS columns native
        try:
            arr = np.asarray(arr, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise _named(exc, name) from exc
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    return np.ascontiguousarray(arr) if arr.ndim else arr


def _named(exc: Exception, name: str) -> Exception:
    """The same kind of error, saying which array could not be read."""
    kind = TypeError if isinstance(exc, TypeError) else ValueError
    return kind(f"{name} could not be converted to float64: {exc}")


def _unwrap(values: object, name: str, time: bool) -> np.ndarray:
    """Strip a container down to a plain ndarray, missing entries as NaN/NaT."""
    # The module test first: it is free, while a failed hasattr on a pandas object
    # walks its columns/index.
    module = type(values).__module__
    if module.startswith("pandas") and hasattr(values, "to_numpy"):
        try:
            return _pandas_to_numpy(values)
        except (TypeError, ValueError) as exc:
            raise _named(exc, name) from exc
    if module.startswith("pyarrow") and hasattr(values, "to_numpy"):
        return arrow_to_numpy(values)
    if _is_astropy_time(values):
        if not time:
            raise TypeError(
                f"{name} is an astropy Time; only the time column may hold times"
            )
        values = _astropy_time_to_days(values)
    if hasattr(values, "filled"):  # numpy masked array / astropy MaskedColumn, Masked
        values = _fill_masked(values)
    # A unit (Quantity, or a Column read from FITS/ECSV) is deliberately not applied:
    # converting only the containers that know how would load the same file 86400x
    # apart as a QTable and as a Table, and survey time units such as TESS's
    # "BJD - 2457000, days" are not parseable anyway. The number is taken as days.
    return np.asarray(values)


def _fill_masked(values: Any) -> Any:
    """``values.filled`` with a fill its storage can hold (NaN, or NaT for times)."""
    kind = getattr(getattr(values, "dtype", None), "kind", "f")
    if kind == "M":
        return values.filled(np.datetime64("NaT"))
    if kind == "m":
        return values.filled(np.timedelta64("NaT"))
    if kind in "iub":  # integer/bool storage cannot hold NaN: cast first
        values = values.astype(np.float64)
    elif kind in "US":
        # Fixed-width text would store "nan" cut to its width ("n" in U1, "na" in
        # U2), which then fails to parse; object storage holds a real float NaN.
        values = values.astype(object)
    return values.filled(np.nan)


def arrow_to_numpy(values: Any) -> np.ndarray:
    """A pyarrow ``Array``/``ChunkedArray`` as ndarray, nulls as NaN/None/NaT.

    Dictionary-encoded columns (what pandas categoricals become in Parquet) are
    decoded first: their ``to_numpy`` reads a null's meaningless index slot, so a
    missing cell silently took a neighbouring category's value.
    """
    import pyarrow as pa

    if pa.types.is_dictionary(values.type):
        values = values.cast(values.type.value_type)
    return np.asarray(values.to_numpy(zero_copy_only=False))


def _pandas_to_numpy(values: Any) -> np.ndarray:
    """A pandas Series/Index/extension array as ndarray, NA as NaN (NaT for times)."""
    dtype = getattr(values, "dtype", None)  # a DataFrame has none
    kind = getattr(dtype, "kind", "O")
    if isinstance(dtype, np.dtype) and kind in "fiub":
        return np.asarray(values.to_numpy())  # numpy-backed: no NA but NaN, no copy
    if kind == "M" and _has_timezone(dtype):
        # tz-aware: pandas converts to naive UTC for a datetime64 target.
        return np.asarray(values.to_numpy(dtype="datetime64[ns]"))
    if kind in "Mm":
        return np.asarray(values.to_numpy())
    # ``na_value`` reaches every nullable/Arrow/object flavour of missing (pd.NA, None,
    # NaN), which a bare float cast does not (``float(pd.NA)`` raises).
    return np.asarray(values.to_numpy(dtype=np.float64, na_value=np.nan))


def _has_timezone(dtype: Any) -> bool:
    if getattr(dtype, "tz", None) is not None:  # DatetimeTZDtype
        return True
    arrow = getattr(dtype, "pyarrow_dtype", None)  # ArrowDtype(timestamp[..., tz])
    return getattr(arrow, "tz", None) is not None


def _is_astropy_time(values: object) -> bool:
    """Duck-type astropy ``Time``/``TimeDelta`` without importing astropy."""
    return (
        hasattr(values, "jd1") and hasattr(values, "jd2") and hasattr(values, "format")
    )


def _astropy_time_to_days(values: Any) -> Any:
    from astropy.time import TimeDelta  # astropy is loaded: values is one of its types

    if isinstance(values, TimeDelta):
        return values.to_value("jd")  # TimeDelta's "jd" is a duration in days
    if values.format in _DAY_FORMATS:
        return values.value
    # ISO strings, Unix/GPS seconds, decimal years, ...: MJD keeps ~1 us resolution
    # where JD would keep ~40 us.
    return values.mjd


def _calendar_to_days(arr: np.ndarray, name: str, time: bool) -> np.ndarray:
    """datetime64 -> MJD days, timedelta64 -> days (NaT -> NaN); refused off-time."""
    if not time:
        raise TypeError(
            f"{name} has dtype {arr.dtype}; cuPeriod needs numbers here (calendar "
            "values are accepted only for the time column)"
        )
    if arr.dtype.kind == "M":
        arr = arr - _MJD_EPOCH
    try:
        days: np.ndarray = arr / _ONE_DAY  # NaT divides to NaN
    except TypeError as exc:  # month/year durations have no fixed length in days
        raise TypeError(
            f"{name} has dtype {arr.dtype}, which has no fixed length in days"
        ) from exc
    return days


__all__ = ["arrow_to_numpy", "as_float_array"]
