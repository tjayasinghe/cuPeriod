"""Light-curve containers: :class:`LightCurve` and :class:`MultiBandLightCurve`.

These are the universal inputs to every periodogram. A :class:`LightCurve` holds the
finite-or-not time/value/error arrays of a single band; a :class:`MultiBandLightCurve`
groups several bands (filters) of the *same* star for the multi-band methods. Both
ingest from raw arrays, a pandas ``DataFrame``, an astropy/pyarrow table, or a file
(CSV/ECSV/FITS/Parquet) with frictionless column mapping (:class:`ColumnMap`).

The time origin is deliberately *not* shifted here — periodograms are invariant to it
and each method references times to their own minimum internally for numerical
stability (the large-absolute-JD pitfall), so the container preserves the input times
verbatim. Calendar times are the one exception: ``datetime64`` columns and astropy
``Time`` objects outside the day-valued formats carry no day scale and are converted
to MJD (see :func:`cuperiod.core._coerce.as_float_array`).

Every array goes through the same coercion whatever the container: masked, null and
NA entries become NaN, so a tuple of masked columns and the table holding them give
the same light curve, and :meth:`LightCurve.finite` drops them.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cuperiod.core._coerce import arrow_to_numpy, as_float_array
from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import ColumnMap, Domain, ResolvedColumns
from cuperiod.core.errors import (
    ColumnResolutionError,
    InsufficientDataError,
    _user_stacklevel,
)

if TYPE_CHECKING:
    import pandas as pd

#: mag = -2.5 log10(flux); d(mag) = (2.5/ln10) d(flux)/flux.
_MAG_PER_DEX: float = 2.5 / np.log(10.0)


def _as_f64(a: object, name: str, *, time: bool = False) -> FloatArray:
    """``a`` as a 1-D contiguous float64 array (masked/missing entries as NaN)."""
    return as_float_array(a, name=name, time=time, ndim=1)


@dataclass(frozen=True, eq=False)
class LightCurve:
    """A single band's time series.

    Parameters
    ----------
    time : numpy.ndarray
        Observation times (JD/HJD/BJD), float64. Units are days throughout cuPeriod.
        ``datetime64`` values (and astropy ``Time`` outside the jd/mjd/btjd/bkjd
        formats) are converted to MJD; ``timedelta64``/``TimeDelta`` to days.
    value : numpy.ndarray
        Brightness in magnitudes or flux (see ``domain``).
    error : numpy.ndarray, optional
        1-sigma uncertainty on ``value``. ``None`` means unweighted.
    domain : Domain or str, default :attr:`Domain.MAGNITUDE`
        Whether ``value`` is a magnitude or a flux.
    meta : Mapping, optional
        Free-form metadata (e.g. object id, band name) carried through results.

    Notes
    -----
    Arrays are coerced to contiguous float64 and length-checked on construction;
    masked, null and NA entries become NaN. Non-finite points are *not* removed
    here; call :meth:`finite` for that.
    """

    time: FloatArray
    value: FloatArray
    error: FloatArray | None = None
    domain: Domain = Domain.MAGNITUDE
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = _as_f64(self.time, "time", time=True)
        value = _as_f64(self.value, "value")
        if time.size != value.size:
            raise ValueError(
                f"time and value length mismatch: {time.size} != {value.size}"
            )
        error = None if self.error is None else _as_f64(self.error, "error")
        if error is not None and error.size != time.size:
            raise ValueError(
                f"error and time length mismatch: {error.size} != {time.size}"
            )
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "error", error)
        # Coerce a string/aliased domain to the enum so identity checks in the
        # flux/magnitude conversions are reliable.
        object.__setattr__(self, "domain", Domain(self.domain))

    # -- properties --------------------------------------------------------------
    @property
    def n(self) -> int:
        """Number of points."""
        return int(self.time.size)

    @property
    def baseline(self) -> float:
        """Time span ``max(time) - min(time)`` in days (0.0 if empty)."""
        if self.time.size == 0:
            return 0.0
        return float(self.time.max() - self.time.min())

    # -- constructors ------------------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        time: object,
        value: object,
        error: object | None = None,
        *,
        domain: Domain | str = Domain.MAGNITUDE,
        meta: Mapping[str, Any] | None = None,
    ) -> LightCurve:
        """Build from raw array-likes.

        Any array container is accepted (lists, ndarrays, numpy/astropy masked
        arrays, pandas/pyarrow columns, astropy ``Quantity``/``Time``); masked,
        null and NA entries become NaN and are dropped by :meth:`finite`. A
        ``Quantity`` is taken as its number: its unit is not applied.
        """
        return cls(
            time=_as_f64(time, "time", time=True),
            value=_as_f64(value, "value"),
            error=None if error is None else _as_f64(error, "error"),
            domain=Domain(domain),
            meta=dict(meta or {}),
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        columns: ColumnMap | None = None,
        domain: Domain | str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> LightCurve:
        """Build from a table with column auto-detection.

        ``df`` may be a pandas ``DataFrame``, an astropy ``Table``/``QTable``/
        ``TimeSeries``, a pyarrow ``Table`` or a ``{column: array}`` mapping.
        """
        names, getter = _adapt_table(df)
        return cls._from_table(names, getter, columns, domain, meta)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        columns: ColumnMap | None = None,
        domain: Domain | str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> LightCurve:
        """Build from a CSV/ECSV/FITS/Parquet file with column auto-detection.

        The format follows the extension; a trailing ``.gz``/``.bz2``/``.xz``
        compression suffix is looked through (``star.fits.gz`` is FITS).
        """
        names, getter = _read_table_file(Path(path))
        base_meta = {"source": str(path), **dict(meta or {})}
        return cls._from_table(names, getter, columns, domain, base_meta)

    @classmethod
    def from_fits(
        cls,
        path: str | Path,
        *,
        hdu: int | str = 1,
        columns: ColumnMap | None = None,
        domain: Domain | str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> LightCurve:
        """Build from a binary-table FITS file (``hdu`` selects the extension)."""
        from astropy.table import Table

        table = Table.read(path, hdu=hdu)
        names, getter = _adapt_table(table)
        base_meta = {"source": str(path), **dict(meta or {})}
        return cls._from_table(names, getter, columns, domain, base_meta)

    @classmethod
    def _from_table(
        cls,
        names: list[str],
        getter: Callable[[str], Any],
        columns: ColumnMap | None,
        domain: Domain | str | None,
        meta: Mapping[str, Any] | None,
    ) -> LightCurve:
        resolved = (columns or ColumnMap()).resolve(names, domain=domain)
        if resolved.band is not None:
            _warn_pooled_bands(resolved.band, getter(resolved.band))
        return cls._from_resolved(getter, resolved, meta)

    @classmethod
    def _from_resolved(
        cls,
        getter: Callable[[str], Any],
        resolved: ResolvedColumns,
        meta: Mapping[str, Any] | None,
    ) -> LightCurve:
        """Read the resolved columns (errors name the offending column)."""
        error = (
            None
            if resolved.error is None
            else _as_f64(getter(resolved.error), f"error column {resolved.error!r}")
        )
        return cls(
            time=_as_f64(
                getter(resolved.time), f"time column {resolved.time!r}", time=True
            ),
            value=_as_f64(getter(resolved.value), f"value column {resolved.value!r}"),
            error=error,
            domain=resolved.domain,
            meta=dict(meta or {}),
        )

    # -- transforms --------------------------------------------------------------
    def _usable(self) -> np.ndarray:
        """The points :meth:`finite` keeps."""
        mask = np.isfinite(self.time) & np.isfinite(self.value)
        if self.error is not None:
            mask &= np.isfinite(self.error) & (self.error > 0.0)
        return mask

    def finite(self) -> LightCurve:
        """Return a copy with non-finite points (and non-positive errors) removed.

        Drops samples where time or value is NaN/inf, and — when errors are present
        — where the error is non-finite or ``<= 0`` (an unusable weight).
        """
        mask = self._usable()
        if mask.all():
            return self
        return LightCurve(
            time=self.time[mask],
            value=self.value[mask],
            error=None if self.error is None else self.error[mask],
            domain=self.domain,
            meta=self.meta,
        )

    def as_flux(self, *, zeropoint: float = 0.0) -> LightCurve:
        """Convert to the flux domain (no-op if already flux).

        Uses ``flux = 10 ** (-0.4 * (mag - zeropoint))`` and propagates the error as
        ``flux_err = 0.4 ln(10) * flux * mag_err``.
        """
        if self.domain is Domain.FLUX:
            return self
        flux = np.power(10.0, -0.4 * (self.value - zeropoint))
        error = None if self.error is None else (0.4 * np.log(10.0)) * flux * self.error
        return LightCurve(self.time, flux, error, Domain.FLUX, self.meta)

    def as_magnitude(self, *, zeropoint: float = 0.0) -> LightCurve:
        """Convert to the magnitude domain (no-op if already magnitude).

        Uses ``mag = -2.5 log10(flux) + zeropoint`` and propagates the error as
        ``mag_err = (2.5 / ln 10) * flux_err / flux``. Non-positive flux has no real
        magnitude; those points become non-finite (without a spurious NumPy warning)
        and are removed by :meth:`finite`.
        """
        if self.domain is Domain.MAGNITUDE:
            return self
        with np.errstate(invalid="ignore", divide="ignore"):
            mag = -2.5 * np.log10(self.value) + zeropoint
            error = (
                None
                if self.error is None
                else _MAG_PER_DEX * self.error / self.value
            )
        return LightCurve(self.time, mag, error, Domain.MAGNITUDE, self.meta)

    def in_domain(
        self, domain: Domain | str, *, zeropoint: float = 0.0
    ) -> LightCurve:
        """Return this light curve converted to ``domain`` (a :class:`Domain` or str).

        Raises
        ------
        ValueError
            If ``domain`` is not a valid domain name.
        """
        # Coerce before the identity test: an uncoerced "flux" used to fall through
        # to the magnitude branch, converting a flux curve to magnitudes and dropping
        # every non-positive flux.
        if Domain(domain) is Domain.FLUX:
            return self.as_flux(zeropoint=zeropoint)
        return self.as_magnitude(zeropoint=zeropoint)

    # -- diagnostics -------------------------------------------------------------
    def unusable_reason(self) -> str | None:
        """Why the errors leave no usable point, or ``None`` when they do not.

        :meth:`finite` drops every point whose error is non-finite or ``<= 0``.
        A placeholder error column (all 0, -1 or NaN) therefore empties the curve,
        and the method that receives it reports a misleading cause ("no usable time
        baseline"). This names the real one. It returns ``None`` when no error is
        attached, when some point survives, and when time/value have no finite
        point at all (a different problem, left to the caller).
        """
        if self.error is None:
            return None
        data = np.isfinite(self.time) & np.isfinite(self.value)
        n_data = int(np.count_nonzero(data))
        if n_data == 0 or self._usable().any():
            return None
        return (
            f"none of the {n_data} points with a finite time and value has a usable "
            "error (all are zero, negative or non-finite), so every point would be "
            "dropped; drop the error column (error=None) to run unweighted"
        )


@dataclass(frozen=True, eq=False)
class MultiBandLightCurve:
    """Several bands (filters) of the same star, for the multi-band methods.

    Parameters
    ----------
    bands : Mapping[str, LightCurve]
        Band name → light curve. Order is preserved (insertion order).
    meta : Mapping, optional
        Free-form metadata (e.g. object id) carried through results.
    """

    bands: Mapping[str, LightCurve]
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bands:
            raise ValueError("MultiBandLightCurve requires at least one band")
        # Checked here, not only in the periodogram entry point, so every consumer
        # (multiband_fap, the period-change scans, the batch runner) gets the same
        # guarantee: a tuple slipped in as a band used to fail deep inside a fit, and
        # a magnitude band next to a flux band varies in anti-phase, which the joint
        # shared-phase models fit as a cancelled signal without any error.
        for name, lc in self.bands.items():
            if not isinstance(lc, LightCurve):
                raise TypeError(
                    f"band {name!r} is a {type(lc).__name__}, not a LightCurve; "
                    "build each band with LightCurve.from_arrays(...)"
                )
        domains = {name: lc.domain for name, lc in self.bands.items()}
        if len(set(domains.values())) > 1:
            listing = ", ".join(f"{k}: {v.value}" for k, v in domains.items())
            raise ValueError(
                f"bands are in different brightness domains ({listing}); convert "
                "them with LightCurve.in_domain(...), or pass the {band: LightCurve} "
                "mapping to periodogram()/to_input() with domain='magnitude' or "
                "domain='flux'"
            )

    @property
    def band_names(self) -> tuple[str, ...]:
        """The band labels, in order."""
        return tuple(self.bands)

    @property
    def n_bands(self) -> int:
        """Number of bands."""
        return len(self.bands)

    @classmethod
    def from_light_curves(
        cls,
        bands: Mapping[str, LightCurve],
        *,
        meta: Mapping[str, Any] | None = None,
    ) -> MultiBandLightCurve:
        """Build from an explicit ``{band: LightCurve}`` mapping."""
        return cls(bands=dict(bands), meta=dict(meta or {}))

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        band_column: str | None = None,
        columns: ColumnMap | None = None,
        domain: Domain | str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> MultiBandLightCurve:
        """Build from one long-format table split on a band column.

        Parameters
        ----------
        df : pandas.DataFrame
            Long-format table with a band/filter column and shared time/value/error
            columns. An astropy or pyarrow table, or a ``{column: array}`` mapping,
            works too.
        band_column : str, optional
            Band column name; auto-detected from :data:`BAND_NAMES` if ``None``.
        columns, domain, meta
            As for :meth:`LightCurve.from_dataframe`.

        Notes
        -----
        Band labels are normalised before grouping, identically here and in
        :meth:`from_file`: byte strings are decoded, padding is stripped and
        integral numbers lose a trailing ``.0`` (so ``1``, ``"1"``, ``"1 "`` and
        ``1.0`` are one band ``"1"``). Rows without a label (blank, null, NaN or
        masked) cannot be assigned to a band; they are dropped with a warning.

        Raises
        ------
        ColumnResolutionError
            If no band column can be resolved, or it holds no label at all.
        """
        return cls._from_table_object(df, band_column, columns, domain, meta)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        band_column: str | None = None,
        columns: ColumnMap | None = None,
        domain: Domain | str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> MultiBandLightCurve:
        """Build from one long-format table file split on a band column.

        Parameters
        ----------
        path : str or Path
            CSV/ECSV/FITS/Parquet/ASCII file (optionally ``.gz``/``.bz2``/``.xz``
            compressed) with a band/filter column and shared time/value/error
            columns.
        band_column : str, optional
            Band column name; auto-detected from the standard filter-column names
            if ``None``.
        columns, domain, meta
            As for :meth:`LightCurve.from_file`.

        Notes
        -----
        Band labels are normalised and unlabelled rows dropped exactly as in
        :meth:`from_dataframe`, so a table gives the same bands either way.

        Raises
        ------
        ColumnResolutionError
            If no band column can be resolved, or it holds no label at all.
        """
        table = _read_table_object(Path(path))
        base_meta = {"source": str(path), **dict(meta or {})}
        return cls._from_table_object(table, band_column, columns, domain, base_meta)

    @classmethod
    def _from_table_object(
        cls,
        table: Any,
        band_column: str | None,
        columns: ColumnMap | None,
        domain: Domain | str | None,
        meta: Mapping[str, Any] | None,
    ) -> MultiBandLightCurve:
        """Split any table object on its band column (both public constructors)."""
        names, getter = _adapt_table(table)
        cmap = columns or ColumnMap(band=band_column)
        if band_column is not None and cmap.band is None:
            cmap = replace(cmap, band=band_column)
        resolved = cmap.resolve(names, domain=domain)
        if resolved.band is None:
            raise ColumnResolutionError(
                "could not resolve a band column; pass band_column=... "
                f"or ColumnMap(band=...). Available columns: {names}"
            )
        codes, labels = _band_codes(getter(resolved.band))
        n_missing = int(np.count_nonzero(codes < 0))
        if not labels:
            raise ColumnResolutionError(
                f"band column {resolved.band!r} holds no labels "
                f"(all {codes.size} rows are blank, null or masked)"
            )
        if n_missing:
            warnings.warn(
                f"{n_missing} of {codes.size} rows have no label in band column "
                f"{resolved.band!r} and were dropped: they cannot be assigned to a "
                "band",
                UserWarning,
                stacklevel=3,
            )
        # Read the columns once and split them, so every band uses the same resolved
        # columns and domain. One stable sort by band code (unlabelled rows, code -1,
        # sort first and are skipped) makes each band a contiguous slice in its
        # original row order -- cheaper than a boolean mask per band on long tables.
        whole = LightCurve._from_resolved(getter, resolved, None)
        order = np.argsort(codes, kind="stable")
        bounds = np.searchsorted(codes[order], np.arange(len(labels) + 1))
        time, value = whole.time[order], whole.value[order]
        error = None if whole.error is None else whole.error[order]
        bands: dict[str, LightCurve] = {}
        for k, label in enumerate(labels):
            rows = slice(int(bounds[k]), int(bounds[k + 1]))
            bands[label] = LightCurve(
                time=time[rows],
                value=value[rows],
                error=None if error is None else error[rows],
                domain=whole.domain,
                meta={"band": label},
            )
        return cls(bands=bands, meta=dict(meta or {}))

    def finite(self) -> MultiBandLightCurve:
        """Return a copy with each band's non-finite points removed."""
        return MultiBandLightCurve(
            bands={k: lc.finite() for k, lc in self.bands.items()}, meta=self.meta
        )

    def unusable_reason(self) -> str | None:
        """Why the errors leave no usable point in any band, or ``None``.

        The multi-band counterpart of :meth:`LightCurve.unusable_reason`: a message
        naming each band whose error column is unusable, returned only when no band
        keeps a point (a band wiped out by its errors does not condemn the others).
        """
        reasons: dict[str, str] = {}
        for name, lc in self.bands.items():
            reason = lc.unusable_reason()
            if reason is not None:
                reasons[name] = reason
            elif lc._usable().any():
                return None
        if not reasons:
            return None
        return "; ".join(f"band {name!r}: {why}" for name, why in reasons.items())

    def stacked(self) -> tuple[FloatArray, FloatArray, FloatArray | None, np.ndarray]:
        """Concatenate all bands into time-sorted ``(time, value, error, band)`` arrays.

        Returns
        -------
        time, value : numpy.ndarray
            Concatenated, ascending in time.
        error : numpy.ndarray or None
            Concatenated errors, or ``None`` if any band lacks errors.
        band : numpy.ndarray
            Per-point band label (object dtype), aligned with the sorted arrays.
        """
        times, values, errors, labels = [], [], [], []
        any_missing_error = False
        for name, lc in self.bands.items():
            times.append(lc.time)
            values.append(lc.value)
            if lc.error is None:
                any_missing_error = True
            else:
                errors.append(lc.error)
            labels.append(np.full(lc.n, name, dtype=object))
        time = np.concatenate(times) if times else np.empty(0)
        value = np.concatenate(values) if values else np.empty(0)
        band = np.concatenate(labels) if labels else np.empty(0, dtype=object)
        error = None if any_missing_error else np.concatenate(errors)
        order = np.argsort(time, kind="stable")
        return (
            time[order],
            value[order],
            None if error is None else error[order],
            band[order],
        )


def stack_bands(
    mblc: MultiBandLightCurve, *, centre: bool
) -> tuple[LightCurve, np.ndarray]:
    """Merge every band into one time-sorted single-band curve, with its band labels.

    Single-band analyses (pre-whitening above all) run on this merge. With
    ``centre=True`` each band's own mean is removed first: pre-whitening fits *one*
    constant for the whole curve, so a raw stack of bands at different zero points
    buries the pulsation under an inter-band offset of a magnitude or more. Centring
    is the least the merge can do to stay usable; a per-band zero point is what a
    session's ``band`` partition then fits properly.

    Parameters
    ----------
    mblc : MultiBandLightCurve
        The bands to merge (non-finite points are dropped first).
    centre : bool
        Subtract each band's mean value before merging.

    Returns
    -------
    lc : LightCurve
        The merged curve, ascending in time, in the first band's domain.
    band : numpy.ndarray
        Per-point band name (object dtype), aligned with ``lc``.
    """
    finite = mblc.finite()
    if centre:
        centred = {
            name: LightCurve.from_arrays(
                band.time,
                band.value - float(np.mean(band.value)) if band.n else band.value,
                band.error,
                domain=band.domain,
            )
            for name, band in finite.bands.items()
        }
        finite = MultiBandLightCurve.from_light_curves(centred).finite()
    time, value, error, labels = finite.stacked()
    domain = next(iter(finite.bands.values())).domain
    return LightCurve.from_arrays(time, value, error, domain=domain), labels


# --- input checks ------------------------------------------------------------


def raise_if_unusable(data: LightCurve | MultiBandLightCurve, label: str) -> None:
    """Raise when an unusable error column would empty ``data`` (see below).

    Call it before building a grid or running a method, so a placeholder error
    column (all 0, -1 or NaN) is reported as such instead of as the "no usable time
    baseline" that the emptied curve otherwise produces.

    Parameters
    ----------
    data : LightCurve or MultiBandLightCurve
        The input about to be analysed.
    label : str
        Message prefix naming the caller, e.g. ``"GLS"``.

    Raises
    ------
    InsufficientDataError
        ``f"{label}: {data.unusable_reason()}"`` when that reason is not ``None``.
    """
    reason = data.unusable_reason()
    if reason is not None:
        raise InsufficientDataError(f"{label}: {reason}")


#: Band labels quoted by the pooled-band warning before it elides the rest.
_MAX_LISTED_LABELS = 5


def _warn_pooled_bands(column_name: str, column: Any) -> None:
    """Warn when a single-band read pools a band column with several labels.

    ``LightCurve`` cannot split a table, so a long-format table read through it
    mixes every filter into one curve -- their different zero points and amplitudes
    then dominate any period search. Reading it that way is the documented default,
    so this warns rather than splits, and names the exact argument that asks for the
    multi-band analysis. Only the single-band table path calls this;
    :class:`MultiBandLightCurve` builds its per-band curves from already-split
    columns and never sees the band column here.
    """
    _, labels = _band_codes(column)
    if len(labels) < 2:
        return
    shown = ", ".join(repr(label) for label in labels[:_MAX_LISTED_LABELS])
    if len(labels) > _MAX_LISTED_LABELS:
        shown += ", ..."
    warnings.warn(
        f"band column {column_name!r} holds {len(labels)} labels ({shown}) but the "
        "table was read as one light curve that pools them. For a multi-band "
        f"analysis pass columns=ColumnMap(band={column_name!r}) to periodogram() or "
        f"to_input() (band_column={column_name!r} to the batch functions), or build "
        "a MultiBandLightCurve.from_dataframe/from_file; to read one pooled band "
        "deliberately, drop the column.",
        UserWarning,
        # The same read is reached from from_dataframe, to_input and periodogram.
        stacklevel=_user_stacklevel(),
    )


# --- table adapters ----------------------------------------------------------


def _adapt_table(obj: Any) -> tuple[list[str], Callable[[str], Any]]:
    """Return ``(column_names, getter)`` for a pandas/astropy/pyarrow/dict table.

    The getter returns the column *as stored* (masked columns, ``Time`` objects,
    pandas/pyarrow columns); callers coerce it with :func:`as_float_array`, which
    knows the column's role. Names are strings; a non-string label (the integer
    columns of ``pandas.read_csv(header=None)``) is listed as its ``str()`` and
    looked up by its original value, the first label winning on a collision.
    """
    # astropy Table (checked before pandas: Tables also expose .loc/.columns).
    if hasattr(obj, "colnames"):
        return list(obj.colnames), lambda name: obj[name]
    # pyarrow Table.
    if hasattr(obj, "column_names") and hasattr(obj, "column"):
        return list(obj.column_names), lambda name: obj.column(name)
    # pandas DataFrame (``iloc`` is pandas-specific), then a plain mapping of arrays.
    if hasattr(obj, "iloc") and hasattr(obj, "columns"):
        keys: Any = obj.columns
    elif isinstance(obj, Mapping):
        keys = obj.keys()
    else:
        raise TypeError(f"unsupported table type: {type(obj)!r}")
    labels: dict[str, Any] = {}
    for key in keys:
        labels.setdefault(str(key), key)
    return list(labels), lambda name: obj[labels[name]]


#: Compression suffixes looked through when choosing a reader by extension.
_COMPRESSION_SUFFIXES = frozenset({".gz", ".bz2", ".xz"})
_FITS_SUFFIXES = frozenset({".fits", ".fit", ".fts", ".fz"})


def _format_suffix(path: Path) -> str:
    """The extension that names the format: ``star.fits.gz`` -> ``.fits``."""
    suffixes = [s.lower() for s in path.suffixes]
    if len(suffixes) >= 2 and suffixes[-1] in _COMPRESSION_SUFFIXES:
        return suffixes[-2]
    return suffixes[-1] if suffixes else ""


def _scan_suffix(path: Path) -> str:
    """:func:`_format_suffix` for a file a folder scan or a drop may take, else ``""``.

    Text and FITS tables are read through ``.gz`` / ``.bz2`` / ``.xz``; Parquet is not
    (pyarrow cannot read a compressed file), so ``star.parquet.gz`` is left out.
    """
    suffix = _format_suffix(path)
    compressed = path.suffix.lower() in _COMPRESSION_SUFFIXES
    return "" if compressed and suffix in (".parquet", ".pq") else suffix


#: A UTF-8 byte-order mark as astropy's ASCII readers decode it: as one character
#: under a UTF-8 locale, byte by byte under a cp1252/latin-1 one (Windows).
_BOM_PREFIXES = ("\ufeff", "\xef\xbb\xbf")


def _without_bom(table: Any) -> Any:
    """``table`` with a UTF-8 byte-order mark dropped from its first column name.

    Excel's "CSV UTF-8" starts the file with one, and astropy's ASCII readers keep it
    in the first header cell (``'\\ufefftime'``), so the time column was not found.
    """
    names = list(table.colnames)
    for bom in _BOM_PREFIXES:
        if names and names[0].startswith(bom):
            bare = names[0][len(bom) :]
            if bare and bare not in names:
                table.rename_column(names[0], bare)
            break
    return table


def _read_table_object(path: Path) -> Any:
    """Read a tabular file into a table object (pyarrow or astropy) by extension."""
    suffix = _format_suffix(path)
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        return pq.read_table(path)

    from astropy.table import Table

    if suffix in _FITS_SUFFIXES:
        # astropy decompresses .gz/.bz2 itself; dispatching on the last suffix
        # alone sent star.fits.gz to the ASCII reader.
        return Table.read(path, format="fits")
    if suffix in {".csv"}:
        return _without_bom(Table.read(path, format="ascii.csv"))
    if suffix in {".ecsv"}:
        return Table.read(path, format="ascii.ecsv")
    if suffix in {".tsv", ".tab"}:
        return _without_bom(Table.read(path, format="ascii.tab"))
    # .dat/.txt and unknowns: let astropy guess the ASCII flavor.
    return _without_bom(Table.read(path, format="ascii"))


def _read_table_file(path: Path) -> tuple[list[str], Callable[[str], Any]]:
    """Read a tabular file into ``(column_names, getter)`` by extension."""
    return _adapt_table(_read_table_object(path))


def _band_codes(column: Any) -> tuple[np.ndarray, list[str]]:
    """Group a band column: per-row codes into the normalised labels.

    Returns
    -------
    codes : numpy.ndarray
        int64 per row: an index into ``labels``, or -1 for a row without a label
        (null, NaN, masked, or blank after stripping).
    labels : list of str
        The distinct normalised labels (:func:`_band_label`) in order of first
        appearance. Cells that normalise alike (``1``, ``"1"``, ``1.0``, ``"1 "``)
        share one label instead of overwriting each other.
    """
    if type(column).__module__.startswith("pandas"):
        import pandas as pd

        # factorize keeps first-appearance order and codes every pandas flavour of
        # missing (NaN/None/pd.NA/NaT) as -1.
        raw_codes, uniques = pd.factorize(column)
        raw_values: list[Any] = list(uniques)
        raw = np.asarray(raw_codes, dtype=np.int64)
    else:
        if type(column).__module__.startswith("pyarrow"):  # nulls become None/NaN
            cells = arrow_to_numpy(column)
            masked = np.zeros(cells.shape, dtype=bool)
        else:  # numpy / astropy / list; a masked cell's data is a fill, not a label
            masked = np.ma.getmaskarray(column)
            cells = np.asarray(np.ma.getdata(column))
        raw = np.full(cells.size, -1, dtype=np.int64)
        present = ~masked.reshape(-1)
        raw[present], raw_values = _factorize(cells.reshape(-1)[present])
    names: dict[str, int] = {}
    remap = np.empty(len(raw_values) + 1, dtype=np.int64)
    remap[-1] = -1  # so a raw code of -1 stays -1 below
    for k, cell in enumerate(raw_values):
        label = _band_label(cell)
        remap[k] = -1 if not label else names.setdefault(label, len(names))
    return remap[raw], list(names)


def _factorize(cells: np.ndarray) -> tuple[np.ndarray, list[Any]]:
    """Codes and distinct values of a 1-D array, values in first-appearance order."""
    if cells.dtype.kind != "O":  # homogeneous: let numpy find the distinct values
        uniq, first, inverse = np.unique(cells, return_index=True, return_inverse=True)
        order = np.argsort(first, kind="stable")
        rank = np.empty(order.size, dtype=np.int64)
        rank[order] = np.arange(order.size)
        return rank[inverse.reshape(-1)], list(uniq[order])
    seen: dict[Any, int] = {}
    codes = np.empty(cells.size, dtype=np.int64)
    for i, cell in enumerate(cells):
        try:
            codes[i] = seen.setdefault(cell, len(seen))
        except TypeError:  # unhashable cell: group on its text
            codes[i] = seen.setdefault(str(cell), len(seen))
    return codes, list(seen)


def _band_label(value: Any) -> str:
    """A band cell as a clean string; ``""`` for a missing cell (None/NaN).

    FITS byte strings are decoded, padding is stripped, and integral numbers lose
    a float's ``.0`` (an integer filter-id column with blanks is float in pandas
    but integer in astropy: both must give ``"1"``).
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))
    return str(value).strip()


__all__ = ["LightCurve", "MultiBandLightCurve", "raise_if_unusable", "stack_bands"]
