"""Light-curve containers: :class:`LightCurve` and :class:`MultiBandLightCurve`.

These are the universal inputs to every periodogram. A :class:`LightCurve` holds the
finite-or-not time/value/error arrays of a single band; a :class:`MultiBandLightCurve`
groups several bands (filters) of the *same* star for the multi-band methods. Both
ingest from raw arrays, a pandas ``DataFrame``, an astropy/pyarrow table, or a file
(CSV/ECSV/FITS/Parquet) with frictionless column mapping (:class:`ColumnMap`).

The time origin is deliberately *not* shifted here — periodograms are invariant to it
and each method references times to their own minimum internally for numerical
stability (the large-absolute-JD pitfall), so the container preserves the input times
verbatim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.errors import ColumnResolutionError

if TYPE_CHECKING:
    import pandas as pd

#: mag = -2.5 log10(flux); d(mag) = (2.5/ln10) d(flux)/flux.
_MAG_PER_DEX: float = 2.5 / np.log(10.0)


def _as_f64(a: object, name: str) -> FloatArray:
    arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


@dataclass(frozen=True, eq=False)
class LightCurve:
    """A single band's time series.

    Parameters
    ----------
    time : numpy.ndarray
        Observation times (JD/HJD/BJD), float64. Units are days throughout cuPeriod.
    value : numpy.ndarray
        Brightness in magnitudes or flux (see ``domain``).
    error : numpy.ndarray, optional
        1-sigma uncertainty on ``value``. ``None`` means unweighted.
    domain : Domain, default :attr:`Domain.MAGNITUDE`
        Whether ``value`` is a magnitude or a flux.
    meta : Mapping, optional
        Free-form metadata (e.g. object id, band name) carried through results.

    Notes
    -----
    Arrays are coerced to contiguous float64 and length-checked on construction.
    Non-finite points are *not* removed here; call :meth:`finite` for that.
    """

    time: FloatArray
    value: FloatArray
    error: FloatArray | None = None
    domain: Domain = Domain.MAGNITUDE
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = _as_f64(self.time, "time")
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
        domain: Domain = Domain.MAGNITUDE,
        meta: Mapping[str, Any] | None = None,
    ) -> LightCurve:
        """Build from raw array-likes."""
        return cls(
            time=_as_f64(time, "time"),
            value=_as_f64(value, "value"),
            error=None if error is None else _as_f64(error, "error"),
            domain=domain,
            meta=dict(meta or {}),
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        columns: ColumnMap | None = None,
        domain: Domain | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> LightCurve:
        """Build from a pandas ``DataFrame`` with column auto-detection."""
        names, getter = _adapt_table(df)
        return cls._from_table(names, getter, columns, domain, meta)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        columns: ColumnMap | None = None,
        domain: Domain | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> LightCurve:
        """Build from a CSV/ECSV/FITS/Parquet file with column auto-detection."""
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
        domain: Domain | None = None,
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
        getter: Callable[[str], FloatArray],
        columns: ColumnMap | None,
        domain: Domain | None,
        meta: Mapping[str, Any] | None,
    ) -> LightCurve:
        resolved = (columns or ColumnMap()).resolve(names, domain=domain)
        error = None if resolved.error is None else getter(resolved.error)
        return cls(
            time=getter(resolved.time),
            value=getter(resolved.value),
            error=error,
            domain=resolved.domain,
            meta=dict(meta or {}),
        )

    # -- transforms --------------------------------------------------------------
    def finite(self) -> LightCurve:
        """Return a copy with non-finite points (and non-positive errors) removed.

        Drops samples where time or value is NaN/inf, and — when errors are present
        — where the error is non-finite or ``<= 0`` (an unusable weight).
        """
        mask = np.isfinite(self.time) & np.isfinite(self.value)
        if self.error is not None:
            mask &= np.isfinite(self.error) & (self.error > 0.0)
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

    def in_domain(self, domain: Domain, *, zeropoint: float = 0.0) -> LightCurve:
        """Return this light curve converted to ``domain``."""
        if domain is Domain.FLUX:
            return self.as_flux(zeropoint=zeropoint)
        return self.as_magnitude(zeropoint=zeropoint)


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
        domain: Domain | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> MultiBandLightCurve:
        """Build from one long-format ``DataFrame`` split on a band column.

        Parameters
        ----------
        df : pandas.DataFrame
            Long-format table with a band/filter column and shared time/value/error
            columns.
        band_column : str, optional
            Band column name; auto-detected from :data:`BAND_NAMES` if ``None``.
        columns, domain, meta
            As for :meth:`LightCurve.from_dataframe`.
        """
        names, _ = _adapt_table(df)
        cmap = columns or ColumnMap(band=band_column)
        if band_column is not None and cmap.band is None:
            cmap = replace(cmap, band=band_column)
        resolved = cmap.resolve(names, domain=domain)
        if resolved.band is None:
            raise ColumnResolutionError(
                "could not resolve a band column; pass band_column=... "
                f"or ColumnMap(band=...). Available columns: {names}"
            )
        bands: dict[str, LightCurve] = {}
        for label, sub in df.groupby(resolved.band, sort=False):
            names_s, getter_s = _adapt_table(sub)
            bands[str(label)] = LightCurve._from_table(
                names_s, getter_s, columns, resolved.domain, {"band": str(label)}
            )
        return cls(bands=bands, meta=dict(meta or {}))

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        band_column: str | None = None,
        columns: ColumnMap | None = None,
        domain: Domain | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> MultiBandLightCurve:
        """Build from one long-format table file split on a band column.

        Parameters
        ----------
        path : str or Path
            CSV/ECSV/FITS/Parquet/ASCII file with a band/filter column and shared
            time/value/error columns.
        band_column : str, optional
            Band column name; auto-detected from the standard filter-column names
            if ``None``.
        columns, domain, meta
            As for :meth:`LightCurve.from_file`.

        Raises
        ------
        ColumnResolutionError
            If no band column can be resolved.
        """
        table = _read_table_object(Path(path))
        names, _ = _adapt_table(table)
        cmap = columns or ColumnMap(band=band_column)
        if band_column is not None and cmap.band is None:
            cmap = replace(cmap, band=band_column)
        resolved = cmap.resolve(names, domain=domain)
        if resolved.band is None:
            raise ColumnResolutionError(
                "could not resolve a band column; pass band_column=... "
                f"or ColumnMap(band=...). Available columns: {names}"
            )
        labels = np.array(
            [_band_label(v) for v in _raw_column(table, resolved.band)], dtype=object
        )
        bands: dict[str, LightCurve] = {}
        for label in dict.fromkeys(labels):  # first-appearance order
            sub = _row_subset(table, labels == label)
            names_s, getter_s = _adapt_table(sub)
            bands[str(label)] = LightCurve._from_table(
                names_s, getter_s, columns, resolved.domain, {"band": str(label)}
            )
        base_meta = {"source": str(path), **dict(meta or {})}
        return cls(bands=bands, meta=base_meta)

    def finite(self) -> MultiBandLightCurve:
        """Return a copy with each band's non-finite points removed."""
        return MultiBandLightCurve(
            bands={k: lc.finite() for k, lc in self.bands.items()}, meta=self.meta
        )

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


# --- table adapters ----------------------------------------------------------


def _column_array(values: object) -> FloatArray:
    """Coerce a table column to a float64 numpy array, filling masked entries."""
    if hasattr(values, "filled"):  # numpy masked array / astropy masked column
        values = values.filled(np.nan)  # type: ignore[attr-defined]
    return np.asarray(values, dtype=np.float64)


def _adapt_table(obj: Any) -> tuple[list[str], Callable[[str], FloatArray]]:
    """Return ``(column_names, getter)`` for a pandas/astropy/pyarrow/dict table."""
    # astropy Table (checked before pandas: Tables also expose .loc/.columns).
    if hasattr(obj, "colnames"):
        return list(obj.colnames), lambda name: _column_array(obj[name])
    # pyarrow Table.
    if hasattr(obj, "column_names") and hasattr(obj, "column"):
        names = list(obj.column_names)
        return names, lambda name: _column_array(
            obj.column(name).to_numpy(zero_copy_only=False)
        )
    # pandas DataFrame (``iloc`` is pandas-specific).
    if hasattr(obj, "iloc") and hasattr(obj, "columns"):
        cols = [str(c) for c in obj.columns]
        return cols, lambda name: _column_array(obj[name].to_numpy())
    # plain mapping of arrays.
    if isinstance(obj, Mapping):
        return list(obj.keys()), lambda name: _column_array(obj[name])
    raise TypeError(f"unsupported table type: {type(obj)!r}")


def _read_table_object(path: Path) -> Any:
    """Read a tabular file into a table object (pyarrow or astropy) by extension."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        return pq.read_table(path)

    from astropy.table import Table

    if suffix in {".fits", ".fit", ".fz"}:
        return Table.read(path)
    if suffix in {".csv"}:
        return Table.read(path, format="ascii.csv")
    if suffix in {".ecsv"}:
        return Table.read(path, format="ascii.ecsv")
    if suffix in {".tsv", ".tab"}:
        return Table.read(path, format="ascii.tab")
    # .dat/.txt and unknowns: let astropy guess the ASCII flavor.
    return Table.read(path, format="ascii")


def _read_table_file(path: Path) -> tuple[list[str], Callable[[str], FloatArray]]:
    """Read a tabular file into ``(column_names, getter)`` by extension."""
    return _adapt_table(_read_table_object(path))


def _raw_column(obj: Any, name: str) -> np.ndarray:
    """A table column as-is (labels stay strings — no float coercion)."""
    if hasattr(obj, "column_names") and hasattr(obj, "column"):  # pyarrow
        return np.asarray(obj.column(name).to_numpy(zero_copy_only=False))
    values = obj[name]
    if hasattr(values, "filled"):
        values = values.filled()  # type: ignore[attr-defined]
    return np.asarray(values)


def _row_subset(obj: Any, mask: np.ndarray) -> Any:
    """The rows of a table object selected by a boolean ``mask``."""
    if hasattr(obj, "column_names") and hasattr(obj, "column"):  # pyarrow
        import pyarrow as pa

        return obj.filter(pa.array(mask))
    return obj[mask]


def _band_label(value: Any) -> str:
    """A band cell as a clean string (FITS byte strings decoded)."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


__all__ = ["LightCurve", "MultiBandLightCurve"]
