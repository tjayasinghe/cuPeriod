"""Adapters for LINCC Frameworks light curves (nested-pandas / lsdb).

The two public entry points are :func:`nested_periodogram` (row-wise) and
:func:`partition_periodogram` (partition-wise, engine-reusing). See the package
docstring of :mod:`cuperiod.interop` for the tier overview.

Nothing here imports nested-pandas, pandas, lsdb, or dask at module scope: the module
is importable in a bare cuPeriod install, and the LINCC stack is touched only when a
function actually runs (:func:`require_nested_pandas`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from pydantic_settings import BaseSettings

from cuperiod.api import _multiband_grid
from cuperiod.core.backend import is_gpu_selector
from cuperiod.core.columns import ColumnMap, Domain, coerce_domain, infer_domain
from cuperiod.core.errors import ColumnResolutionError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve, raise_if_unusable
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod, get_method

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

#: Nest name used when neither ``nested=`` nor a preset names one. The parameter
#: itself defaults to ``None``, so an explicitly passed ``"lc"`` is told apart from
#: "not given" and always wins over a preset's own nest.
_DEFAULT_NEST: Final[str] = "lc"

_NAN: Final[float] = float("nan")


# --- optional-dependency guard -----------------------------------------------


def require_nested_pandas() -> None:
    """Raise a helpful :class:`ImportError` if nested-pandas is not installed.

    Raises
    ------
    ImportError
        With the ``pip install`` line for the ``[nested]`` extra.
    """
    try:
        import nested_pandas  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "cuperiod.interop needs nested-pandas. Install it with:\n"
            "    pip install 'cuperiod[nested]'\n"
            "(or 'cuperiod[lsdb]' to also get lsdb for HATS catalogs)"
        ) from exc


# --- survey column presets ----------------------------------------------------

#: Verified column layouts of the common LINCC-stack catalogs. Each entry carries
#: ``nested``, ``time``, ``value``, ``error``, ``band`` and ``domain``; sub-column names
#: are bare (they are prefixed with whichever nest name is in use). Pass one by name as
#: ``preset=`` to fill every column parameter you did not set yourself — except that a
#: ``value=`` naming a *different* column (e.g. ``psfMag`` under a ``psfFlux`` preset)
#: drops the preset's ``error`` and ``domain``, which describe the preset's own value.
#:
#: * ``"ztf_dr22"`` — ZTF DR22 object table: nest ``lc`` with ``hmjd``/``mag``/
#:   ``magerr``/``catflags``. The band is *not* in the nest: DR22 has one row per
#:   (object, filter) with the filter in the **base** column ``filterid``
#:   (1=g, 2=r, 3=i), so the preset sets ``band=None`` and each row is a single-band
#:   search. A multi-band run needs a self-join that nests the per-filter rows of one
#:   object into one row first (``NestedFrame.join_nested``).
#: * ``"ztf_alerts"`` — ALeRCE ZTF alert HATS catalogs: nest ``lc`` with
#:   ``lc_mjd``/``lc_magpsf``/``lc_sigmapsf`` and the in-nest band ``lc_fid``.
#: * ``"rubin_dp1_object"`` / ``"rubin_dp1_dia"`` — Rubin DP1 forced photometry,
#:   nest ``objectForcedSource`` / ``diaObjectForcedSource`` with ``midpointMjdTai``,
#:   ``psfFlux`` (nJy), ``psfFluxErr`` and the in-nest band ``band`` (``ugrizy``).
#:   Difference-imaging fluxes are legitimately negative, so both presets search in
#:   :attr:`~cuperiod.Domain.FLUX` and never convert to magnitudes.
COLUMN_PRESETS: Final[dict[str, dict[str, Any]]] = {
    "ztf_dr22": {
        "nested": "lc",
        "time": "hmjd",
        "value": "mag",
        "error": "magerr",
        "band": None,
        "domain": Domain.MAGNITUDE,
    },
    "ztf_alerts": {
        "nested": "lc",
        "time": "lc_mjd",
        "value": "lc_magpsf",
        "error": "lc_sigmapsf",
        "band": "lc_fid",
        "domain": Domain.MAGNITUDE,
    },
    "rubin_dp1_object": {
        "nested": "objectForcedSource",
        "time": "midpointMjdTai",
        "value": "psfFlux",
        "error": "psfFluxErr",
        "band": "band",
        "domain": Domain.FLUX,
    },
    "rubin_dp1_dia": {
        "nested": "diaObjectForcedSource",
        "time": "midpointMjdTai",
        "value": "psfFlux",
        "error": "psfFluxErr",
        "band": "band",
        "domain": Domain.FLUX,
    },
}

#: The keys every entry of :data:`COLUMN_PRESETS` defines.
PRESET_KEYS: Final[tuple[str, ...]] = (
    "nested", "time", "value", "error", "band", "domain",
)


# --- column resolution --------------------------------------------------------


@dataclass(frozen=True)
class NestedColumns:
    """Resolved, fully dotted column names for one nested light-curve column.

    Attributes
    ----------
    nested : str
        The nested column's name (e.g. ``"lc"``).
    time, value : str
        Dotted sub-column names (e.g. ``"lc.hmjd"``).
    error, band : str or None
        Dotted sub-column names, or ``None`` when absent/not requested.
    domain : Domain
        The brightness domain of ``value``.
    """

    nested: str
    time: str
    value: str
    error: str | None
    band: str | None
    domain: Domain

    @property
    def read_columns(self) -> list[str]:
        """The dotted columns to request from ``map_rows``, in a stable order."""
        names = [self.time, self.value]
        if self.error is not None:
            names.append(self.error)
        if self.band is not None:
            names.append(self.band)
        return names

    @property
    def subcolumns(self) -> list[str]:
        """The same columns as bare sub-column names."""
        return [name.split(".", 1)[1] for name in self.read_columns]


def _name_list(obj: Any) -> list[str] | None:
    """Coerce a dtype attribute to a list of sub-column names, or ``None``."""
    if obj is None or callable(obj) or isinstance(obj, (str, bytes)):
        return None
    if isinstance(obj, Mapping):
        return [str(k) for k in obj]
    if isinstance(obj, Iterable):
        return [str(k) for k in obj]
    return None


def _subcolumn_names(dtype: Any) -> list[str]:
    """Sub-column names of a ``NestedDtype``, without touching any data.

    nested-pandas renamed this attribute across releases: ``columns`` /
    ``column_dtypes`` (0.7) and ``field_names`` / ``fields`` (0.6). Every spelling is
    tried so one code path covers the lsdb-pinned 0.6.x line and 0.7.x.
    """
    for attr in ("columns", "column_dtypes", "field_names", "fields"):
        names = _name_list(getattr(dtype, attr, None))
        if names:
            return names
    raise TypeError(
        f"{dtype!r} does not look like a nested-pandas NestedDtype: no sub-column "
        "names found (tried .columns/.column_dtypes/.field_names/.fields)"
    )


def _schema(data: Any) -> tuple[Any, list[str]]:
    """``(dtypes, column names)`` of a frame or catalog (lazy: reads the schema only).

    ``.dtypes`` is the adapter's whole contract with its input: a ``NestedFrame`` and
    a lazy lsdb ``Catalog`` both expose it without computing anything.
    """
    dtypes = getattr(data, "dtypes", None)
    if dtypes is None:
        raise TypeError(
            f"expected a nested-pandas NestedFrame or an lsdb Catalog, got "
            f"{type(data).__name__}"
        )
    return dtypes, [str(c) for c in getattr(dtypes, "index", dtypes)]


def _nested_dtype(data: Any, nested: str) -> Any:
    """The ``NestedDtype`` of column ``nested`` (lazy: reads the schema only)."""
    dtypes, available = _schema(data)
    if nested not in available:
        raise ColumnResolutionError(
            f"no nested column {nested!r}; available columns: {available}"
        )
    return dtypes[nested]


def _bare(name: str | None, nested: str, role: str) -> str | None:
    """Normalize ``"lc.hmjd"`` / ``"hmjd"`` to the bare sub-column name."""
    if name is None:
        return None
    if "." not in name:
        return name
    prefix, rest = name.split(".", 1)
    if prefix != nested:
        raise ColumnResolutionError(
            f"{role} column {name!r} is dotted but does not belong to the nested "
            f"column {nested!r}; pass a sub-column of {nested!r} (bare or dotted)"
        )
    return rest


def _preset_entry(preset: str | None) -> dict[str, Any]:
    """Look up a preset by name (empty dict when ``preset`` is ``None``)."""
    if preset is None:
        return {}
    try:
        return COLUMN_PRESETS[preset]
    except KeyError:
        raise ValueError(
            f"unknown preset {preset!r}; choose from {sorted(COLUMN_PRESETS)}"
        ) from None


def resolve_nested_columns(
    data: Any,
    nested: str | None = None,
    *,
    time: str | None = None,
    value: str | None = None,
    error: str | None = None,
    band: str | None = None,
    domain: Domain | str | None = None,
    preset: str | None = None,
) -> NestedColumns:
    """Resolve the light-curve sub-columns of a nested column, without computing.

    Only the nested column's *schema* is read (the ``NestedDtype`` sub-column names),
    so this is safe on a lazy lsdb ``Catalog``. Unset roles fall back to the preset (if
    any) and then to cuPeriod's :class:`~cuperiod.ColumnMap` auto-detection.

    Parameters
    ----------
    data : NestedFrame or lsdb Catalog
        Anything exposing ``.dtypes``.
    nested : str, optional
        Name of the nested column holding the light curves. ``None`` takes the
        preset's nest, or ``"lc"`` without a preset; any name passed here — ``"lc"``
        included — wins over the preset's.
    time, value, error : str, optional
        Sub-column names, bare (``"hmjd"``) or dotted (``"lc.hmjd"``). Auto-detected
        when ``None``.
    band : str, optional
        In-nest band/filter sub-column. Multi-band mode is used **only** when this
        resolves; it is never auto-detected, so a nest that happens to carry a
        ``band`` column is not silently reinterpreted.
    domain : Domain or str, optional
        Brightness domain override; inferred from the value column's name otherwise.
    preset : str, optional
        A key of :data:`COLUMN_PRESETS`. Preset ``time``/``value`` are required to
        exist; preset ``error``/``band`` are dropped when the nest does not have them.
        A ``value`` naming a different column than the preset's (case-insensitive)
        does not inherit the preset's ``error`` or ``domain`` — those describe the
        preset's value: the error is then paired with the new value column
        (``psfMag`` -> ``psfMagErr``; none found -> unweighted) and the domain is
        inferred from its name, falling back to the preset's only when the name says
        nothing. An explicit ``error=``/``domain=`` always wins.

    Returns
    -------
    NestedColumns
        Dotted names plus the resolved :class:`~cuperiod.Domain`.

    Raises
    ------
    ColumnResolutionError
        If the nested column is missing, or time/value cannot be resolved.

    Examples
    --------
    >>> resolve_nested_columns(frame, preset="rubin_dp1_object")   # doctest: +SKIP
    NestedColumns(nested='objectForcedSource',
                  time='objectForcedSource.midpointMjdTai', ...)
    """
    entry = _preset_entry(preset)
    if nested is None:
        nested = str(entry["nested"]) if entry else _DEFAULT_NEST
    subcols = _subcolumn_names(_nested_dtype(data, nested))
    lower = {c.lower(): c for c in subcols}

    def soft(name: str | None) -> str | None:
        """Keep a preset-supplied name only when the nest actually has it."""
        return name if name is not None and name.lower() in lower else None

    time_name = _bare(time, nested, "time") or entry.get("time")
    explicit_value = _bare(value, nested, "value")
    value_name = explicit_value or entry.get("value")
    # A preset's error column and domain describe the preset's *value* column. When
    # value= picks another one (psfMag beside a psfFlux preset), inheriting them
    # weighted magnitudes by nJy errors and searched them as fluxes — BLS/TLS then
    # chased an inverted signal. Let the new value pick its own pairing instead.
    overridden = bool(
        entry
        and explicit_value is not None
        and explicit_value.lower() != str(entry.get("value", "")).lower()
    )
    if error is not None:
        error_name = _bare(error, nested, "error")
    elif overridden:
        error_name = None  # ColumnMap pairs it with the value (psfMag -> psfMagErr)
    else:
        error_name = soft(entry.get("error"))
    if band is not None:
        band_name = _bare(band, nested, "band")
        if band_name is not None and band_name.lower() not in lower:
            raise ColumnResolutionError(
                f"band column {band!r} is not a sub-column of {nested!r}; "
                f"available: {subcols}"
            )
    else:
        band_name = soft(entry.get("band"))

    resolved_domain = coerce_domain(domain)
    if resolved_domain is None and entry:
        preset_domain = coerce_domain(entry.get("domain"))
        if overridden and explicit_value is not None:
            resolved_domain = infer_domain(explicit_value) or preset_domain
        else:
            resolved_domain = preset_domain
    resolved = ColumnMap(
        time=time_name, value=value_name, error=error_name
    ).resolve(subcols, domain=resolved_domain)
    band_actual = None if band_name is None else lower[band_name.lower()]
    return NestedColumns(
        nested=nested,
        time=f"{nested}.{resolved.time}",
        value=f"{nested}.{resolved.value}",
        error=None if resolved.error is None else f"{nested}.{resolved.error}",
        band=None if band_actual is None else f"{nested}.{band_actual}",
        domain=resolved.domain,
    )


# --- shared kernel ------------------------------------------------------------


def _unique_in_order(labels: np.ndarray) -> list[Any]:
    """Distinct band labels, first-seen order (stable band ordering per object)."""
    seen: dict[Any, None] = {}
    for label in labels.tolist():
        seen.setdefault(label, None)
    return list(seen)


@dataclass
class _Kernel:
    """Column/method configuration shared by both tiers.

    Deliberately holds no engine and no light-curve data, so dask can pickle it to a
    worker; the engine is built inside the worker by :class:`_PartitionKernel`.
    """

    columns: NestedColumns
    method: str
    settings: BaseSettings | None
    backend: str
    grid: GridSpec | None
    n_best: int
    prefix: str
    _prepared: tuple[PeriodogramMethod, BaseSettings, str] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def out_columns(self) -> tuple[str, ...]:
        """Names of the result columns this kernel produces."""
        p = self.prefix
        names = [f"{p}best_period", f"{p}best_power", f"{p}fap"]
        names += [f"{p}period_{i}" for i in range(2, max(self.n_best, 1) + 1)]
        return tuple(names)

    def validate(self) -> PeriodogramMethod:
        """Look the method up and reject a configuration no object can satisfy.

        A multi-band run with a single-band-only method cannot produce a result for
        *any* row, and :meth:`evaluate` turns per-object failures into NaN rows — so
        the combination has to raise here, before a single object is touched, or the
        misconfiguration ships as a silently all-NaN frame. A settings object of the
        wrong method is rejected here too, for the same reason.
        """
        method = get_method(self.method)
        if self.columns.band is not None and not method.supports_multiband:
            raise ValueError(f"{method.name} does not support multi-band input")
        method.coerce_settings(self.settings)
        return method

    def prepare(self) -> tuple[PeriodogramMethod, BaseSettings, str]:
        """Resolve method, settings and backend once (raises on misconfiguration).

        The backend goes through
        :meth:`~cuperiod.methods.base.PeriodogramMethod.resolve_backend_for`, so a
        ``backend="auto"`` request defers to ``settings.backend`` (and its
        ``CUPERIOD_<METHOD>_BACKEND`` override) exactly as :func:`cuperiod.periodogram`
        does. Resolution is memoised by the backend module, and this kernel keeps the
        result, so the row-wise tier does not re-probe the environment per object.
        """
        if self._prepared is None:
            method = self.validate()
            settings = method.coerce_settings(self.settings)
            self._prepared = (
                method,
                settings,
                method.resolve_backend_for(self.backend, settings),
            )
        return self._prepared

    def empty_result(self, index: Any = None) -> pd.DataFrame:
        """A correctly-typed zero-row result (dask/lsdb meta, empty inputs)."""
        import pandas as pd

        data = {name: pd.Series(dtype="float64") for name in self.out_columns}
        return pd.DataFrame(data, index=index)

    # -- per-object work ------------------------------------------------------
    def nan_row(self) -> dict[str, float]:
        """An all-NaN result row (a failed or unusable object)."""
        return dict.fromkeys(self.out_columns, _NAN)

    def light_curve(
        self,
        time: Any,
        value: Any,
        error: Any,
        band: Any,
    ) -> LightCurve | MultiBandLightCurve:
        """Assemble one object's light curve from its per-epoch arrays."""
        t = np.asarray(time, dtype=np.float64)
        v = np.asarray(value, dtype=np.float64)
        e = None if error is None else np.asarray(error, dtype=np.float64)
        if band is None:
            return LightCurve.from_arrays(t, v, e, domain=self.columns.domain)
        labels = np.asarray(band)
        bands: dict[str, LightCurve] = {}
        for label in _unique_in_order(labels):
            mask = labels == label
            bands[str(label)] = LightCurve.from_arrays(
                t[mask],
                v[mask],
                None if e is None else e[mask],
                domain=self.columns.domain,
                meta={"band": str(label)},
            )
        return MultiBandLightCurve.from_light_curves(bands)

    def compute(
        self,
        method: PeriodogramMethod,
        settings: BaseSettings,
        backend: str,
        lc: LightCurve | MultiBandLightCurve,
        engine: object | None,
    ) -> Periodogram:
        """Run one light curve, reusing ``engine`` when the method has one.

        The result carries the settings' peak-selection defaults (``n_peaks``,
        ``peak_separation_rayleigh``, ``alias_freq_tolerance``, ...), as
        :func:`cuperiod.periodogram` attaches them, so ``period_2 ...`` are picked
        by the same rule as in a single-curve call.
        """
        raise_if_unusable(lc, method.name)
        if isinstance(lc, MultiBandLightCurve):
            if not method.supports_multiband:
                raise ValueError(f"{method.name} does not support multi-band input")
            grid = self.grid or _multiband_grid(method, lc, settings)
            pg = method.multiband_power(grid, lc, settings, backend, engine=engine)
        else:
            single = (
                lc.in_domain(method.natural_domain) if method.natural_domain else lc
            )
            grid = self.grid or method.default_grid(single, settings)
            pg = method.power(grid, single, settings, backend, engine=engine)
        return pg.with_peak_defaults(settings)

    def peaks_row(self, pg: Periodogram) -> dict[str, float]:
        """Flatten a periodogram to this kernel's result columns."""
        row = self.nan_row()
        peaks = pg.best_periods(max(self.n_best, 1))
        if not peaks:
            return row
        best = peaks[0]
        p = self.prefix
        row[f"{p}best_period"] = float(best.period)
        row[f"{p}best_power"] = float(best.power)
        row[f"{p}fap"] = float(best.extra.get("fap", _NAN))
        for rank in range(2, max(self.n_best, 1) + 1):
            if len(peaks) >= rank:
                row[f"{p}period_{rank}"] = float(peaks[rank - 1].period)
        return row

    def evaluate(
        self,
        prepared: tuple[PeriodogramMethod, BaseSettings, str],
        time: Any,
        value: Any,
        error: Any,
        band: Any,
        engine: object | None = None,
    ) -> dict[str, float]:
        """One object, from raw arrays to result columns. Never raises.

        A per-object failure (too few detections, no baseline, a degenerate fit) must
        not abort a survey-scale run, so it becomes an all-NaN row.
        """
        method, settings, backend = prepared
        try:
            lc = self.light_curve(time, value, error, band)
            return self.peaks_row(self.compute(method, settings, backend, lc, engine))
        except Exception:  # a bad object must not kill the partition
            return self.nan_row()


@dataclass
class _RowKernel(_Kernel):
    """Tier 1: called by ``map_rows`` with one row's arrays as a dict."""

    def __call__(self, row: Mapping[str, Any]) -> dict[str, float]:
        cols = self.columns
        prepared = self.prepare()
        return self.evaluate(
            prepared,
            row[cols.time],
            row[cols.value],
            None if cols.error is None else row.get(cols.error),
            None if cols.band is None else row.get(cols.band),
        )


@dataclass
class _PartitionKernel(_Kernel):
    """Tier 2: called with a whole partition; one engine serves every object."""

    def __call__(self, df: Any) -> pd.DataFrame:
        import pandas as pd

        if len(df) == 0:
            return self.empty_result(df.index[:0] if hasattr(df, "index") else None)
        cols = self.columns
        prepared = self.prepare()
        method, settings, backend = prepared
        starts, ends, flat = _flat_buffers(df[cols.nested], cols.subcolumns)
        time = flat[cols.time.split(".", 1)[1]]
        value = flat[cols.value.split(".", 1)[1]]
        error = None if cols.error is None else flat[cols.error.split(".", 1)[1]]
        band = None if cols.band is None else flat[cols.band.split(".", 1)[1]]

        engine = method.make_engine(backend, settings)
        try:
            rows = [
                self.evaluate(
                    prepared,
                    time[i:j],
                    value[i:j],
                    None if error is None else error[i:j],
                    None if band is None else band[i:j],
                    engine=engine,
                )
                for i, j in zip(starts, ends, strict=True)
            ]
        finally:
            gpu_engine = engine is not None and is_gpu_selector(backend)
            del engine
            # Only a GPU engine leaves blocks in cupy's pool. A CPU engine (GLS on
            # "finufft" builds one too) must not import cupy, where it is installed,
            # and flush a device pool this partition never touched.
            if gpu_engine:
                from cuperiod.core.device import free_gpu_memory

                free_gpu_memory()
        return pd.DataFrame(rows, index=df.index, columns=list(self.out_columns))


def _flat_buffers(
    series: Any, subcolumns: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read a nested column's Arrow buffers once: per-object slices + flat arrays.

    The nested extension array is CSR-like — one flat array per sub-column plus a
    shared list-offset array — so a partition's whole light-curve payload comes out in
    a handful of buffer reads instead of one materialized DataFrame per object.

    Returns
    -------
    starts, ends : numpy.ndarray
        Per-object slice bounds into the flat arrays.
    flat : dict of str to numpy.ndarray
        The concatenated per-epoch values, keyed by bare sub-column name.
    """
    array = series.array
    offsets = np.asarray(array.list_offsets, dtype=np.int64)
    offsets = offsets - offsets[0]  # a sliced partition may start mid-buffer
    struct = array.struct_array
    if hasattr(struct, "combine_chunks"):  # a ChunkedArray in nested-pandas >= 0.7
        struct = struct.combine_chunks()
    total = int(offsets[-1])
    flat: dict[str, np.ndarray] = {}
    for name in subcolumns:
        field_array = struct.field(name)
        values = np.asarray(field_array.flatten().to_numpy(zero_copy_only=False))
        if values.size != total and hasattr(field_array, "values"):
            # Nulls/slicing can make flatten() disagree with the raw offsets; the
            # child array always indexes by them.
            values = np.asarray(
                field_array.values.to_numpy(zero_copy_only=False)
            )
        flat[name] = values
    return offsets[:-1], offsets[1:], flat


# --- lsdb detection -----------------------------------------------------------


def _is_lsdb_catalog(data: Any) -> bool:
    """Whether ``data`` is a lazy lsdb ``Catalog`` (no lsdb import needed).

    lsdb's ``Catalog.map_rows`` requires ``meta=`` and returns a lazy catalog, while
    ``NestedFrame.map_rows`` computes eagerly and rejects ``meta`` — the two paths
    differ only in that argument, so the class's defining module is enough to pick one.
    """
    module = type(data).__module__ or ""
    return module.startswith("lsdb") and hasattr(data, "map_rows")


def _catalog_meta(out_columns: tuple[str, ...]) -> dict[str, type]:
    """The lsdb ``meta`` for the result columns (all float64).

    With ``append_columns=True`` lsdb wants *only* the added columns, which is exactly
    what a kernel returns, so one mapping serves both modes.
    """
    return dict.fromkeys(out_columns, float)


def _refuse_column_clash(data: Any, out_columns: tuple[str, ...]) -> None:
    """Raise when appending would duplicate a column name already on ``data``.

    ``map_rows(append_columns=True)`` ends in a plain ``concat`` with no uniqueness
    check, so a second method run on the first one's output (or a base table that
    already has a ``fap`` column) produced two ``best_period`` columns: selection
    then returns a frame instead of a Series, and Parquet export refuses the table.
    Read from the schema only, so a lazy catalog stays lazy.
    """
    _, existing = _schema(data)
    taken = set(existing)
    clash = [name for name in out_columns if name in taken]
    if clash:
        raise ValueError(
            f"the input already has result column(s) {clash}; appending would "
            "duplicate them. Pass prefix= (e.g. prefix='pdm_') to name this run's "
            "columns apart, or append_columns=False to get the results alone."
        )


# --- tier 1: row-wise ---------------------------------------------------------


def nested_periodogram(
    data: Any,
    nested: str | None = None,
    *,
    time: str | None = None,
    value: str | None = None,
    error: str | None = None,
    band: str | None = None,
    preset: str | None = None,
    method: str = "GLS",
    settings: BaseSettings | None = None,
    backend: str = "auto",
    grid: GridSpec | None = None,
    domain: Domain | str | None = None,
    n_best: int = 1,
    prefix: str = "",
    append_columns: bool = True,
    meta: Any = None,
) -> Any:
    """Run a periodogram on every row of a nested light-curve table (Tier 1).

    One row = one object = one light curve = one periodogram. The nested column is
    read through ``map_rows``, so this works identically on an in-memory
    ``NestedFrame`` and on a lazy lsdb ``Catalog`` (which stays lazy — call
    ``.compute()`` when you want the answer).

    Objects that cannot be searched (too few detections, no time baseline, a
    degenerate fit) yield NaN result columns instead of raising: at survey scale a
    single bad light curve must never abort the run.

    Parameters
    ----------
    data : NestedFrame or lsdb Catalog
        A table with one row per object and the light curves in a nested column.
    nested : str, optional
        The nested column's name. ``None`` takes the preset's nest, or ``"lc"``
        without a preset; a name passed here (``"lc"`` included) always wins.
    time, value, error : str, optional
        Sub-column names, bare (``"hmjd"``) or dotted (``"lc.hmjd"``). Auto-detected
        from the nest's schema when ``None``.
    band : str, optional
        In-nest band sub-column (e.g. ``"lc.band"``). When given, each row is built as
        a :class:`~cuperiod.MultiBandLightCurve` grouped on that column and the
        method's multi-band model runs; otherwise every epoch is one band.
    preset : str, optional
        A key of :data:`COLUMN_PRESETS` (``"ztf_dr22"``, ``"ztf_alerts"``,
        ``"rubin_dp1_object"``, ``"rubin_dp1_dia"``) filling every column parameter you
        did not set. A ``value`` naming another column than the preset's keeps none of
        the preset's ``error``/``domain`` (see :func:`resolve_nested_columns`).
    method : str, default "GLS"
        Any registered method name.
    settings : settings model, optional
        The method's settings (e.g. :class:`~cuperiod.GLSSettings`). Its peak fields
        (``peak_separation_rayleigh``, ...) shape the ``period_i`` columns.
    backend : str, default "auto"
        ``"auto"``/``"cpu"``/``"gpu"`` or a concrete backend. ``"auto"`` defers to
        ``settings.backend`` (``CUPERIOD_<METHOD>_BACKEND``) when that is pinned.
    grid : GridSpec, optional
        One explicit trial grid for every object; by default each object gets the
        method's own grid from its baseline and sampling.
    domain : Domain or str, optional
        Brightness-domain override (``"flux"`` for difference-imaging fluxes, which
        are legitimately negative and must not be converted to magnitudes).
    n_best : int, default 1
        Peaks to report. ``> 1`` adds ``period_2 … period_n`` columns.
    prefix : str, default ""
        Prepended to every result-column name (e.g. ``"gls_"``).
    append_columns : bool, default True
        Keep the input columns and append the results; ``False`` returns only the
        result columns. Appending onto a table that already has a column of a result
        name raises — use ``prefix=`` to run several methods on one frame.
    meta : dict, optional
        lsdb ``meta`` override. Built automatically (``{column: float}``) otherwise;
        ignored for an in-memory ``NestedFrame``.

    Returns
    -------
    NestedFrame or lsdb Catalog
        The same kind of object that came in, with ``{prefix}best_period``,
        ``{prefix}best_power``, ``{prefix}fap`` (NaN when the method reports none) and
        any ``{prefix}period_i`` columns. A zero-row frame gives zero rows with the
        same float64 result columns.

    Raises
    ------
    ValueError
        If ``band`` resolves but ``method`` has no multi-band model, or if
        ``append_columns`` would duplicate an existing column name. Per-object
        failures become NaN rows, so a configuration that can never work is rejected
        up front instead.

    See Also
    --------
    partition_periodogram : the batched, engine-reusing tier (GPU throughput).

    Notes
    -----
    On dask (lsdb), run **one worker per GPU** — e.g. a
    ``dask_cuda.LocalCUDACluster`` — and let each worker own its device. Kernels are
    plain picklable configuration; compute engines are never pickled but built lazily
    inside the worker, so a GPU context is created in the process that uses it.

    Examples
    --------
    A ZTF DR22-style ``NestedFrame`` (single band per row: DR22 keeps the filter in
    the base column ``filterid``):

    >>> from cuperiod.interop import nested_periodogram
    >>> out = nested_periodogram(frame, "lc", preset="ztf_dr22")   # doctest: +SKIP
    >>> out[["best_period", "best_power", "fap"]].head()           # doctest: +SKIP

    A lazy lsdb catalog of Rubin DP1 forced photometry, multi-band in flux:

    >>> import lsdb                                                # doctest: +SKIP
    >>> cat = lsdb.open_catalog("dp1_object")                      # doctest: +SKIP
    >>> res = nested_periodogram(                                  # doctest: +SKIP
    ...     cat, preset="rubin_dp1_object", method="GLS", n_best=3, prefix="gls_"
    ... )
    >>> res[["gls_best_period", "gls_period_2"]].compute()         # doctest: +SKIP
    """
    require_nested_pandas()
    columns = resolve_nested_columns(
        data,
        nested,
        time=time,
        value=value,
        error=error,
        band=band,
        domain=domain,
        preset=preset,
    )
    kernel = _RowKernel(
        columns=columns,
        method=method,
        settings=settings,
        backend=backend,
        grid=grid,
        n_best=n_best,
        prefix=prefix,
    )
    kernel.validate()  # fail now, not once per object as an all-NaN row
    if append_columns:
        _refuse_column_clash(data, kernel.out_columns)
    read = columns.read_columns
    if _is_lsdb_catalog(data):
        return data.map_rows(
            kernel,
            read,
            meta=_catalog_meta(kernel.out_columns) if meta is None else meta,
            infer_nesting=False,
            append_columns=append_columns,
        )
    if len(data) == 0:
        # nested-pandas never calls the kernel on zero rows and returns a frame with
        # no result columns at all; give the typed empty result Tier 2 gives.
        return _empty_row_result(data, kernel, append_columns)
    return data.map_rows(
        kernel, read, infer_nesting=False, append_columns=append_columns
    )


def _empty_row_result(data: Any, kernel: _Kernel, append_columns: bool) -> Any:
    """Tier 1's answer for a zero-row ``NestedFrame``: typed, correctly named."""
    import pandas as pd
    from nested_pandas import NestedFrame

    results = NestedFrame(kernel.empty_result(data.index[:0]))
    if not append_columns:
        return results
    return pd.concat([data, results], axis=1)


# --- tier 2: partition-wise ---------------------------------------------------


def partition_periodogram(
    data: Any,
    nested: str | None = None,
    *,
    time: str | None = None,
    value: str | None = None,
    error: str | None = None,
    band: str | None = None,
    preset: str | None = None,
    method: str = "GLS",
    settings: BaseSettings | None = None,
    backend: str = "auto",
    grid: GridSpec | None = None,
    domain: Domain | str | None = None,
    n_best: int = 1,
    prefix: str = "",
    meta: Any = None,
) -> Any:
    """Run a periodogram on every object of a partition, batched (Tier 2).

    Same inputs and same result columns as :func:`nested_periodogram`, different
    execution: a partition's nested column is read **once** through its Arrow buffers
    (list offsets + struct fields), each object is a slice of those flat arrays, and
    every object in the partition is evaluated against **one** method engine built
    per partition (:meth:`~cuperiod.methods.base.PeriodogramMethod.make_engine`).

    That is the GPU differentiator: plan/kernel setup is amortized over a whole
    partition of stars instead of paid per star. On a CPU backend (no engine) it is
    still the cheaper path, because the per-object DataFrame round-trip is skipped.

    Parameters
    ----------
    data : NestedFrame or lsdb Catalog
        A ``NestedFrame`` is processed immediately; a ``Catalog`` is wired up with
        ``map_partitions`` and stays lazy.
    nested, time, value, error, band, preset : str, optional
        Column selection, exactly as in :func:`nested_periodogram`.
    method, settings, backend, grid, domain, n_best, prefix
        As in :func:`nested_periodogram`.
    meta : pandas.DataFrame, optional
        lsdb ``meta`` override; an empty float-typed frame is built otherwise.

    Returns
    -------
    pandas.DataFrame or lsdb Catalog
        One row per object, indexed like the input, with the same result columns as
        :func:`nested_periodogram` (result columns only — the input columns are not
        copied). An empty input yields a correctly-typed empty frame, which is what
        lets lsdb infer a schema by calling the kernel on an empty partition.

    Raises
    ------
    ValueError
        If ``band`` resolves but ``method`` has no multi-band model, exactly as in
        :func:`nested_periodogram`.

    See Also
    --------
    nested_periodogram : the row-wise tier.

    Notes
    -----
    One worker per GPU (``dask_cuda.LocalCUDACluster``). The callable dask ships to
    the workers holds only picklable configuration — column names, method name,
    settings, grid — and the engine is created inside the worker on its own device,
    then released when the partition finishes. Engines are never pickled.

    Examples
    --------
    In memory, on a ``NestedFrame``:

    >>> from cuperiod.interop import partition_periodogram
    >>> res = partition_periodogram(frame, "lc", method="GLS")    # doctest: +SKIP
    >>> res["best_period"].head()                                 # doctest: +SKIP

    Over an lsdb catalog, one GPU worker per device:

    >>> from dask_cuda import LocalCUDACluster                    # doctest: +SKIP
    >>> from dask.distributed import Client                       # doctest: +SKIP
    >>> client = Client(LocalCUDACluster())                       # doctest: +SKIP
    >>> res = partition_periodogram(                              # doctest: +SKIP
    ...     cat, preset="ztf_alerts", method="GLS", backend="gpu"
    ... )
    >>> res.compute()                                             # doctest: +SKIP
    """
    require_nested_pandas()
    columns = resolve_nested_columns(
        data,
        nested,
        time=time,
        value=value,
        error=error,
        band=band,
        domain=domain,
        preset=preset,
    )
    kernel = _PartitionKernel(
        columns=columns,
        method=method,
        settings=settings,
        backend=backend,
        grid=grid,
        n_best=n_best,
        prefix=prefix,
    )
    kernel.validate()  # fail now, not once per object as an all-NaN row
    if _is_lsdb_catalog(data):
        return data.map_partitions(
            kernel, meta=kernel.empty_result() if meta is None else meta
        )
    return kernel(data)


__all__ = [
    "COLUMN_PRESETS",
    "PRESET_KEYS",
    "NestedColumns",
    "nested_periodogram",
    "partition_periodogram",
    "require_nested_pandas",
    "resolve_nested_columns",
]
