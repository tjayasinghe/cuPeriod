"""The single-light-curve API: :func:`periodogram`.

This is the one entry point most users need. It accepts a light curve in any convenient
form (a :class:`LightCurve`/:class:`MultiBandLightCurve`, a ``(t, y[, dy])`` tuple, a
pandas/astropy/pyarrow table, or a dict of arrays), one or several method names, and an
optional settings/grid/backend, and returns a :class:`Periodogram` (single method) or a
:class:`MultiResult` (several methods).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_settings import BaseSettings

from cuperiod.core.columns import ColumnMap, Domain
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import MultiResult, Periodogram
from cuperiod.methods.base import PeriodogramMethod, get_method

#: A single method's settings, a ``{method: settings}`` mapping, or ``None`` (defaults).
SettingsInput = BaseSettings | Mapping[str, BaseSettings] | None


def _looks_like_arrays(obj: Sequence[Any]) -> bool:
    """Whether ``obj`` is a ``(t, y)`` / ``(t, y, dy)`` tuple of array-likes."""
    if not isinstance(obj, (tuple, list)) or not (2 <= len(obj) <= 3):
        return False
    first = obj[0]
    return hasattr(first, "__len__") and not isinstance(first, (str, bytes))


def to_input(
    data: Any,
    *,
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
) -> LightCurve | MultiBandLightCurve:
    """Coerce a user input into a :class:`LightCurve` or :class:`MultiBandLightCurve`.

    Parameters
    ----------
    data : various
        A :class:`LightCurve`/:class:`MultiBandLightCurve`, a ``(t, y[, dy])`` tuple, a
        ``{band: LightCurve}`` mapping, a pandas/astropy/pyarrow table, or a
        ``{column: array}`` mapping.
    columns : ColumnMap, optional
        Column overrides for table inputs.
    domain : Domain, optional
        Brightness-domain override.

    Returns
    -------
    LightCurve or MultiBandLightCurve
    """
    if isinstance(data, (LightCurve, MultiBandLightCurve)):
        if domain is not None and isinstance(data, LightCurve):
            return data.in_domain(domain)
        return data
    if _looks_like_arrays(data):
        t, v = data[0], data[1]
        e = data[2] if len(data) == 3 else None
        return LightCurve.from_arrays(t, v, e, domain=domain or Domain.MAGNITUDE)
    if isinstance(data, Mapping):
        values = list(data.values())
        if values and all(isinstance(v, LightCurve) for v in values):
            return MultiBandLightCurve.from_light_curves(data)
        # dict of arrays -> a table
        names, getter = _mapping_table(data)
        return LightCurve._from_table(names, getter, columns, domain, None)
    # pandas / astropy / pyarrow table
    return LightCurve.from_dataframe(data, columns=columns, domain=domain)


def _mapping_table(data: Mapping[str, Any]) -> tuple[list[str], Any]:
    from cuperiod.core.lightcurve import _adapt_table

    return _adapt_table(data)


def _select_settings(
    method: PeriodogramMethod, settings: SettingsInput
) -> BaseSettings:
    """Pick the settings object that applies to ``method``."""
    if settings is None:
        return method.coerce_settings(None)
    if isinstance(settings, BaseSettings):
        if isinstance(settings, method.settings_cls):
            return settings
        return method.coerce_settings(None)
    # mapping of {method_name: settings}
    for key, value in settings.items():
        if key.upper() == method.name:
            return method.coerce_settings(value)
    return method.coerce_settings(None)


def _multiband_grid(
    method: PeriodogramMethod, mblc: MultiBandLightCurve, settings: BaseSettings
) -> GridSpec:
    """Build a method's default grid from a multi-band stacked baseline."""
    time, value, error, _ = mblc.finite().stacked()
    synthetic = LightCurve.from_arrays(time, value, error)
    return method.default_grid(synthetic, settings)


def _run_one(
    method: PeriodogramMethod,
    lc: LightCurve | MultiBandLightCurve,
    settings: BaseSettings,
    backend: str,
    grid: GridSpec | None,
) -> Periodogram:
    resolved_backend = method.resolve_backend(backend)
    if isinstance(lc, MultiBandLightCurve):
        if not method.supports_multiband:
            raise ValueError(f"{method.name} does not support multi-band input")
        g = grid or _multiband_grid(method, lc, settings)
        return method.multiband_power(g, lc, settings, resolved_backend)
    single = lc.in_domain(method.natural_domain) if method.natural_domain else lc
    g = grid or method.default_grid(single, settings)
    return method.power(g, single, settings, resolved_backend, engine=None)


def periodogram(
    data: Any,
    method: str | Sequence[str] = "GLS",
    *,
    backend: str = "auto",
    grid: GridSpec | None = None,
    settings: SettingsInput = None,
    columns: ColumnMap | None = None,
    domain: Domain | None = None,
) -> Periodogram | MultiResult:
    """Compute one or more periodograms for a single light curve.

    Parameters
    ----------
    data : various
        The light curve (see :func:`to_input` for accepted forms).
    method : str or sequence of str, default "GLS"
        One method name, or several. Case-insensitive (``"gls"`` == ``"GLS"``).
    backend : str, default "auto"
        ``"auto"`` (GPU when available, else CPU), ``"cpu"``, ``"gpu"``, or a concrete
        backend name.
    grid : GridSpec, optional
        Custom trial grid; defaults to the method's grid for this light curve.
    settings : settings model or mapping, optional
        A single method's settings, or a ``{method: settings}`` mapping.
    columns : ColumnMap, optional
        Column overrides for table inputs.
    domain : Domain, optional
        Brightness-domain override.

    Returns
    -------
    Periodogram or MultiResult
        A :class:`Periodogram` if ``method`` is a single string, else a
        :class:`MultiResult` keyed by method name.

    Examples
    --------
    >>> pg = periodogram((t, mag, err), "GLS")          # doctest: +SKIP
    >>> pg.best_period()                                # doctest: +SKIP
    >>> res = periodogram(df, ["GLS", "BLS"])           # doctest: +SKIP
    """
    lc = to_input(data, columns=columns, domain=domain)
    names = [method] if isinstance(method, str) else list(method)
    results: dict[str, Periodogram] = {}
    for name in names:
        m = get_method(name)
        results[m.name] = _run_one(m, lc, _select_settings(m, settings), backend, grid)
    if isinstance(method, str):
        return results[get_method(method).name]
    return MultiResult(results)


def best_periods(
    result: Periodogram | MultiResult, n: int = 10, **kwargs: Any
) -> Any:
    """Convenience wrapper around ``result.best_periods(n)``."""
    return result.best_periods(n, **kwargs)


__all__ = ["best_periods", "periodogram", "to_input"]
