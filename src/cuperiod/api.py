"""The single-light-curve API: :func:`periodogram`.

This is the one entry point most users need. It accepts a light curve in any convenient
form (a :class:`LightCurve`/:class:`MultiBandLightCurve`, a ``(t, y[, dy])`` tuple, a
pandas/astropy/pyarrow table, or a dict of arrays), one or several method names, and an
optional settings/grid/backend, and returns a :class:`Periodogram` (single method) or a
:class:`MultiResult` (several methods).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pydantic_settings import BaseSettings

from cuperiod.core.columns import ColumnMap, Domain, coerce_domain
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import (
    LightCurve,
    MultiBandLightCurve,
    raise_if_unusable,
)
from cuperiod.core.result import MultiResult, Periodogram
from cuperiod.methods.base import (
    PeriodogramMethod,
    SettingsInput,
    get_method,
    select_settings,
)

#: Median of an unlabelled ``(t, y, dy)`` tuple inside which ``y`` looks like a
#: normalised flux rather than a magnitude (see :func:`_warn_flux_tuple`).
_NORMALISED_FLUX_RANGE = (0.5, 1.5)


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
    domain: Domain | str | None = None,
) -> LightCurve | MultiBandLightCurve:
    """Coerce a user input into a :class:`LightCurve` or :class:`MultiBandLightCurve`.

    Parameters
    ----------
    data : various
        A :class:`LightCurve`/:class:`MultiBandLightCurve`, a ``(t, y[, dy])`` tuple, a
        ``{band: LightCurve}`` mapping, a pandas/astropy/pyarrow table, or a
        ``{column: array}`` mapping. A bare ``(t, y[, dy])`` tuple is read as
        magnitudes unless ``domain`` says otherwise.
    columns : ColumnMap, optional
        Column overrides for table inputs. Naming a band column
        (``ColumnMap(band=...)``) reads a long-format table as a
        :class:`MultiBandLightCurve` split on it; without one a table is a single
        light curve.
    domain : Domain or {"magnitude", "flux"}, optional
        Brightness-domain override. For multi-band input it applies to every band.

    Returns
    -------
    LightCurve or MultiBandLightCurve
    """
    dom = coerce_domain(domain)
    if isinstance(data, LightCurve):
        return data if dom is None else data.in_domain(dom)
    if isinstance(data, MultiBandLightCurve):
        _check_bands(data.bands, dom)
        if dom is None:
            return data
        return MultiBandLightCurve(
            bands={k: lc.in_domain(dom) for k, lc in data.bands.items()},
            meta=data.meta,
        )
    if _looks_like_arrays(data):
        t, v = data[0], data[1]
        e = data[2] if len(data) == 3 else None
        return LightCurve.from_arrays(t, v, e, domain=dom or Domain.MAGNITUDE)
    band_split = columns is not None and columns.band is not None
    if isinstance(data, Mapping):
        values = list(data.values())
        if values and all(isinstance(v, LightCurve) for v in values):
            _check_bands(data, dom)
            bands = data if dom is None else {
                k: lc.in_domain(dom) for k, lc in data.items()
            }
            return MultiBandLightCurve.from_light_curves(bands)
        if band_split:
            return _split_bands(data, columns, dom)
        # dict of arrays -> a table
        names, getter = _mapping_table(data)
        return LightCurve._from_table(names, getter, columns, dom, None)
    if band_split:
        return _split_bands(data, columns, dom)
    # pandas / astropy / pyarrow table
    return LightCurve.from_dataframe(data, columns=columns, domain=dom)


def _check_bands(bands: Mapping[str, Any], domain: Domain | None) -> None:
    """Refuse multi-band input the joint fits would silently get wrong.

    Every band must be a :class:`LightCurve`, and — unless ``domain=`` is about to
    convert them all — every band must be in the same brightness domain: a magnitude
    band and a flux band of one star vary in anti-phase, which the shared-phase
    multi-band models fit as a cancelled signal.
    """
    for name, lc in bands.items():
        if not isinstance(lc, LightCurve):
            raise TypeError(
                f"band {name!r} is a {type(lc).__name__}, not a LightCurve; build "
                "each band with LightCurve.from_arrays(...)"
            )
    if domain is None:
        domains = {name: lc.domain for name, lc in bands.items()}
        if len(set(domains.values())) > 1:
            listing = ", ".join(f"{k}: {v}" for k, v in domains.items())
            raise ValueError(
                f"bands are in different brightness domains ({listing}); pass "
                "domain='magnitude' or domain='flux' to convert them all, or convert "
                "the bands with LightCurve.in_domain(...)"
            )


def _split_bands(
    data: Any, columns: ColumnMap | None, domain: Domain | None
) -> MultiBandLightCurve:
    """A long-format table (any supported kind) split on ``columns.band``.

    The caller named the band column explicitly, so the table is multi-band by request
    — as the batch file loader already treats ``ColumnMap(band=...)``. Without a named
    band column a table stays single-band (the documented default).
    :meth:`MultiBandLightCurve.from_dataframe` accepts pandas, astropy and pyarrow
    tables and ``{column: array}`` mappings alike.
    """
    return MultiBandLightCurve.from_dataframe(data, columns=columns, domain=domain)


def _mapping_table(data: Mapping[str, Any]) -> tuple[list[str], Any]:
    from cuperiod.core.lightcurve import _adapt_table

    return _adapt_table(data)


def _select_settings(
    method: PeriodogramMethod, settings: SettingsInput
) -> BaseSettings:
    """Pick the settings object that applies to ``method`` (see ``select_settings``)."""
    return select_settings([method], settings)[method.name]


def _warn_flux_tuple(
    data: Any,
    lc: LightCurve | MultiBandLightCurve,
    methods: Sequence[PeriodogramMethod],
) -> None:
    """Warn when a box/transit search is about to read a normalised flux as magnitudes.

    A bare ``(t, y[, dy])`` tuple is magnitudes by documented default, and BLS/TLS
    convert to flux with ``10**(-0.4 y)`` — which turns every dip of a normalised flux
    into a brightening the dip-only searches cannot see. Values centred near 1 are the
    tell-tale of a normalised flux (real magnitudes are rarely in 0.5-1.5), so only
    then is the assumption flagged. Zero-centred fluxes stay indistinguishable from
    differential magnitudes and are left alone.
    """
    if not isinstance(lc, LightCurve) or not _looks_like_arrays(data):
        return
    flux_methods = [m.name for m in methods if m.natural_domain is Domain.FLUX]
    if not flux_methods or lc.value.size == 0:
        return
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = float(np.nanmedian(lc.value))
    lo, hi = _NORMALISED_FLUX_RANGE
    if lo < median < hi:
        warnings.warn(
            f"{', '.join(flux_methods)}: the (t, y, dy) tuple is read as magnitudes "
            f"(median {median:.3g}), but its values look like a normalised flux; pass "
            "domain='flux' if they are fluxes, or a LightCurve with an explicit domain "
            "to silence this.",
            UserWarning,
            stacklevel=3,
        )


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
    backend: str | None,
    grid: GridSpec | None,
) -> Periodogram:
    resolved_backend = method.resolve_backend_for(backend, settings)
    # A placeholder error column (all 0, -1 or NaN) empties the curve once finite()
    # drops its unusable weights. Name it here, before a grid is built: otherwise the
    # empty curve surfaces as "no usable time baseline" from the grid builder, or as
    # whatever the method's own guard happens to say.
    if isinstance(lc, MultiBandLightCurve):
        if not method.supports_multiband:
            raise ValueError(f"{method.name} does not support multi-band input")
        raise_if_unusable(lc, f"{method.name} multiband")
        g = grid or _multiband_grid(method, lc, settings)
        pg = method.multiband_power(g, lc, settings, resolved_backend)
    else:
        single = lc.in_domain(method.natural_domain) if method.natural_domain else lc
        raise_if_unusable(single, method.name)
        g = grid or method.default_grid(single, settings)
        pg = method.power(g, single, settings, resolved_backend, engine=None)
    # The settings' peak-selection and stored-spectrum fields (n_peaks,
    # peak_separation_rayleigh, alias_freq_tolerance, harmonic_max, downsample_points)
    # become this result's defaults for best_periods()/downsample()/to_dict().
    return pg.with_peak_defaults(settings)


def periodogram(
    data: Any,
    method: str | Sequence[str] = "GLS",
    *,
    backend: str | None = "auto",
    grid: GridSpec | None = None,
    settings: SettingsInput = None,
    columns: ColumnMap | None = None,
    domain: Domain | str | None = None,
) -> Periodogram | MultiResult:
    """Compute one or more periodograms for a single light curve.

    Parameters
    ----------
    data : various
        The light curve (see :func:`to_input` for accepted forms). A bare
        ``(t, y[, dy])`` tuple is read as magnitudes unless ``domain`` says otherwise.
    method : str or sequence of str, default "GLS"
        One method name, or several. Case-insensitive (``"gls"`` == ``"GLS"``).
    backend : str, default "auto"
        ``"auto"`` (GPU when available, else CPU), ``"cpu"``, ``"gpu"``, or a concrete
        backend name (case-insensitive). ``"auto"`` (or ``None``) defers to each
        method's ``settings.backend`` — itself ``"auto"`` unless set, including via
        ``CUPERIOD_<METHOD>_BACKEND`` — while any other value overrides it.
    grid : GridSpec, optional
        Custom trial grid; defaults to the method's grid for this light curve.
    settings : settings model or mapping, optional
        A single method's settings, or a ``{method: settings}`` mapping whose keys are
        matched like method names (``"String-Length"`` works). A settings object of
        the wrong class, or a key naming no method, raises.
    columns : ColumnMap, optional
        Column overrides for table inputs; ``ColumnMap(band=...)`` splits a
        long-format table into a multi-band light curve.
    domain : Domain or {"magnitude", "flux"}, optional
        Brightness-domain override (applied to every band of multi-band input).

    Returns
    -------
    Periodogram or MultiResult
        A :class:`Periodogram` if ``method`` is a single string, else a
        :class:`MultiResult` keyed by method name.

    Raises
    ------
    TypeError
        If ``settings`` holds a settings object of the wrong class.
    UnknownMethodError
        If a method name, or a ``settings`` mapping key, names no registered method.
    BackendUnavailableError
        If the requested (or settings-pinned) backend cannot run here.

    Warns
    -----
    UserWarning
        When BLS/TLS receive a bare tuple without ``domain`` whose values look like a
        normalised flux (median between 0.5 and 1.5).

    Examples
    --------
    >>> pg = periodogram((t, mag, err), "GLS")          # doctest: +SKIP
    >>> pg.best_period()                                # doctest: +SKIP
    >>> res = periodogram(df, ["GLS", "BLS"])           # doctest: +SKIP
    """
    lc = to_input(data, columns=columns, domain=domain)
    names = [method] if isinstance(method, str) else list(method)
    methods = [get_method(name) for name in names]
    chosen = select_settings(methods, settings)
    if domain is None:
        _warn_flux_tuple(data, lc, methods)
    results: dict[str, Periodogram] = {}
    for m in methods:
        results[m.name] = _run_one(m, lc, chosen[m.name], backend, grid)
    if isinstance(method, str):
        return results[methods[0].name]
    return MultiResult(results)


def best_periods(
    result: Periodogram | MultiResult, n: int | None = None, **kwargs: Any
) -> Any:
    """Convenience wrapper around ``result.best_periods(n)``.

    ``n=None`` takes the result's own default — the method settings' ``n_peaks``
    (10 unless set).
    """
    return result.best_periods(n, **kwargs)


__all__ = ["best_periods", "periodogram", "to_input"]
