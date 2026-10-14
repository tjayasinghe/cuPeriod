"""Box Least Squares (BLS) for eclipsing binaries and box-like transit signals.

The search runs in log-spaced period segments so the absolute box-duration grid tracks
the period, exactly as a transit's duration scales. Within a segment the box search is
delegated to one of three backends — astropy's compiled ``BoxLeastSquares`` (the fast
CPU product path), the in-house vectorized ``numpy`` reference, or the ``cupy`` CUDA
kernel (the full-catalog GPU path) — all of which agree to floating-point round-off.

BLS works in the flux domain, where a transit/eclipse is a *dip*; magnitude inputs are
converted to flux automatically. Each peak carries the box parameters (depth, duration,
mid-transit time, depth-SNR) and the global signal-detection efficiency (SDE).
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

import numpy as np

from cuperiod.core._arrayapi import resolve_torch_device
from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import Domain
from cuperiod.core.config import BLSSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec, check_grid_size
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods._bls_core import bls_power
from cuperiod.methods.base import PeriodogramMethod, register

#: Per-period fields shared by every backend (astropy + the in-house search).
_SEGMENT_FIELDS = ("period", "power", "depth", "depth_snr", "duration", "transit_time")

#: ``(periods, durations)`` of one log period segment.
Segment = tuple[FloatArray, FloatArray]


def _segment_durations(p_lo: float, settings: BLSSettings) -> FloatArray:
    """Box durations (days) for the period segment starting at ``p_lo``.

    Fractions ``duration_min_frac``-``duration_max_frac`` of the segment's minimum
    period, floored at ``min_duration_days``.
    """
    d_hi = settings.duration_max_frac * p_lo
    d_lo = max(settings.duration_min_frac * p_lo, settings.min_duration_days)
    if d_lo >= d_hi:
        return np.asarray([d_hi], dtype=np.float64)
    return np.asarray(np.geomspace(d_lo, d_hi, settings.n_durations), dtype=np.float64)


def _max_period(baseline: float, settings: BLSSettings) -> float:
    """Largest searchable period: capped so at least ``min_transits`` transits fit."""
    return min(settings.max_period_days, baseline / settings.min_transits)


def _segment_bounds(
    baseline: float, settings: BLSSettings
) -> list[tuple[float, float]]:
    """``[(p_lo, p_hi), ...]`` of the default log period segments, ascending."""
    max_period = _max_period(baseline, settings)
    bounds: list[tuple[float, float]] = []
    p_lo = settings.min_period_days
    while p_lo < max_period:
        p_hi = min(p_lo * settings.segment_factor, max_period)
        bounds.append((p_lo, p_hi))
        p_lo = p_hi
    return bounds


def _segment_periods(
    baseline: float,
    settings: BLSSettings,
    *,
    time: FloatArray | None = None,
    context: str = "BLS default grid",
) -> list[tuple[float, FloatArray]]:
    """``[(p_lo, periods), ...]`` of the non-empty default log segments, ascending.

    Each segment is uniform in frequency at ``grid_duration_frac / (grid_oversample
    * baseline)``. The total size is checked before anything is allocated
    (:func:`~cuperiod.core.grid.check_grid_size`; ``time``, when given, lets the
    message diagnose an outlier-inflated baseline, the usual cause of a huge grid).
    """
    bounds = _segment_bounds(baseline, settings)
    if not bounds:
        return []
    df = settings.grid_duration_frac / (settings.grid_oversample * baseline)
    # np.arange's own length rule, ceil((stop - start) / step), summed over segments.
    total = sum(
        max(0, math.ceil((1.0 / p_lo - 1.0 / p_hi) / df)) for p_lo, p_hi in bounds
    )
    check_grid_size(
        total, automatic=True, time=time, baseline=baseline, context=context
    )
    parts: list[tuple[float, FloatArray]] = []
    for p_lo, p_hi in bounds:
        freq = np.arange(1.0 / p_hi, 1.0 / p_lo, df)
        # arange can overshoot its stop by a rounding quantum: that sample would be a
        # period at (or just below) p_lo -- a duplicate of the previous segment's
        # longest period, or below min_period_days in the first segment.
        freq = freq[freq < 1.0 / p_lo]
        if freq.size:
            parts.append((p_lo, np.asarray(1.0 / freq[::-1], dtype=np.float64)))
    return parts


def _segment_grids(
    baseline: float,
    settings: BLSSettings,
    *,
    time: FloatArray | None = None,
    context: str = "BLS default grid",
) -> list[Segment]:
    """``[(periods, durations), ...]`` for each log period segment, ascending period.

    The default grid of a light curve with this ``baseline`` (see
    :func:`_segment_periods`), each segment with its own duration set.
    """
    return [
        (periods, _segment_durations(p_lo, settings))
        for p_lo, periods in _segment_periods(
            baseline, settings, time=time, context=context
        )
    ]


def _log_segments(periods: FloatArray, settings: BLSSettings) -> list[Segment]:
    """Split an arbitrary period grid into the log segments the default grid uses.

    Segment ``k`` holds the periods in ``[p0 * f**k, p0 * f**(k+1))`` (``p0`` the
    shortest period, ``f = segment_factor``) and is searched with the durations of
    a segment starting at ``p0 * f**k``, so a box duration tracks the period as on
    the default grid -- one duration set scaled to the shortest period would fit a
    long-period eclipse with boxes far too short, and blow the phase-bin count (the
    bin is a tenth of the shortest box) up to ~1e5 per period on a 1-100 d grid.
    The grid's periods are kept exactly (in their order within each segment).
    """
    p0 = float(periods.min())
    with np.errstate(divide="ignore", invalid="ignore"):
        index = np.floor(
            np.log(periods / p0) / math.log(settings.segment_factor) + 1e-9
        ).astype(np.int64)
    segments: list[Segment] = []
    for k in np.unique(index).tolist():
        segments.append(
            (
                np.ascontiguousarray(periods[index == k]),
                _segment_durations(p0 * settings.segment_factor**k, settings),
            )
        )
    return segments


def _grid_segments(grid: GridSpec, settings: BLSSettings) -> list[Segment]:
    """The ``(periods, durations)`` segments that search exactly ``grid``'s periods.

    A grid from :meth:`BLSMethod.default_grid` records its segments
    (``meta["segment_starts"]`` / ``meta["segment_sizes"]``) and is split exactly
    there, whatever light curve or settings built it; any other grid is split into
    log segments by :func:`_log_segments`. The grid is never rebuilt from the light
    curve, so a grid made with other settings, or for another light curve, is
    searched as given. Shared by the single- and multi-band searches.
    """
    periods = np.asarray(grid.period, dtype=np.float64)
    if periods.size == 0:
        return []
    if not np.all(np.isfinite(periods)):
        # A frequency grid may start at 0 (period inf); a box cannot be folded on it.
        raise ValueError(
            "BLS: trial periods must be finite; drop frequency 0 from the grid"
        )
    starts = grid.meta.get("segment_starts")
    sizes = grid.meta.get("segment_sizes")
    if (
        starts is not None
        and sizes is not None
        and len(starts) == len(sizes)
        and sum(int(s) for s in sizes) == periods.size
    ):
        segments: list[Segment] = []
        offset = 0
        for p_lo, size in zip(starts, sizes, strict=True):
            stop = offset + int(size)
            segments.append(
                (periods[offset:stop], _segment_durations(float(p_lo), settings))
            )
            offset = stop
        return segments
    return _log_segments(periods, settings)


def _segment_power(
    backend: str,
    jd: FloatArray,
    flux: FloatArray,
    err: FloatArray,
    periods: FloatArray,
    durations: FloatArray,
    settings: BLSSettings,
    device_cache: dict[str, Any] | None = None,
) -> dict[str, FloatArray]:
    """One segment's per-period box maxima via the configured backend.

    ``device_cache`` is a caller-owned dict shared by the segments of one light
    curve, so the GPU backends upload the curve once per run rather than per segment.
    """
    if backend == "astropy":
        from astropy.timeseries import BoxLeastSquares

        result = BoxLeastSquares(jd, flux, err).power(
            periods,
            durations,
            objective=settings.objective,
            oversample=settings.bins_per_duration,
        )
        return {
            name: np.asarray(getattr(result, name), dtype=np.float64)
            for name in _SEGMENT_FIELDS
        }
    power = bls_power(
        jd,
        flux,
        err,
        periods,
        durations,
        settings.bins_per_duration,
        objective=settings.objective,
        backend=backend,
        batch=settings.batch_periods,
        precision=settings.precision,
        device_cache=device_cache,
    )
    return {name: getattr(power, name) for name in _SEGMENT_FIELDS}


def _assemble(
    chunks: dict[str, list[FloatArray]], n: int, baseline: float, backend: str,
    meta: Any,
) -> Periodogram:
    """Concatenate segment outputs into a single :class:`Periodogram` with SDE."""
    arrays = {key: np.concatenate(parts) for key, parts in chunks.items()}
    power = np.nan_to_num(arrays["power"], nan=0.0, posinf=0.0, neginf=0.0)
    mean, std = float(power.mean()), float(power.std())
    sde = (power - mean) / std if std > 0.0 else np.zeros_like(power)
    frequency = 1.0 / arrays["period"]
    extras = {
        "depth": arrays["depth"],
        "depth_snr": arrays["depth_snr"],
        "duration": arrays["duration"],
        "t0": arrays["transit_time"],
        "sde": sde,
    }
    return Periodogram.from_spectrum(
        method="BLS",
        backend=backend,
        frequency=frequency,
        power=power,
        objective_sense="max",
        n_samples=n,
        baseline=baseline,
        extras=extras,
        meta=meta,
    )


class BLSMethod(PeriodogramMethod):
    """Box Least Squares method (numba / astropy CPU, cupy GPU).

    The CPU product path prefers the multicore ``numba`` box search when numba is
    installed (the ``[fast]`` extra) — a faithful, parallel port of the CUDA kernel
    that beats astropy's compiled ``BoxLeastSquares`` by an order of magnitude — and
    falls back to astropy otherwise. ``numpy`` is the array-module-generic reference
    that shares its source with the ``cupy`` kernel (for floating-point parity tests).
    """

    name: ClassVar[str] = "BLS"
    objective_sense: ClassVar[Literal["max", "min"]] = "max"
    supports_multiband: ClassVar[bool] = True
    natural_domain: ClassVar[Domain] = Domain.FLUX
    settings_cls: ClassVar[type] = BLSSettings
    cpu_backend: ClassVar[str] = "astropy"
    fast_cpu_backend: ClassVar[str | None] = "numba"
    gpu_backend: ClassVar[str | None] = "cupy"
    portable_gpu_backend: ClassVar[str | None] = "torch"
    all_backends: ClassVar[tuple[str, ...]] = (
        "numba", "numpy", "astropy", "cupy", "torch",
    )

    def default_grid(self, lc: LightCurve, settings: BLSSettings) -> GridSpec:  # type: ignore[override]
        """Log period segments, each uniform in frequency, up to the period cap.

        The cap is ``min(max_period_days, baseline / min_transits)``. ``meta``
        records the segmentation (``segment_starts``, ``segment_sizes``) so
        :meth:`power` searches exactly these periods with these segments' box
        durations, whichever light curve the grid is later used with.

        Raises
        ------
        InsufficientDataError
            If the baseline is too short for the configured period range.
        GridTooLargeError
            If the grid would exceed :data:`~cuperiod.core.grid.MAX_GRID_SIZE`
            periods (checked before allocation; usually a mis-converted timestamp).
        """
        finite = lc.finite()
        parts = _segment_periods(finite.baseline, settings, time=finite.time)
        if not parts:
            raise InsufficientDataError(
                "BLS: baseline too short for the configured period range"
            )
        periods = np.concatenate([p for _, p in parts])
        return GridSpec(
            kind="period",
            values=periods,
            uniform=False,
            meta={
                "segmented": True,
                "segment_starts": tuple(float(p_lo) for p_lo, _ in parts),
                "segment_sizes": tuple(int(p.size) for _, p in parts),
            },
        )

    def power(  # type: ignore[override]
        self,
        grid: GridSpec,
        lc: LightCurve,
        settings: BLSSettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        finite = lc.finite()
        n = finite.n
        if n < settings.min_detections:
            raise InsufficientDataError(
                f"BLS: {n} finite points < min_detections {settings.min_detections}"
            )
        if finite.baseline <= 0.0:
            raise InsufficientDataError("BLS: no usable time baseline")

        jd = finite.time
        flux = finite.value
        err = finite.error if finite.error is not None else np.ones_like(flux)
        if backend == "torch" or backend.startswith("torch:"):
            # Fully-qualify the device so both dispatch and the recorded backend are
            # concrete (e.g. "torch:cpu"); settings.device picks the device for bare
            # "torch"/"auto"/"gpu".
            backend = f"torch:{resolve_torch_device(backend, settings.device)}"

        segments = _grid_segments(grid, settings)
        if not segments:
            raise InsufficientDataError("BLS: empty trial grid")

        chunks: dict[str, list[FloatArray]] = {name: [] for name in _SEGMENT_FIELDS}
        device_cache: dict[str, Any] = {}
        for periods, durations in segments:
            seg = _segment_power(
                backend, jd, flux, err, periods, durations, settings, device_cache
            )
            for name in _SEGMENT_FIELDS:
                chunks[name].append(seg[name])
        return _assemble(chunks, n, finite.baseline, backend, finite.meta)

    def multiband_power(  # type: ignore[override]
        self,
        grid: GridSpec,
        mblc: MultiBandLightCurve,
        settings: BLSSettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        from cuperiod.multiband.bls_mb import bls_multiband_power

        if backend == "torch" or backend.startswith("torch:"):
            # As in power(): a bare "torch" would otherwise run on torch's CPU device,
            # ignore settings.device and be recorded without its device.
            backend = f"torch:{resolve_torch_device(backend, settings.device)}"
        return bls_multiband_power(grid, mblc, settings, backend, self)

    def estimate_device_bytes(self, n_points: int) -> int:
        return 128 * 1024**2 + n_points * 8 * 10


register(BLSMethod())

__all__ = ["BLSMethod"]
