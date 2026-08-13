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

from typing import Any, ClassVar, Literal

import numpy as np

from cuperiod.core._arrayapi import resolve_torch_device
from cuperiod.core._typing import FloatArray
from cuperiod.core.columns import Domain
from cuperiod.core.config import BLSSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods._bls_core import bls_power
from cuperiod.methods.base import PeriodogramMethod, register

#: Per-period fields shared by every backend (astropy + the in-house search).
_SEGMENT_FIELDS = ("period", "power", "depth", "depth_snr", "duration", "transit_time")


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


def _segment_grids(
    baseline: float, settings: BLSSettings
) -> list[tuple[FloatArray, FloatArray]]:
    """``[(periods, durations), ...]`` for each log period segment, ascending period."""
    max_period = _max_period(baseline, settings)
    if max_period <= settings.min_period_days:
        return []
    df = settings.grid_duration_frac / (settings.grid_oversample * baseline)
    grids: list[tuple[FloatArray, FloatArray]] = []
    p_lo = settings.min_period_days
    while p_lo < max_period:
        p_hi = min(p_lo * settings.segment_factor, max_period)
        freq = np.arange(1.0 / p_hi, 1.0 / p_lo, df)
        if freq.size:
            periods = np.asarray(1.0 / freq[::-1], dtype=np.float64)
            grids.append((periods, _segment_durations(p_lo, settings)))
        p_lo = p_hi
    return grids


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
        finite = lc.finite()
        grids = _segment_grids(finite.baseline, settings)
        if not grids:
            raise InsufficientDataError(
                "BLS: baseline too short for the configured period range"
            )
        periods = np.concatenate([p for p, _ in grids])
        return GridSpec(
            kind="period",
            values=periods,
            uniform=False,
            meta={"segmented": True},
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

        if grid.meta.get("segmented", False):
            segments = _segment_grids(finite.baseline, settings)
        else:  # user-supplied flat period grid
            periods = grid.period
            durations = _segment_durations(float(periods.min()), settings)
            segments = [(periods, durations)]
        if not segments:
            raise InsufficientDataError(
                "BLS: baseline too short for the configured period range"
            )

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

        return bls_multiband_power(grid, mblc, settings, backend, self)

    def estimate_device_bytes(self, n_points: int) -> int:
        return 128 * 1024**2 + n_points * 8 * 10


register(BLSMethod())

__all__ = ["BLSMethod"]
