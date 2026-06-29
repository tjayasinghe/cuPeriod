"""Conditional-entropy period search (Graham et al. 2013).

For a trial period the folded data are binned into a 2-D phase-magnitude histogram,
and the Shannon conditional entropy ``H(m | phase)`` of that distribution is computed.
At the true period magnitude is well predicted by phase, so the distribution
concentrates and the conditional entropy drops — this is a *minimization* method. CE is
notably robust to the sparse, aliased sampling of wide-field surveys.

One vectorized kernel runs on numpy (CPU) and cupy (GPU), so the two backends agree
exactly.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, ClassVar, Final, Literal

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.backend import ensure_cuda_dll_path
from cuperiod.core.config import CESettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import (
    GridSpec,
    pseudo_nyquist_frequency,
    uniform_frequency_grid,
)
from cuperiod.core.lightcurve import LightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod, register

CEBackend = Literal["numpy", "cupy"]

#: Trial periods per vectorized batch.
DEFAULT_BATCH: Final = 1024


def _entropy_batch(
    xp: ModuleType,
    tau: Any,
    mag_bin: Any,
    periods: Any,
    *,
    n_phase: int,
    n_mag: int,
    batch: int,
) -> Any:
    """Conditional entropy H(m|phase) for each trial period, vectorized over periods."""
    n_points = int(tau.shape[0])
    n_periods = int(periods.shape[0])
    n_cells = n_phase * n_mag
    entropy = xp.empty(n_periods, dtype=np.float64)
    for start in range(0, n_periods, batch):
        stop = min(start + batch, n_periods)
        pb = periods[start:stop]
        n_p = int(pb.shape[0])
        rows = xp.arange(n_p)
        phase = xp.mod(tau[None, :] / pb[:, None], 1.0)
        phase_bin = (phase * n_phase).astype(np.int64)
        xp.clip(phase_bin, 0, n_phase - 1, out=phase_bin)
        cell = phase_bin * n_mag + mag_bin[None, :]  # (P, N) in [0, n_cells)
        flat = (rows[:, None] * n_cells + cell).ravel()
        count = xp.zeros(n_p * n_cells, dtype=np.float64)
        xp.add.at(count, flat, xp.broadcast_to(xp.ones(1), (n_p, n_points)).ravel())
        count = count.reshape(n_p, n_phase, n_mag)

        phase_total = count.sum(axis=2, keepdims=True)  # (P, n_phase, 1)
        mask = count > 0.0
        safe_count = xp.where(mask, count, 1.0)
        safe_total = xp.where(phase_total > 0.0, phase_total, 1.0)
        term = xp.where(
            mask, count * (xp.log(safe_total) - xp.log(safe_count)), 0.0
        )
        entropy[start:stop] = term.sum(axis=(1, 2)) / n_points
    return entropy


def conditional_entropy(
    t: FloatArray,
    y: FloatArray,
    periods: FloatArray,
    *,
    n_phase_bins: int = 10,
    n_mag_bins: int = 10,
    backend: CEBackend = "numpy",
    batch: int = DEFAULT_BATCH,
) -> FloatArray:
    """Conditional entropy for each trial period (minimized at the true period).

    Parameters
    ----------
    t, y : numpy.ndarray
        Finite times (days) and values of one band.
    periods : numpy.ndarray
        Trial periods (days).
    n_phase_bins, n_mag_bins : int, default 10
        Histogram resolution in phase and magnitude.
    backend : {"numpy", "cupy"}, default "numpy"
        CPU or GPU.
    batch : int, default 1024
        Trial periods per vectorized batch.

    Returns
    -------
    numpy.ndarray
        Conditional entropy per period.
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    periods_host = np.ascontiguousarray(periods, dtype=np.float64)
    if periods_host.size == 0:
        return np.zeros(0, dtype=np.float64)
    tau = t - t.min()
    span = float(y.max() - y.min())
    if span <= 0.0:
        return np.zeros(periods_host.size, dtype=np.float64)
    mag_bin = np.clip(
        ((y - y.min()) / span * n_mag_bins).astype(np.int64), 0, n_mag_bins - 1
    )

    if backend == "cupy":
        ensure_cuda_dll_path()
        import cupy as cp

        entropy = _entropy_batch(
            cp, cp.asarray(tau), cp.asarray(mag_bin), cp.asarray(periods_host),
            n_phase=n_phase_bins, n_mag=n_mag_bins, batch=batch,
        )
        return np.asarray(cp.asnumpy(entropy), dtype=np.float64)
    if backend != "numpy":
        raise ValueError(f"unknown backend {backend!r}")
    return np.asarray(
        _entropy_batch(
            np, tau, mag_bin, periods_host,
            n_phase=n_phase_bins, n_mag=n_mag_bins, batch=batch,
        ),
        dtype=np.float64,
    )


class ConditionalEntropyMethod(PeriodogramMethod):
    """Conditional-entropy period search (numpy CPU, cupy GPU)."""

    name: ClassVar[str] = "CE"
    objective_sense: ClassVar[Literal["max", "min"]] = "min"
    supports_multiband: ClassVar[bool] = False
    settings_cls: ClassVar[type] = CESettings
    cpu_backend: ClassVar[str] = "numpy"
    gpu_backend: ClassVar[str | None] = "cupy"
    all_backends: ClassVar[tuple[str, ...]] = ("numpy", "cupy")

    def default_grid(self, lc: LightCurve, settings: CESettings) -> GridSpec:  # type: ignore[override]
        finite = lc.finite()
        if finite.baseline <= 0.0:
            raise InsufficientDataError("CE: no usable time baseline")
        minimum = settings.minimum_frequency or 1.0 / finite.baseline
        maximum = settings.maximum_frequency or pseudo_nyquist_frequency(
            finite.time, settings.nyquist_factor
        )
        return uniform_frequency_grid(
            finite.baseline,
            maximum_frequency=maximum,
            minimum_frequency=minimum,
            samples_per_peak=settings.samples_per_peak,
        )

    def power(  # type: ignore[override]
        self,
        grid: GridSpec,
        lc: LightCurve,
        settings: CESettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        finite = lc.finite()
        n = finite.n
        if n < settings.min_detections:
            raise InsufficientDataError(
                f"CE: {n} finite points < min_detections {settings.min_detections}"
            )
        if finite.baseline <= 0.0:
            raise InsufficientDataError("CE: no usable time baseline")
        periods = grid.period
        entropy = conditional_entropy(
            finite.time, finite.value, periods,
            n_phase_bins=settings.n_phase_bins, n_mag_bins=settings.n_mag_bins,
            backend=backend, batch=settings.batch_periods,  # type: ignore[arg-type]
        )
        return Periodogram.from_spectrum(
            method="CE",
            backend=backend,
            frequency=1.0 / periods,
            power=entropy,
            objective_sense="min",
            n_samples=n,
            baseline=finite.baseline,
            meta=finite.meta,
        )

    def estimate_device_bytes(self, n_points: int) -> int:
        return 128 * 1024**2 + n_points * 8 * 8


register(ConditionalEntropyMethod())

__all__ = [
    "CEBackend",
    "ConditionalEntropyMethod",
    "DEFAULT_BATCH",
    "conditional_entropy",
]
