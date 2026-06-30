"""String-length period search (Lafler-Kinman 1965 / Dworetsky 1983).

The cheapest useful period statistic: fold the data, sort by phase, and measure the
total length of the "string" joining consecutive points (with wrap-around). At the true
period a smooth folded light curve gives a short string, so this is a *minimization*
method — excellent for eclipsing and eccentric shapes and nearly free to evaluate.

Magnitudes are scaled to a span of 0.5 (Dworetsky's normalization) so the magnitude
and phase axes contribute comparably to the Euclidean step length. One vectorized
kernel runs on numpy (CPU) and cupy (GPU).
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, ClassVar, Final, Literal

import numpy as np

from cuperiod.core._arrayapi import (
    array_namespace,
    resolve_precision,
    resolve_torch_device,
    to_device_array,
    to_host,
)
from cuperiod.core._typing import FloatArray
from cuperiod.core.backend import ensure_cuda_dll_path
from cuperiod.core.config import StringLengthSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import (
    GridSpec,
    pseudo_nyquist_frequency,
    uniform_frequency_grid,
)
from cuperiod.core.lightcurve import LightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod, register

SLBackend = Literal["numpy", "cupy"]

#: Trial periods per vectorized batch (bounds the (P, N) sort/gather arrays).
DEFAULT_BATCH: Final = 1024


def _length_batch(
    xp: ModuleType, tau: Any, m_scaled: Any, periods: Any, *, batch: int
) -> Any:
    """Total folded string length for each trial period, vectorized over periods.

    Array-API generic (numpy/cupy/torch). Uses only fancy indexing and slicing for the
    per-row sort/gather and the consecutive differences, avoiding ``take_along_axis`` /
    ``diff`` (not in every namespace); float dtype follows ``periods``.
    """
    idtype = xp.int64
    n_periods = int(periods.shape[0])
    length = xp.empty(n_periods, dtype=periods.dtype)
    for start in range(0, n_periods, batch):
        stop = min(start + batch, n_periods)
        pb = periods[start:stop]
        n_p = int(pb.shape[0])
        phase = xp.remainder(tau[None, :] / pb[:, None], 1.0)  # (P, N)
        order = xp.argsort(phase, axis=1)
        rows = xp.arange(n_p, dtype=idtype)[:, None]
        ph = phase[rows, order]            # phase sorted per row
        mm = m_scaled[order]               # magnitudes gathered in the same order
        dphi = ph[:, 1:] - ph[:, :-1]
        dmag = mm[:, 1:] - mm[:, :-1]
        total = xp.sum(xp.sqrt(dphi * dphi + dmag * dmag), axis=1)
        wrap_phi = (ph[:, 0] + 1.0) - ph[:, -1]
        wrap_mag = mm[:, 0] - mm[:, -1]
        total = total + xp.sqrt(wrap_phi * wrap_phi + wrap_mag * wrap_mag)
        length[start:stop] = total
    return length


def string_length(
    t: FloatArray,
    y: FloatArray,
    periods: FloatArray,
    *,
    backend: str = "numpy",
    batch: int = DEFAULT_BATCH,
    precision: str = "auto",
) -> FloatArray:
    """String length for each trial period (minimized at the true period).

    Parameters
    ----------
    t, y : numpy.ndarray
        Finite times (days) and values of one band.
    periods : numpy.ndarray
        Trial periods (days).
    backend : {"numpy", "cupy"}, default "numpy"
        CPU or GPU.
    batch : int, default 1024
        Trial periods per vectorized batch.

    Returns
    -------
    numpy.ndarray
        String length per period.
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    periods_host = np.ascontiguousarray(periods, dtype=np.float64)
    if periods_host.size == 0:
        return np.zeros(0, dtype=np.float64)
    tau = t - t.min()
    span = float(y.max() - y.min())
    m_scaled = (y - y.min()) / span * 0.5 - 0.25 if span > 0.0 else np.zeros_like(y)

    if backend == "cupy":
        ensure_cuda_dll_path()
        import cupy as cp

        length = _length_batch(
            cp, cp.asarray(tau), cp.asarray(m_scaled), cp.asarray(periods_host),
            batch=batch,
        )
        return np.asarray(cp.asnumpy(length), dtype=np.float64)
    if backend == "torch" or backend.startswith("torch:"):
        import torch

        device = backend.split(":", 1)[1] if ":" in backend else "cpu"
        fdt = (torch.float32
               if resolve_precision(precision, device) == "float32" else torch.float64)
        tau_d = to_device_array(tau, device=device, dtype=fdt)
        m_d = to_device_array(m_scaled, device=device, dtype=fdt)
        per_d = to_device_array(periods_host, device=device, dtype=fdt)
        return to_host(
            _length_batch(array_namespace(per_d), tau_d, m_d, per_d, batch=batch)
        )
    if backend != "numpy":
        raise ValueError(f"unknown backend {backend!r}")
    return np.asarray(
        _length_batch(np, tau, m_scaled, periods_host, batch=batch), dtype=np.float64
    )


class StringLengthMethod(PeriodogramMethod):
    """String-length period search (numpy CPU, cupy GPU)."""

    name: ClassVar[str] = "STRINGLENGTH"
    objective_sense: ClassVar[Literal["max", "min"]] = "min"
    supports_multiband: ClassVar[bool] = False
    settings_cls: ClassVar[type] = StringLengthSettings
    cpu_backend: ClassVar[str] = "numpy"
    gpu_backend: ClassVar[str | None] = "cupy"
    portable_gpu_backend: ClassVar[str | None] = "torch"
    all_backends: ClassVar[tuple[str, ...]] = ("numpy", "cupy", "torch")

    def default_grid(self, lc: LightCurve, settings: StringLengthSettings) -> GridSpec:  # type: ignore[override]
        finite = lc.finite()
        if finite.baseline <= 0.0:
            raise InsufficientDataError("string-length: no usable time baseline")
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
        settings: StringLengthSettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        finite = lc.finite()
        n = finite.n
        if n < settings.min_detections:
            raise InsufficientDataError(
                f"string-length: {n} finite points < min_detections "
                f"{settings.min_detections}"
            )
        if finite.baseline <= 0.0:
            raise InsufficientDataError("string-length: no usable time baseline")
        periods = grid.period
        if backend == "torch" or backend.startswith("torch:"):
            backend = f"torch:{resolve_torch_device(backend, settings.device)}"
        length = string_length(
            finite.time, finite.value, periods,
            backend=backend, batch=settings.batch_periods,
            precision=settings.precision,
        )
        return Periodogram.from_spectrum(
            method="STRINGLENGTH",
            backend=backend,
            frequency=1.0 / periods,
            power=length,
            objective_sense="min",
            n_samples=n,
            baseline=finite.baseline,
            meta=finite.meta,
        )

    def estimate_device_bytes(self, n_points: int) -> int:
        return 128 * 1024**2 + n_points * 8 * 8


register(StringLengthMethod())

__all__ = ["DEFAULT_BATCH", "SLBackend", "StringLengthMethod", "string_length"]
