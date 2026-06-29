"""Multiharmonic Analysis of Variance (Schwarzenberg-Czerny 1996).

MHAOV fits a trigonometric polynomial of order ``H`` (a truncated Fourier series with
``2H+1`` terms) to the phased data at each trial frequency and forms the analysis-of-
variance F-statistic

    AOV = (model_SS / 2H) / (residual_SS / (N - 2H - 1)),

which is *maximized* at the true frequency. Several harmonics make it far more sensitive
than single-harmonic Lomb-Scargle to sharply non-sinusoidal signals (RR Lyrae, eclipsing
binaries, cepheids).

The model sum of squares is the squared norm of the projection of the data onto the
harmonic basis. Schwarzenberg-Czerny computes that projection with orthogonal
trigonometric polynomials for numerical stability; this implementation evaluates the
mathematically identical least-squares projection, vectorized over the frequency grid,
so the same code runs on numpy (CPU) and cupy (GPU). Errors are not used (AOV is an
unweighted variance analysis).
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, ClassVar, Final, Literal

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.backend import ensure_cuda_dll_path
from cuperiod.core.config import MHAOVSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import (
    GridSpec,
    pseudo_nyquist_frequency,
    uniform_frequency_grid,
)
from cuperiod.core.lightcurve import LightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod, register

MHAOVBackend = Literal["numpy", "cupy"]

#: Trial frequencies per vectorized batch (bounds the (F, N, 2H+1) design tensor).
DEFAULT_BATCH: Final = 512

#: Diagonal ridge to keep the normal equations solvable at degenerate frequencies.
_RIDGE: Final = 1e-10


def _design(xp: ModuleType, angle: Any, n_harmonics: int) -> Any:
    """Trig-polynomial design tensor ``(F, N, 2H+1)`` = [1, cos kθ, sin kθ]."""
    n_freq, n_points = angle.shape
    d = 2 * n_harmonics + 1
    design = xp.empty((n_freq, n_points, d), dtype=np.float64)
    design[:, :, 0] = 1.0
    for k in range(1, n_harmonics + 1):
        design[:, :, 2 * k - 1] = xp.cos(k * angle)
        design[:, :, 2 * k] = xp.sin(k * angle)
    return design


def _aov_batch(
    xp: ModuleType,
    tau: Any,
    y: Any,
    frequencies: Any,
    *,
    n_harmonics: int,
    total_ss: float,
    y_mean: float,
    n_points: int,
    batch: int,
) -> Any:
    """AOV F-statistic for each trial frequency, vectorized over frequencies."""
    d = 2 * n_harmonics + 1
    dof_model = float(2 * n_harmonics)
    dof_resid = float(n_points - d)
    n_freq = int(frequencies.shape[0])
    eye = xp.eye(d, dtype=np.float64) * _RIDGE
    out = xp.empty(n_freq, dtype=np.float64)
    two_pi = 2.0 * np.pi

    for start in range(0, n_freq, batch):
        stop = min(start + batch, n_freq)
        fb = frequencies[start:stop]
        angle = (two_pi * fb)[:, None] * tau[None, :]
        design = _design(xp, angle, n_harmonics)
        gram = xp.einsum("fni,fnj->fij", design, design) + eye
        proj = xp.einsum("fni,n->fi", design, y)
        # numpy 2.x batched solve treats a 2-D RHS as matrices, so add a trailing
        # singleton to keep it a per-frequency vector solve.
        beta = xp.linalg.solve(gram, proj[..., None])[..., 0]
        model_ss = (beta * proj).sum(axis=1) - n_points * y_mean * y_mean
        model_ss = xp.clip(model_ss, 0.0, total_ss)
        resid_ss = xp.clip(total_ss - model_ss, 1e-300, None)
        out[start:stop] = (model_ss / dof_model) / (resid_ss / dof_resid)
    return out


def aov_power(
    t: FloatArray,
    y: FloatArray,
    frequencies: FloatArray,
    *,
    n_harmonics: int = 3,
    backend: MHAOVBackend = "numpy",
    batch: int = DEFAULT_BATCH,
) -> FloatArray:
    """Multiharmonic AOV statistic for each trial frequency.

    Parameters
    ----------
    t, y : numpy.ndarray
        Finite times (days) and values of one band.
    frequencies : numpy.ndarray
        Trial frequencies (cycles/day).
    n_harmonics : int, default 3
        Harmonic order ``H`` (model has ``2H+1`` terms).
    backend : {"numpy", "cupy"}, default "numpy"
        CPU or GPU.
    batch : int, default 512
        Trial frequencies per vectorized batch.

    Returns
    -------
    numpy.ndarray
        AOV F-statistic per frequency (maximized at the true frequency). All-zero for a
        constant signal or when there are too few points.
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    freqs = np.ascontiguousarray(frequencies, dtype=np.float64)
    if freqs.size == 0:
        return np.zeros(0, dtype=np.float64)
    n = t.size
    if n <= 2 * n_harmonics + 1:
        return np.zeros(freqs.size, dtype=np.float64)
    tau = t - t.min()
    y_mean = float(y.mean())
    total_ss = float(((y - y_mean) ** 2).sum())
    if total_ss <= 0.0:
        return np.zeros(freqs.size, dtype=np.float64)

    if backend == "cupy":
        ensure_cuda_dll_path()
        import cupy as cp

        out = _aov_batch(
            cp, cp.asarray(tau), cp.asarray(y), cp.asarray(freqs),
            n_harmonics=n_harmonics, total_ss=total_ss, y_mean=y_mean,
            n_points=n, batch=batch,
        )
        return np.asarray(cp.asnumpy(out), dtype=np.float64)
    if backend != "numpy":
        raise ValueError(f"unknown backend {backend!r}")
    out = _aov_batch(
        np, tau, y, freqs,
        n_harmonics=n_harmonics, total_ss=total_ss, y_mean=y_mean,
        n_points=n, batch=batch,
    )
    return np.asarray(out, dtype=np.float64)


class MHAOVMethod(PeriodogramMethod):
    """Multiharmonic Analysis of Variance (numpy CPU, cupy GPU)."""

    name: ClassVar[str] = "MHAOV"
    objective_sense: ClassVar[Literal["max", "min"]] = "max"
    supports_multiband: ClassVar[bool] = False
    settings_cls: ClassVar[type] = MHAOVSettings
    cpu_backend: ClassVar[str] = "numpy"
    gpu_backend: ClassVar[str | None] = "cupy"
    all_backends: ClassVar[tuple[str, ...]] = ("numpy", "cupy")

    def default_grid(self, lc: LightCurve, settings: MHAOVSettings) -> GridSpec:  # type: ignore[override]
        finite = lc.finite()
        if finite.baseline <= 0.0:
            raise InsufficientDataError("MHAOV: no usable time baseline")
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
        settings: MHAOVSettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        finite = lc.finite()
        n = finite.n
        if n < settings.min_detections or n <= 2 * settings.n_harmonics + 1:
            raise InsufficientDataError(
                f"MHAOV: {n} finite points are too few for {settings.n_harmonics} "
                "harmonics"
            )
        if finite.baseline <= 0.0:
            raise InsufficientDataError("MHAOV: no usable time baseline")
        frequency = grid.frequency
        power = aov_power(
            finite.time, finite.value, frequency,
            n_harmonics=settings.n_harmonics,
            backend=backend, batch=settings.batch_periods,  # type: ignore[arg-type]
        )
        return Periodogram.from_spectrum(
            method="MHAOV",
            backend=backend,
            frequency=frequency,
            power=power,
            objective_sense="max",
            n_samples=n,
            baseline=finite.baseline,
            meta=finite.meta,
        )

    def estimate_device_bytes(self, n_points: int) -> int:
        return 128 * 1024**2 + n_points * 8 * 12


register(MHAOVMethod())

__all__ = ["DEFAULT_BATCH", "MHAOVBackend", "MHAOVMethod", "aov_power"]
