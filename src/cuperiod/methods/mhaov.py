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

from cuperiod.core._arrayapi import (
    array_namespace,
    resolve_precision,
    resolve_torch_device,
    to_device_array,
    to_host,
)
from cuperiod.core._typing import FloatArray
from cuperiod.core.backend import ensure_cuda_dll_path
from cuperiod.core.config import MHAOVSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import (
    GridSpec,
    pseudo_nyquist_frequency,
    uniform_frequency_grid,
)
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
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
    design = xp.empty((n_freq, n_points, d), dtype=angle.dtype)
    design[:, :, 0] = 1.0
    for k in range(1, n_harmonics + 1):
        design[:, :, 2 * k - 1] = xp.cos(k * angle)
        design[:, :, 2 * k] = xp.sin(k * angle)
    return design


def _model_ss_batch(
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
    """Regression sum of squares (about the mean) per trial frequency.

    This is the projection norm of the data onto the ``2H+1`` harmonic basis; the AOV
    F-statistic (single- or multi-band) is formed from it by the callers.
    """
    d = 2 * n_harmonics + 1
    n_freq = int(frequencies.shape[0])
    fdtype = frequencies.dtype
    eye = xp.eye(d, dtype=fdtype) * _RIDGE
    out = xp.empty(n_freq, dtype=fdtype)
    two_pi = 2.0 * float(np.pi)

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
        model_ss = xp.sum(beta * proj, axis=1) - n_points * y_mean * y_mean
        out[start:stop] = xp.clip(model_ss, 0.0, total_ss)
    return out


def _compute_model_ss(
    t: FloatArray,
    y: FloatArray,
    frequencies: FloatArray,
    *,
    n_harmonics: int,
    backend: str,
    batch: int,
    precision: str = "auto",
) -> tuple[FloatArray, float, int]:
    """Host-side regression SS per frequency, plus ``(total_ss, n)`` for one band.

    Returns all-zero model SS (and ``total_ss=0``) when the band has too few points or
    no variance, so a caller can skip it.
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    freqs = np.ascontiguousarray(frequencies, dtype=np.float64)
    n = t.size
    if freqs.size == 0 or n <= 2 * n_harmonics + 1:
        return np.zeros(freqs.size, dtype=np.float64), 0.0, n
    tau = t - t.min()
    y_mean = float(y.mean())
    total_ss = float(((y - y_mean) ** 2).sum())
    if total_ss <= 0.0:
        return np.zeros(freqs.size, dtype=np.float64), 0.0, n

    if backend == "cupy":
        ensure_cuda_dll_path()
        import cupy as cp

        out = _model_ss_batch(
            cp, cp.asarray(tau), cp.asarray(y), cp.asarray(freqs),
            n_harmonics=n_harmonics, total_ss=total_ss, y_mean=y_mean,
            n_points=n, batch=batch,
        )
        return np.asarray(cp.asnumpy(out), dtype=np.float64), total_ss, n
    if backend == "torch" or backend.startswith("torch:"):
        import torch

        device = backend.split(":", 1)[1] if ":" in backend else "cpu"
        fdt = (torch.float32
               if resolve_precision(precision, device) == "float32" else torch.float64)
        out = _model_ss_batch(
            array_namespace(to_device_array(freqs, device=device, dtype=fdt)),
            to_device_array(tau, device=device, dtype=fdt),
            to_device_array(y, device=device, dtype=fdt),
            to_device_array(freqs, device=device, dtype=fdt),
            n_harmonics=n_harmonics, total_ss=total_ss, y_mean=y_mean,
            n_points=n, batch=batch,
        )
        return to_host(out), total_ss, n
    if backend != "numpy":
        raise ValueError(f"unknown backend {backend!r}")
    out = _model_ss_batch(
        np, tau, y, freqs,
        n_harmonics=n_harmonics, total_ss=total_ss, y_mean=y_mean,
        n_points=n, batch=batch,
    )
    return np.asarray(out, dtype=np.float64), total_ss, n


def aov_power(
    t: FloatArray,
    y: FloatArray,
    frequencies: FloatArray,
    *,
    n_harmonics: int = 3,
    backend: str = "numpy",
    batch: int = DEFAULT_BATCH,
    precision: str = "auto",
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
    model_ss, total_ss, n = _compute_model_ss(
        t, y, frequencies, n_harmonics=n_harmonics, backend=backend, batch=batch,
        precision=precision,
    )
    if total_ss <= 0.0:
        return model_ss
    dof_model = float(2 * n_harmonics)
    dof_resid = float(n - (2 * n_harmonics + 1))
    resid_ss = np.clip(total_ss - model_ss, 1e-300, None)
    return (model_ss / dof_model) / (resid_ss / dof_resid)


def aov_multiband_power(
    frequencies: FloatArray,
    bands: list[tuple[FloatArray, FloatArray]],
    *,
    n_harmonics: int = 3,
    backend: str = "numpy",
    batch: int = DEFAULT_BATCH,
    precision: str = "auto",
) -> FloatArray:
    """Pooled multiband AOV F-statistic at a shared frequency, per-band amplitudes.

    Each band is fit with its own ``2H+1`` trig-polynomial (independent amplitudes and
    phases) at the *same* trial frequency; the regression and residual sums of squares
    are pooled across bands into one F-statistic with ``B*2H`` model and
    ``N_total - B*(2H+1)`` residual degrees of freedom.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Shared trial frequencies (cycles/day).
    bands : list of (time, value)
        Per-band finite ``(t, y)`` arrays.
    n_harmonics, backend, batch
        As for :func:`aov_power`.

    Returns
    -------
    numpy.ndarray
        Pooled AOV F-statistic per frequency.
    """
    freqs = np.ascontiguousarray(frequencies, dtype=np.float64)
    if freqs.size == 0:
        return np.zeros(0, dtype=np.float64)
    model_sum = np.zeros(freqs.size, dtype=np.float64)
    resid_sum = np.zeros(freqs.size, dtype=np.float64)
    n_total = 0
    n_used_bands = 0
    for t, y in bands:
        model_ss, total_ss, n = _compute_model_ss(
            t, y, freqs, n_harmonics=n_harmonics, backend=backend, batch=batch,
            precision=precision,
        )
        if total_ss <= 0.0:
            continue
        model_sum += model_ss
        resid_sum += total_ss - model_ss
        n_total += n
        n_used_bands += 1
    d = 2 * n_harmonics + 1
    dof_resid = n_total - n_used_bands * d
    if n_used_bands == 0 or dof_resid <= 0:
        return np.zeros(freqs.size, dtype=np.float64)
    dof_model = float(n_used_bands * 2 * n_harmonics)
    resid = np.clip(resid_sum, 1e-300, None)
    return (model_sum / dof_model) / (resid / dof_resid)


class MHAOVMethod(PeriodogramMethod):
    """Multiharmonic Analysis of Variance (numpy CPU, cupy GPU)."""

    name: ClassVar[str] = "MHAOV"
    objective_sense: ClassVar[Literal["max", "min"]] = "max"
    supports_multiband: ClassVar[bool] = True
    settings_cls: ClassVar[type] = MHAOVSettings
    cpu_backend: ClassVar[str] = "numpy"
    gpu_backend: ClassVar[str | None] = "cupy"
    portable_gpu_backend: ClassVar[str | None] = "torch"
    all_backends: ClassVar[tuple[str, ...]] = ("numpy", "cupy", "torch")

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
        if backend == "torch" or backend.startswith("torch:"):
            backend = f"torch:{resolve_torch_device(backend, settings.device)}"
        power = aov_power(
            finite.time, finite.value, frequency,
            n_harmonics=settings.n_harmonics,
            backend=backend, batch=settings.batch_periods,
            precision=settings.precision,
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

    def multiband_power(  # type: ignore[override]
        self,
        grid: GridSpec,
        mblc: MultiBandLightCurve,
        settings: MHAOVSettings,
        backend: str,
    ) -> Periodogram:
        from cuperiod.multiband.mhaov_mb import mhaov_multiband_power

        if backend == "torch" or backend.startswith("torch:"):
            backend = f"torch:{resolve_torch_device(backend, settings.device)}"
        return mhaov_multiband_power(grid, mblc, settings, backend)

    def estimate_device_bytes(self, n_points: int) -> int:
        return 128 * 1024**2 + n_points * 8 * 12


register(MHAOVMethod())

__all__ = [
    "DEFAULT_BATCH",
    "MHAOVBackend",
    "MHAOVMethod",
    "aov_multiband_power",
    "aov_power",
]
