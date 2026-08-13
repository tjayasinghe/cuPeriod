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
    device_ref,
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

#: Trial frequencies per vectorized batch on host backends when ``batch`` is auto (0).
DEFAULT_BATCH: Final = 512

#: Transient byte budgets for auto-sized frequency chunks (``batch <= 0``): the chunk
#: adapts to the light-curve length, so long curves cannot blow memory. Device
#: backends get a far larger budget — their single-shot latency is dominated by
#: per-chunk dispatch (kernel launches plus the batched-solve sync), so fewer,
#: larger chunks are strictly faster at identical results.
_CHUNK_BYTES: Final = 1 << 27
_DEVICE_CHUNK_BYTES: Final = 1 << 29

#: Rough number of (chunk, N)-sized float64 workspaces alive at once in
#: :func:`_model_ss_batch` (angle, cos/sin pairs, Chebyshev recurrence temps, the
#: weighted product), used to convert the byte budgets into a chunk length.
_WORKSPACES: Final = 12


def _resolve_batch(batch: int, n_points: int, on_device: bool) -> int:
    """Effective frequency-chunk length: ``batch`` verbatim, or auto when ``<= 0``."""
    if batch > 0:
        return batch
    budget = _DEVICE_CHUNK_BYTES if on_device else _CHUNK_BYTES
    auto = max(1, budget // max(1, n_points * 8 * _WORKSPACES))
    return auto if on_device else min(DEFAULT_BATCH, auto)

#: Diagonal ridge that keeps the harmonic normal equations solvable at degenerate
#: frequencies (f→0, where the cosine columns collapse onto the constant column). It is
#: applied as ``_RIDGE_EPS · eps(dtype) · n_points``: scaling by the working precision's
#: machine epsilon and the Gram diagonal magnitude (≈ ``n_points``, from the all-ones
#: constant column) makes it representable in float32. A fixed absolute 1e-10 underflows
#: against the ~N-sized diagonal on a float32 device (1e-10 ≪ eps_f32·N), leaving the
#: matrix singular so ``linalg.solve`` raises. In float64 this reproduces the previous
#: ~1e-10 ridge to within rounding, so float64 results are unchanged.
_RIDGE_EPS: Final = 1.0e3


def _harmonic_sums(
    xp: ModuleType, angle: Any, y: Any, n_harmonics: int
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Per-frequency harmonic trig sums from one ``(F, N)`` angle matrix.

    Returns ``(C, S, A, B)`` with ``C[m] = Σ_n cos(m·θ)``, ``S[m] = Σ_n sin(m·θ)`` for
    ``m = 1..2H`` (index 0 unused: those sums are the constants ``N`` and ``0``) and
    ``A[k] = Σ_n y_n cos(k·θ)``, ``B[k] = Σ_n y_n sin(k·θ)`` for ``k = 1..H``. The
    higher harmonics come from the Chebyshev recurrence ``T_m = 2·cosθ·T_{m-1} −
    T_{m-2}``, so ``cos``/``sin`` are evaluated **once** regardless of ``H``.
    """
    m_max = 2 * n_harmonics
    c1 = xp.cos(angle)
    s1 = xp.sin(angle)
    C: list[Any] = [None] * (m_max + 1)
    S: list[Any] = [None] * (m_max + 1)
    A: list[Any] = [None] * (n_harmonics + 1)
    B: list[Any] = [None] * (n_harmonics + 1)
    two_c1 = 2.0 * c1
    cm, sm = c1, s1
    prev_c: Any = None
    prev_s: Any = None
    for m in range(1, m_max + 1):
        C[m] = xp.sum(cm, axis=1)
        S[m] = xp.sum(sm, axis=1)
        if m <= n_harmonics:
            A[m] = xp.sum(y[None, :] * cm, axis=1)
            B[m] = xp.sum(y[None, :] * sm, axis=1)
        if m < m_max:
            if m == 1:  # T_0 is the constant 1 / 0, kept out of the arrays
                next_c = two_c1 * cm - 1.0
                next_s = two_c1 * sm
            else:
                next_c = two_c1 * cm - prev_c
                next_s = two_c1 * sm - prev_s
            prev_c, prev_s = cm, sm
            cm, sm = next_c, next_s
    return C, S, A, B


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

    Every entry of the normal equations is analytically a harmonic trig sum — by the
    product-to-sum identities, ``Σ cos(kθ)cos(lθ) = ½(C_{|k−l|} + C_{k+l})`` and so on
    — so instead of materializing the ``(F, N, 2H+1)`` design tensor and forming its
    ``O(F·N·d²)`` Gram, the sums ``C_m``/``S_m`` (``m ≤ 2H``) are computed in
    ``O(F·N·H)`` from one ``cos``/``sin`` evaluation (:func:`_harmonic_sums`) and the
    tiny ``(F, d, d)`` Gram is assembled from them. Mathematically identical (same
    normal equations, same ridge), gemm-free on every backend, and ``O(d)`` less
    transient memory.
    """
    d = 2 * n_harmonics + 1
    n_freq = int(frequencies.shape[0])
    fdtype = frequencies.dtype
    dev = device_ref(frequencies)
    ridge = _RIDGE_EPS * float(xp.finfo(fdtype).eps) * float(n_points)
    out = xp.empty(n_freq, dtype=fdtype, device=dev)
    two_pi = 2.0 * float(np.pi)
    sum_y = float(xp.sum(y))
    h = n_harmonics

    for start in range(0, n_freq, batch):
        stop = min(start + batch, n_freq)
        fb = frequencies[start:stop]
        n_f = int(fb.shape[0])
        angle = (two_pi * fb)[:, None] * tau[None, :]
        C, S, A, B = _harmonic_sums(xp, angle, y, h)
        del angle

        gram = xp.zeros((n_f, d, d), dtype=fdtype, device=dev)
        proj = xp.empty((n_f, d), dtype=fdtype, device=dev)
        gram[:, 0, 0] = float(n_points) + ridge
        proj[:, 0] = sum_y
        for k in range(1, h + 1):
            gram[:, 0, 2 * k - 1] = C[k]
            gram[:, 2 * k - 1, 0] = C[k]
            gram[:, 0, 2 * k] = S[k]
            gram[:, 2 * k, 0] = S[k]
            proj[:, 2 * k - 1] = A[k]
            proj[:, 2 * k] = B[k]
            for line in range(1, h + 1):
                # Σ cos·cos, Σ sin·sin, Σ cos·sin over n from the m-sums; C_0 = N,
                # S_0 = 0, S_{-m} = -S_m.
                m_diff = abs(k - line)
                c_diff = C[m_diff] if m_diff else float(n_points)
                cc = 0.5 * (c_diff + C[k + line])
                ss = 0.5 * (c_diff - C[k + line])
                if k == line:
                    s_diff = 0.0
                elif k > line:
                    s_diff = S[m_diff]
                else:
                    s_diff = -S[m_diff]
                cs = 0.5 * (S[k + line] - s_diff)
                gram[:, 2 * k - 1, 2 * line - 1] = cc + (ridge if k == line else 0.0)
                gram[:, 2 * k, 2 * line] = ss + (ridge if k == line else 0.0)
                gram[:, 2 * k - 1, 2 * line] = cs
                gram[:, 2 * line, 2 * k - 1] = cs
        # numpy 2.x batched solve treats a 2-D RHS as matrices, so add a trailing
        # singleton to keep it a per-frequency vector solve.
        beta = xp.linalg.solve(gram, proj[..., None])[..., 0]
        model_ss = xp.sum(beta * proj, axis=1) - n_points * y_mean * y_mean
        out[start:stop] = xp.clip(model_ss, 0.0, total_ss)
    return out


# --- CPU fast path: numba-parallel, one loop-iteration per trial frequency ----

_NUMBA_MHAOV_KERNEL: Any = None


def _numba_mhaov_kernel() -> Any:
    """Lazily compile (once) and cache the numba MHAOV kernel.

    The same harmonic-trig-sum normal equations as :func:`_model_ss_batch`, with the
    Chebyshev recurrence run per point in scalar registers — no ``(F, N)`` transients
    at all — and the tiny ``d×d`` solve done per frequency. ``prange`` over the
    frequency grid.
    """
    global _NUMBA_MHAOV_KERNEL
    if _NUMBA_MHAOV_KERNEL is not None:
        return _NUMBA_MHAOV_KERNEL
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=False)  # pragma: no cover - njit
    def _kernel(tau, y, freqs, n_harmonics, ridge, sum_y, y_mean, total_ss):  # type: ignore[no-untyped-def]
        n_freq = freqs.shape[0]
        n_points = tau.shape[0]
        h = n_harmonics
        m_max = 2 * h
        d = 2 * h + 1
        two_pi = 2.0 * np.pi
        out = np.empty(n_freq)
        for fi in prange(n_freq):
            f = freqs[fi]
            c_sums = np.zeros(m_max + 1)
            s_sums = np.zeros(m_max + 1)
            a_sums = np.zeros(h + 1)
            b_sums = np.zeros(h + 1)
            for j in range(n_points):
                theta = two_pi * f * tau[j]
                c1 = np.cos(theta)
                s1 = np.sin(theta)
                two_c1 = 2.0 * c1
                cm = c1
                sm = s1
                c_prev = 1.0
                s_prev = 0.0
                yj = y[j]
                for m in range(1, m_max + 1):
                    c_sums[m] += cm
                    s_sums[m] += sm
                    if m <= h:
                        a_sums[m] += yj * cm
                        b_sums[m] += yj * sm
                    c_next = two_c1 * cm - c_prev
                    s_next = two_c1 * sm - s_prev
                    c_prev = cm
                    s_prev = sm
                    cm = c_next
                    sm = s_next
            gram = np.zeros((d, d))
            proj = np.zeros(d)
            gram[0, 0] = n_points + ridge
            proj[0] = sum_y
            for k in range(1, h + 1):
                gram[0, 2 * k - 1] = c_sums[k]
                gram[2 * k - 1, 0] = c_sums[k]
                gram[0, 2 * k] = s_sums[k]
                gram[2 * k, 0] = s_sums[k]
                proj[2 * k - 1] = a_sums[k]
                proj[2 * k] = b_sums[k]
                for line in range(1, h + 1):
                    m_diff = k - line if k >= line else line - k
                    c_diff = c_sums[m_diff] if m_diff else float(n_points)
                    cc = 0.5 * (c_diff + c_sums[k + line])
                    ss = 0.5 * (c_diff - c_sums[k + line])
                    if k == line:
                        s_diff = 0.0
                    elif k > line:
                        s_diff = s_sums[m_diff]
                    else:
                        s_diff = -s_sums[m_diff]
                    cs = 0.5 * (s_sums[k + line] - s_diff)
                    extra = ridge if k == line else 0.0
                    gram[2 * k - 1, 2 * line - 1] = cc + extra
                    gram[2 * k, 2 * line] = ss + extra
                    gram[2 * k - 1, 2 * line] = cs
                    gram[2 * line, 2 * k - 1] = cs
            beta = np.linalg.solve(gram, proj)
            model_ss = 0.0
            for i in range(d):
                model_ss += beta[i] * proj[i]
            model_ss -= n_points * y_mean * y_mean
            if model_ss < 0.0:
                model_ss = 0.0
            elif model_ss > total_ss:
                model_ss = total_ss
            out[fi] = model_ss
        return out

    _NUMBA_MHAOV_KERNEL = _kernel
    return _kernel


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

    if backend == "numba":
        kernel = _numba_mhaov_kernel()
        ridge = _RIDGE_EPS * float(np.finfo(np.float64).eps) * float(n)
        out = kernel(
            tau, y, freqs, n_harmonics, ridge, float(y.sum()), y_mean, total_ss
        )
        return np.asarray(out, dtype=np.float64), total_ss, n
    if backend == "cupy":
        ensure_cuda_dll_path()
        import cupy as cp

        # Route through the array-API namespace (not raw cupy) so device-matched
        # creation (``xp.zeros(..., device=)``) works — raw cupy rejects ``device=``.
        freqs_cp = cp.asarray(freqs)
        out = _model_ss_batch(
            array_namespace(freqs_cp), cp.asarray(tau), cp.asarray(y), freqs_cp,
            n_harmonics=n_harmonics, total_ss=total_ss, y_mean=y_mean,
            n_points=n, batch=_resolve_batch(batch, n, on_device=True),
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
            n_points=n, batch=_resolve_batch(batch, n, on_device=device != "cpu"),
        )
        return to_host(out), total_ss, n
    if backend != "numpy":
        raise ValueError(f"unknown backend {backend!r}")
    out = _model_ss_batch(
        np, tau, y, freqs,
        n_harmonics=n_harmonics, total_ss=total_ss, y_mean=y_mean,
        n_points=n, batch=_resolve_batch(batch, n, on_device=False),
    )
    return np.asarray(out, dtype=np.float64), total_ss, n


def aov_power(
    t: FloatArray,
    y: FloatArray,
    frequencies: FloatArray,
    *,
    n_harmonics: int = 3,
    backend: str = "numpy",
    batch: int = 0,
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
    batch : int, default 0
        Trial frequencies per vectorized batch; 0 auto-sizes the batch from a
        transient-memory budget (much larger on device backends, whose
        single-shot latency is per-batch dispatch overhead, not arithmetic).
        The batch does not affect the result.

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
    batch: int = 0,
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
    """Multiharmonic Analysis of Variance (numba/numpy CPU, cupy GPU)."""

    name: ClassVar[str] = "MHAOV"
    objective_sense: ClassVar[Literal["max", "min"]] = "max"
    supports_multiband: ClassVar[bool] = True
    settings_cls: ClassVar[type] = MHAOVSettings
    cpu_backend: ClassVar[str] = "numpy"
    fast_cpu_backend: ClassVar[str | None] = "numba"
    gpu_backend: ClassVar[str | None] = "cupy"
    portable_gpu_backend: ClassVar[str | None] = "torch"
    all_backends: ClassVar[tuple[str, ...]] = ("numba", "numpy", "cupy", "torch")

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
        engine: object | None = None,
    ) -> Periodogram:
        from cuperiod.multiband.mhaov_mb import mhaov_multiband_power

        if backend == "torch" or backend.startswith("torch:"):
            backend = f"torch:{resolve_torch_device(backend, settings.device)}"
        return mhaov_multiband_power(grid, mblc, settings, backend)

    def estimate_device_bytes(self, n_points: int) -> int:
        return _DEVICE_CHUNK_BYTES + 64 * 1024**2 + n_points * 8 * 12


register(MHAOVMethod())

__all__ = [
    "DEFAULT_BATCH",
    "MHAOVBackend",
    "MHAOVMethod",
    "aov_multiband_power",
    "aov_power",
]
