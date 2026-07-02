"""Generalized Lomb-Scargle (GLS) via non-uniform FFT.

GLS is the workhorse for most periodic variable stars. cuPeriod computes the
generalized (floating-mean) Lomb-Scargle power of Zechmeister & Kürster (2009) — the
same statistic as astropy's ``LombScargle(..., fit_mean=True).power(normalization=
"standard")`` — but evaluates the trigonometric sums with a type-1 NUFFT instead of
extirpolation. For a uniform frequency grid ``f_k = f0 + k*df``::

    S(f_k) = sum_j  c_j * exp(2j*pi * f_k * tau_j)

is one type-1 transform (nonuniform ``tau_j`` → uniform modes ``k``). The GLS needs
three per band: the weights ``w`` and weighted data ``w*y`` on the base grid, and
``w`` on the doubled grid for the cos²/sin²/cos·sin terms. finufft (CPU) and cufinufft
(GPU) share one assembly and match astropy to ~1e-9 in power.

Backends
--------
finufft
    CPU; cross-platform wheels. The portable accelerator and the fallback when no GPU
    is present.
cufinufft
    GPU; the full-catalog path. :class:`CufinufftGLS` reuses cufinufft plans across
    light curves for batch throughput.
astropy
    The reference ``LombScargle`` implementation, available as an explicit backend and
    used for false-alarm probabilities and custom (non-uniform) grids.
"""

from __future__ import annotations

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
from cuperiod.core.backend import array_module, ensure_cuda_dll_path
from cuperiod.core.config import GLSSettings
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.peaks import local_maxima
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod, register

NufftBackend = Literal["finufft", "cufinufft"]

#: NUFFT relative tolerance: keeps GLS power within ~1e-9 of the exact direct sum.
DEFAULT_EPS: Final = 1e-9


def _trig_sums(
    tau: Any,
    strengths: Any,
    f0: float,
    df: float,
    nf: int,
    backend: NufftBackend,
    eps: float,
) -> Any:
    """``out[i, k] = sum_j strengths[i, j] exp(2j*pi*(f0 + k*df)*tau_j)`` via NUFFT.

    ``strengths`` is a ``(n_trans, N)`` stack: transforms that share the same
    nonuniform points (the base-grid pair ``w`` and ``w*y``) go through **one**
    batched NUFFT call, sharing the point sort/spread setup. finufft's default mode
    ordering puts mode ``m = k - nf//2`` at output index ``k``, so the strengths are
    modulated by ``exp(2j*pi*f_center*tau)`` with ``f_center = f0 + (nf//2)*df`` and
    the times scaled to ``x = 2*pi*df*tau`` (wrapped to ``[-pi, pi)``); output index
    ``k`` then lands on frequency ``f0 + k*df``. For ``backend="cufinufft"`` the
    inputs must already be cupy arrays and the ``(n_trans, nf)`` output **stays on
    device** so the caller can assemble the power there.
    """
    if backend == "finufft":
        import finufft

        xp: Any = np
        nufft1d1 = finufft.nufft1d1
    elif backend == "cufinufft":
        ensure_cuda_dll_path()
        import cufinufft
        import cupy as cp

        xp = cp
        nufft1d1 = cufinufft.nufft1d1
    else:
        raise ValueError(f"unknown NUFFT backend {backend!r}")

    f_center = f0 + (nf // 2) * df
    x = (2.0 * np.pi * df) * tau
    x = xp.mod(x + np.pi, 2.0 * np.pi) - np.pi
    c = (strengths * xp.exp(2j * np.pi * f_center * tau)[None, :]).astype(np.complex128)
    out = nufft1d1(x, c, nf, eps=eps, isign=1)
    return out if out.ndim == 2 else out[None, :]


def _assemble_power_parts(
    xp: Any,
    c: Any, s: Any, yc: Any, ys: Any, c2: Any, s2: Any,
    y_mean: float, yy: float, fit_mean: bool,
) -> Any:
    """Zechmeister-Kürster generalized-LS power from the real trig-sum components.

    ``c``/``s`` are the cos/sin sums of the weights, ``yc``/``ys`` of the weighted data,
    and ``c2``/``s2`` the doubled-frequency cos/sin sums of the weights. Uses only
    array-API-standard ops, so it runs unchanged on numpy, cupy, and torch (any device)
    — no complex dtype is needed, so the Apple-MPS path (no complex128) works.
    Degenerate frequencies map to 0.
    """
    cc = 0.5 * (1.0 + c2)
    ss = 0.5 * (1.0 - c2)
    cs = 0.5 * s2
    yc = yc - y_mean * c
    ys = ys - y_mean * s
    if fit_mean:
        cc = cc - c * c
        ss = ss - s * s
        cs = cs - c * s

    denom = cc * ss - cs * cs
    power = (ss * yc * yc + cc * ys * ys - 2.0 * cs * yc * ys) / (yy * denom)
    return xp.where(xp.isfinite(power), power, xp.zeros_like(power))


def _assemble_power(
    sw: Any, swy: Any, sw2: Any, y_mean: float, yy: float, fit_mean: bool
) -> Any:
    """GLS power from three *complex* trig sums (the NUFFT path).

    Splits each complex sum into its real/imaginary parts and delegates the Zechmeister-
    Kürster math to :func:`_assemble_power_parts`. Works on numpy or cupy arrays (GPU
    path keeps everything on device); ``y_mean``/``yy`` are host scalars.
    """
    xp = array_module(sw)
    parts = (sw.real, sw.imag, swy.real, swy.imag, sw2.real, sw2.imag)
    if xp is np:
        with np.errstate(divide="ignore", invalid="ignore"):
            return _assemble_power_parts(xp, *parts, y_mean, yy, fit_mean)
    return _assemble_power_parts(xp, *parts, y_mean, yy, fit_mean)


def _prep(
    t: FloatArray, y: FloatArray, dy: FloatArray | None
) -> tuple[FloatArray, FloatArray, FloatArray, float, float]:
    """Shared host prep: time origin shift, normalized weights, weighted mean/variance.

    Returns ``(tau, w, y, y_mean, yy)`` with ``tau = t - t.min()`` and ``w`` summing to
    1 (inverse-variance, or uniform when ``dy is None``).
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    tau = t - t.min()
    if dy is None:
        w = np.full(t.shape, 1.0 / t.size, dtype=np.float64)
    else:
        dy = np.ascontiguousarray(dy, dtype=np.float64)
        w = 1.0 / (dy * dy)
        w /= w.sum()
    y_mean = float(np.dot(w, y))
    yy = float(np.dot(w, y * y)) - y_mean * y_mean
    return tau, w, y, y_mean, yy


def lombscargle_power(
    t: FloatArray,
    y: FloatArray,
    dy: FloatArray | None,
    f0: float,
    df: float,
    nf: int,
    *,
    fit_mean: bool = True,
    backend: NufftBackend = "finufft",
    eps: float = DEFAULT_EPS,
) -> FloatArray:
    """GLS power on the grid ``f0 + df*arange(nf)`` via NUFFT.

    Matches astropy ``normalization="standard"`` with ``fit_mean`` and inverse-variance
    weights ``w = 1/dy**2`` (uniform when ``dy is None``). The returned power is in
    ``[0, 1]``; degenerate frequencies come back as 0. ``t``/``y``/``dy`` must already
    be the finite single-band selection.

    Parameters
    ----------
    t, y : numpy.ndarray
        Times (days) and values.
    dy : numpy.ndarray or None
        1-sigma errors, or ``None`` for uniform weights.
    f0, df, nf : float, float, int
        Frequency grid start, step, and count.
    fit_mean : bool, default True
        Floating-mean (generalized) variant.
    backend : {"finufft", "cufinufft"}, default "finufft"
        NUFFT backend.
    eps : float, default 1e-9
        NUFFT relative tolerance.

    Returns
    -------
    numpy.ndarray
        Power of length ``nf``.
    """
    if nf <= 0:
        return np.zeros(0, dtype=np.float64)
    tau, w, y, y_mean, yy = _prep(t, y, dy)
    base = np.stack([w, w * y])
    tau_b: Any = tau
    if backend == "cufinufft":
        # Device inputs in, device sums out: the power is assembled on the GPU and
        # only the final spectrum crosses back to the host.
        ensure_cuda_dll_path()
        import cupy as cp

        tau_b = cp.asarray(tau)
        base = cp.asarray(base)
    pair = _trig_sums(tau_b, base, f0, df, nf, backend, eps)
    sw2 = _trig_sums(tau_b, base[:1], 2.0 * f0, 2.0 * df, nf, backend, eps)
    return to_host(_assemble_power(pair[0], pair[1], sw2[0], y_mean, yy, fit_mean))


#: Elements per transient ``(chunk, N)`` matrix in the direct path (64 MB float64);
#: the frequency chunk is capped so long light curves cannot blow device memory.
_DIRECT_CHUNK_ELEMS: Final = 1 << 23


def _gls_sums_direct(
    xp: Any, tau: Any, w: Any, wy: Any, f0: float, df: float, nf: int, *,
    freq_batch: int,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """All six GLS trig sums on ``f_k = f0 + k*df`` from one cos/sin evaluation.

    Returns ``(c, s, yc, ys, c2, s2)`` — the cos/sin sums of the weights ``w`` and the
    weighted data ``wy`` on the base grid, and of ``w`` on the doubled grid (the
    ``isign=+1`` convention of the NUFFT path). The base-grid sums share one angle
    matrix, and the doubled-frequency sums follow from the double-angle identities
    ``cos2θ = 2cos²θ − 1`` and ``sin2θ = 2·sinθ·cosθ``, so ``cos``/``sin`` are
    evaluated **once** per frequency instead of six times (twice per grid in the
    three-call arrangement). ``O(N*nf)`` rather than the NUFFT's ``O(nf log nf)``, but
    pure array-API — the portable GLS path for AMD/Intel/Mac. Batched over frequency
    to bound the transient ``(chunk, N)`` matrices, with the chunk auto-capped so the
    transients stay ~64 MB regardless of ``N``.
    """
    two_pi = 2.0 * float(np.pi)
    fdtype = tau.dtype
    dev = device_ref(tau)
    n_points = int(tau.shape[0])
    chunk = max(1, min(freq_batch, _DIRECT_CHUNK_ELEMS // max(1, n_points)))
    freqs = f0 + df * xp.arange(nf, dtype=fdtype, device=dev)
    c, s, yc, ys, c2, s2 = (xp.empty(nf, dtype=fdtype, device=dev) for _ in range(6))
    sum_w = float(xp.sum(w))
    w_row = w[None, :]
    wy_row = wy[None, :]
    for start in range(0, nf, chunk):
        stop = min(start + chunk, nf)
        ang = (two_pi * freqs[start:stop])[:, None] * tau[None, :]
        cos_a = xp.cos(ang)
        sin_a = xp.sin(ang)
        wc = w_row * cos_a
        c[start:stop] = xp.sum(wc, axis=1)
        s[start:stop] = xp.sum(w_row * sin_a, axis=1)
        yc[start:stop] = xp.sum(wy_row * cos_a, axis=1)
        ys[start:stop] = xp.sum(wy_row * sin_a, axis=1)
        c2[start:stop] = 2.0 * xp.sum(wc * cos_a, axis=1) - sum_w
        s2[start:stop] = 2.0 * xp.sum(wc * sin_a, axis=1)
    return c, s, yc, ys, c2, s2


def lombscargle_power_torch(
    t: FloatArray,
    y: FloatArray,
    dy: FloatArray | None,
    f0: float,
    df: float,
    nf: int,
    *,
    fit_mean: bool = True,
    device: str = "cpu",
    precision: str = "auto",
    freq_batch: int = 4096,
) -> FloatArray:
    """GLS power on ``f0 + df*arange(nf)`` via the portable torch direct trig-sum path.

    Numerically matches :func:`lombscargle_power` (to the working precision): same
    Zechmeister-Kürster assembly, just NUFFT-free trig sums so it runs on any torch
    device (CUDA/ROCm/MPS/XPU/CPU). ``precision="auto"`` is float64 except on MPS, where
    float64 is impossible and float32 is used. Returns numpy float64.
    """
    if nf <= 0:
        return np.zeros(0, dtype=np.float64)
    import torch

    tau, w, y, y_mean, yy = _prep(t, y, dy)
    prec = resolve_precision(precision, device)
    tdtype = torch.float32 if prec == "float32" else torch.float64
    tau_d = to_device_array(tau, device=device, dtype=tdtype)
    w_d = to_device_array(w, device=device, dtype=tdtype)
    wy_d = to_device_array(w * y, device=device, dtype=tdtype)
    xp = array_namespace(tau_d)
    c, s, yc, ys, c2, s2 = _gls_sums_direct(
        xp, tau_d, w_d, wy_d, f0, df, nf, freq_batch=freq_batch
    )
    power = _assemble_power_parts(xp, c, s, yc, ys, c2, s2, y_mean, yy, fit_mean)
    return to_host(power)


class CufinufftGLS:
    """Plan-reusing GPU Lomb-Scargle for batch processing.

    cufinufft cannot batch transforms over *different* nonuniform points in one call
    (each light curve has its own observation times), so the GPU win comes from reusing
    cufinufft plans across light curves: a plan is fixed by its mode count alone, while
    the per-star frequency grid lives in the points and strengths. One cached plan
    serves many stars and all three trig sums of a run.

    The mode count ``nf`` is near-unique per star, so the plan is sized to ``nf``
    rounded up to a multiple of ``nf_bucket``: similar-baseline stars share a plan, the
    extra modes land above the grid top and are sliced off, and index ``k`` still maps
    to ``f0 + k*df``. One instance per GPU-owning process (not thread-safe). Results are
    identical to :func:`lombscargle_power` with ``backend="cufinufft"``.

    Parameters
    ----------
    eps : float, default 1e-9
        NUFFT relative tolerance.
    nf_bucket : int, default 65536
        Plan mode-count rounding granularity.
    """

    def __init__(self, eps: float = DEFAULT_EPS, nf_bucket: int = 1 << 16) -> None:
        ensure_cuda_dll_path()
        import cufinufft
        import cupy

        self._cufinufft = cufinufft
        self._cp = cupy
        self._eps = eps
        self._bucket = max(1, int(nf_bucket))
        self._plans: dict[tuple[int, int], Any] = {}

    def _plan(self, nf: int, n_trans: int) -> Any:
        plan = self._plans.get((nf, n_trans))
        if plan is None:
            plan = self._cufinufft.Plan(
                1, (nf,), n_trans=n_trans, eps=self._eps, isign=1, dtype="complex128"
            )
            self._plans[(nf, n_trans)] = plan
        return plan

    def power(
        self,
        t: FloatArray,
        y: FloatArray,
        dy: FloatArray | None,
        f0: float,
        df: float,
        nf: int,
        *,
        fit_mean: bool = True,
    ) -> FloatArray:
        """GLS power on ``f0 + df*arange(nf)`` via reused cufinufft plans.

        The base-grid pair (``w`` and ``w*y`` share the same points) runs as one
        ``n_trans=2`` batched transform — one ``setpts``/execute instead of two — and
        the doubled-grid sum uses an ``n_trans=1`` plan of the same bucketed size.
        """
        if nf <= 0:
            return np.zeros(0, dtype=np.float64)
        cp = self._cp
        tau, w, y, y_mean, yy = _prep(t, y, dy)
        tau_g = cp.asarray(tau)
        base = cp.asarray(np.stack([w, w * y]))
        nf_plan = ((nf + self._bucket - 1) // self._bucket) * self._bucket
        two_pi = 2.0 * np.pi

        def sums(f0_: float, df_: float, strengths: Any) -> Any:
            x = (two_pi * df_) * tau_g
            x = cp.mod(x + np.pi, two_pi) - np.pi
            mod = cp.exp(2j * np.pi * (f0_ + (nf_plan // 2) * df_) * tau_g)
            plan = self._plan(nf_plan, int(strengths.shape[0]))
            plan.setpts(x)
            out = plan.execute((strengths * mod[None, :]).astype(cp.complex128))
            return out if out.ndim == 2 else out[None, :]

        pair = sums(f0, df, base)
        sw2 = sums(2.0 * f0, 2.0 * df, base[:1])
        power = _assemble_power(
            pair[0, :nf], pair[1, :nf], sw2[0, :nf], y_mean, yy, fit_mean
        )
        return np.asarray(cp.asnumpy(power), dtype=np.float64)


# --- method wrapper ----------------------------------------------------------


def _astropy_lombscargle(lc: LightCurve, settings: GLSSettings) -> Any:
    from astropy.timeseries import LombScargle

    return LombScargle(lc.time, lc.value, lc.error, fit_mean=settings.fit_mean)


def _frequency_faps(
    ls: Any, power: FloatArray, settings: GLSSettings, f0: float, f_max: float
) -> dict[str, FloatArray]:
    """False-alarm probabilities at the local maxima (NaN elsewhere)."""
    if settings.fap_method == "none":
        return {}
    candidates = local_maxima(power)
    if not candidates.size:
        return {}
    fap = np.full(power.shape, np.nan, dtype=np.float64)
    try:
        values = np.atleast_1d(
            np.asarray(
                ls.false_alarm_probability(
                    power[candidates],
                    method=settings.fap_method,
                    minimum_frequency=f0,
                    maximum_frequency=f_max,
                ),
                dtype=np.float64,
            )
        )
    except (ImportError, ValueError):
        # FAP needs scipy for some methods; skip rather than fail the periodogram.
        return {}
    fap[candidates] = values
    return {"fap": fap}


class GLSMethod(PeriodogramMethod):
    """Generalized Lomb-Scargle method (NUFFT-accelerated)."""

    name: ClassVar[str] = "GLS"
    objective_sense: ClassVar[Literal["max", "min"]] = "max"
    supports_multiband: ClassVar[bool] = True
    settings_cls: ClassVar[type] = GLSSettings
    cpu_backend: ClassVar[str] = "finufft"
    gpu_backend: ClassVar[str | None] = "cufinufft"
    portable_gpu_backend: ClassVar[str | None] = "torch"
    all_backends: ClassVar[tuple[str, ...]] = (
        "finufft", "cufinufft", "torch", "astropy",
    )

    def default_grid(self, lc: LightCurve, settings: GLSSettings) -> GridSpec:  # type: ignore[override]
        finite = lc.finite()
        if finite.baseline <= 0.0:
            raise InsufficientDataError("GLS: no usable time baseline")
        ls = _astropy_lombscargle(finite, settings)
        minimum = settings.minimum_frequency
        if minimum is None:
            minimum = 1.0 / finite.baseline
        freq = np.asarray(
            ls.autofrequency(
                minimum_frequency=minimum,
                maximum_frequency=settings.maximum_frequency,
                samples_per_peak=settings.samples_per_peak,
                nyquist_factor=settings.nyquist_factor,
            ),
            dtype=np.float64,
        )
        return GridSpec(kind="frequency", values=freq, uniform=True)

    def power(  # type: ignore[override]
        self,
        grid: GridSpec,
        lc: LightCurve,
        settings: GLSSettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        finite = lc.finite()
        n = finite.n
        if n < settings.min_detections:
            raise InsufficientDataError(
                f"GLS: {n} finite points < min_detections {settings.min_detections}"
            )
        if finite.baseline <= 0.0:
            raise InsufficientDataError("GLS: no usable time baseline")
        if grid.size == 0:
            raise InsufficientDataError("GLS: empty trial grid")

        ls = _astropy_lombscargle(finite, settings)
        if backend == "astropy" or not grid.uniform:
            # A non-uniform grid is evaluated by astropy regardless of the requested
            # NUFFT backend, so report astropy as the backend that actually ran.
            actual_backend = "astropy"
            frequency = grid.frequency
            power = np.asarray(
                ls.power(frequency, normalization="standard"), dtype=np.float64
            )
        else:
            f0, df, nf = grid.uniform_frequency_params()
            frequency = f0 + df * np.arange(nf, dtype=np.float64)
            if backend == "torch" or backend.startswith("torch:"):
                device = resolve_torch_device(backend, settings.device)
                actual_backend = f"torch:{device}"
                power = lombscargle_power_torch(
                    finite.time, finite.value, finite.error, f0, df, nf,
                    fit_mean=settings.fit_mean,
                    device=device,
                    precision=settings.precision,
                    freq_batch=settings.direct_freq_batch,
                )
            elif backend == "cufinufft" and engine is not None:
                actual_backend = backend
                power = engine.power(  # type: ignore[attr-defined]
                    finite.time, finite.value, finite.error, f0, df, nf,
                    fit_mean=settings.fit_mean,
                )
            else:
                actual_backend = backend
                power = lombscargle_power(
                    finite.time, finite.value, finite.error, f0, df, nf,
                    fit_mean=settings.fit_mean,
                    backend=backend,  # type: ignore[arg-type]
                    eps=settings.nufft_eps,
                )
        power = np.nan_to_num(
            np.asarray(power, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        extras = _frequency_faps(
            ls, power, settings, float(frequency[0]), float(frequency[-1])
        )
        return Periodogram.from_spectrum(
            method="GLS",
            backend=actual_backend,
            frequency=frequency,
            power=power,
            objective_sense="max",
            n_samples=n,
            baseline=finite.baseline,
            extras=extras,
            meta=finite.meta,
        )

    def multiband_power(  # type: ignore[override]
        self,
        grid: GridSpec,
        mblc: MultiBandLightCurve,
        settings: GLSSettings,
        backend: str,
    ) -> Periodogram:
        from cuperiod.multiband.gls_mb import gls_multiband_power

        return gls_multiband_power(grid, mblc, settings)

    def make_engine(self, backend: str, settings: GLSSettings) -> object | None:  # type: ignore[override]
        if backend == "cufinufft":
            return CufinufftGLS(eps=settings.nufft_eps)
        return None

    def estimate_device_bytes(self, n_points: int) -> int:
        # Bucketed plan + a few complex128 buffers over the (padded) grid.
        bucket = 1 << 16
        padded = ((n_points + bucket - 1) // bucket) * bucket
        return 128 * 1024**2 + padded * 16 * 8


register(GLSMethod())

__all__ = [
    "CufinufftGLS",
    "DEFAULT_EPS",
    "GLSMethod",
    "NufftBackend",
    "lombscargle_power",
    "lombscargle_power_torch",
]
