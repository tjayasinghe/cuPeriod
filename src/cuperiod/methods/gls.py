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
    is present. Small transforms run on one thread (see :func:`_finufft_options`);
    :class:`FinufftGLS` reuses plans across the many small transforms of a scan.
cufinufft
    GPU; the full-catalog path. :class:`CufinufftGLS` reuses cufinufft plans across
    light curves for batch throughput.
astropy
    The reference ``LombScargle`` implementation, available as an explicit backend and
    used for false-alarm probabilities and custom (non-uniform) grids.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
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
from cuperiod.core.grid import GridSpec, check_grid_size, regular_cadence_nyquist
from cuperiod.core.lightcurve import LightCurve, MultiBandLightCurve
from cuperiod.core.result import Periodogram
from cuperiod.methods.base import PeriodogramMethod, register

NufftBackend = Literal["finufft", "cufinufft"]

#: NUFFT relative tolerance: keeps GLS power within ~1e-9 of the exact direct sum.
DEFAULT_EPS: Final = 1e-9

#: Below this much work (``n_trans * n_points + n_modes``) a finufft transform runs on
#: one thread. Light-curve-sized transforms are dominated by the OpenMP start-up of a
#: threaded call: on a 32-thread desktop a 300-point, 4000-mode transform takes
#: ~3 ms threaded and ~0.4 ms single-threaded, and single-threading keeps winning up
#: to ~1e5 point-strengths or ~1.5e5 modes (measured crossover; the threaded call
#: wins beyond it, by up to ~1.6x at 2e5 points).
_FINUFFT_SINGLE_THREAD_WORK: Final = 150_000


def _finufft_options(n_trans: int, n_points: int, n_modes: int) -> dict[str, int]:
    """finufft options for one transform: ``nthreads=1`` when it is small.

    Large transforms keep finufft's default (all cores, or ``OMP_NUM_THREADS``), so
    long light curves and very fine grids still use every core.
    """
    if n_trans * n_points + n_modes <= _FINUFFT_SINGLE_THREAD_WORK:
        return {"nthreads": 1}
    return {}


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
    device** so the caller can assemble the power there. finufft runs small
    transforms on one thread (:func:`_finufft_options`).
    """
    options: dict[str, int] = {}
    if backend == "finufft":
        import finufft

        xp: Any = np
        nufft1d1 = finufft.nufft1d1
        options = _finufft_options(
            int(strengths.shape[0]), int(tau.shape[0]), int(nf)
        )
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
    out = nufft1d1(x, c, nf, eps=eps, isign=1, **options)
    return out if out.ndim == 2 else out[None, :]


def _assemble_power_parts(
    xp: Any,
    c: Any, s: Any, yc: Any, ys: Any, c2: Any, s2: Any,
    yy: float, fit_mean: bool,
) -> Any:
    """Zechmeister-Kürster generalized-LS power from the real trig-sum components.

    ``c``/``s`` are the cos/sin sums of the weights, ``yc``/``ys`` of the weighted
    *centred* data (see :func:`_prep`), and ``c2``/``s2`` the doubled-frequency
    cos/sin sums of the weights. Uses only array-API-standard ops, so it runs unchanged
    on numpy, cupy, and torch (any device) — no complex dtype is needed, so the
    Apple-MPS path (no complex128) works. Degenerate frequencies map to 0, and the
    power is clipped to its mathematical range ``[0, 1]``: where the denominator
    degenerates (``f*T << 1``, or ``f = k/cadence`` on a lattice) round-off can
    otherwise push it slightly, or at ``f*T << 1`` grossly, outside.
    """
    cc = 0.5 * (1.0 + c2)
    ss = 0.5 * (1.0 - c2)
    cs = 0.5 * s2
    if fit_mean:
        cc = cc - c * c
        ss = ss - s * s
        cs = cs - c * s

    denom = cc * ss - cs * cs
    power = (ss * yc * yc + cc * ys * ys - 2.0 * cs * yc * ys) / (yy * denom)
    usable = xp.isfinite(power) & (power > 0.0)
    power = xp.where(usable, power, xp.zeros_like(power))
    return xp.where(power < 1.0, power, xp.ones_like(power))


def _assemble_power(sw: Any, swy: Any, sw2: Any, yy: float, fit_mean: bool) -> Any:
    """GLS power from three *complex* trig sums (the NUFFT path).

    Splits each complex sum into its real/imaginary parts and delegates the Zechmeister-
    Kürster math to :func:`_assemble_power_parts`. Works on numpy or cupy arrays (GPU
    path keeps everything on device); ``yy`` is a host scalar.
    """
    xp = array_module(sw)
    parts = (sw.real, sw.imag, swy.real, swy.imag, sw2.real, sw2.imag)
    if xp is np:
        with np.errstate(divide="ignore", invalid="ignore"):
            return _assemble_power_parts(xp, *parts, yy, fit_mean)
    return _assemble_power_parts(xp, *parts, yy, fit_mean)


def _prep(
    t: FloatArray, y: FloatArray, dy: FloatArray | None
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """Shared host prep: time origin shift, normalized weights, centred data.

    Returns ``(tau, w, y_c, yy)`` with ``tau = t - t.min()``, ``w`` summing to 1
    (inverse-variance, or uniform when ``dy is None``), the data centred on their
    weighted mean ``y_c = y - sum(w*y)``, and ``yy = sum(w*y_c**2)``.

    Centring here, in float64 before any transform, is what astropy does too
    (``center_data``). The algebra is the same as subtracting ``mean*sum(w*cos)`` from
    the trig sums afterwards, but that difference of two large, NUFFT-approximated
    numbers loses ``(mean/scatter)**2`` of the precision: normalized flux ``1 ± 1e-6``
    was ~3e-5 off astropy, and a sine on a ``1e4`` pedestal reached power ~1.03.
    ``yy`` is 0 for data with no scatter above round-off (see :func:`_centred`).
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    tau = t - t.min()
    w = _weights(t.size, dy)
    y_c, yy = _centred(w, y)
    return tau, w, y_c, yy


def _weights(n: int, dy: FloatArray | None) -> FloatArray:
    """Normalized inverse-variance weights (uniform when ``dy is None``)."""
    if dy is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    dy = np.ascontiguousarray(dy, dtype=np.float64)
    w = 1.0 / (dy * dy)
    w /= w.sum()
    return w


#: Scatter below this many ulps of the data's RMS level is round-off, not signal.
_FLAT_ULPS: Final = 64.0


def _centred(w: FloatArray, y: FloatArray) -> tuple[FloatArray, float]:
    """``(y_c, yy)``: the data about their weighted mean and its weighted variance.

    Two passes (the second removes the rounding error of the first mean), so the
    centred data really have zero weighted mean, as the power assembly assumes. Data
    with no scatter above round-off -- an exactly constant or flat-lined series --
    get ``yy = 0``, which the assembly maps to power 0: their centred residual is
    rounding noise, and a power formed from it is a ratio of rounding errors (astropy
    reports up to ~0.08 on a constant light curve, a peak its own FAP calls
    overwhelmingly significant).
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    y_c = y - float(np.dot(w, y))
    y_c -= float(np.dot(w, y_c))
    yy = float(np.dot(w, y_c * y_c))
    floor = _FLAT_ULPS * float(np.finfo(np.float64).eps) * float(np.max(np.abs(y)))
    if not yy > floor * floor:
        yy = 0.0
    return y_c, yy


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
    weights ``w = 1/dy**2`` (uniform when ``dy is None``). The returned power is
    clipped to ``[0, 1]``; degenerate frequencies come back as 0. ``t``/``y``/``dy``
    must already be the finite single-band selection.

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
    tau, w, y_c, yy = _prep(t, y, dy)
    base = np.stack([w, w * y_c])
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
    return to_host(_assemble_power(pair[0], pair[1], sw2[0], yy, fit_mean))


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

    tau, w, y_c, yy = _prep(t, y, dy)
    prec = resolve_precision(precision, device)
    tdtype = torch.float32 if prec == "float32" else torch.float64
    tau_d = to_device_array(tau, device=device, dtype=tdtype)
    w_d = to_device_array(w, device=device, dtype=tdtype)
    wy_d = to_device_array(w * y_c, device=device, dtype=tdtype)
    xp = array_namespace(tau_d)
    c, s, yc, ys, c2, s2 = _gls_sums_direct(
        xp, tau_d, w_d, wy_d, f0, df, nf, freq_batch=freq_batch
    )
    power = _assemble_power_parts(xp, c, s, yc, ys, c2, s2, yy, fit_mean)
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

    The multi-band paths reuse the same plan cache through :meth:`trig_sums` — the
    offsets model's ``K + 2`` transforms and the flex model's harmonic sums all run
    on grids of the same mode count, so one bucketed plan pair serves every band,
    every harmonic, and every star of a batch.

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

    def trig_sums(
        self, tau: Any, strengths: Any, f0: float, df: float, nf: int
    ) -> Any:
        """Plan-cached type-1 trig sums on ``f0 + df*arange(nf)``.

        The plan-reusing equivalent of :func:`_trig_sums` with
        ``backend="cufinufft"``: ``tau`` and ``strengths`` (``(n_trans, N)``) are
        cupy arrays and the ``(n_trans, nf)`` complex sums stay on device. The plan
        is sized to the bucketed mode count and keyed by ``(modes, n_trans)``, so
        transforms of the same shape — across bands, harmonics, and light curves —
        share one plan.
        """
        cp = self._cp
        nf_plan = ((nf + self._bucket - 1) // self._bucket) * self._bucket
        two_pi = 2.0 * np.pi
        x = (two_pi * df) * tau
        x = cp.mod(x + np.pi, two_pi) - np.pi
        mod = cp.exp(2j * np.pi * (f0 + (nf_plan // 2) * df) * tau)
        plan = self._plan(nf_plan, int(strengths.shape[0]))
        plan.setpts(x)
        out = plan.execute((strengths * mod[None, :]).astype(cp.complex128))
        out = out if out.ndim == 2 else out[None, :]
        return out[:, :nf]

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
        tau, w, y_c, yy = _prep(t, y, dy)
        tau_g = cp.asarray(tau)
        base = cp.asarray(np.stack([w, w * y_c]))
        pair = self.trig_sums(tau_g, base, f0, df, nf)
        sw2 = self.trig_sums(tau_g, base[:1], 2.0 * f0, 2.0 * df, nf)
        power = _assemble_power(pair[0], pair[1], sw2[0], yy, fit_mean)
        return np.asarray(cp.asnumpy(power), dtype=np.float64)


#: Plan modes kept alive by one :class:`FinufftGLS` (summed over ``n_trans``); about
#: 64 MB of finufft work arrays at most.
_FINUFFT_CACHED_MODES: Final = 1 << 21


def _bucket_modes(nf: int) -> int:
    """``nf`` rounded up to 8 buckets per octave (at most 12.5% extra modes)."""
    step = 1 << max(0, nf.bit_length() - 4)
    return ((nf + step - 1) // step) * step


class FinufftGLS:
    """Plan-reusing CPU Lomb-Scargle for many small transforms.

    A one-shot ``finufft.nufft1d1`` builds and destroys a plan per call, and for the
    light-curve-sized transforms of a period-change scan or a batch the plan set-up
    costs about as much as the transform itself. This engine keeps single-threaded
    plans keyed by ``(modes, n_trans)``; like :class:`CufinufftGLS` the mode count is
    rounded up (to :func:`_bucket_modes`, at most 12.5% more), so neighbouring grid
    sizes share a plan, and the extra modes are sliced off. Transforms too large for
    the single-thread regime (:func:`_finufft_options`) fall back to the one-shot,
    multi-threaded call. Results agree with :func:`lombscargle_power` to the NUFFT
    tolerance.

    One instance per thread or process: a finufft plan is not re-entrant.

    Parameters
    ----------
    eps : float, default 1e-9
        NUFFT relative tolerance.
    max_cached_modes : int, default 2**21
        Least-recently-used plans are dropped once the cached plans hold more modes
        than this (summed over ``n_trans``).
    """

    def __init__(
        self, eps: float = DEFAULT_EPS, *, max_cached_modes: int = _FINUFFT_CACHED_MODES
    ) -> None:
        import finufft

        self._finufft = finufft
        self._eps = eps
        self._max_modes = max(1, int(max_cached_modes))
        self._plans: OrderedDict[tuple[int, int], Any] = OrderedDict()

    def _plan(self, nf: int, n_trans: int) -> Any:
        key = (nf, n_trans)
        plan = self._plans.get(key)
        if plan is not None:
            self._plans.move_to_end(key)
            return plan
        plan = self._finufft.Plan(
            1, (nf,), n_trans=n_trans, eps=self._eps, isign=1, dtype="complex128",
            nthreads=1,
        )
        self._plans[key] = plan
        while len(self._plans) > 1 and (
            sum(m * k for m, k in self._plans) > self._max_modes
        ):
            self._plans.popitem(last=False)
        return plan

    def trig_sums(
        self, tau: FloatArray, strengths: Any, f0: float, df: float, nf: int
    ) -> Any:
        """Plan-cached type-1 trig sums on ``f0 + df*arange(nf)`` (host arrays).

        The plan-reusing equivalent of :func:`_trig_sums` with ``backend="finufft"``:
        ``strengths`` is ``(n_trans, N)`` and the result is ``(n_trans, nf)``.
        """
        n_trans = int(strengths.shape[0])
        nf_plan = _bucket_modes(int(nf))
        if not _finufft_options(n_trans, int(tau.shape[0]), nf_plan):
            return _trig_sums(tau, strengths, f0, df, nf, "finufft", self._eps)
        x = (2.0 * np.pi * df) * tau
        x = np.mod(x + np.pi, 2.0 * np.pi) - np.pi
        mod = np.exp(2j * np.pi * (f0 + (nf_plan // 2) * df) * tau)
        plan = self._plan(nf_plan, n_trans)
        plan.setpts(np.ascontiguousarray(x))
        data = np.ascontiguousarray((strengths * mod[None, :]).astype(np.complex128))
        out = plan.execute(data if n_trans > 1 else data[0])
        out = out if out.ndim == 2 else out[None, :]
        return out[:, :nf]

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
        """GLS power on ``f0 + df*arange(nf)`` via reused finufft plans."""
        if nf <= 0:
            return np.zeros(0, dtype=np.float64)
        tau, w, y_c, yy = _prep(t, y, dy)
        base = np.stack([w, w * y_c])
        pair = self.trig_sums(tau, base, f0, df, nf)
        sw2 = self.trig_sums(tau, base[:1], 2.0 * f0, 2.0 * df, nf)
        return np.asarray(
            _assemble_power(pair[0], pair[1], sw2[0], yy, fit_mean), dtype=np.float64
        )


# --- method wrapper ----------------------------------------------------------


def _astropy_lombscargle(lc: LightCurve, settings: GLSSettings) -> Any:
    from astropy.timeseries import LombScargle

    # Relative times: astropy's Baluev/Davies FAP takes the effective baseline from
    # the one-pass variance <t^2> - <t>^2 of exactly these times. On absolute JDs
    # (t^2 ~ 6e12, ulp ~1e-3 d^2) that is rounding noise for runs shorter than about
    # a day -- zero, negative (NaN FAP) or plainly wrong -- while the power and the
    # autofrequency grid are invariant to the time origin.
    time = lc.time - lc.time.min()
    return LombScargle(time, lc.value, lc.error, fit_mean=settings.fit_mean)


def _frequency_faps(
    ls: Any, pg: Periodogram, settings: GLSSettings
) -> FloatArray | None:
    """False-alarm probabilities at every peak candidate of ``pg`` (NaN elsewhere).

    The candidates are exactly the samples :meth:`Periodogram.best_periods` can
    report -- the interior local maxima *and* a peak on either edge of the grid, the
    usual place for a period comparable to the baseline -- so every reported peak
    carries a FAP. At a grid edge the true maximum may lie just outside the band, so
    the FAP of the edge power is an upper bound (conservative). Returns None when no
    FAP is computed (``fap_method="none"``, or a method that needs a missing
    optional dependency).
    """
    if settings.fap_method == "none" or pg.size == 0:
        return None
    # One candidate rule for peaks and FAPs: the periodogram's own.
    candidates = pg._peak_candidates(pg._score())
    if not candidates.size:
        return None
    fap = np.full(pg.power.shape, np.nan, dtype=np.float64)
    try:
        values = np.atleast_1d(
            np.asarray(
                ls.false_alarm_probability(
                    pg.power[candidates],
                    method=settings.fap_method,
                    minimum_frequency=float(pg.frequency[0]),
                    maximum_frequency=float(pg.frequency[-1]),
                ),
                dtype=np.float64,
            )
        )
    except (ImportError, ValueError):
        # FAP needs scipy for some methods; skip rather than fail the periodogram.
        return None
    fap[candidates] = values
    return fap


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
        """astropy's ``autofrequency`` grid, capped at a regular cadence's Nyquist.

        The grid runs from ``minimum_frequency`` (default ``1/baseline``) to
        ``maximum_frequency`` (default ``nyquist_factor`` times astropy's average
        Nyquist frequency ``0.5*N/baseline``) in steps of
        ``1/(samples_per_peak*baseline)``, sample for sample as astropy builds it.

        When the sampling is strictly regular (:func:`~cuperiod.core.grid.
        regular_cadence_nyquist`: a constant cadence, gaps allowed, coherent over the
        run) and neither ``maximum_frequency`` nor ``nyquist_factor`` was set, the
        automatic ceiling stops at the true Nyquist frequency ``0.5/cadence``: above
        it the spectrum is a mirror image, every peak has images of the same height,
        and the best period would be one of them about as often as the truth. Set
        ``maximum_frequency`` or ``nyquist_factor`` to search past it (super-Nyquist
        work on long, barycentre-modulated baselines is not capped in any case).

        Raises
        ------
        InsufficientDataError
            If the times span no positive baseline, or the minimum frequency is not
            below the maximum.
        GridTooLargeError
            If the grid would exceed :data:`~cuperiod.core.grid.MAX_GRID_SIZE`
            samples (checked before allocation; e.g. a mis-converted timestamp with
            an explicit ``maximum_frequency``).
        """
        finite = lc.finite()
        if finite.baseline <= 0.0:
            raise InsufficientDataError("GLS: no usable time baseline")
        ls = _astropy_lombscargle(finite, settings)
        minimum = settings.minimum_frequency
        if minimum is None:
            minimum = 1.0 / finite.baseline
        # astropy's own size arithmetic (its baseline is that of the relative times),
        # evaluated before anything is allocated.
        tau = finite.time - finite.time.min()
        span = float(tau.max() - tau.min())
        requested = settings.maximum_frequency
        automatic = requested is None
        maximum = (
            settings.nyquist_factor * (0.5 * finite.n / span)
            if requested is None
            else float(requested)
        )
        if automatic and "nyquist_factor" not in settings.model_fields_set:
            nyquist = regular_cadence_nyquist(finite.time)
            if nyquist is not None and minimum < nyquist < maximum:
                maximum = requested = nyquist
        if not maximum > minimum:
            raise InsufficientDataError(
                f"GLS: the minimum frequency ({minimum:.6g} c/d) is not below the "
                f"maximum frequency ({maximum:.6g} c/d); set maximum_frequency"
            )
        df = 1.0 / span / settings.samples_per_peak
        check_grid_size(
            1 + int(np.round((maximum - minimum) / df)),
            automatic=automatic,
            time=finite.time,
            baseline=finite.baseline,
            maximum_frequency=maximum,
            context="GLS default grid",
        )
        freq = np.asarray(
            ls.autofrequency(
                minimum_frequency=minimum,
                maximum_frequency=requested,
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
        if backend == "astropy" or not (grid.uniform and grid.kind == "frequency"):
            # A non-uniform grid is evaluated by astropy regardless of the requested
            # NUFFT backend, so report astropy as the backend that actually ran.
            # (GridSpec checks that a uniform flag is truthful; only a frequency
            # progression gives the NUFFT its (f0, df, nf).)
            actual_backend = "astropy"
            frequency = grid.frequency
            power = np.asarray(
                ls.power(frequency, normalization="standard"), dtype=np.float64
            )
            weights = _weights(finite.value.size, finite.error)
            if _centred(weights, finite.value)[1] == 0.0:
                # Flat data: astropy's power is a ratio of rounding errors here.
                power = np.zeros_like(power)
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
            elif (backend == "cufinufft" and isinstance(engine, CufinufftGLS)) or (
                backend == "finufft" and isinstance(engine, FinufftGLS)
            ):
                # Each plan-reusing engine only serves its own backend.
                actual_backend = backend
                power = engine.power(
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
        # The standard-normalized power lies in [0, 1]; astropy's own evaluation can
        # stray by round-off where the fit degenerates, like the NUFFT assembly.
        np.clip(power, 0.0, 1.0, out=power)
        pg = Periodogram.from_spectrum(
            method="GLS",
            backend=actual_backend,
            frequency=frequency,
            power=power,
            objective_sense="max",
            n_samples=n,
            baseline=finite.baseline,
            meta=finite.meta,
        )
        fap = _frequency_faps(ls, pg, settings)
        if fap is None:
            return pg
        return dataclasses.replace(pg, extras={"fap": fap})

    def multiband_power(  # type: ignore[override]
        self,
        grid: GridSpec,
        mblc: MultiBandLightCurve,
        settings: GLSSettings,
        backend: str,
        engine: object | None = None,
    ) -> Periodogram:
        from cuperiod.multiband.gls_mb import gls_multiband_power

        return gls_multiband_power(grid, mblc, settings, backend, engine=engine)

    def make_engine(self, backend: str, settings: GLSSettings) -> object | None:  # type: ignore[override]
        """A plan-reusing engine for repeated transforms on ``backend``.

        :class:`CufinufftGLS` for ``"cufinufft"``; :class:`FinufftGLS` (CPU,
        single-threaded small-transform plans) for ``"finufft"``; None otherwise.
        """
        if backend == "cufinufft":
            return CufinufftGLS(eps=settings.nufft_eps)
        if backend == "finufft":
            return FinufftGLS(eps=settings.nufft_eps)
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
    "FinufftGLS",
    "GLSMethod",
    "NufftBackend",
    "lombscargle_power",
    "lombscargle_power_torch",
]
