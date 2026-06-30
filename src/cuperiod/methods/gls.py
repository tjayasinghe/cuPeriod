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
    tau: FloatArray,
    strengths: FloatArray,
    f0: float,
    df: float,
    nf: int,
    backend: NufftBackend,
    eps: float,
) -> np.ndarray:
    """``out[k] = sum_j strengths_j exp(2j*pi*(f0 + k*df)*tau_j)`` via a type-1 NUFFT.

    finufft's default mode ordering puts mode ``m = k - nf//2`` at output index ``k``,
    so the strengths are modulated by ``exp(2j*pi*f_center*tau)`` with
    ``f_center = f0 + (nf//2)*df`` and the times scaled to ``x = 2*pi*df*tau`` (wrapped
    to ``[-pi, pi)``); output index ``k`` then lands on frequency ``f0 + k*df``.
    """
    f_center = f0 + (nf // 2) * df
    x = (2.0 * np.pi * df) * tau
    x = np.mod(x + np.pi, 2.0 * np.pi) - np.pi
    c = (strengths * np.exp(2j * np.pi * f_center * tau)).astype(np.complex128)

    if backend == "finufft":
        import finufft

        out = finufft.nufft1d1(x, c, nf, eps=eps, isign=1)
        return np.asarray(out, dtype=np.complex128)
    if backend == "cufinufft":
        ensure_cuda_dll_path()
        import cufinufft
        import cupy as cp

        out_g = cufinufft.nufft1d1(cp.asarray(x), cp.asarray(c), nf, eps=eps, isign=1)
        return np.asarray(cp.asnumpy(out_g), dtype=np.complex128)
    raise ValueError(f"unknown NUFFT backend {backend!r}")


def _assemble_power(
    sw: Any, swy: Any, sw2: Any, y_mean: float, yy: float, fit_mean: bool
) -> Any:
    """Zechmeister-Kürster generalized-LS power (standard norm) from three trig sums.

    Works on numpy or cupy arrays (the GPU path keeps everything on device);
    ``y_mean``/``yy`` are host scalars. Degenerate frequencies map to 0.
    """
    xp = array_module(sw)
    c, s = sw.real, sw.imag
    yc, ys = swy.real, swy.imag
    c2, s2 = sw2.real, sw2.imag

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
    if xp is np:
        with np.errstate(divide="ignore", invalid="ignore"):
            power = (ss * yc * yc + cc * ys * ys - 2.0 * cs * yc * ys) / (yy * denom)
    else:
        power = (ss * yc * yc + cc * ys * ys - 2.0 * cs * yc * ys) / (yy * denom)
    return xp.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)


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
    sw = _trig_sums(tau, w, f0, df, nf, backend, eps)
    swy = _trig_sums(tau, w * y, f0, df, nf, backend, eps)
    sw2 = _trig_sums(tau, w, 2.0 * f0, 2.0 * df, nf, backend, eps)
    return np.asarray(
        _assemble_power(sw, swy, sw2, y_mean, yy, fit_mean), dtype=np.float64
    )


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
        self._plans: dict[int, Any] = {}

    def _plan(self, nf: int) -> Any:
        plan = self._plans.get(nf)
        if plan is None:
            plan = self._cufinufft.Plan(
                1, (nf,), eps=self._eps, isign=1, dtype="complex128"
            )
            self._plans[nf] = plan
        return plan

    @staticmethod
    def _squeeze(arr: Any) -> Any:
        return arr[0] if getattr(arr, "ndim", 1) == 2 else arr

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
        """GLS power on ``f0 + df*arange(nf)`` via a reused cufinufft plan."""
        if nf <= 0:
            return np.zeros(0, dtype=np.float64)
        cp = self._cp
        tau, w, y, y_mean, yy = _prep(t, y, dy)
        tau_g = cp.asarray(tau)
        wg = cp.asarray(w)
        wyg = cp.asarray(w * y)
        nf_plan = ((nf + self._bucket - 1) // self._bucket) * self._bucket
        plan = self._plan(nf_plan)
        two_pi = 2.0 * np.pi

        def sums(f0_: float, df_: float, strengths: tuple[Any, ...]) -> list[Any]:
            x = (two_pi * df_) * tau_g
            x = cp.mod(x + np.pi, two_pi) - np.pi
            mod = cp.exp(2j * np.pi * (f0_ + (nf_plan // 2) * df_) * tau_g)
            plan.setpts(x)
            return [
                self._squeeze(plan.execute((c * mod).astype(cp.complex128)))
                for c in strengths
            ]

        sw, swy = sums(f0, df, (wg, wyg))
        (sw2,) = sums(2.0 * f0, 2.0 * df, (wg,))
        power = _assemble_power(sw[:nf], swy[:nf], sw2[:nf], y_mean, yy, fit_mean)
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
    all_backends: ClassVar[tuple[str, ...]] = ("finufft", "cufinufft", "astropy")

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
            actual_backend = backend
            f0, df, nf = grid.uniform_frequency_params()
            frequency = f0 + df * np.arange(nf, dtype=np.float64)
            if backend == "cufinufft" and engine is not None:
                power = engine.power(  # type: ignore[attr-defined]
                    finite.time, finite.value, finite.error, f0, df, nf,
                    fit_mean=settings.fit_mean,
                )
            else:
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
]
