"""Amplitude spectra — the search statistic that drives pre-whitening.

Classical-pulsator work is done in *amplitude*, not power: a δ Scuti mode is quoted in
millimagnitudes, and the Breger signal-to-noise criterion compares a peak's amplitude
with the mean amplitude of the surrounding residual spectrum. This module computes the
least-squares amplitude spectrum

.. math::

    y(t) \\approx c_0 + a\\cos(2\\pi f \\Delta t) + b\\sin(2\\pi f \\Delta t),
    \\qquad A(f) = \\sqrt{a^2 + b^2},

on a uniform frequency grid, reusing cuPeriod's NUFFT machinery: the six weighted
trigonometric sums of the Zechmeister-Kürster normal equations are exactly the sums the
GLS periodogram already evaluates with one type-1 transform each.

The load-bearing observation for pre-whitening is that **three of those sums do not
depend on the data**. ``cc``, ``ss`` and ``cs`` are properties of the sampling and the
weights alone, so :class:`SpectrumEngine` computes them once and every subsequent
iteration of the extraction loop costs a *single* NUFFT of the current residuals. A
50-frequency solution therefore costs ~52 transforms rather than ~150.

Backends
--------
cufinufft
    NVIDIA fast path; the plan and its ``setpts`` are built once and reused across every
    pre-whitening iteration (only the strengths change).
finufft
    CPU fast path, same plan reuse. The default when no CUDA device is present.
torch
    Portable direct trig-sum path (AMD/Intel/Apple/CPU), ``O(N \\cdot n_f)`` but
    device-agnostic.
numpy
    The same direct path on the host — the transparent reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from cuperiod.core._arrayapi import (
    array_namespace,
    device_ref,
    resolve_precision,
    resolve_torch_device,
    to_device_array,
    to_host,
)
from cuperiod.core._typing import FloatArray, IntArray
from cuperiod.core.backend import (
    available_backends,
    ensure_cuda_dll_path,
    torch_available,
    torch_gpu_available,
)
from cuperiod.core.errors import BackendUnavailableError, InsufficientDataError
from cuperiod.core.grid import GridSpec
from cuperiod.core.peaks import local_maxima

#: How the trigonometric sums are turned into an amplitude.
Normalization = Literal["lsq", "dft"]

#: Robust/classical estimators for the local noise level of a spectrum.
NoiseEstimator = Literal["mean", "median"]

#: Concrete backends the amplitude spectrum can run on.
SPECTRUM_BACKENDS: Final = ("finufft", "cufinufft", "torch", "numpy")

#: Elements per transient ``(chunk, N)`` matrix in the direct path (64 MB float64).
_DIRECT_CHUNK_ELEMS: Final = 1 << 23

#: Default NUFFT relative tolerance (matches the GLS default).
DEFAULT_EPS: Final = 1e-9


def weighted_epoch(time: FloatArray, weight: FloatArray | None = None) -> float:
    """The reference epoch that decorrelates phase from frequency.

    A sinusoid's phase and frequency are correlated through the reference epoch:
    shifting
    the epoch by ``dt`` shifts the phase by ``2 pi f dt``, so an epoch far from the data
    makes the phase uncertainty balloon with the frequency uncertainty. The correlation
    vanishes exactly at the weighted mean of the observation times, which is therefore
    the epoch cuPeriod references phases to (and, incidentally, the one that keeps
    ``t - t_ref`` smallest for real-scale Julian dates).

    Parameters
    ----------
    time : numpy.ndarray
        Observation times (days).
    weight : numpy.ndarray, optional
        Per-point weights; uniform when omitted.

    Returns
    -------
    float
        The reference epoch in days.
    """
    t = np.asarray(time, dtype=np.float64)
    if t.size == 0:
        return 0.0
    if weight is None:
        return float(np.mean(t))
    w = np.asarray(weight, dtype=np.float64)
    total = float(w.sum())
    return float(np.dot(w, t) / total) if total > 0.0 else float(np.mean(t))


def resolve_spectrum_backend(requested: str = "auto") -> str:
    """Resolve ``auto``/``cpu``/``gpu``/concrete to a runnable spectrum backend.

    Mirrors :meth:`cuperiod.methods.base.PeriodogramMethod.resolve_backend`: ``"auto"``
    prefers the NVIDIA cufinufft fast path, then a non-CPU torch device, then the
    finufft CPU path, and finally the pure-numpy reference.

    Parameters
    ----------
    requested : str, default "auto"
        ``"auto"``, ``"cpu"``, ``"gpu"``, ``"torch"``, ``"torch:<device>"``, or one of
        :data:`SPECTRUM_BACKENDS`.

    Returns
    -------
    str
        A concrete, available backend name.

    Raises
    ------
    BackendUnavailableError
        If the request cannot be satisfied here.
    """
    available = available_backends()
    if requested == "auto":
        if "cufinufft" in available:
            return "cufinufft"
        if torch_gpu_available():
            return "torch"
        return "finufft" if "finufft" in available else "numpy"
    if requested == "cpu":
        return "finufft" if "finufft" in available else "numpy"
    if requested == "gpu":
        if "cufinufft" in available:
            return "cufinufft"
        if torch_gpu_available():
            return "torch"
        raise BackendUnavailableError(
            "pre-whitening: no GPU backend available (need the [gpu] extra and a CUDA "
            "device, or the [torch] extra and a CUDA/ROCm/MPS/XPU device)"
        )
    if requested == "torch" or requested.startswith("torch:"):
        if not torch_available():
            raise BackendUnavailableError(
                "pre-whitening: backend 'torch' needs the [torch] extra "
                "(pip install 'cuperiod[torch]')"
            )
        return requested
    # GLS names astropy as its exact-but-slow reference; the amplitude spectrum's
    # equivalent is the direct numpy path.
    if requested == "astropy":
        return "numpy"
    if requested not in SPECTRUM_BACKENDS:
        raise BackendUnavailableError(
            f"pre-whitening: unknown backend {requested!r}; "
            f"choose from {SPECTRUM_BACKENDS}"
        )
    if requested in {"finufft", "cufinufft"} and requested not in available:
        raise BackendUnavailableError(
            f"pre-whitening: backend {requested!r} is not available here"
        )
    return requested


# --- noise / peak utilities ---------------------------------------------------


def noise_level(
    frequency: FloatArray,
    amplitude: FloatArray,
    at: float,
    *,
    window: float = 1.0,
    estimator: NoiseEstimator = "mean",
    min_samples: int = 25,
) -> float:
    """Local noise level of an amplitude spectrum around ``at``.

    The Breger et al. (1993) signal-to-noise criterion divides a peak amplitude by the
    *average amplitude* of the residual spectrum in a box centred on that frequency;
    this is that average. When the box holds fewer than ``min_samples`` grid points (a
    narrow window on a coarse grid, or a box clipped by the edge of the grid), the
    nearest ``min_samples`` samples are used instead so the estimate never rests on a
    handful of points.

    Parameters
    ----------
    frequency, amplitude : numpy.ndarray
        The spectrum, ascending in frequency.
    at : float
        Centre of the box (cycles/day).
    window : float, default 1.0
        Half-width of the box (cycles/day).
    estimator : {"mean", "median"}, default "mean"
        ``"mean"`` reproduces the classical Breger noise; ``"median"`` is robust to
        residual peaks left inside the box.
    min_samples : int, default 25
        Minimum number of grid samples averaged.

    Returns
    -------
    float
        The local noise amplitude (NaN if the spectrum is empty).
    """
    n = int(frequency.size)
    if n == 0:
        return float("nan")
    lo = int(np.searchsorted(frequency, at - window, side="left"))
    hi = int(np.searchsorted(frequency, at + window, side="right"))
    if hi - lo < min(min_samples, n):
        centre = int(np.searchsorted(frequency, at))
        half = max(1, min(min_samples, n) // 2)
        lo = max(0, min(centre - half, n - min(min_samples, n)))
        hi = min(n, lo + min(min_samples, n))
    box = amplitude[lo:hi]
    box = box[np.isfinite(box)]
    if box.size == 0:
        return float("nan")
    return float(np.mean(box) if estimator == "mean" else np.median(box))


def _blocked_mask(
    frequency: FloatArray, exclude: FloatArray, separation: float
) -> np.ndarray:
    """Boolean mask of grid samples within ``separation`` of any excluded frequency."""
    blocked = np.zeros(frequency.size, dtype=bool)
    if exclude.size == 0 or separation <= 0.0:
        return blocked
    lo = np.searchsorted(frequency, exclude - separation, side="left")
    hi = np.searchsorted(frequency, exclude + separation, side="right")
    for a, b in zip(lo, hi, strict=True):
        blocked[int(a) : int(b)] = True
    return blocked


def _candidate_indices(amplitude: FloatArray) -> IntArray:
    """Interior local maxima plus boundary samples that exceed their one neighbour."""
    candidates = local_maxima(amplitude)
    n = int(amplitude.size)
    edges: list[int] = []
    if n == 1:
        edges.append(0)
    elif n >= 2:
        if amplitude[0] >= amplitude[1]:
            edges.append(0)
        if amplitude[-1] > amplitude[-2]:
            edges.append(n - 1)
    if edges:
        candidates = np.concatenate([candidates, np.asarray(edges, dtype=np.int64)])
    return candidates


@dataclass(frozen=True)
class AmplitudeSpectrum:
    """A least-squares amplitude spectrum on a uniform frequency grid.

    Attributes
    ----------
    frequency : numpy.ndarray
        Trial frequencies (cycles/day), ascending and uniformly spaced.
    amplitude : numpy.ndarray
        Best-fit sinusoid amplitude at each frequency, in the units of the input.
    phase : numpy.ndarray
        Best-fit phase in radians for the convention
        ``A * sin(2*pi*f*(t - t_ref) + phase)``, wrapped to ``[0, 2*pi)``.
    power : numpy.ndarray
        The equivalent normalized Lomb-Scargle power (0-1), handy for false-alarm
        probabilities. Approximate when ``normalization="dft"``.
    t_ref : float
        Time origin the phases are referenced to (days).
    backend : str
        Concrete backend that produced the spectrum.
    normalization : {"lsq", "dft"}
        Amplitude convention (see :class:`SpectrumEngine`).
    n_samples : int
        Number of light-curve points used.
    baseline : float
        Time span of the light curve (days).
    """

    frequency: FloatArray
    amplitude: FloatArray
    phase: FloatArray
    power: FloatArray
    t_ref: float
    backend: str
    normalization: str
    n_samples: int
    baseline: float

    @property
    def size(self) -> int:
        """Number of grid samples."""
        return int(self.frequency.size)

    @property
    def period(self) -> FloatArray:
        """The grid as periods in days."""
        with np.errstate(divide="ignore"):
            return 1.0 / self.frequency

    @property
    def rayleigh(self) -> float:
        """Rayleigh frequency resolution ``1/baseline`` (cycles/day)."""
        return 1.0 / self.baseline if self.baseline > 0.0 else float("inf")

    def peak_index(
        self,
        *,
        exclude: FloatArray | None = None,
        separation: float = 0.0,
    ) -> int | None:
        """Index of the highest peak, skipping neighbourhoods of ``exclude``.

        Parameters
        ----------
        exclude : numpy.ndarray, optional
            Frequencies whose ``+/- separation`` neighbourhood is off limits (the
            already-extracted components).
        separation : float, default 0.0
            Exclusion half-width in cycles/day.

        Returns
        -------
        int or None
            Grid index of the tallest admissible peak, or ``None`` if there is none.
        """
        if self.size == 0:
            return None
        candidates = _candidate_indices(self.amplitude)
        if candidates.size == 0:
            candidates = np.arange(self.size, dtype=np.int64)
        finite = np.isfinite(self.amplitude[candidates])
        candidates = candidates[finite]
        if candidates.size == 0:
            return None
        if exclude is not None and exclude.size and separation > 0.0:
            blocked = _blocked_mask(self.frequency, exclude, separation)
            candidates = candidates[~blocked[candidates]]
            if candidates.size == 0:
                return None
        return int(candidates[int(np.argmax(self.amplitude[candidates]))])

    def refine_peak(self, index: int) -> tuple[float, float]:
        """Sub-grid ``(frequency, amplitude)`` at ``index`` by parabolic interpolation.

        A three-point parabola through the peak sample and its neighbours locates the
        apex to a small fraction of the grid step, which is a much better starting
        point for the non-linear fit than the grid sample itself.
        """
        i = int(index)
        f = float(self.frequency[i])
        a = float(self.amplitude[i])
        if i <= 0 or i >= self.size - 1:
            return f, a
        y0, y1, y2 = (float(self.amplitude[j]) for j in (i - 1, i, i + 1))
        denom = y0 - 2.0 * y1 + y2
        if denom >= 0.0 or not np.isfinite(denom):
            return f, a
        shift = 0.5 * (y0 - y2) / denom
        if not (-1.0 < shift < 1.0):
            return f, a
        step = float(self.frequency[i + 1] - self.frequency[i])
        apex = y1 - 0.25 * (y0 - y2) * shift
        return f + shift * step, max(apex, a)

    def noise_at(
        self,
        frequency: float,
        *,
        window: float = 1.0,
        estimator: NoiseEstimator = "mean",
        min_samples: int = 25,
    ) -> float:
        """Local noise amplitude around ``frequency`` (see :func:`noise_level`)."""
        return noise_level(
            self.frequency,
            self.amplitude,
            frequency,
            window=window,
            estimator=estimator,
            min_samples=min_samples,
        )

    def downsample(self, n_points: int = 2000) -> tuple[FloatArray, FloatArray]:
        """Peak-preserving ``(frequency, amplitude)`` downsample for plots/storage."""
        from cuperiod.core.peaks import peak_preserving_downsample

        freq, amp = peak_preserving_downsample(
            self.frequency, self.amplitude, n_points
        )
        return freq.astype(np.float64), amp.astype(np.float64)


# --- trig-sum kernels ---------------------------------------------------------


def _wrapped_angles(xp: Any, tau: Any, df: float) -> Any:
    """``2*pi*df*tau`` wrapped to ``[-pi, pi)`` — the NUFFT nonuniform points."""
    x = (2.0 * np.pi * df) * tau
    return xp.mod(x + np.pi, 2.0 * np.pi) - np.pi


def _direct_trig_sums(
    xp: Any, tau: Any, strengths: Any, f0: float, df: float, nf: int, chunk: int
) -> tuple[Any, Any]:
    """``(sum_j s_j cos(2*pi*f_k*tau_j), sum_j s_j sin(...))`` by direct evaluation.

    Pure array-API, so it runs unchanged on numpy, cupy and torch (any device). The
    frequency loop is chunked so the transient ``(chunk, N)`` angle matrix stays bounded
    regardless of the light-curve length.
    """
    dtype = tau.dtype
    dev = device_ref(tau)
    n_points = int(tau.shape[0])
    chunk = max(1, min(chunk, _DIRECT_CHUNK_ELEMS // max(1, n_points)))
    freqs = f0 + df * xp.arange(nf, dtype=dtype, device=dev)
    cos_sum = xp.empty(nf, dtype=dtype, device=dev)
    sin_sum = xp.empty(nf, dtype=dtype, device=dev)
    row = strengths[None, :]
    two_pi = 2.0 * float(np.pi)
    for start in range(0, nf, chunk):
        stop = min(start + chunk, nf)
        ang = (two_pi * freqs[start:stop])[:, None] * tau[None, :]
        cos_sum[start:stop] = xp.sum(row * xp.cos(ang), axis=1)
        sin_sum[start:stop] = xp.sum(row * xp.sin(ang), axis=1)
    return cos_sum, sin_sum


class SpectrumEngine:
    """Reusable amplitude-spectrum evaluator for one light curve's sampling.

    The sampling (``time``), the weights (``error``) and the frequency grid are fixed at
    construction; :meth:`spectrum` then evaluates any number of *value* arrays on them.
    Everything that depends only on the sampling — the NUFFT plan and its point sort,
    and the ``cc``/``ss``/``cs`` window sums of the normal equations — is computed once,
    so each call costs a single transform. That is what makes iterative pre-whitening
    cheap.

    Parameters
    ----------
    time : numpy.ndarray
        Observation times in days (finite, at least two distinct values).
    error : numpy.ndarray, optional
        1-sigma uncertainties. ``None`` gives uniform weights.
    grid : GridSpec
        A **uniform frequency** grid (``kind="frequency"``, ``uniform=True``).
    normalization : {"lsq", "dft"}, default "lsq"
        ``"lsq"`` solves the weighted least-squares normal equations at every trial
        frequency — the statistically correct amplitude for irregular sampling.
        ``"dft"`` is the classical Deeming amplitude ``2|sum w y e^{2 pi i f t}|`` used
        by Period04; cheaper to set up and identical for even, unweighted sampling.
    backend : str, default "auto"
        See :func:`resolve_spectrum_backend`.
    device, precision : str
        Torch device and compute precision (see :class:`cuperiod.GLSSettings`).
    eps : float, default 1e-9
        NUFFT relative tolerance.
    freq_batch : int, default 4096
        Frequency chunk of the direct (torch/numpy) path.
    t_ref : float, optional
        Phase reference epoch. Defaults to :func:`weighted_epoch`.

    Notes
    -----
    One instance is not thread-safe (a cufinufft plan holds device state). Build one per
    worker.
    """

    def __init__(
        self,
        time: FloatArray,
        error: FloatArray | None = None,
        *,
        grid: GridSpec,
        normalization: Normalization = "lsq",
        backend: str = "auto",
        device: str = "auto",
        precision: str = "auto",
        eps: float = DEFAULT_EPS,
        freq_batch: int = 4096,
        t_ref: float | None = None,
    ) -> None:
        t = np.ascontiguousarray(np.asarray(time, dtype=np.float64))
        if t.ndim != 1 or t.size < 3:
            raise InsufficientDataError(
                "pre-whitening: need at least 3 finite points to build a spectrum"
            )
        if grid.kind != "frequency" or not grid.uniform:
            raise ValueError(
                "SpectrumEngine needs a uniform frequency grid; build one with "
                "cuperiod.uniform_frequency_grid(...)"
            )
        f0, df, nf = grid.uniform_frequency_params()
        if nf <= 0 or df <= 0.0:
            raise ValueError("SpectrumEngine needs a non-empty, increasing grid")

        self.n_samples = int(t.size)
        self.baseline = float(t.max() - t.min())
        self.normalization: Normalization = normalization
        self.backend = resolve_spectrum_backend(backend)
        self.frequency: FloatArray = f0 + df * np.arange(nf, dtype=np.float64)
        self._f0, self._df, self._nf = f0, df, int(nf)
        self._eps = float(eps)
        self._freq_batch = int(freq_batch)

        if error is None:
            w = np.full(t.shape, 1.0 / t.size, dtype=np.float64)
        else:
            dy = np.ascontiguousarray(np.asarray(error, dtype=np.float64))
            if dy.shape != t.shape:
                raise ValueError("error and time must have the same length")
            w = 1.0 / (dy * dy)
            w /= w.sum()
        self.t_ref = weighted_epoch(t, w) if t_ref is None else float(t_ref)
        self._tau = t - self.t_ref
        self._weight: FloatArray = w

        self._plan: Any = None
        self._plan_mod: Any = None
        self._xp: Any = None
        self._tau_dev: Any = None
        self._torch_dtype: Any = None
        self._device = "cpu"
        self._precision = "float64"
        self._setup(device, precision)
        # The weights sum to 1, so a well-conditioned frequency has det ~ 0.25 and
        # rounding noise ~eps. Anything far below that is a degenerate design (a zero
        # frequency, or sampling that cannot separate cosine from sine there) and is
        # reported as zero amplitude rather than as a huge spurious peak.
        self._det_floor = 1e-10 if self._precision == "float64" else 1e-4
        self._cc, self._ss, self._cs, self._det = self._window_terms()

    # -- setup -------------------------------------------------------------------
    def _setup(self, device: str, precision: str) -> None:
        """Build the per-backend state that depends only on the sampling."""
        backend = self.backend
        if backend in {"finufft", "cufinufft"}:
            self._plan, self._plan_mod = self._make_plan(self._df, self._f0)
            return
        if backend == "numpy":
            self._tau_dev = self._tau
            self._xp = array_namespace(self._tau)
            return
        # torch (any device)
        import torch

        self._device = resolve_torch_device(backend, device)
        self.backend = f"torch:{self._device}"
        self._precision = resolve_precision(precision, self._device)
        self._torch_dtype = (
            torch.float32 if self._precision == "float32" else torch.float64
        )
        self._tau_dev = to_device_array(
            self._tau, device=self._device, dtype=self._torch_dtype
        )
        self._xp = array_namespace(self._tau_dev)

    def _import_nufft(self) -> tuple[Any, Any]:
        """``(plan_factory, array_namespace)`` for the resolved NUFFT backend."""
        if self.backend == "finufft":
            import finufft

            return finufft.Plan, np
        ensure_cuda_dll_path()
        import cufinufft
        import cupy as cp

        return cufinufft.Plan, cp

    def _make_plan(self, df: float, f0: float) -> tuple[Any, Any]:
        """A type-1 NUFFT plan with ``setpts`` done, plus its modulation array.

        finufft puts output index ``k`` at mode ``m = k - nf//2``; modulating the
        strengths by ``exp(2j*pi*f_center*tau)`` with ``f_center = f0 + (nf//2)*df``
        maps index ``k`` onto frequency ``f0 + k*df``.
        """
        nf = self._nf
        plan_factory, xp = self._import_nufft()
        if self._tau_dev is None:
            self._xp = xp
            self._tau_dev = xp.asarray(self._tau)
        plan = plan_factory(
            1, (nf,), n_trans=1, eps=self._eps, isign=1, dtype="complex128"
        )
        plan.setpts(_wrapped_angles(xp, self._tau_dev, df))
        mod = xp.exp(2j * np.pi * (f0 + (nf // 2) * df) * self._tau_dev)
        return plan, mod

    def _plan_sums(
        self, plan: Any, mod: Any, strengths: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Execute ``plan`` for ``strengths`` and split the result into cos/sin sums."""
        out = plan.execute((self._xp.asarray(strengths) * mod).astype(np.complex128))
        out = out[0] if out.ndim == 2 else out
        return to_host(out.real), to_host(out.imag)

    def _base_sums(self, strengths: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Cos/sin sums of ``strengths`` on the base grid, returned on the host."""
        if self._plan is not None:
            return self._plan_sums(self._plan, self._plan_mod, strengths)
        cos_sum, sin_sum = _direct_trig_sums(
            self._xp,
            self._tau_dev,
            self._to_dev(strengths),
            self._f0,
            self._df,
            self._nf,
            self._freq_batch,
        )
        return to_host(cos_sum), to_host(sin_sum)

    def _doubled_sums(self, strengths: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Cos/sin sums of ``strengths`` on the doubled grid ``2*f_k`` (host)."""
        if self._plan is not None:
            plan, mod = self._make_plan(2.0 * self._df, 2.0 * self._f0)
            return self._plan_sums(plan, mod, strengths)
        cos_sum, sin_sum = _direct_trig_sums(
            self._xp,
            self._tau_dev,
            self._to_dev(strengths),
            2.0 * self._f0,
            2.0 * self._df,
            self._nf,
            self._freq_batch,
        )
        return to_host(cos_sum), to_host(sin_sum)

    def _to_dev(self, host: FloatArray) -> Any:
        """Move a host float64 array onto the compute device at the working dtype."""
        if self._torch_dtype is not None:
            return to_device_array(host, device=self._device, dtype=self._torch_dtype)
        return self._xp.asarray(host)

    def _window_terms(
        self,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """The data-independent ``cc``, ``ss``, ``cs`` sums and their determinant.

        These are the Gram entries of the floating-mean design at every trial frequency
        — functions of the observation times and weights only — so the extraction loop
        pays for them exactly once.
        """
        nf = self._nf
        if self.normalization == "dft":
            zero = np.zeros(nf, dtype=np.float64)
            return zero, zero, zero, zero
        c, s = self._base_sums(self._weight)
        c2, s2 = self._doubled_sums(self._weight)
        cc = 0.5 * (1.0 + c2) - c * c
        ss = 0.5 * (1.0 - c2) - s * s
        cs = 0.5 * s2 - c * s
        det = cc * ss - cs * cs
        return cc, ss, cs, det

    # -- evaluation --------------------------------------------------------------
    def spectrum(self, values: FloatArray) -> AmplitudeSpectrum:
        """Amplitude spectrum of ``values`` on this engine's sampling and grid.

        Parameters
        ----------
        values : numpy.ndarray
            Brightness values aligned with the ``time`` this engine was built from —
            typically the current pre-whitening residuals.

        Returns
        -------
        AmplitudeSpectrum
        """
        y = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        if y.shape != self._tau.shape:
            raise ValueError(
                f"values length {y.size} does not match the engine's "
                f"{self._tau.size} time samples"
            )
        w = self._weight
        y_mean = float(np.dot(w, y))
        centred = y - y_mean
        yy = float(np.dot(w, centred * centred))
        yc, ys = self._base_sums(w * centred)

        if self.normalization == "dft":
            a = 2.0 * yc
            b = 2.0 * ys
        else:
            det = self._det
            safe = det > self._det_floor
            inv = np.where(safe, 1.0 / np.where(safe, det, 1.0), 0.0)
            a = (self._ss * yc - self._cs * ys) * inv
            b = (self._cc * ys - self._cs * yc) * inv

        amplitude = np.hypot(a, b)
        phase = np.mod(np.arctan2(a, b), 2.0 * np.pi)
        if yy > 0.0:
            power = np.clip((a * yc + b * ys) / yy, 0.0, 1.0)
        else:
            power = np.zeros_like(amplitude)
        bad = ~np.isfinite(amplitude)
        if bad.any():
            amplitude = np.where(bad, 0.0, amplitude)
            phase = np.where(bad, 0.0, phase)
            power = np.where(bad, 0.0, power)
        return AmplitudeSpectrum(
            frequency=self.frequency,
            amplitude=amplitude,
            phase=phase,
            power=power,
            t_ref=self.t_ref,
            backend=self.backend,
            normalization=self.normalization,
            n_samples=self.n_samples,
            baseline=self.baseline,
        )


def amplitude_spectrum(
    time: FloatArray,
    value: FloatArray,
    error: FloatArray | None = None,
    *,
    grid: GridSpec,
    normalization: Normalization = "lsq",
    backend: str = "auto",
    device: str = "auto",
    precision: str = "auto",
    eps: float = DEFAULT_EPS,
    freq_batch: int = 4096,
    t_ref: float | None = None,
) -> AmplitudeSpectrum:
    """One-shot amplitude spectrum (a :class:`SpectrumEngine` used once).

    Use the engine directly when evaluating many value arrays on the same sampling —
    that is what the pre-whitening loop does.

    Parameters
    ----------
    time, value : numpy.ndarray
        The light curve (finite points only).
    error : numpy.ndarray, optional
        1-sigma uncertainties; ``None`` gives uniform weights.
    grid : GridSpec
        A uniform frequency grid.
    normalization, backend, device, precision, eps, freq_batch, t_ref
        See :class:`SpectrumEngine`.

    Returns
    -------
    AmplitudeSpectrum
    """
    engine = SpectrumEngine(
        time,
        error,
        grid=grid,
        normalization=normalization,
        backend=backend,
        device=device,
        precision=precision,
        eps=eps,
        freq_batch=freq_batch,
        t_ref=t_ref,
    )
    return engine.spectrum(value)


__all__ = [
    "DEFAULT_EPS",
    "SPECTRUM_BACKENDS",
    "AmplitudeSpectrum",
    "NoiseEstimator",
    "Normalization",
    "SpectrumEngine",
    "amplitude_spectrum",
    "noise_level",
    "resolve_spectrum_backend",
    "weighted_epoch",
]
