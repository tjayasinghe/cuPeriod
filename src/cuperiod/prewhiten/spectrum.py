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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from cuperiod.core._arrayapi import (
    array_namespace,
    device_ref,
    is_cupy_array,
    resolve_precision,
    resolve_torch_device,
    to_device_array,
    to_host,
)
from cuperiod.core._typing import FloatArray, IntArray
from cuperiod.core.backend import (
    available_backends,
    ensure_cuda_dll_path,
    has_module,
    normalize_backend,
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

#: ``points + modes`` up to which a CPU finufft plan runs on one thread. Waking the
#: thread pool costs more than a small transform: on 32 cores a 300-point,
#: 3746-mode spectrum took 1.40 ms threaded and 0.21 ms on one thread, and the gain
#: holds to ~1e5 (the rule and value of ``gls._FINUFFT_SINGLE_THREAD_WORK``).
_FINUFFT_SINGLE_THREAD_WORK: Final = 150_000

#: Conditioning below which the least-squares amplitude is not trusted (see
#: :meth:`SpectrumEngine._window_terms`): the smallest eigenvalue of the centred
#: cos/sin Gram matrix over its ideal value of 1/2. Below 0.1 the noise in the
#: normal-equation solution is amplified more than threefold; a higher cut starts
#: replacing genuinely informative frequencies of nightly ground-based sampling,
#: where it flipped a real 2.0037 c/d mode to its 1 c/d aliases.
_ILL_CONDITIONED: Final = 0.1

#: Smallest eigenvalue of the Gram matrix below which a trial frequency carries no
#: information at all and reads zero amplitude: comfortably above the NUFFT's own
#: ~1e-9 error in the sums (float64), or the float32 direct path's ~1e-6.
_DEAD_EIGENVALUE: Final = {"float64": 1e-6, "float32": 1e-3}

#: Frequencies per re-anchoring of the float32 direct path's phase recurrence.
_PHASE_ANCHOR: Final = 512


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
    requested = normalize_backend(requested)
    # Requests that need no GPU knowledge are answered first: probing for CUDA imports
    # cupy and loads the CUDA runtime (a few hundred ms and ~200 MB per process), which
    # a run that asked for the CPU must never pay for.
    if requested in {"numpy", "astropy"}:
        # GLS names astropy as its exact-but-slow reference; the amplitude spectrum's
        # equivalent is the direct numpy path.
        return "numpy"
    if requested == "cpu":
        return "finufft" if has_module("finufft") else "numpy"
    if requested == "finufft":
        if not has_module("finufft"):
            raise BackendUnavailableError(
                "pre-whitening: backend 'finufft' is not available here"
            )
        return requested
    if requested == "torch" or requested.startswith("torch:"):
        if not torch_available():
            raise BackendUnavailableError(
                "pre-whitening: backend 'torch' needs the [torch] extra "
                "(pip install 'cuperiod[torch]')"
            )
        return requested
    if requested not in {"auto", "gpu", "cufinufft"}:
        raise BackendUnavailableError(
            f"pre-whitening: unknown backend {requested!r}; choose from "
            f"{('auto', 'cpu', 'gpu', *SPECTRUM_BACKENDS)} (or 'torch:<device>')"
        )
    available = available_backends()
    if requested == "auto":
        if "cufinufft" in available:
            return "cufinufft"
        if torch_gpu_available():
            return "torch"
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
    if requested not in available:
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


class _SpectrumReadout:
    """Peak, noise and point readers over a ``(frequency, amplitude)`` pair.

    Shared by the public :class:`AmplitudeSpectrum` and the extraction loop's lean
    :class:`_SpectrumScan`, which carries only what those readers need.
    """

    frequency: FloatArray
    amplitude: FloatArray

    @property
    def size(self) -> int:
        """Number of grid samples."""
        return int(self.frequency.size)

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

    def amplitude_at(self, frequency: FloatArray | float) -> Any:
        """Amplitude at the grid sample(s) nearest ``frequency``.

        The direct, single-frequency reading of the spectrum — what a marker drawn on
        this curve sits at, and the reference the extraction compares each fitted
        amplitude against (see :attr:`~cuperiod.Sinusoid.amplitude_ratio`).

        Parameters
        ----------
        frequency : float or numpy.ndarray
            Frequencies in cycles/day. Values outside the grid clamp to its ends.

        Returns
        -------
        float or numpy.ndarray
            Matching the input's scalar-or-array shape.
        """
        wanted = np.asarray(frequency, dtype=np.float64)
        if self.size == 0:
            empty = np.full(wanted.shape, np.nan)
            return empty if np.ndim(frequency) else float("nan")
        right = np.clip(np.searchsorted(self.frequency, wanted), 0, self.size - 1)
        left = np.clip(right - 1, 0, self.size - 1)
        nearer_left = np.abs(self.frequency[left] - wanted) <= np.abs(
            self.frequency[right] - wanted
        )
        out = self.amplitude[np.where(nearer_left, left, right)]
        return out if np.ndim(frequency) else float(out)

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


# Equality is identity (eq=False): a generated __eq__ would compare the arrays and
# raise on their ambiguous truth value, and frozen+eq would make hash() fail on them.
@dataclass(frozen=True, eq=False)
class AmplitudeSpectrum(_SpectrumReadout):
    """A least-squares amplitude spectrum on a uniform frequency grid.

    Compares (and hashes) by identity; compare the arrays themselves to test two
    spectra for equal content.

    Attributes
    ----------
    frequency : numpy.ndarray
        Trial frequencies (cycles/day), ascending and uniformly spaced.
    amplitude : numpy.ndarray
        Best-fit sinusoid amplitude at each frequency, in the units of the input.
        Where the sampling barely separates a sinusoid from the constant (see
        :class:`SpectrumEngine`) this is the bounded power-equivalent amplitude
        ``sqrt(2 * power * var)`` rather than the ill-conditioned normal-equation
        solution, and where it cannot separate them at all it is zero.
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
    def period(self) -> FloatArray:
        """The grid as periods in days."""
        with np.errstate(divide="ignore"):
            return 1.0 / self.frequency

    @property
    def rayleigh(self) -> float:
        """Rayleigh frequency resolution ``1/baseline`` (cycles/day)."""
        return 1.0 / self.baseline if self.baseline > 0.0 else float("inf")

    def downsample(self, n_points: int = 2000) -> tuple[FloatArray, FloatArray]:
        """Peak-preserving ``(frequency, amplitude)`` downsample for plots/storage."""
        from cuperiod.core.peaks import peak_preserving_downsample

        freq, amp = peak_preserving_downsample(
            self.frequency, self.amplitude, n_points
        )
        return freq.astype(np.float64), amp.astype(np.float64)

    def noise_curve(
        self,
        *,
        window: float = 1.0,
        estimator: NoiseEstimator = "mean",
        min_samples: int = 25,
    ) -> FloatArray:
        """The local noise level at every grid frequency (see :func:`noise_level`).

        The curve a significance threshold is drawn from: ``k * noise_curve()`` is the
        S/N = k line over the spectrum. Each value equals :meth:`noise_at` at that grid
        frequency (the same box, the same ``min_samples`` fallback, non-finite samples
        ignored); the mean is evaluated with running sums, so it agrees to rounding.

        Parameters
        ----------
        window : float, default 1.0
            Half-width of the box (cycles/day).
        estimator : {"mean", "median"}, default "mean"
            As for :func:`noise_level`.
        min_samples : int, default 25
            Minimum number of grid samples per box.

        Returns
        -------
        numpy.ndarray
            One noise amplitude per grid frequency (NaN where a box holds no finite
            sample).
        """
        freq = self.frequency
        n = int(freq.size)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        need = min(min_samples, n)
        lo = np.searchsorted(freq, freq - window, side="left")
        hi = np.searchsorted(freq, freq + window, side="right")
        short = (hi - lo) < need
        if short.any():
            # noise_level's fallback: the nearest `need` samples around the centre.
            centre = np.searchsorted(freq, freq[short])
            half = max(1, need // 2)
            low = np.maximum(0, np.minimum(centre - half, n - need))
            lo[short] = low
            hi[short] = np.minimum(n, low + need)
        amplitude = self.amplitude
        finite = np.isfinite(amplitude)
        out = np.empty(n, dtype=np.float64)
        if estimator == "mean":
            sums = np.concatenate([[0.0], np.cumsum(np.where(finite, amplitude, 0.0))])
            counts = np.concatenate([[0], np.cumsum(finite)])
            count = counts[hi] - counts[lo]
            with np.errstate(invalid="ignore", divide="ignore"):
                out[:] = (sums[hi] - sums[lo]) / count
            out[count == 0] = np.nan
            return out
        for i in range(n):
            box = amplitude[lo[i] : hi[i]]
            box = box[finite[lo[i] : hi[i]]]
            out[i] = float(np.median(box)) if box.size else float("nan")
        return out


@dataclass(frozen=True, eq=False)
class _NormalTerms:
    """The data-independent half of the floating-mean normal equations, per frequency.

    ``cc``, ``ss`` and ``cs`` are the centred Gram entries; ``inv`` is ``1/det`` where
    the design is usable and 0 where it carries no information; ``ill`` marks the usable
    frequencies whose smallest Gram eigenvalue is below :data:`_ILL_CONDITIONED` of its
    ideal, and ``ill_index`` lists them (usually a small fraction of the grid).
    """

    cc: FloatArray
    ss: FloatArray
    cs: FloatArray
    inv: FloatArray
    ill: np.ndarray
    ill_index: IntArray


class _SpectrumScan(_SpectrumReadout):
    """An amplitude-only evaluation of a spectrum, for the extraction loop.

    The loop reads the amplitude everywhere (peak picking, the noise box) but the power
    at a single index and the phase nowhere, so building the full-grid phase and power
    arrays on every iteration would cost about twice the transform itself. This keeps
    the solved coefficients instead, answers :meth:`power_at` for one index, and builds
    the public :class:`AmplitudeSpectrum` only for spectra that are kept
    (:meth:`to_spectrum`), with values identical to :meth:`SpectrumEngine.spectrum`.
    """

    def __init__(
        self,
        engine: SpectrumEngine,
        amplitude: FloatArray,
        a: FloatArray,
        b: FloatArray,
        yc: FloatArray,
        ys: FloatArray,
        yy: float,
        bad: IntArray,
    ) -> None:
        self.frequency = engine.frequency
        self.amplitude = amplitude
        self._a, self._b, self._yc, self._ys = a, b, yc, ys
        self._yy = yy
        self._bad = bad
        self._meta = (
            engine.t_ref,
            engine.backend,
            engine.normalization,
            engine.n_samples,
            engine.baseline,
        )

    def _power(self, sl: slice | None = None) -> FloatArray:
        a, b, yc, ys = (
            (self._a, self._b, self._yc, self._ys)
            if sl is None
            else (self._a[sl], self._b[sl], self._yc[sl], self._ys[sl])
        )
        if self._yy > 0.0:
            return np.clip((a * yc + b * ys) / self._yy, 0.0, 1.0)
        return np.zeros(a.shape, dtype=np.float64)

    def power_at(self, index: int) -> float:
        """The normalized power at one grid index (equal to ``to_spectrum().power``)."""
        i = int(index)
        if self._bad.size and np.any(self._bad == i):
            return 0.0
        return float(self._power(slice(i, i + 1))[0])

    def to_spectrum(self) -> AmplitudeSpectrum:
        """The full :class:`AmplitudeSpectrum` (phase and power on the whole grid)."""
        phase = np.mod(np.arctan2(self._a, self._b), 2.0 * np.pi)
        power = self._power()
        if self._bad.size:
            phase[self._bad] = 0.0
            power[self._bad] = 0.0
        t_ref, backend, normalization, n_samples, baseline = self._meta
        return AmplitudeSpectrum(
            frequency=self.frequency,
            amplitude=self.amplitude,
            phase=phase,
            power=power,
            t_ref=t_ref,
            backend=backend,
            normalization=normalization,
            n_samples=n_samples,
            baseline=baseline,
        )


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

    On the host the reduction is a matrix-vector product (one BLAS pass, one fewer
    ``(chunk, N)`` temporary). Device arrays keep the elementwise product and sum: on
    torch it was measured slower as a mat-vec (float32, or small ``N``) and the GPU
    variants have not been benchmarked.
    """
    dtype = tau.dtype
    dev = device_ref(tau)
    n_points = int(tau.shape[0])
    chunk = max(1, min(chunk, _DIRECT_CHUNK_ELEMS // max(1, n_points)))
    freqs = f0 + df * xp.arange(nf, dtype=dtype, device=dev)
    cos_sum = xp.empty(nf, dtype=dtype, device=dev)
    sin_sum = xp.empty(nf, dtype=dtype, device=dev)
    two_pi = 2.0 * float(np.pi)
    if isinstance(tau, np.ndarray):
        for start in range(0, nf, chunk):
            stop = min(start + chunk, nf)
            ang = (two_pi * freqs[start:stop])[:, None] * tau[None, :]
            cos_sum[start:stop] = np.cos(ang) @ strengths
            sin_sum[start:stop] = np.sin(ang) @ strengths
        return cos_sum, sin_sum
    row = strengths[None, :]
    for start in range(0, nf, chunk):
        stop = min(start + chunk, nf)
        ang = (two_pi * freqs[start:stop])[:, None] * tau[None, :]
        cos_sum[start:stop] = xp.sum(row * xp.cos(ang), axis=1)
        sin_sum[start:stop] = xp.sum(row * xp.sin(ang), axis=1)
    return cos_sum, sin_sum


def _wrap_phase(x: FloatArray) -> FloatArray:
    """Angles (float64) reduced to ``[-pi, pi)``."""
    return np.mod(x + np.pi, 2.0 * np.pi) - np.pi


def _direct_trig_sums_anchored(
    xp: Any,
    host_tau: FloatArray,
    to_dev: Callable[[FloatArray], Any],
    strengths: Any,
    f0: float,
    df: float,
    nf: int,
    chunk: int,
) -> tuple[Any, Any]:
    """:func:`_direct_trig_sums` for reduced (float32) working precision.

    Forming ``2*pi*f*tau`` in float32 loses the phase as ``f*T`` grows: the argument
    reaches ``pi*f*T`` radians with a 6e-8 relative step, so at ``f*T ~ 1e6`` (a
    multi-year curve at a few hundred cycles/day) the spectrum is off by ~15% and the
    grid itself quantises. Here the phases are reduced in float64 on the host instead —
    the start phase of every block of :data:`_PHASE_ANCHOR` frequencies and the
    per-step increment, both wrapped to ``[-pi, pi)`` — and only the short recurrence
    ``phi0 + k * delta`` runs at the working precision on the device.
    """
    dtype = strengths.dtype
    dev = device_ref(strengths)
    n_points = int(host_tau.shape[0])
    chunk = max(1, min(chunk, _PHASE_ANCHOR, _DIRECT_CHUNK_ELEMS // max(1, n_points)))
    two_pi = 2.0 * float(np.pi)
    delta = to_dev(_wrap_phase((two_pi * df) * host_tau))[None, :]
    steps = xp.arange(chunk, dtype=dtype, device=dev)[:, None]
    cos_sum = xp.empty(nf, dtype=dtype, device=dev)
    sin_sum = xp.empty(nf, dtype=dtype, device=dev)
    row = strengths[None, :]
    for start in range(0, nf, chunk):
        stop = min(start + chunk, nf)
        phi0 = to_dev(_wrap_phase((two_pi * (f0 + start * df)) * host_tau))
        ang = phi0[None, :] + steps[: stop - start] * delta
        cos_sum[start:stop] = xp.sum(row * xp.cos(ang), axis=1)
        sin_sum[start:stop] = xp.sum(row * xp.sin(ang), axis=1)
    return cos_sum, sin_sum


def _complex_to_host(out: Any) -> tuple[FloatArray, FloatArray]:
    """Split a (device or host) complex result into host cos/sin parts.

    A device array crosses to the host once, as one contiguous complex buffer, rather
    than as two strided real/imaginary views that each need a device-side copy first.
    On the host the parts stay views of that buffer: copying them out costs more (~5%
    of a spectrum evaluation) than their stride does in the arithmetic that follows.
    """
    if is_cupy_array(out):
        import cupy

        out = cupy.asnumpy(out)
    host = np.asarray(out, dtype=np.complex128)
    return host.real, host.imag


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
        Observation times in days (finite, at least three points spanning a non-zero
        baseline).
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

    Raises
    ------
    InsufficientDataError
        For fewer than three points.
    ValueError
        For a grid that is not uniform in frequency, a non-finite time, or an error
        that is not finite and positive (drop such points first, e.g. with
        :meth:`cuperiod.LightCurve.finite`).

    Notes
    -----
    One instance is not thread-safe (a cufinufft plan holds device state, and the
    normal-equation terms are built lazily on first use). Build one per worker.

    Where the sampling can barely tell a sinusoid from the constant — near multiples of
    the cadence and half-cadence of regular sampling, near 1, 2, ... c/d for
    single-site nightly sampling, and as ``f -> 0`` — the floating-mean normal
    equations are nearly singular and their least-squares amplitude is noise divided
    by a tiny eigenvalue: giant, spurious peaks. There the ``"lsq"`` spectrum reports
    the bounded power-equivalent amplitude instead, and zero where the design carries
    no information at all.
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
        # Validated before anything reaches a transform: a NaN or inf time does not
        # merely poison the sums, it crashes the finufft/cufinufft point sort outright.
        if not np.all(np.isfinite(t)):
            raise ValueError(
                "SpectrumEngine: time contains non-finite values; drop those points "
                "first (LightCurve.finite() does)"
            )
        if grid.kind != "frequency" or not grid.uniform:
            raise ValueError(
                "SpectrumEngine needs a uniform frequency grid; build one with "
                "cuperiod.uniform_frequency_grid(...)"
            )
        f0, df, nf = grid.uniform_frequency_params()
        if nf <= 0 or df <= 0.0:
            raise ValueError("SpectrumEngine needs a non-empty, increasing grid")

        if error is None:
            w = np.full(t.shape, 1.0 / t.size, dtype=np.float64)
        else:
            dy = np.ascontiguousarray(np.asarray(error, dtype=np.float64))
            if dy.shape != t.shape:
                raise ValueError("error and time must have the same length")
            if not np.all(np.isfinite(dy) & (dy > 0.0)):
                raise ValueError(
                    "SpectrumEngine: every error must be finite and positive; drop "
                    "the other points first (LightCurve.finite() does)"
                )
            w = 1.0 / (dy * dy)
            w /= w.sum()

        self.n_samples = int(t.size)
        self.baseline = float(t.max() - t.min())
        self.normalization: Normalization = normalization
        self.backend = resolve_spectrum_backend(backend)
        self.frequency: FloatArray = f0 + df * np.arange(nf, dtype=np.float64)
        self._f0, self._df, self._nf = f0, df, int(nf)
        self._eps = float(eps)
        self._freq_batch = int(freq_batch)
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
        self._window_sums: tuple[FloatArray, FloatArray] | None = None
        self._terms: _NormalTerms | None = None
        self._released = False
        self._setup(device, precision)
        # The weights sum to 1, so a well-conditioned frequency has det ~ 0.25 and
        # rounding noise ~eps. Anything far below that is a degenerate design (a zero
        # frequency, or sampling that cannot separate cosine from sine there) and is
        # reported as zero amplitude rather than as a huge spurious peak.
        self._det_floor = 1e-10 if self._precision == "float64" else 1e-4
        self._dead_floor = _DEAD_EIGENVALUE[self._precision]

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
        options: dict[str, int] = {}
        if (
            self.backend == "finufft"
            and self._tau.size + nf <= _FINUFFT_SINGLE_THREAD_WORK
        ):
            # The sums then differ from the threaded plan's at the NUFFT tolerance
            # only (<= 2e-11 relative measured), far inside the eps the engine asks.
            options["nthreads"] = 1
        plan = plan_factory(
            1, (nf,), n_trans=1, eps=self._eps, isign=1, dtype="complex128",
            **options,
        )
        plan.setpts(_wrapped_angles(xp, self._tau_dev, df))
        mod = xp.exp(2j * np.pi * (f0 + (nf // 2) * df) * self._tau_dev)
        return plan, mod

    def _plan_sums(
        self, plan: Any, mod: Any, strengths: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Execute ``plan`` for ``strengths`` and split the result into cos/sin sums."""
        # Real strengths times the complex128 modulation are already complex128.
        out = plan.execute(self._xp.asarray(strengths) * mod)
        out = out[0] if out.ndim == 2 else out
        return _complex_to_host(out)

    def _direct_sums(
        self, strengths: FloatArray, f0: float, df: float
    ) -> tuple[FloatArray, FloatArray]:
        """Cos/sin sums of ``strengths`` on ``f0 + k*df`` by the direct path (host)."""
        dev_strengths = self._to_dev(strengths)
        if self._precision == "float32":
            cos_sum, sin_sum = _direct_trig_sums_anchored(
                self._xp, self._tau, self._to_dev, dev_strengths,
                f0, df, self._nf, self._freq_batch,
            )
        else:
            cos_sum, sin_sum = _direct_trig_sums(
                self._xp, self._tau_dev, dev_strengths,
                f0, df, self._nf, self._freq_batch,
            )
        return to_host(cos_sum), to_host(sin_sum)

    def release(self, *, force: bool = False) -> None:
        """Drop the device-side state: the NUFFT plan and the device copy of the times.

        The host-side window and normal-equation terms are kept, and the next
        evaluation re-plans lazily, so an idle engine holds no GPU memory and a
        released one gives the same spectra (bitwise wherever re-planning is
        deterministic). On a CPU backend this is a no-op unless ``force`` is set.

        Parameters
        ----------
        force : bool, default False
            Release a CPU plan too (tests; freeing a large finufft plan).
        """
        on_device = self.backend == "cufinufft" or (
            self.backend.startswith("torch:") and self._device != "cpu"
        )
        if not (on_device or force) or self._released:
            return
        self._plan = None
        self._plan_mod = None
        self._tau_dev = None
        self._released = True

    def _ensure_ready(self) -> None:
        """Re-create what :meth:`release` dropped (no-op otherwise)."""
        if not self._released:
            return
        self._released = False
        if self.backend in {"finufft", "cufinufft"}:
            self._plan, self._plan_mod = self._make_plan(self._df, self._f0)
        elif self.backend == "numpy":
            self._tau_dev = self._tau
        else:
            self._tau_dev = to_device_array(
                self._tau, device=self._device, dtype=self._torch_dtype
            )
            self._xp = array_namespace(self._tau_dev)

    def _base_sums(self, strengths: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Cos/sin sums of ``strengths`` on the base grid, returned on the host."""
        self._ensure_ready()
        if self._plan is not None:
            return self._plan_sums(self._plan, self._plan_mod, strengths)
        return self._direct_sums(strengths, self._f0, self._df)

    def _doubled_sums(self, strengths: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Cos/sin sums of ``strengths`` on the doubled grid ``2*f_k`` (host)."""
        self._ensure_ready()
        if self._plan is None:
            return self._direct_sums(strengths, 2.0 * self._f0, 2.0 * self._df)
        # Re-point the base plan at the doubled angles rather than building a second
        # one (on a GPU that is a second device workspace per light curve), then put
        # its points back: the base sums afterwards are bit-identical.
        xp, plan, nf = self._xp, self._plan, self._nf
        f0, df = 2.0 * self._f0, 2.0 * self._df
        plan.setpts(_wrapped_angles(xp, self._tau_dev, df))
        try:
            mod = xp.exp(2j * np.pi * (f0 + (nf // 2) * df) * self._tau_dev)
            return self._plan_sums(plan, mod, strengths)
        finally:
            plan.setpts(_wrapped_angles(xp, self._tau_dev, self._df))

    def _to_dev(self, host: FloatArray) -> Any:
        """Move a host float64 array onto the compute device at the working dtype."""
        if self._torch_dtype is not None:
            return to_device_array(host, device=self._device, dtype=self._torch_dtype)
        return self._xp.asarray(host)

    def _window_terms(self) -> _NormalTerms:
        """The data-independent ``cc``, ``ss``, ``cs`` sums, ``1/det`` and conditioning.

        These are the Gram entries of the floating-mean design at every trial frequency
        — functions of the observation times and weights only — so the extraction loop
        pays for them exactly once (see :meth:`_normal_terms`).

        The smallest eigenvalue of ``[[cc, cs], [cs, ss]]`` measures how well the
        sampling separates a sinusoid at that frequency from the constant; its ideal is
        1/2 (the weights sum to 1). Where it falls below ``_dead_floor`` the frequency
        carries no information and ``inv`` is zeroed, so the amplitude reads 0; where it
        falls below :data:`_ILL_CONDITIONED` of the ideal the frequency is flagged
        ``ill`` and :meth:`spectrum` reports the bounded power-equivalent amplitude.
        """
        if self._window_sums is None:
            self._window_sums = self._base_sums(self._weight)
        c, s = self._window_sums  # exactly the spectral window; see :meth:`window`
        c2, s2 = self._doubled_sums(self._weight)
        cc = 0.5 * (1.0 + c2) - c * c
        ss = 0.5 * (1.0 - c2) - s * s
        cs = 0.5 * s2 - c * s
        det = cc * ss - cs * cs
        # det / lam_max rather than half - root: no cancellation if lam_min << lam_max.
        lam_max = 0.5 * (cc + ss) + np.sqrt(0.25 * (cc - ss) ** 2 + cs * cs)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam_min = np.where(lam_max > 0.0, det / lam_max, 0.0)
        safe = (det > self._det_floor) & (lam_min > self._dead_floor)
        inv = np.where(safe, 1.0 / np.where(safe, det, 1.0), 0.0)
        ill = safe & (lam_min < 0.5 * _ILL_CONDITIONED)
        return _NormalTerms(
            cc=cc, ss=ss, cs=cs, inv=inv, ill=ill, ill_index=np.flatnonzero(ill)
        )

    def _normal_terms(self) -> _NormalTerms:
        """The cached :meth:`_window_terms` (built on first use, then reused)."""
        if self._terms is None:
            self._terms = self._window_terms()
        return self._terms

    def _conditioning_at(self, frequency: float) -> float:
        """Conditioning of the floating-mean design at one frequency, ``O(N)``.

        The smallest eigenvalue of the centred cos/sin Gram matrix over its ideal 1/2,
        evaluated directly in float64 at an arbitrary (off-grid) frequency: 1 for a
        frequency the sampling resolves perfectly, 0 for one it cannot see at all.
        """
        theta = (2.0 * np.pi * float(frequency)) * self._tau
        w = self._weight
        c, s = float(np.dot(w, np.cos(theta))), float(np.dot(w, np.sin(theta)))
        c2, s2 = float(np.dot(w, np.cos(2.0 * theta))), float(
            np.dot(w, np.sin(2.0 * theta))
        )
        cc = 0.5 * (1.0 + c2) - c * c
        ss = 0.5 * (1.0 - c2) - s * s
        cs = 0.5 * s2 - c * s
        lam_max = 0.5 * (cc + ss) + float(np.sqrt(0.25 * (cc - ss) ** 2 + cs * cs))
        if not lam_max > 0.0:
            return 0.0
        return max((cc * ss - cs * cs) / lam_max, 0.0) / 0.5

    # -- evaluation --------------------------------------------------------------
    def _scan(self, values: FloatArray) -> _SpectrumScan:
        """Amplitude-only evaluation of ``values`` (see :class:`_SpectrumScan`)."""
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
        if not np.isfinite(yy):
            raise ValueError(
                "SpectrumEngine: values contain non-finite entries; drop those points "
                "first (LightCurve.finite() does)"
            )
        yc, ys = self._base_sums(w * centred)

        if self.normalization == "dft":
            a = 2.0 * yc
            b = 2.0 * ys
            amplitude = np.hypot(a, b)
        else:
            terms = self._normal_terms()
            inv = terms.inv
            a = (terms.ss * yc - terms.cs * ys) * inv
            b = (terms.cc * ys - terms.cs * yc) * inv
            amplitude = np.hypot(a, b)
            index = terms.ill_index
            if index.size:
                # The power is bounded however ill-conditioned the solve; the amplitude
                # of the sinusoid explaining that much variance, sqrt(2 * power * yy),
                # equals the least-squares one where the design is well conditioned.
                explained = a[index] * yc[index] + b[index] * ys[index]
                amplitude[index] = np.sqrt(2.0 * np.clip(explained, 0.0, max(yy, 0.0)))
        finite = np.isfinite(amplitude)
        bad = np.flatnonzero(~finite)
        if bad.size:
            amplitude[bad] = 0.0
        return _SpectrumScan(self, amplitude, a, b, yc, ys, yy, bad)

    def spectrum(self, values: FloatArray) -> AmplitudeSpectrum:
        """Amplitude spectrum of ``values`` on this engine's sampling and grid.

        Parameters
        ----------
        values : numpy.ndarray
            Brightness values aligned with the ``time`` this engine was built from —
            typically the current pre-whitening residuals. Must be finite.

        Returns
        -------
        AmplitudeSpectrum

        Raises
        ------
        ValueError
            If ``values`` does not match the sampling or holds a non-finite entry.
        """
        return self._scan(values).to_spectrum()

    def window(self) -> AmplitudeSpectrum:
        """The spectral window ``|W(f)|`` of this sampling, on the engine's grid.

        The window function :math:`W(f) = \\sum_j w_j e^{2\\pi i f \\Delta t_j}`
        is what convolves the true spectrum when the sampling is irregular: every real
        peak is dressed with the window's sidelobes, so a candidate frequency sitting
        where a stronger component's window has a lobe (classically at ±1 cycle/day for
        single-site ground-based data) is suspect. Period04 displays it for exactly
        this reason, and comparing a doubtful peak against the window is the standard
        alias check before believing a close pair.

        The returned object reuses :class:`AmplitudeSpectrum`: ``amplitude`` is
        ``|W(f)|`` (dimensionless, ``|W| <= 1``, and ``|W| -> 1`` as ``f -> 0``),
        ``phase`` is ``arg W(f)``, and ``power`` is ``|W(f)|^2``. Costs one transform
        on first call, then nothing: the sums are shared with the cached normal
        equations, so after a :meth:`spectrum` call (or before one) neither pays for
        them twice — and an engine that only ever serves the window never computes
        the doubled-grid normal-equation terms at all.

        Returns
        -------
        AmplitudeSpectrum
        """
        if self._window_sums is None:
            self._window_sums = self._base_sums(self._weight)
        c, s = self._window_sums
        amplitude = np.hypot(c, s)
        return AmplitudeSpectrum(
            frequency=self.frequency,
            amplitude=amplitude,
            phase=np.mod(np.arctan2(s, c), 2.0 * np.pi),
            power=amplitude * amplitude,
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


def spectral_window(
    time: FloatArray,
    error: FloatArray | None = None,
    *,
    grid: GridSpec,
    backend: str = "auto",
    device: str = "auto",
    precision: str = "auto",
    eps: float = DEFAULT_EPS,
    freq_batch: int = 4096,
    t_ref: float | None = None,
) -> AmplitudeSpectrum:
    """One-shot spectral window of a sampling (see :meth:`SpectrumEngine.window`).

    The window depends only on the observation times and weights — no brightness
    values enter — so this is the diagnostic to run when deciding whether a peak in an
    amplitude spectrum is real or an alias of a stronger one.

    Parameters
    ----------
    time : numpy.ndarray
        Observation times in days (finite points only).
    error : numpy.ndarray, optional
        1-sigma uncertainties; the window is weighted exactly as the spectrum is.
    grid : GridSpec
        A uniform frequency grid.
    backend, device, precision, eps, freq_batch, t_ref
        See :class:`SpectrumEngine`.

    Returns
    -------
    AmplitudeSpectrum
        ``amplitude`` holds ``|W(f)| <= 1``.

    Examples
    --------
    >>> grid = cup.uniform_frequency_grid(t.max() - t.min(),    # doctest: +SKIP
    ...                                   maximum_frequency=5.0)
    >>> window = cup.spectral_window(t, dy, grid=grid)          # doctest: +SKIP
    """
    engine = SpectrumEngine(
        time,
        error,
        grid=grid,
        backend=backend,
        device=device,
        precision=precision,
        eps=eps,
        freq_batch=freq_batch,
        t_ref=t_ref,
    )
    return engine.window()


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
    "spectral_window",
    "weighted_epoch",
]
