"""Frequency / period grid construction.

A :class:`GridSpec` is the trial grid a periodogram is evaluated on. Most Fourier
methods (GLS, MHAOV) want a *uniform frequency* grid; box/fold methods (BLS, TLS,
PDM, CE, string-length) want a *period* grid, often log-spaced. :class:`GridSpec`
stores whichever is natural and exposes both views (``frequency`` / ``period``).

The default GLS frequency grid follows the standard Lomb-Scargle heuristic (Rayleigh
resolution ``1/baseline`` divided by an oversampling factor), identical to astropy's
``autofrequency`` so power spectra line up sample-for-sample with the reference.

Grid size policy
----------------
The size of an automatic grid is ``f_max * samples_per_peak * baseline``, and both
factors come from the data, so one bad timestamp or one unusual cadence can ask for
billions of samples. Three guards keep that bounded without touching the grids of
ordinary sampling:

* :func:`pseudo_nyquist_frequency` recognises *multi-exposure visits* (dithers,
  filter sequences, stacked near-simultaneous bands), whose seconds-long spacing would
  otherwise set ``f_max`` to thousands of cycles/day, and caps the automatic ceiling
  at :data:`MULTI_EXPOSURE_RAYLEIGH_BUDGET` resolution elements (scaled up by a
  ``nyquist_factor`` above 5).
* Every :func:`uniform_frequency_grid` call is refused with a
  :class:`GridTooLargeError` above :data:`MAX_GRID_SIZE` samples, before allocating.
  This is the only refusal.
* :func:`check_grid_size` (used by :func:`default_frequency_grid`) additionally warns
  about automatic grids above :data:`AUTO_GRID_WARN_SIZE` samples, and says so when
  the baseline is inflated by a few outlying timestamps (mis-converted times, or
  genuine archival epochs).

:func:`regular_cadence_nyquist` detects strictly regular (lattice) sampling, whose
spectrum above the true Nyquist frequency ``0.5/cadence`` is a mirror image of the
band below it.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, NoReturn

import numpy as np

from cuperiod.core._typing import FloatArray
from cuperiod.core.errors import CuPeriodError, InsufficientDataError

GridKind = Literal["frequency", "period"]

#: Hard ceiling on the number of samples :func:`uniform_frequency_grid` allocates
#: (2 GB per float64 array). Legitimate grids stay well below it (four years of
#: Kepler 1-minute data at the default oversampling is ~5e7); grids above it come
#: from a corrupted baseline or a mistyped bound. Raise it (module attribute, or the
#: ``max_size=`` argument) for a deliberate larger grid.
MAX_GRID_SIZE: int = 250_000_000

#: Automatic grids larger than this emit a :class:`GridSizeWarning` (with an
#: outlying-timestamp diagnosis when a few epochs dominate the baseline) and are
#: still built; only :data:`MAX_GRID_SIZE` refuses a grid.
AUTO_GRID_WARN_SIZE: int = 5_000_000

#: Resolution budget, in Rayleigh widths ``1/baseline``, of the automatic frequency
#: ceiling on a multi-exposure cadence (see :func:`pseudo_nyquist_frequency`): at the
#: default 5 samples per peak, a grid of at most 500,000 samples. A
#: ``nyquist_factor`` above 5 scales it by ``nyquist_factor / 5``.
MULTI_EXPOSURE_RAYLEIGH_BUDGET: Final = 100_000.0

#: A gap longer than this multiple of the median gap ends a visit.
_VISIT_GAP_RATIO: Final = 10.0

#: Visits of at most this many samples are back-to-back exposures, not a time series.
_MAX_EXPOSURES_PER_VISIT: Final = 10

#: A baseline longer than this multiple of the central 98% span is outlier-dominated.
_BASELINE_OUTLIER_RATIO: Final = 10.0

#: Lattice detection: fraction of gaps that must be integer multiples of the cadence.
_LATTICE_MIN_FRACTION: Final = 0.9

#: Lattice detection: the cadence may be up to this many times the median gap's
#: divisor, so a lattice with most samples dropped is still recognised.
_LATTICE_MAX_DIVISOR: Final = 10

#: Grids are validated in chunks of this many samples to bound temporary memory.
_CHECK_CHUNK: Final = 1 << 20


class GridTooLargeError(CuPeriodError, ValueError):
    """A trial grid would exceed the size limit; nothing was allocated.

    The message names the requested size, its memory footprint, the baseline and
    frequency ceiling that produced it, and the remedies (an explicit
    ``maximum_frequency``, a smaller ``samples_per_peak``, cleaning outlying
    timestamps, or a deliberately raised limit).
    """


class GridSizeWarning(UserWarning):
    """An automatically chosen trial grid is unusually large (still computed)."""


@dataclass(frozen=True, eq=False)
class GridSpec:
    """A trial grid in either the frequency or period domain.

    Grids compare and hash by identity: two grids holding equal arrays are distinct
    objects (comparing the arrays element-wise has no single truth value).

    Parameters
    ----------
    kind : {"frequency", "period"}
        Which quantity ``values`` holds.
    values : numpy.ndarray
        Finite, ascending grid samples (cycles/day for frequency, ``>= 0``; days for
        period, ``> 0``). Repeated values are allowed, and so are rounding-level
        steps back (at most ``1e-9 * max|values|``, as where two concatenated
        ``np.arange`` segments meet).
    uniform : bool, default False
        True if ``values`` is an arithmetic progression. Uniform frequency grids let
        the NUFFT GLS backend skip re-deriving ``(f0, df, nf)`` -- it rebuilds the grid
        from the first two samples, so the flag is checked rather than trusted.
    meta : Mapping, optional
        Method-specific grid parameters (e.g. BLS duration fractions / segments).

    Raises
    ------
    ValueError
        On an unknown ``kind``; non-1-D, non-finite, descending or out-of-domain
        values; or ``uniform=True`` on values that are not an arithmetic progression
        (or on a period grid).
    """

    kind: GridKind
    values: FloatArray
    uniform: bool = False
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("frequency", "period"):
            raise ValueError(
                f"grid kind must be 'frequency' or 'period', got {self.kind!r}"
            )
        values = np.ascontiguousarray(np.asarray(self.values, dtype=np.float64))
        if values.ndim != 1:
            raise ValueError(f"grid values must be 1-D, got shape {values.shape}")
        _validate_grid_values(values, self.kind, bool(self.uniform))
        object.__setattr__(self, "values", values)

    @property
    def size(self) -> int:
        """Number of grid samples."""
        return int(self.values.size)

    @property
    def frequency(self) -> FloatArray:
        """The grid as frequencies (cycles/day), ascending."""
        if self.kind == "frequency":
            return self.values
        return _reciprocal_reversed(self.values)

    @property
    def period(self) -> FloatArray:
        """The grid as periods (days), ascending."""
        if self.kind == "period":
            return self.values
        return _reciprocal_reversed(self.values)

    def uniform_frequency_params(self) -> tuple[float, float, int]:
        """Return ``(f0, df, nf)`` for a uniform frequency grid.

        Raises
        ------
        ValueError
            If this is not a frequency grid.
        """
        if self.kind != "frequency":
            raise ValueError("uniform_frequency_params requires a frequency grid")
        freq = self.values
        nf = int(freq.size)
        if nf == 0:
            return 0.0, 0.0, 0
        f0 = float(freq[0])
        df = float(freq[1] - freq[0]) if nf > 1 else 0.0
        return f0, df, nf


def _reciprocal_reversed(values: FloatArray) -> FloatArray:
    """``1/values`` in reverse order, as a fresh array with positive strides.

    Reversing *before* dividing gives the same numbers as ``(1/values)[::-1]`` but a
    new contiguous array: the reversed view of a one-sample array keeps a negative
    stride while claiming to be C-contiguous, so ``np.ascontiguousarray`` passes it
    through and ``torch.as_tensor`` then rejects it.
    """
    with np.errstate(divide="ignore"):
        return 1.0 / values[::-1]


def _validate_grid_values(values: FloatArray, kind: str, uniform: bool) -> None:
    """Check finiteness, order, domain and (if flagged) uniformity, chunk by chunk."""
    n = int(values.size)
    if n == 0:
        return
    first, last = float(values[0]), float(values[-1])
    if not (math.isfinite(first) and math.isfinite(last)):
        raise ValueError("grid values must be finite")
    _check_domain(first, kind)
    if n == 1:
        return
    check_uniform = uniform
    if check_uniform and kind != "frequency":
        raise ValueError(
            "uniform=True is only meaningful for a frequency grid; pass uniform=False"
        )
    df = float(values[1] - values[0])
    if check_uniform and not df > 0.0:
        raise ValueError("uniform=True needs a strictly increasing frequency grid")
    scale = max(abs(first), abs(last))
    # Rounding of f0 + df*k is ~eps*|f| per sample, independent of df, so a narrow
    # window at a high frequency needs the eps*max|f| term as well as the 1e-6 of df.
    tolerance = 1e-6 * abs(df) + 8.0 * np.finfo(np.float64).eps * scale
    # Segment-wise builders (BLS concatenates one np.arange per log period segment)
    # overlap by a rounding quantum at a join: arange may overshoot its stop, so the
    # first period of a segment can sit ~1e-11 (relative) below the last one of the
    # previous segment -- about 200 eps of max|values| at worst, independent of the
    # baseline. That is not a reversed grid, so steps back of up to 1e-9 of the
    # grid's scale are accepted; a genuine reversal is many orders larger.
    reversal = 1e-9 * scale
    if not last >= first - reversal:
        # A dense descending grid can have every step within ``reversal``.
        _raise_not_ascending(f"values[-1]={last!r} < values[0]={first!r}")
    for start in range(0, n - 1, _CHECK_CHUNK):
        chunk = values[start : start + _CHECK_CHUNK + 1]
        step = np.diff(chunk)
        smallest = float(step.min())  # NaN propagates, and fails the test below
        if not smallest >= -reversal:
            if not bool(np.all(np.isfinite(chunk))):
                raise ValueError("grid values must be finite")
            i = start + int(np.argmin(step))
            _raise_not_ascending(
                f"values[{i}]={float(values[i])!r} > "
                f"values[{i + 1}]={float(values[i + 1])!r}"
            )
        if smallest < 0.0:
            # A tolerated step back means values[0] need not be the minimum.
            _check_domain(float(chunk.min()), kind)
        if check_uniform:
            deviation = max(float(step.max()) - df, df - smallest)
            if deviation > tolerance:
                raise ValueError(
                    "uniform=True but the values are not an arithmetic progression "
                    f"(a step differs from values[1]-values[0]={df:.6g} by "
                    f"{deviation:.3g}); pass uniform=False"
                )


def _check_domain(minimum: float, kind: str) -> None:
    if kind == "frequency" and minimum < 0.0:
        raise ValueError(f"grid frequencies must be >= 0, got minimum {minimum}")
    if kind == "period" and minimum <= 0.0:
        raise ValueError(f"grid periods must be > 0, got minimum {minimum}")


def _raise_not_ascending(detail: str) -> NoReturn:
    raise ValueError(
        f"grid values must be ascending ({detail}); sort them (e.g. np.sort), or "
        "build a period grid as kind='period' instead of reversing 1/frequency"
    )


def _as_finite(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value}")
    return out


def uniform_grid_size(
    baseline: float,
    *,
    maximum_frequency: float,
    minimum_frequency: float | None = None,
    samples_per_peak: float = 5,
) -> int:
    """Number of samples :func:`uniform_frequency_grid` would build (no allocation).

    Validates the arguments exactly as :func:`uniform_frequency_grid` does, so a
    caller (a GUI hint, another method's default grid) can size a grid up front.

    Parameters
    ----------
    baseline, maximum_frequency, minimum_frequency, samples_per_peak
        As for :func:`uniform_frequency_grid`.

    Returns
    -------
    int
        The grid size ``ceil((f_max - f0) * samples_per_peak * baseline) + 1``.

    Raises
    ------
    ValueError
        On a non-finite or non-positive baseline / ``samples_per_peak``, a
        non-finite ``maximum_frequency``, a negative or non-finite
        ``minimum_frequency``, or ``minimum >= maximum``.
    """
    return _uniform_params(baseline, maximum_frequency, minimum_frequency,
                           samples_per_peak)[2]


def _uniform_params(
    baseline: float,
    maximum_frequency: float,
    minimum_frequency: float | None,
    samples_per_peak: float,
) -> tuple[float, float, int]:
    """Validated ``(f0, df, nf)`` of a uniform grid (astropy autofrequency rules)."""
    baseline = float(baseline)
    if not (math.isfinite(baseline) and baseline > 0.0):
        raise ValueError(f"baseline must be positive and finite, got {baseline}")
    spp = float(samples_per_peak)
    if not (math.isfinite(spp) and spp > 0.0):
        raise ValueError(
            f"samples_per_peak must be a positive finite number, got {samples_per_peak}"
        )
    f_max = _as_finite("maximum_frequency", maximum_frequency)
    df = 1.0 / (spp * baseline)
    if minimum_frequency is None:
        f0 = df
    else:
        f0 = _as_finite("minimum_frequency", minimum_frequency)
        if f0 < 0.0:
            raise ValueError(f"minimum_frequency must be >= 0, got {minimum_frequency}")
    if f0 >= f_max:
        raise ValueError(
            f"minimum frequency ({f0}) must be < "
            f"maximum_frequency ({maximum_frequency})"
        )
    nf = int(np.ceil((f_max - f0) / df)) + 1
    return f0, df, max(nf, 1)


def uniform_frequency_grid(
    baseline: float,
    *,
    maximum_frequency: float,
    minimum_frequency: float | None = None,
    samples_per_peak: float = 5,
    max_size: int | None = None,
) -> GridSpec:
    """Build a uniform frequency grid, matching astropy ``autofrequency``.

    Parameters
    ----------
    baseline : float
        Time span ``max(t) - min(t)`` in days (positive, finite).
    maximum_frequency : float
        Highest trial frequency (cycles/day, finite).
    minimum_frequency : float, optional
        Lowest trial frequency (``>= 0``). Defaults to one Rayleigh step
        ``1/(s*baseline)``.
    samples_per_peak : float, default 5
        Oversampling factor; the frequency step is ``1/(samples_per_peak*baseline)``.
    max_size : int, optional
        Refuse grids with more samples than this. Defaults to :data:`MAX_GRID_SIZE`
        (read at call time, so raising the module attribute raises it everywhere).

    Returns
    -------
    GridSpec
        A uniform frequency grid.

    Raises
    ------
    GridTooLargeError
        If the grid would exceed ``max_size`` samples (checked before allocating).
    ValueError
        On invalid arguments (see :func:`uniform_grid_size`).
    """
    f0, df, nf = _uniform_params(
        baseline, maximum_frequency, minimum_frequency, samples_per_peak
    )
    check_grid_size(
        nf,
        limit=max_size,
        baseline=float(baseline),
        maximum_frequency=float(maximum_frequency),
        context="uniform_frequency_grid",
    )
    freq = f0 + df * np.arange(nf, dtype=np.float64)
    return GridSpec(kind="frequency", values=freq, uniform=True)


def _baseline_outlier_ratio(time: FloatArray) -> tuple[float, float, float]:
    """``(baseline, central span, ratio)`` with the 1% extreme samples trimmed.

    Trimming by *rank* (at least one sample per side) rather than by interpolated
    percentile, so a single sentinel timestamp in a few dozen points is caught.
    """
    t = np.sort(np.asarray(time, dtype=np.float64))
    t = t[np.isfinite(t)]
    n = int(t.size)
    if n < 4:
        return (float(t[-1] - t[0]) if n else 0.0), 0.0, 1.0
    k = max(1, math.ceil(0.01 * n))
    if n - 2 * k < 2:
        return float(t[-1] - t[0]), 0.0, 1.0
    baseline = float(t[-1] - t[0])
    central = float(t[n - 1 - k] - t[k])
    if central <= 0.0:
        return baseline, central, (math.inf if baseline > 0.0 else 1.0)
    return baseline, central, baseline / central


def check_grid_size(
    n_points: int,
    *,
    limit: int | None = None,
    automatic: bool = False,
    time: FloatArray | None = None,
    baseline: float | None = None,
    maximum_frequency: float | None = None,
    context: str = "trial grid",
) -> None:
    """Refuse (or warn about) a trial grid of ``n_points`` samples before allocating.

    Every grid above ``limit`` (default :data:`MAX_GRID_SIZE`) raises; that is the
    only refusal. An *automatic* grid -- one whose ceiling came from the data rather
    than the user -- above :data:`AUTO_GRID_WARN_SIZE` emits a
    :class:`GridSizeWarning` and is still built. When ``time`` shows the baseline is
    dominated by a few outlying timestamps, the warning (or error) says so: that is
    the usual cause of an oversized grid when the timestamps are wrong (a sentinel
    ``0``, a JD/MJD mix-up), but it is also what genuine archival epochs combined
    with a new campaign look like, so it is a diagnosis rather than a refusal.

    Parameters
    ----------
    n_points : int
        Number of trial frequencies / periods the caller is about to allocate (for
        segmented grids, the total over segments).
    limit : int, optional
        Hard ceiling; defaults to :data:`MAX_GRID_SIZE`.
    automatic : bool, default False
        Whether the grid's extent was chosen automatically.
    time : numpy.ndarray, optional
        Observation times, used to diagnose an outlier-inflated baseline.
    baseline, maximum_frequency : float, optional
        Quoted in the message when given.
    context : str, default "trial grid"
        What is being built (e.g. ``"PDM default grid"``), for the message.

    Raises
    ------
    GridTooLargeError
        When the grid exceeds ``limit``.

    Warns
    -----
    GridSizeWarning
        For an automatic grid above :data:`AUTO_GRID_WARN_SIZE` (but within
        ``limit``).
    """
    hard = MAX_GRID_SIZE if limit is None else int(limit)
    n_points = int(n_points)
    over_hard = n_points > hard
    over_soft = automatic and n_points > AUTO_GRID_WARN_SIZE
    if not (over_hard or over_soft):
        return
    outlier = ""
    if time is not None and np.size(time) > 0:
        span, central, ratio = _baseline_outlier_ratio(time)
        baseline = span if baseline is None else baseline
        if ratio > _BASELINE_OUTLIER_RATIO:
            outlier = (
                f" The baseline ({span:.6g} d) is {ratio:.3g}x the span of the central "
                f"98% of the samples ({central:.6g} d), so a few outlying epochs set "
                "the grid size. Genuine archival epochs combined with new data are "
                "legitimate; otherwise check for mis-converted timestamps (a 0 or "
                "sentinel time, a JD/MJD mix-up)."
            )
    gib = n_points * 8.0 / 2.0**30
    origin = ""
    if maximum_frequency is not None:
        origin += f" up to {maximum_frequency:.6g} cycles/day"
    if baseline is not None:
        origin += f" over a {baseline:.6g}-day baseline"
    remedy = (
        " Set maximum_frequency (and/or a smaller samples_per_peak) or pass an "
        "explicit grid="
    )
    if over_hard:
        raise GridTooLargeError(
            f"{context}: {n_points:,} trial samples{origin} ({gib:.3g} GiB per "
            f"float64 array) exceeds the limit of {hard:,}.{outlier}{remedy}; for a "
            "deliberately larger grid raise max_size= or "
            "cuperiod.core.grid.MAX_GRID_SIZE."
        )
    message = (
        f"{context}: the automatic grid has {n_points:,} trial samples{origin} "
        f"({gib:.3g} GiB per float64 array).{outlier}{remedy} to bound it. "
        "Computing it anyway."
    )
    warnings.warn(GridSizeWarning(message), stacklevel=3)


def pseudo_nyquist_frequency(time: FloatArray, nyquist_factor: float = 5) -> float:
    """A pseudo-Nyquist maximum frequency from the median sampling interval.

    Irregular sampling has no true Nyquist limit; this returns
    ``nyquist_factor * 0.5 / median(diff(sort(time)))`` as a practical upper bound for
    the trial-frequency grid of fold-based methods (PDM, CE, string-length).

    **Multi-exposure visits.** When most consecutive gaps fall inside short *visits*
    -- runs of at most ten samples, split wherever a gap exceeds ten times the median
    gap: dithered triplets, filter sequences, near-simultaneous bands stacked by a
    multi-band fit -- the median gap measures the exposure spacing (seconds) rather
    than the cadence, and the ceiling would reach thousands of cycles/day, i.e. grids
    of 1e7-1e9 samples on multi-year baselines. For such cadences the ceiling is
    limited to ``max(B / baseline, f_visit)`` with the budget
    ``B = MULTI_EXPOSURE_RAYLEIGH_BUDGET * max(1, nyquist_factor / 5)`` Rayleigh
    widths, where ``f_visit`` is the same estimate computed from visit start times.
    At the default factor that is ~55 cycles/day on a five-year baseline and ~27 on
    a ten-year survey such as LSST: RR Lyrae, Cepheids and most delta Scuti stars
    stay in band, the fastest delta Scuti modes may not. The resulting grid is a
    prefix of the uncapped one, so any signal below the new ceiling is evaluated at
    exactly the same frequencies; an explicit ``maximum_frequency`` (or a
    ``nyquist_factor`` above 5, which widens the budget in proportion) searches
    higher. Ordinary sampling (single exposures, continuous space photometry, dense
    nightly runs) is unaffected, and so is any cadence whose classic ceiling stays
    within the budget.

    Parameters
    ----------
    time : numpy.ndarray
        Observation times (days); non-finite entries are ignored.
    nyquist_factor : float, default 5
        Multiple of the median-sampling Nyquist frequency to allow.

    Returns
    -------
    float
        A maximum trial frequency in cycles/day (``nyquist_factor`` when there are
        no two distinct times).
    """
    t = np.sort(np.asarray(time, dtype=np.float64))
    t = t[np.isfinite(t)]
    gaps = np.diff(t)
    dt = gaps[gaps > 0.0]
    if dt.size == 0:
        return float(nyquist_factor)
    median_gap = float(np.median(dt))
    classic = float(nyquist_factor) * 0.5 / median_gap
    baseline = float(t[-1] - t[0])
    # A factor above the default asks for a wider search, so it widens the budget in
    # proportion; at or below the default (WWZ uses 1) the budget is the constant.
    budget = MULTI_EXPOSURE_RAYLEIGH_BUDGET * max(1.0, float(nyquist_factor) / 5.0)
    if classic * baseline <= budget:
        return classic
    visit_cap = _multi_exposure_ceiling(t, gaps, dt.size, median_gap, nyquist_factor)
    if visit_cap is None:
        return classic
    return min(classic, max(budget / baseline, visit_cap))


def _multi_exposure_ceiling(
    t: FloatArray,
    gaps: FloatArray,
    n_positive: int,
    median_gap: float,
    nyquist_factor: float,
) -> float | None:
    """Visit-level pseudo-Nyquist when the median gap is a multi-exposure spacing.

    Returns None unless at least half of the positive gaps lie inside visits of at
    most :data:`_MAX_EXPOSURES_PER_VISIT` samples (so the median measures exposures,
    not the cadence). Otherwise returns ``nyquist_factor * 0.5 / median(visit gap)``
    over the visits' start times (0.0 when there is a single visit).
    """
    breaks = gaps > _VISIT_GAP_RATIO * median_gap
    visit = np.concatenate([[0], np.cumsum(breaks)])
    sizes = np.bincount(visit)
    small = sizes[visit[:-1]] <= _MAX_EXPOSURES_PER_VISIT
    in_small_visit = (~breaks) & (gaps > 0.0) & small
    if int(np.count_nonzero(in_small_visit)) * 2 < n_positive:
        return None
    starts = t[np.concatenate([[True], breaks])]
    visit_gaps = np.diff(starts)
    visit_gaps = visit_gaps[visit_gaps > 0.0]
    if visit_gaps.size == 0:
        return 0.0
    return float(nyquist_factor) * 0.5 / float(np.median(visit_gaps))


def regular_cadence_nyquist(
    time: FloatArray,
    *,
    tolerance: float = 0.01,
    min_coherence: float = 0.9,
) -> float | None:
    """The true Nyquist frequency ``0.5/cadence`` of strictly regular sampling.

    Sampling on a lattice ``t_j = t_0 + n_j * cadence`` (gaps and dropped cadences
    allowed) has a spectral window equal to one at ``1/cadence``, so every spectrum
    above ``0.5/cadence`` is a mirror image of the band below it and the images of a
    real signal have the same height as the signal. Default grids that run past that
    frequency then report a mirror alias about as often as the truth.

    The test is two-staged. First, at least 90% of the positive gaps must lie within
    ``tolerance`` (a fraction of the cadence) of an integer multiple of the cadence,
    which is taken as the median gap divided by the smallest integer (up to 10) that
    makes this true -- so a lattice with most samples dropped is still recognised.
    Second, the *global* coherence ``|mean(exp(2 pi i r_j))|`` of the residuals
    ``r_j`` (in cadences) from a least-squares lattice fit must reach
    ``min_coherence``. This is ``|W(1/cadence)|``, the height of the mirror image
    relative to the signal: a smooth barycentric timing term over a few months keeps
    it near 1, while lattices whose phase restarts every night, or multi-year
    barycentric drift, decorrelate it -- there super-Nyquist frequencies remain
    distinguishable and no cap should be applied.

    Parameters
    ----------
    time : numpy.ndarray
        Observation times (days); non-finite entries are ignored.
    tolerance : float, default 0.01
        Per-gap tolerance, as a fraction of the cadence.
    min_coherence : float, default 0.9
        Minimum lattice coherence (``0`` keeps only the gap test).

    Returns
    -------
    float or None
        ``0.5/cadence`` in cycles/day, or None when the sampling is not a lattice
        (or has fewer than eight gaps).
    """
    t = np.sort(np.asarray(time, dtype=np.float64))
    t = t[np.isfinite(t)]
    gaps = np.diff(t)
    dt = gaps[gaps > 0.0]
    if dt.size < 8:
        return None
    median_gap = float(np.median(dt))
    cadence = 0.0
    for divisor in range(1, _LATTICE_MAX_DIVISOR + 1):
        trial = median_gap / divisor
        ratio = dt / trial
        on_lattice = np.abs(ratio - np.rint(ratio)) <= tolerance
        if float(np.mean(on_lattice)) >= _LATTICE_MIN_FRACTION:
            cadence = trial
            break
    if cadence <= 0.0:
        return None
    index = np.concatenate([[0.0], np.cumsum(np.rint(gaps / cadence))])
    offset = t - t[0]
    index_mean = float(np.mean(index))
    centred = index - index_mean
    denom = float(np.dot(centred, centred))
    if denom <= 0.0:
        return None
    step = float(np.dot(centred, offset - float(np.mean(offset)))) / denom
    if not step > 0.0:
        return None
    intercept = float(np.mean(offset)) - step * index_mean
    residual = (offset - intercept - step * index) / step
    coherence = float(np.abs(np.mean(np.exp(2j * np.pi * residual))))
    if coherence < min_coherence:
        return None
    return 0.5 / step


def default_frequency_grid(
    time: FloatArray,
    *,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
    samples_per_peak: float = 5,
    nyquist_factor: float = 5,
    maximum_frequency_floor: float = 0.0,
    stop_at_nyquist: bool = False,
    context: str = "default frequency grid",
) -> GridSpec:
    """The automatic uniform frequency grid for observation times ``time``.

    The shared recipe of the fold/AOV methods and pre-whitening: from ``1/T`` (or
    ``minimum_frequency``) to ``maximum_frequency``, or, when that is None, to
    ``max(pseudo_nyquist_frequency(time, nyquist_factor), maximum_frequency_floor)``
    -- further limited to the true Nyquist frequency of a strictly regular cadence
    when ``stop_at_nyquist`` is set -- at ``samples_per_peak`` samples per Rayleigh
    width. The size is checked before allocation (:func:`check_grid_size`), with the
    automatic-grid warning only when the ceiling was automatic.

    Parameters
    ----------
    time : numpy.ndarray
        Observation times (days); non-finite entries are ignored.
    minimum_frequency, maximum_frequency : float, optional
        Explicit band edges (cycles/day); None means automatic.
    samples_per_peak : float, default 5
        Oversampling factor.
    nyquist_factor : float, default 5
        Passed to :func:`pseudo_nyquist_frequency` for the automatic ceiling.
    maximum_frequency_floor : float, default 0.0
        Lower bound on the automatic ceiling (e.g. 50 c/d for pre-whitening).
    stop_at_nyquist : bool, default False
        Cap the automatic ceiling at :func:`regular_cadence_nyquist` when the
        sampling is a lattice.
    context : str
        Label used in size warnings / errors (e.g. ``"PDM default grid"``).

    Returns
    -------
    GridSpec
        A uniform frequency grid, identical to ``uniform_frequency_grid(T,
        maximum_frequency=..., minimum_frequency=1/T, ...)``.

    Raises
    ------
    InsufficientDataError
        If the times span no positive baseline.
    GridTooLargeError
        If the grid would exceed :data:`MAX_GRID_SIZE` samples.

    Warns
    -----
    GridSizeWarning
        If an automatic grid exceeds :data:`AUTO_GRID_WARN_SIZE` samples.
    """
    t = np.asarray(time, dtype=np.float64)
    t = t[np.isfinite(t)]
    baseline = float(t.max() - t.min()) if t.size else 0.0
    if baseline <= 0.0:
        raise InsufficientDataError(f"{context}: no usable time baseline")
    automatic = maximum_frequency is None
    if maximum_frequency is None:
        f_max = max(
            pseudo_nyquist_frequency(t, nyquist_factor), maximum_frequency_floor
        )
        if stop_at_nyquist:
            nyquist = regular_cadence_nyquist(t)
            if nyquist is not None:
                f_max = min(f_max, nyquist)
    else:
        f_max = float(maximum_frequency)
    f_min = 1.0 / baseline if minimum_frequency is None else float(minimum_frequency)
    n_points = uniform_grid_size(
        baseline,
        maximum_frequency=f_max,
        minimum_frequency=f_min,
        samples_per_peak=samples_per_peak,
    )
    check_grid_size(
        n_points,
        automatic=automatic,
        time=t,
        baseline=baseline,
        maximum_frequency=f_max,
        context=context,
    )
    return uniform_frequency_grid(
        baseline,
        maximum_frequency=f_max,
        minimum_frequency=f_min,
        samples_per_peak=samples_per_peak,
    )


def log_period_grid(
    *,
    minimum_period: float,
    maximum_period: float,
    n_periods: int,
) -> GridSpec:
    """Build a log-spaced period grid (general-purpose; BLS uses its own segments).

    Parameters
    ----------
    minimum_period, maximum_period : float
        Finite period bounds in days (``0 < minimum_period < maximum_period``).
    n_periods : int
        Number of grid points (at most :data:`MAX_GRID_SIZE`).

    Returns
    -------
    GridSpec
        A period grid, ascending.
    """
    if not (math.isfinite(minimum_period) and math.isfinite(maximum_period)):
        raise ValueError(
            f"minimum_period and maximum_period must be finite, got "
            f"({minimum_period}, {maximum_period})"
        )
    if not 0.0 < minimum_period < maximum_period:
        raise ValueError("require 0 < minimum_period < maximum_period")
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1")
    check_grid_size(n_periods, context="log_period_grid")
    periods = np.geomspace(minimum_period, maximum_period, n_periods)
    return GridSpec(kind="period", values=periods, uniform=False)


__all__ = [
    "AUTO_GRID_WARN_SIZE",
    "MAX_GRID_SIZE",
    "MULTI_EXPOSURE_RAYLEIGH_BUDGET",
    "GridKind",
    "GridSizeWarning",
    "GridSpec",
    "GridTooLargeError",
    "check_grid_size",
    "default_frequency_grid",
    "log_period_grid",
    "pseudo_nyquist_frequency",
    "regular_cadence_nyquist",
    "uniform_frequency_grid",
    "uniform_grid_size",
]
