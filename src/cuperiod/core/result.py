"""Result objects: :class:`Peak`, :class:`Periodogram`, :class:`MultiResult`.

A :class:`Periodogram` is the raw spectrum (frequency, period, power) plus any
method-specific per-grid quantities (BLS depth/duration/t0, ...). It is stored in
ascending-frequency order so peak finding is uniform across methods, and it knows
whether its statistic is maximized (GLS/BLS/MHAOV/TLS) or minimized (PDM/CE/string-
length) so :meth:`Periodogram.best_periods` always returns the most significant peaks.
"""

from __future__ import annotations

import dataclasses
from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np

from cuperiod.core._typing import Float32Array, FloatArray, IntArray
from cuperiod.core.peaks import (
    _interior_maxima,
    _sum_is_finite,
    peak_preserving_downsample,
    select_diverse_peaks,
    select_top_peaks,
)

ObjectiveSense = Literal["max", "min"]

#: Default number of peaks :meth:`Periodogram.best_periods` returns.
DEFAULT_N_PEAKS: Final = 10

#: Default minimum peak separation, in Rayleigh widths ``1/baseline``.
DEFAULT_SEPARATION_RAYLEIGH: Final = 3.0

#: Default highest harmonic order tested by alias-diverse selection.
DEFAULT_HARMONIC_MAX: Final = 8

#: Default size of the peak-preserving downsampled spectrum.
DEFAULT_DOWNSAMPLE_POINTS: Final = 2000

#: Settings fields that default the peak-selection / stored-spectrum parameters,
#: keyed by the parameter they default (the keys of :attr:`Periodogram.peak_defaults`).
PEAK_DEFAULT_FIELDS: Final[Mapping[str, str]] = {
    "n": "n_peaks",
    "min_separation_rayleigh": "peak_separation_rayleigh",
    "alias_tolerance": "alias_freq_tolerance",
    "harmonic_max": "harmonic_max",
    "n_points": "downsample_points",
}


def peak_defaults_from_settings(settings: object | None) -> dict[str, Any]:
    """The peak-selection defaults a method settings object carries.

    Maps ``n_peaks`` -> ``n``, ``peak_separation_rayleigh`` ->
    ``min_separation_rayleigh``, ``alias_freq_tolerance`` -> ``alias_tolerance``,
    ``harmonic_max`` -> ``harmonic_max`` and ``downsample_points`` -> ``n_points``
    (see :data:`PEAK_DEFAULT_FIELDS`); fields the model does not define are omitted.

    Parameters
    ----------
    settings : object or None
        A method settings model (e.g. :class:`~cuperiod.BLSSettings`).

    Returns
    -------
    dict
        Suitable for :meth:`Periodogram.with_peak_defaults`.
    """
    out: dict[str, Any] = {}
    if settings is None:
        return out
    for parameter, field_name in PEAK_DEFAULT_FIELDS.items():
        value = getattr(settings, field_name, None)
        if value is not None:
            out[parameter] = value
    return out


@dataclass(frozen=True)
class Peak:
    """A single periodogram peak.

    Peaks compare by value and are hashable (on their scalar fields).

    Parameters
    ----------
    period : float
        Period in days.
    frequency : float
        Frequency in cycles/day.
    power : float
        The method's statistic at this peak (large = significant for max-objective
        methods; small = significant for min-objective methods).
    rank : int
        1-based significance rank within the result.
    extra : Mapping[str, float], optional
        Method-specific scalars at this peak (e.g. ``depth``, ``duration``, ``t0``,
        ``depth_snr``, ``sde`` for BLS; ``fap`` for GLS).
    """

    period: float
    frequency: float
    power: float
    rank: int
    extra: Mapping[str, float] = field(default_factory=dict, hash=False)

    def to_dict(self) -> dict[str, float | int]:
        """Flatten to a plain dict (peak scalars + extras)."""
        out: dict[str, float | int] = {
            "period": self.period,
            "frequency": self.frequency,
            "power": self.power,
            "rank": self.rank,
        }
        out.update(self.extra)
        return out


@dataclass(frozen=True, eq=False)
class Periodogram:
    """A computed periodogram for a single light curve and method.

    Periodograms compare and hash by identity (their arrays have no single-valued
    equality), like :class:`~cuperiod.LightCurve`.

    Attributes
    ----------
    method, backend : str
        The method name and the concrete backend that produced this spectrum.
    frequency, period, power : numpy.ndarray
        The raw spectrum, ascending in frequency (``period = 1/frequency``).
    objective_sense : {"max", "min"}
        Whether peaks are maxima or minima of ``power``.
    n_samples : int
        Number of finite light-curve points used.
    baseline : float
        Time baseline in days (sets the Rayleigh resolution ``1/baseline``).
    extras : Mapping[str, numpy.ndarray]
        Per-grid method-specific arrays, aligned with ``frequency``.
    meta : Mapping
        Free-form metadata carried from the light curve.
    peak_defaults : Mapping
        Defaults for the parameters of :meth:`best_periods` (``n``,
        ``min_separation_rayleigh``, ``alias_tolerance``, ``harmonic_max``) and
        :meth:`downsample` (``n_points``) used when a call leaves them unset --
        normally the method settings' ``n_peaks``, ``peak_separation_rayleigh``,
        ``alias_freq_tolerance``, ``harmonic_max`` and ``downsample_points`` (see
        :func:`peak_defaults_from_settings` and :meth:`with_peak_defaults`). Empty
        means the built-in defaults (10 peaks, 3 Rayleigh widths, harmonic order 8,
        2000 points).
    """

    method: str
    backend: str
    frequency: FloatArray
    period: FloatArray
    power: FloatArray
    objective_sense: ObjectiveSense
    n_samples: int
    baseline: float
    extras: Mapping[str, FloatArray] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)
    peak_defaults: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.objective_sense not in ("max", "min"):
            raise ValueError(
                "objective_sense must be 'max' or 'min', got "
                f"{self.objective_sense!r}"
            )

    @classmethod
    def from_spectrum(
        cls,
        *,
        method: str,
        backend: str,
        frequency: FloatArray,
        power: FloatArray,
        objective_sense: ObjectiveSense,
        n_samples: int,
        baseline: float,
        extras: Mapping[str, FloatArray] | None = None,
        meta: Mapping[str, Any] | None = None,
        peak_defaults: Mapping[str, Any] | None = None,
    ) -> Periodogram:
        """Build a :class:`Periodogram`, sorting everything into ascending frequency.

        Raises
        ------
        ValueError
            If ``frequency`` is not 1-D, ``power`` or an ``extras`` array does not
            have the shape of ``frequency``, or ``objective_sense`` is neither
            ``"max"`` nor ``"min"``.
        """
        if objective_sense not in ("max", "min"):
            raise ValueError(
                f"from_spectrum: objective_sense must be 'max' or 'min', got "
                f"{objective_sense!r}"
            )
        frequency = np.ascontiguousarray(frequency, dtype=np.float64)
        power = np.ascontiguousarray(power, dtype=np.float64)
        if frequency.ndim != 1:
            raise ValueError(
                f"from_spectrum: frequency must be 1-D, got shape {frequency.shape}"
            )
        if power.shape != frequency.shape:
            raise ValueError(
                f"from_spectrum: power shape {power.shape} does not match frequency "
                f"shape {frequency.shape}"
            )
        converted: dict[str, FloatArray] = {}
        for key, values in (extras or {}).items():
            array = np.ascontiguousarray(values, dtype=np.float64)
            if array.shape != frequency.shape:
                raise ValueError(
                    f"from_spectrum: extras[{key!r}] shape {array.shape} does not "
                    f"match frequency shape {frequency.shape}"
                )
            converted[key] = array
        order = np.argsort(frequency, kind="stable")
        frequency = frequency[order]
        power = power[order]
        sorted_extras = {k: v[order] for k, v in converted.items()}
        with np.errstate(divide="ignore"):
            period = 1.0 / frequency
        return cls(
            method=method,
            backend=backend,
            frequency=frequency,
            period=period,
            power=power,
            objective_sense=objective_sense,
            n_samples=n_samples,
            baseline=baseline,
            extras=sorted_extras,
            meta=dict(meta or {}),
            peak_defaults=dict(peak_defaults or {}),
        )

    def with_peak_defaults(self, defaults: Mapping[str, Any] | object) -> Periodogram:
        """A copy whose peak-selection defaults are updated from ``defaults``.

        Parameters
        ----------
        defaults : Mapping or settings object
            Either a mapping keyed like :attr:`peak_defaults` (``n``,
            ``min_separation_rayleigh``, ``alias_tolerance``, ``harmonic_max``,
            ``n_points``) or a method settings object, read with
            :func:`peak_defaults_from_settings`. Entries override existing ones.

        Returns
        -------
        Periodogram
            A new result sharing this one's arrays.
        """
        new = (
            dict(defaults)
            if isinstance(defaults, Mapping)
            else peak_defaults_from_settings(defaults)
        )
        unknown = sorted(set(new) - set(PEAK_DEFAULT_FIELDS))
        if unknown:
            raise ValueError(
                f"unknown peak default(s) {unknown}; expected a subset of "
                f"{sorted(PEAK_DEFAULT_FIELDS)}"
            )
        return dataclasses.replace(self, peak_defaults={**self.peak_defaults, **new})

    def _peak_default(self, name: str, fallback: Any) -> Any:
        value = self.peak_defaults.get(name)
        return fallback if value is None else value

    @property
    def size(self) -> int:
        """Number of grid samples."""
        return int(self.frequency.size)

    def _score(self) -> FloatArray:
        """Power converted to a maximize-me score."""
        return self.power if self.objective_sense == "max" else -self.power

    def best_periods(
        self,
        n: int | None = None,
        *,
        min_separation_rayleigh: float | None = None,
        alias_diverse: bool = False,
        harmonic_max: int | None = None,
        alias_tolerance: float | None = None,
    ) -> list[Peak]:
        """Return the ``n`` most significant peaks.

        Parameters left as None take their value from :attr:`peak_defaults` (the
        method settings), then from the built-in default.

        Parameters
        ----------
        n : int, optional
            Maximum number of peaks (default 10; ``<= 0`` returns no peaks).
        min_separation_rayleigh : float, optional
            Minimum peak separation in Rayleigh widths ``1/baseline`` (default 3.0):
            closer candidates are the same peak.
        alias_diverse : bool, default False
            If True, skip candidates alias- or harmonic-related to a chosen peak
            (recommended for box searches).
        harmonic_max : int, optional
            Highest harmonic order considered when ``alias_diverse`` is set
            (default 8).
        alias_tolerance : float, optional
            Frequency tolerance (cycles/day) of the alias / harmonic relations when
            ``alias_diverse`` is set; around the ``m``-th harmonic it widens to
            ``m*alias_tolerance``. Defaults to the settings' ``alias_freq_tolerance``
            (BLS/TLS), else one Rayleigh width ``1/baseline`` (or the separation, if
            smaller) -- the resolution below which two frequencies cannot be told
            apart. It is deliberately not the separation: three Rayleigh widths,
            multiplied by the harmonic order, excluded most of a short-baseline
            search band and hid genuine second signals.

        Returns
        -------
        list of Peak
            Ranked most-significant first.
        """
        count = int(self._peak_default("n", DEFAULT_N_PEAKS)) if n is None else int(n)
        if count <= 0 or self.size == 0:
            return []
        score = self._score()
        candidates = self._peak_candidates(score)
        if candidates.size == 0:
            return []
        separation_rayleigh = float(
            self._peak_default("min_separation_rayleigh", DEFAULT_SEPARATION_RAYLEIGH)
            if min_separation_rayleigh is None
            else min_separation_rayleigh
        )
        resolved = self.baseline > 0.0
        separation = separation_rayleigh / self.baseline if resolved else 0.0
        if alias_diverse:
            order = int(
                self._peak_default("harmonic_max", DEFAULT_HARMONIC_MAX)
                if harmonic_max is None
                else harmonic_max
            )
            tolerance = (
                self._peak_default("alias_tolerance", None)
                if alias_tolerance is None
                else alias_tolerance
            )
            if tolerance is None:
                tolerance = min(separation, 1.0 / self.baseline) if resolved else 0.0
            chosen = select_diverse_peaks(
                self.frequency,
                score,
                candidates,
                count,
                max(float(tolerance), 1e-12),
                order,
                min_separation=max(separation, 1e-12),
            )
        else:
            chosen = select_top_peaks(
                self.frequency, score, candidates, count, separation
            )
        return [self._make_peak(int(idx), rank) for rank, idx in enumerate(chosen, 1)]

    def _peak_candidates(self, score: FloatArray) -> IntArray:
        """Finite peak candidates: interior local maxima plus boundary peaks.

        :func:`~cuperiod.core.peaks.local_maxima` excludes the grid endpoints by
        contract, but a true peak can sit at the lowest or highest frequency (a signal
        whose period is comparable to the baseline), so boundary samples that exceed
        their single neighbor are added too. Non-finite scores and non-finite periods
        (e.g. a ``frequency == 0`` sample, period ``inf``) are dropped so they can never
        be reported as a best period -- and they compare as ``-inf`` so a NaN next to
        the true maximum (astropy's power at ``f = 0``) cannot hide it.
        """
        if _sum_is_finite(score) and self._periods_all_finite():
            safe = score
        else:
            usable = np.isfinite(score) & np.isfinite(self.period)
            safe = np.where(usable, score, -np.inf)
        candidates = _interior_maxima(safe)
        boundary: list[int] = []
        if self.size == 1:
            boundary.append(0)
        elif self.size >= 2:
            if safe[0] >= safe[1]:
                boundary.append(0)
            if safe[-1] > safe[-2]:
                boundary.append(self.size - 1)
        if boundary:
            candidates = np.concatenate(
                [candidates, np.asarray(boundary, dtype=np.int64)]
            )
        if candidates.size:
            ok = np.isfinite(score[candidates]) & np.isfinite(self.period[candidates])
            candidates = candidates[ok]
        if candidates.size == 0:
            finite = np.flatnonzero(np.isfinite(score) & np.isfinite(self.period))
            if finite.size == 0:
                return np.empty(0, dtype=np.int64)
            return finite[[int(np.argmax(score[finite]))]]
        return candidates

    def _periods_all_finite(self) -> bool:
        """Whether every period is finite, from the ascending frequencies in O(log n).

        ``period = 1/frequency`` is non-finite only at ``frequency == 0`` (located by
        bisection) or at a non-finite frequency (NaN sorts last, ``-inf`` first).
        """
        freq = self.frequency
        n = int(freq.size)
        if n == 0:
            return True
        if not (np.isfinite(freq[0]) and np.isfinite(freq[-1])):
            return False
        at_zero = int(np.searchsorted(freq, 0.0))
        if at_zero < n and freq[at_zero] == 0.0:
            return False
        return self.period.size == n

    def _make_peak(self, idx: int, rank: int) -> Peak:
        extra = {k: float(v[idx]) for k, v in self.extras.items()}
        return Peak(
            period=float(self.period[idx]),
            frequency=float(self.frequency[idx]),
            power=float(self.power[idx]),
            rank=rank,
            extra=extra,
        )

    def best_period(self) -> float:
        """The single most significant period in days (NaN if none is finite)."""
        peaks = self.best_periods(1)
        return peaks[0].period if peaks else float("nan")

    def downsample(
        self, n_points: int | None = None
    ) -> tuple[Float32Array, Float32Array]:
        """Peak-preserving downsample of ``(frequency, power)`` to ``n_points``.

        Each bin keeps its most significant sample -- the maximum for max-objective
        methods, the minimum for min-objective ones -- so the detection survives.
        ``n_points`` defaults to :attr:`peak_defaults` (``downsample_points``), else
        2000.
        """
        count = (
            int(self._peak_default("n_points", DEFAULT_DOWNSAMPLE_POINTS))
            if n_points is None
            else int(n_points)
        )
        return peak_preserving_downsample(
            self.frequency, self.power, count, sense=self.objective_sense
        )

    def to_dict(
        self, *, n_best: int | None = None, include_raw: bool = False
    ) -> dict[str, Any]:
        """Serialize to a plain dict (peaks always; raw spectrum optional).

        ``n_best`` and the downsampled size default as in :meth:`best_periods` and
        :meth:`downsample`.
        """
        out: dict[str, Any] = {
            "method": self.method,
            "backend": self.backend,
            "n_samples": self.n_samples,
            "baseline": self.baseline,
            "objective_sense": self.objective_sense,
            "peaks": [p.to_dict() for p in self.best_periods(n_best)],
        }
        if include_raw:
            freq, power = self.downsample()
            out["frequency"] = freq.tolist()
            out["power"] = power.tolist()
        return out


def _normalize_method_key(key: str) -> str:
    return "".join(ch for ch in key if ch.isalnum()).upper()


@dataclass(frozen=True, eq=False)
class MultiResult:
    """Several methods' periodograms for one light curve, keyed by method name.

    A read-only mapping whose lookups (``[]``, ``in``, :meth:`get`) ignore case and
    separators, so ``res["String-Length"]``, ``"gls" in res`` and
    ``res.get("STRINGLENGTH")`` all work. Compares and hashes by identity.
    """

    results: Mapping[str, Periodogram]

    def _resolve(self, method: object) -> str | None:
        """The stored key ``method`` refers to, or None."""
        if not isinstance(method, str):
            return None
        if method in self.results:
            return method
        wanted = _normalize_method_key(method)
        if wanted in self.results:
            return wanted
        for key in self.results:
            if _normalize_method_key(key) == wanted:
                return key
        return None

    def __getitem__(self, method: str) -> Periodogram:
        key = self._resolve(method)
        if key is None:
            raise KeyError(method)
        return self.results[key]

    def __contains__(self, method: object) -> bool:
        return self._resolve(method) is not None

    def __iter__(self) -> Iterator[str]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def get(
        self, method: str, default: Periodogram | None = None
    ) -> Periodogram | None:
        """The periodogram for ``method`` (any case/separators), or ``default``."""
        key = self._resolve(method)
        return default if key is None else self.results[key]

    def keys(self) -> KeysView[str]:
        """Method names present."""
        return self.results.keys()

    def values(self) -> ValuesView[Periodogram]:
        """The periodograms, in method order."""
        return self.results.values()

    def items(self) -> ItemsView[str, Periodogram]:
        """``(method, Periodogram)`` pairs."""
        return self.results.items()

    def best_periods(
        self, n: int | None = None, **kwargs: Any
    ) -> dict[str, list[Peak]]:
        """The N-best peaks per method (each defaulting from its own settings)."""
        return {k: v.best_periods(n, **kwargs) for k, v in self.results.items()}

    def to_dict(
        self, *, n_best: int | None = None, include_raw: bool = False
    ) -> dict[str, Any]:
        """Serialize every method's result."""
        return {
            k: v.to_dict(n_best=n_best, include_raw=include_raw)
            for k, v in self.results.items()
        }


__all__ = [
    "DEFAULT_DOWNSAMPLE_POINTS",
    "DEFAULT_HARMONIC_MAX",
    "DEFAULT_N_PEAKS",
    "DEFAULT_SEPARATION_RAYLEIGH",
    "PEAK_DEFAULT_FIELDS",
    "MultiResult",
    "ObjectiveSense",
    "Peak",
    "Periodogram",
    "peak_defaults_from_settings",
]
