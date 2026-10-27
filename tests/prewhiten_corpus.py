"""The pre-whitening golden corpus: data sets, the settings matrix, and the summary.

:func:`cuperiod.prewhiten` is about to be re-expressed as a stepper over an
interactive session, and the promise is that its results do not move. This module
defines what "its results" means: a fixed spread of synthetic pulsators (resolved and
close pairs, harmonics and combinations, gapped nightly sampling, pure noise,
heteroscedastic and absent error bars) crossed with a matrix of settings that drives
every branch of the extraction loop (all four refine policies, every stop criterion,
all three uncertainty estimators with fixed seeds, ``fit_mean`` either way, the
polish and prune switches, narrow grids that run out of resolvable peaks, and the
blocked-candidate budget).

``tests/make_prewhiten_golden.py`` runs every case and records :func:`summarize` of
the result in ``tests/data/prewhiten_golden.json``; ``tests/test_prewhiten_golden.py``
re-runs them and compares. Almost every case uses the ``numpy`` spectrum backend, which
is bitwise reproducible; the few ``finufft`` cases are compared with the tolerance its
run-to-run noise needs (:attr:`Case.rtol_floor`).

The data sets are generated from fixed seeds here rather than stored: a few hundred
points each, so the corpus stays small and the definitions stay readable.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import cuperiod as cup
from cuperiod.core.grid import uniform_frequency_grid

FloatArray = Any
Data = tuple[FloatArray, FloatArray, FloatArray | None]

#: Format of the golden file; bump when :func:`summarize` changes shape.
GOLDEN_FORMAT = 1

#: Where the golden file lives.
GOLDEN_PATH = Path(__file__).parent / "data" / "prewhiten_golden.json"

#: Relative tolerances of the comparison (see tests/test_prewhiten_golden.py). Another
#: numpy / scipy / BLAS build or OS moves the well-conditioned cases by up to ~5e-7
#: (the seeded bootstrap errors; measured across numpy 2.4.6 / 2.5.4, scipy 1.17.1 /
#: 1.18, OpenBLAS 0.3.31 / 0.3.34 on Windows and Linux), so 1e-6 is the bound there.
RTOL_SAME_ENVIRONMENT = 1e-12
RTOL_OTHER_ENVIRONMENT = 1e-6

#: The finufft backend is not run-to-run reproducible (its spectrum moves at ~1e-15,
#: which shifts each peak seed and so the optimiser's stopping point by ~1e-11); the
#: final numbers stay put to far better than this.
RTOL_FINUFFT = 1e-7

#: Residual samples recorded per case (evenly spaced indices).
RESIDUAL_SAMPLES = 16

_BASE_JD = 2459000.0


def _sines(
    t: FloatArray, freqs: tuple[float, ...], amps: tuple[float, ...],
    phases: tuple[float, ...],
) -> FloatArray:
    out = np.zeros_like(t)
    for f, a, p in zip(freqs, amps, phases, strict=True):
        out += a * np.sin(2.0 * np.pi * f * (t - t.min()) + p)
    return out


def _uniform_times(rng: np.random.Generator, n: int, span: float) -> FloatArray:
    return np.sort(rng.uniform(0.0, span, n)) + _BASE_JD


def _triple() -> Data:
    """A δ Scuti-like triple; the third mode is the exact harmonic of the first."""
    rng = np.random.default_rng(4)
    t = _uniform_times(rng, 500, 20.0)
    y = 10.0 + _sines(t, (12.34, 17.81, 24.68), (0.012, 0.007, 0.004),
                      (0.5, 2.0, 1.0))
    return t, y + rng.normal(0.0, 0.0015, t.size), np.full(t.size, 0.0015)


def _triple_no_errors() -> Data:
    t, y, _ = _triple()
    return t, y, None


def _heteroscedastic() -> Data:
    """Error bars spanning a factor of four, the noise drawn from them."""
    rng = np.random.default_rng(11)
    t = _uniform_times(rng, 500, 20.0)
    err = rng.uniform(0.001, 0.004, t.size)
    y = 10.0 + _sines(t, (9.13, 15.71), (0.010, 0.006), (1.1, 4.0))
    return t, y + rng.normal(0.0, 1.0, t.size) * err, err


def _close_pair() -> Data:
    """Two modes 1.6 Rayleigh widths apart plus an isolated third."""
    rng = np.random.default_rng(21)
    span = 30.0
    t = _uniform_times(rng, 600, span)
    y = 5.0 + _sines(t, (8.0, 8.0 + 1.6 / span, 13.1), (0.010, 0.008, 0.005),
                     (0.3, 2.2, 5.0))
    return t, y + rng.normal(0.0, 0.001, t.size), np.full(t.size, 0.001)


def _combinations() -> Data:
    """Two parents with a sum, a harmonic and a difference frequency."""
    rng = np.random.default_rng(31)
    t = _uniform_times(rng, 700, 25.0)
    freqs = (5.3, 7.1, 12.4, 10.6, 1.8)
    y = 12.0 + _sines(t, freqs, (0.020, 0.012, 0.004, 0.003, 0.0025),
                      (0.1, 1.7, 3.3, 4.4, 0.9))
    return t, y + rng.normal(0.0, 0.001, t.size), np.full(t.size, 0.001)


def _nightly() -> Data:
    """Ground-based nights: 14 eight-hour windows, so 1 c/d aliases abound."""
    rng = np.random.default_rng(41)
    nights = np.arange(14, dtype=np.float64)
    t = np.sort(
        np.concatenate([n + rng.uniform(0.0, 0.33, 35) for n in nights])
    ) + _BASE_JD
    y = 8.0 + _sines(t, (9.7, 14.2), (0.015, 0.008), (2.5, 0.4))
    return t, y + rng.normal(0.0, 0.002, t.size), np.full(t.size, 0.002)


def _noise_only() -> Data:
    rng = np.random.default_rng(51)
    t = _uniform_times(rng, 400, 20.0)
    return t, 10.0 + rng.normal(0.0, 0.002, t.size), np.full(t.size, 0.002)


def _modulated() -> Data:
    """One strong mode whose amplitude doubles across the run.

    Its residual leaves sidelobes one Rayleigh width either side, which a tight
    separation guard admits as candidates and the prune later removes.
    """
    rng = np.random.default_rng(61)
    t = _uniform_times(rng, 400, 20.0)
    ramp = 0.5 + (t - t.min()) / (t.max() - t.min())
    y = 10.0 + 0.02 * ramp * np.sin(2.0 * np.pi * 6.2 * (t - t.min()) + 0.7)
    return t, y + rng.normal(0.0, 0.0005, t.size), np.full(t.size, 0.0005)


def _drift() -> Data:
    """One strong mode with a slow linear frequency drift.

    The residual of a constant-frequency fit is a cluster of peaks hugging the parent;
    with a tight separation guard and a wide refinement box, candidates drawn from it
    slide back onto the parent and are blocked, which exercises the blocked-candidate
    budget.
    """
    rng = np.random.default_rng(61)
    t = _uniform_times(rng, 400, 20.0)
    x = t - t.min()
    y = 10.0 + 0.02 * np.sin(2.0 * np.pi * (6.2 * x + 0.002 * x * x) + 0.7)
    return t, y + rng.normal(0.0, 0.0005, t.size), np.full(t.size, 0.0005)


DATASETS: Mapping[str, Callable[[], Data]] = {
    "triple": _triple,
    "triple_no_errors": _triple_no_errors,
    "heteroscedastic": _heteroscedastic,
    "close_pair": _close_pair,
    "combinations": _combinations,
    "nightly": _nightly,
    "noise_only": _noise_only,
    "modulated": _modulated,
    "drift": _drift,
}

#: Settings every case starts from: a small grid keeps the numpy backend quick.
BASE_SETTINGS: Mapping[str, Any] = {
    "backend": "numpy",
    "maximum_frequency": 30.0,
    "samples_per_peak": 5,
    "max_frequencies": 8,
    "n_resamples": 20,
    "seed": 3,
}


@dataclass(frozen=True)
class Case:
    """One corpus entry: a data set, settings overrides and an optional custom grid.

    ``grid`` is ``(minimum, maximum, samples_per_peak)`` in cycles/day for a uniform
    grid built by :func:`~cuperiod.core.grid.uniform_frequency_grid`.
    """

    name: str
    dataset: str
    settings: Mapping[str, Any] = field(default_factory=dict)
    grid: tuple[float, float, float] | None = None
    #: Tolerance in another numerical environment, when RTOL_OTHER_ENVIRONMENT is too
    #: tight for this case.
    other_environment_rtol: float | None = None
    #: Ill-conditioned by construction: a separation guard below the refinement box
    #: lets two components converge on one frequency with large cancelling
    #: amplitudes, which only rounding fixes. Its numbers are pinned in the recording
    #: environment; elsewhere only the run's :func:`outline` must match.
    chaotic: bool = False

    def resolved_settings(self) -> cup.PreWhitenSettings:
        return cup.PreWhitenSettings(**{**BASE_SETTINGS, **self.settings})

    @property
    def rtol_floor(self) -> float:
        """Smallest relative tolerance this case can be held to."""
        backend = self.resolved_settings().backend
        return 0.0 if backend == "numpy" else RTOL_FINUFFT


CASES: tuple[Case, ...] = (
    # -- refine policies and loop switches on the resolved triple -----------------
    Case("triple-last", "triple"),
    Case("triple-none", "triple", {"refine": "none"}),
    Case("triple-cyclic", "triple", {"refine": "cyclic", "sweeps": 2}),
    Case("triple-simultaneous", "triple", {"refine": "simultaneous"}),
    Case("triple-no-polish", "triple", {"final_refine": False}),
    Case("triple-no-polish-none", "triple",
         {"final_refine": False, "refine": "none"}),
    Case("triple-no-prune", "triple", {"prune": False}),
    Case("triple-fit-mean-false", "triple", {"fit_mean": False}),
    Case("triple-dft", "triple", {"normalization": "dft"}),
    Case("triple-max-simultaneous-1", "triple", {"max_simultaneous": 1}),
    Case("triple-max-frequencies-0", "triple", {"max_frequencies": 0}),
    Case("triple-max-frequencies-2", "triple", {"max_frequencies": 2}),
    Case("triple-no-spectra", "triple", {"store_spectra": False}),
    Case("triple-no-combinations", "triple", {"combinations": False}),
    # -- stop criteria -------------------------------------------------------------
    Case("triple-stop-fap", "triple", {"stop_criteria": ("fap",)}),
    Case("triple-stop-bic", "triple", {"stop_criteria": ("bic",)}),
    Case("triple-stop-amplitude", "triple",
         {"stop_criteria": ("amplitude",), "min_amplitude": 0.005}),
    Case("triple-stop-all", "triple",
         {"stop_criteria": ("snr", "fap", "bic", "amplitude"),
          "min_amplitude": 0.001, "snr_threshold": 5.0}),
    Case("triple-stop-snr-median", "triple",
         {"noise_estimator": "median", "snr_window": 2.0}),
    # -- uncertainty estimators ----------------------------------------------------
    Case("triple-analytic", "triple", {"uncertainty": "analytic"}),
    Case("triple-analytic-uncorrected", "triple",
         {"uncertainty": "analytic", "correlation_correction": False}),
    Case("triple-covariance-uncorrected", "triple",
         {"correlation_correction": False}),
    Case("triple-bootstrap", "triple", {"uncertainty": "bootstrap"}),
    Case("triple-bootstrap-cyclic", "triple",
         {"uncertainty": "bootstrap", "max_simultaneous": 1, "seed": 17}),
    # -- error bars ------------------------------------------------------------------
    Case("noerr-last", "triple_no_errors"),
    Case("noerr-bootstrap", "triple_no_errors", {"uncertainty": "bootstrap"}),
    Case("noerr-fit-mean-false", "triple_no_errors", {"fit_mean": False}),
    Case("hetero-last", "heteroscedastic"),
    Case("hetero-analytic", "heteroscedastic", {"uncertainty": "analytic"}),
    Case("hetero-fit-mean-false", "heteroscedastic", {"fit_mean": False}),
    # -- close pair, combinations, aliasing ----------------------------------------
    Case("pair-last", "close_pair"),
    Case("pair-simultaneous", "close_pair", {"refine": "simultaneous"}),
    Case("pair-tight-guard", "close_pair",
         {"min_separation_rayleigh": 0.8, "refine_bound_rayleigh": 0.5}),
    Case("combos-last", "combinations"),
    Case("combos-order-3", "combinations",
         {"combination_max_order": 3, "combination_parents": 3}),
    Case("nightly-last", "nightly"),
    Case("nightly-cyclic-analytic", "nightly",
         {"refine": "cyclic", "uncertainty": "analytic"}),
    Case("nightly-fit-mean-false", "nightly", {"fit_mean": False}),
    # -- nothing to find -------------------------------------------------------------
    Case("noise-last", "noise_only"),
    Case("noise-low-threshold", "noise_only", {"snr_threshold": 3.0}),
    # -- narrow grids and blocked candidates -----------------------------------------
    Case("narrow-no-peak", "triple", {"min_separation_rayleigh": 8.0},
         grid=(12.0, 12.7, 5.0)),
    Case("modulated-prune", "modulated",
         {"min_separation_rayleigh": 0.2, "refine_bound_rayleigh": 5.0,
          "snr_threshold": 2.0, "max_frequencies": 6}, chaotic=True),
    Case("drift-blocked", "drift",
         {"min_separation_rayleigh": 0.3, "refine_bound_rayleigh": 3.0,
          "snr_threshold": 2.0, "max_frequencies": 6, "refine": "simultaneous"},
         chaotic=True),
    # Moves ~5e-6 between environments (measured): near-degenerate, not chaotic.
    Case("drift-budget", "drift",
         {"min_separation_rayleigh": 0.05, "refine_bound_rayleigh": 8.0,
          "snr_threshold": 0.5, "max_frequencies": 10, "refine": "simultaneous"},
         other_environment_rtol=1e-4),
    # -- finufft, compared at RTOL_FINUFFT --------------------------------------------
    Case("finufft-triple", "triple", {"backend": "finufft"}),
    Case("finufft-pair", "close_pair", {"backend": "finufft"}),
    Case("finufft-combos-analytic", "combinations",
         {"backend": "finufft", "uncertainty": "analytic"}),
)


def load(case: Case) -> tuple[Data, cup.PreWhitenSettings, Any]:
    """The data, resolved settings and custom grid (or ``None``) of ``case``."""
    data = DATASETS[case.dataset]()
    settings = case.resolved_settings()
    grid = None
    if case.grid is not None:
        low, high, oversampling = case.grid
        baseline = float(np.ptp(data[0]))
        grid = uniform_frequency_grid(
            baseline, minimum_frequency=low, maximum_frequency=high,
            samples_per_peak=oversampling,
        )
    return data, settings, grid


def run(case: Case) -> cup.PreWhitenResult:
    """Run :func:`cuperiod.prewhiten` on ``case``."""
    data, settings, grid = load(case)
    t, y, e = data
    payload = (t, y) if e is None else (t, y, e)
    return cup.prewhiten(payload, settings=settings, grid=grid)


def _spectrum_summary(spectrum: Any) -> dict[str, Any] | None:
    if spectrum is None:
        return None
    amplitude = np.asarray(spectrum.amplitude, dtype=np.float64)
    return {
        "size": int(amplitude.size),
        "first": float(spectrum.frequency[0]),
        "last": float(spectrum.frequency[-1]),
        "argmax": int(np.argmax(amplitude)),
        "max": float(np.max(amplitude)),
        "mean": float(np.mean(amplitude)),
    }


#: Sinusoid fields of the v1.3 result; later trailing fields are not part of the
#: promise (they did not exist when the golden was written).
SINUSOID_FIELDS: tuple[str, ...] = (
    "rank", "label", "frequency", "frequency_error", "amplitude", "amplitude_error",
    "phase", "phase_error", "snr", "fap", "delta_bic", "combination",
    "spectrum_amplitude", "blended",
)

COMBINATION_FIELDS: tuple[str, ...] = (
    "index", "parents", "coefficients", "order", "label", "predicted", "residual",
    "tolerance", "expected_false",
)


def summarize(result: cup.PreWhitenResult) -> dict[str, Any]:
    """Everything the golden pins down, as JSON-ready values."""
    residuals = np.asarray(result.residuals, dtype=np.float64)
    picks = np.unique(
        np.linspace(0, max(residuals.size - 1, 0), RESIDUAL_SAMPLES).astype(np.int64)
    )
    components = [
        {name: _plain(getattr(c, name)) for name in SINUSOID_FIELDS}
        for c in result.components
    ]
    combinations = [
        {name: _plain(getattr(c, name)) for name in COMBINATION_FIELDS}
        for c in result.combinations
    ]
    return {
        "components": components,
        "combinations": combinations,
        "stop_reason": result.stop_reason,
        "n_iterations": int(result.n_iterations),
        "n_pruned": int(result.n_pruned),
        "n_samples": int(result.n_samples),
        "uncertainty_method": result.uncertainty_method,
        "backend": result.backend,
        "scalars": {
            "offset": float(result.offset),
            "offset_error": float(result.offset_error),
            "t_ref": float(result.t_ref),
            "baseline": float(result.baseline),
            "rms": float(result.rms),
            "chi2": float(result.chi2),
            "reduced_chi2": float(result.reduced_chi2),
            "bic": float(result.bic),
            "correlation_factor": float(result.correlation_factor),
        },
        "residuals": {
            "size": int(residuals.size),
            "index": [int(i) for i in picks],
            "sample": [float(residuals[i]) for i in picks],
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
        },
        "spectrum": _spectrum_summary(result.spectrum),
        "residual_spectrum": _spectrum_summary(result.residual_spectrum),
        "window": _spectrum_summary(result.window),
    }


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def fingerprint() -> dict[str, str]:
    """The numerical environment: equal fingerprints should give bitwise equal runs."""
    import scipy

    try:
        import finufft

        finufft_version = str(getattr(finufft, "__version__", "unknown"))
    except ImportError:  # pragma: no cover - finufft is a core dependency
        finufft_version = "absent"
    blas = lapack = "unknown"
    try:
        config = np.show_config(mode="dicts")
        deps = config.get("Build Dependencies", {})
        blas = "{} {}".format(deps.get("blas", {}).get("name", "?"),
                              deps.get("blas", {}).get("version", "?"))
        lapack = "{} {}".format(deps.get("lapack", {}).get("name", "?"),
                                deps.get("lapack", {}).get("version", "?"))
    except (TypeError, ValueError, AttributeError):  # pragma: no cover - old numpy
        pass
    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "finufft": finufft_version,
        "blas": blas,
        "lapack": lapack,
        "machine": platform.machine(),
        "system": platform.system(),
        "python": ".".join(str(v) for v in sys.version_info[:3]),
    }


def source_digest() -> dict[str, str]:
    """sha256 of every pre-whitening source file (informational, for forensics)."""
    package = Path(cup.__file__).parent / "prewhiten"
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        for path in sorted(package.glob("*.py"))
    }


def case_by_name(name: str) -> Case:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(name)


def outline(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The parts of a summary that hold even for a :attr:`Case.chaotic` case."""
    return {
        "stop_reason": summary["stop_reason"],
        "n_iterations": summary["n_iterations"],
        "n_pruned": summary["n_pruned"],
        "n_samples": summary["n_samples"],
        "n_components": len(summary["components"]),
        "uncertainty_method": summary["uncertainty_method"],
        "backend": summary["backend"],
    }


def describe(case: Case) -> dict[str, Any]:
    """A JSON-ready description of the case (recorded beside its summary)."""
    return {
        "dataset": case.dataset,
        "settings": {k: _plain(v) for k, v in case.settings.items()},
        "grid": list(case.grid) if case.grid is not None else None,
    }
