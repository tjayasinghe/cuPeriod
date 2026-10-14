"""Grid construction."""

from __future__ import annotations

import math
import tracemalloc
import warnings

import numpy as np
import pytest

import cuperiod.core.grid as G
from cuperiod.core.errors import CuPeriodError, InsufficientDataError
from cuperiod.core.grid import (
    GridSpec,
    log_period_grid,
    pseudo_nyquist_frequency,
    uniform_frequency_grid,
)

DAY_S = 86400.0


def test_uniform_frequency_grid_spacing() -> None:
    grid = uniform_frequency_grid(
        100.0, maximum_frequency=2.0, minimum_frequency=0.01, samples_per_peak=5
    )
    assert grid.kind == "frequency"
    assert grid.uniform
    df = np.diff(grid.values)
    assert np.allclose(df, df[0])
    assert df[0] == pytest.approx(1.0 / (5 * 100.0))
    assert grid.values[-1] >= 2.0


def test_grid_frequency_period_views() -> None:
    grid = GridSpec(kind="frequency", values=np.array([0.5, 1.0, 2.0]), uniform=False)
    assert np.allclose(grid.frequency, [0.5, 1.0, 2.0])
    assert np.allclose(grid.period, [0.5, 1.0, 2.0])  # 1/f reversed -> ascending


def test_uniform_frequency_params() -> None:
    grid = uniform_frequency_grid(50.0, maximum_frequency=5.0, samples_per_peak=4)
    f0, df, nf = grid.uniform_frequency_params()
    assert nf == grid.size
    assert df == pytest.approx(1.0 / (4 * 50.0))


def test_log_period_grid() -> None:
    grid = log_period_grid(minimum_period=0.5, maximum_period=50.0, n_periods=100)
    assert grid.kind == "period"
    assert grid.size == 100
    assert grid.values[0] == pytest.approx(0.5)
    assert grid.values[-1] == pytest.approx(50.0)


def test_invalid_grids_raise() -> None:
    with pytest.raises(ValueError):
        uniform_frequency_grid(-1.0, maximum_frequency=2.0)
    with pytest.raises(ValueError):
        log_period_grid(minimum_period=5.0, maximum_period=1.0, n_periods=10)


def test_uniform_frequency_grid_rejects_min_ge_max() -> None:
    # Regression: transposed bounds must raise, not return a 1-sample grid above max.
    with pytest.raises(ValueError, match="must be <"):
        uniform_frequency_grid(10.0, maximum_frequency=0.5, minimum_frequency=2.0)


# --- CORE-B-16: argument validation ------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"baseline": float("nan"), "maximum_frequency": 2.0}, "baseline"),
        ({"baseline": float("inf"), "maximum_frequency": 2.0}, "baseline"),
        ({"baseline": 0.0, "maximum_frequency": 2.0}, "baseline"),
        ({"baseline": 10.0, "maximum_frequency": float("nan")}, "maximum_frequency"),
        ({"baseline": 10.0, "maximum_frequency": float("inf")}, "maximum_frequency"),
        (
            {"baseline": 10.0, "maximum_frequency": 2.0, "samples_per_peak": 0},
            "samples_per_peak",
        ),
        (
            {"baseline": 10.0, "maximum_frequency": 2.0, "samples_per_peak": -5},
            "samples_per_peak",
        ),
        (
            {
                "baseline": 10.0,
                "maximum_frequency": 2.0,
                "samples_per_peak": float("nan"),
            },
            "samples_per_peak",
        ),
        (
            {"baseline": 10.0, "maximum_frequency": 2.0, "minimum_frequency": -1.0},
            "minimum_frequency",
        ),
        (
            {
                "baseline": 10.0,
                "maximum_frequency": 2.0,
                "minimum_frequency": float("nan"),
            },
            "minimum_frequency",
        ),
    ],
)
def test_uniform_frequency_grid_validates_arguments(
    kwargs: dict[str, float], match: str
) -> None:
    baseline = kwargs.pop("baseline")
    with pytest.raises(ValueError, match=match):
        uniform_frequency_grid(baseline, **kwargs)  # type: ignore[arg-type]


def test_uniform_grid_accepts_zero_minimum_and_fractional_oversampling() -> None:
    grid = uniform_frequency_grid(
        10.0, maximum_frequency=1.0, minimum_frequency=0.0, samples_per_peak=2.5
    )
    assert grid.values[0] == 0.0
    assert grid.values[1] == pytest.approx(1.0 / 25.0)


def test_uniform_frequency_grid_bit_identical_to_formula() -> None:
    """The validated builder keeps the astropy-autofrequency arithmetic exactly."""
    rng = np.random.default_rng(0)
    for _ in range(120):
        baseline = float(rng.uniform(0.5, 4000.0))
        spp = int(rng.integers(1, 12))
        fmax = float(rng.uniform(0.5, 50.0))
        fmin = None if rng.random() < 0.5 else float(rng.uniform(0.0, 0.4))
        grid = uniform_frequency_grid(
            baseline, maximum_frequency=fmax, minimum_frequency=fmin,
            samples_per_peak=spp,
        )
        df = 1.0 / (spp * baseline)
        f0 = df if fmin is None else fmin
        nf = max(int(np.ceil((fmax - f0) / df)) + 1, 1)
        assert np.array_equal(grid.values, f0 + df * np.arange(nf, dtype=np.float64))
        assert G.uniform_grid_size(
            baseline, maximum_frequency=fmax, minimum_frequency=fmin,
            samples_per_peak=spp,
        ) == nf


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_log_period_grid_rejects_non_finite_bounds(bad: float) -> None:
    with pytest.raises(ValueError, match="period"):
        log_period_grid(minimum_period=1.0, maximum_period=bad, n_periods=10)
    with pytest.raises(ValueError, match="period"):
        log_period_grid(minimum_period=bad, maximum_period=10.0, n_periods=10)


# --- CORE-B-9: GridSpec validation -----------------------------------------------


@pytest.mark.parametrize("kind", ["periods", "Period", "freq", "FREQUENCY", 7])
def test_gridspec_rejects_unknown_kind(kind: object) -> None:
    with pytest.raises(ValueError, match="kind"):
        GridSpec(kind=kind, values=np.geomspace(0.3, 30.0, 40))  # type: ignore[arg-type]


def test_gridspec_rejects_non_uniform_values_flagged_uniform() -> None:
    # A log grid flagged uniform made GLS evaluate f0 + df*k up to 0.23 c/d only.
    with pytest.raises(ValueError, match="uniform"):
        GridSpec("frequency", np.geomspace(0.05, 2.0, 1500), uniform=True)


def test_gridspec_rejects_uniform_period_grid() -> None:
    with pytest.raises(ValueError, match="uniform"):
        GridSpec("period", np.linspace(1.0, 2.0, 10), uniform=True)


@pytest.mark.parametrize(
    "values",
    [
        np.array([0.1, 0.3, 0.2]),  # not ascending
        np.array([2.0, 1.0, 0.5]),  # descending
        np.array([0.1, np.nan, 0.3]),
        np.array([0.1, 0.2, np.inf]),
    ],
)
def test_gridspec_rejects_unordered_or_non_finite_values(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="ascending|finite"):
        GridSpec("frequency", values)


@pytest.mark.parametrize(
    ("baseline", "overrides"),
    [
        (10.8, {}),
        (16.6, {}),
        (1092.66, {}),
        (81.0, {"segment_factor": 1.5}),
    ],
)
def test_bls_default_grid_accepted_at_segment_joins(
    baseline: float, overrides: dict[str, float]
) -> None:
    """BLS concatenates one ``np.arange`` per log segment; arange can overshoot its
    stop by a rounding quantum, so a join steps back by ~1e-11 of the period. That is
    not a reversed grid and must not crash the default BLS search (it did for ~1% of
    round-number baselines)."""
    from cuperiod.core.config import BLSSettings
    from cuperiod.core.lightcurve import LightCurve
    from cuperiod.methods.base import get_method

    time = np.linspace(0.0, baseline, 500)
    lc = LightCurve.from_arrays(
        time, np.ones(time.size), np.full(time.size, 1e-3), domain="flux"
    )
    grid = get_method("BLS").default_grid(lc, BLSSettings(**overrides))
    assert grid.size > 0


def test_gridspec_tolerates_rounding_level_reversals_only() -> None:
    GridSpec("period", np.array([0.2, 0.4, 0.4 - 3e-13, 0.8]))
    GridSpec("frequency", np.array([1.0, 2.0, 2.0 - 1e-12, 3.0]))
    with pytest.raises(ValueError, match="ascending"):
        GridSpec("period", np.array([0.4, 0.2]))
    with pytest.raises(ValueError, match="ascending"):
        GridSpec("period", np.array([0.2, 0.4, 0.4 - 1e-6, 0.8]))
    # A dense but wholly descending window: every step is tiny, the span is not.
    with pytest.raises(ValueError, match="ascending"):
        GridSpec("frequency", np.linspace(1.0, 0.99, 2_000_001))
    # A tolerated step back must not slip a value out of the domain.
    with pytest.raises(ValueError, match="frequencies must be >= 0"):
        GridSpec("frequency", np.array([0.0, -1e-12, 1.0, 10.0]))
    with pytest.raises(ValueError, match="periods must be > 0"):
        GridSpec("period", np.array([1e-12, -1e-12, 1.0, 10.0]))


def test_gridspec_ascending_error_points_at_the_offending_sample() -> None:
    with pytest.raises(ValueError, match=r"values\[2\]"):
        GridSpec("frequency", np.array([0.1, 0.2, 0.3, 0.25, 0.4]))


def test_gridspec_rejects_negative_frequency_and_non_positive_period() -> None:
    with pytest.raises(ValueError, match="frequenc"):
        GridSpec("frequency", np.array([-1.0, 0.5, 1.0]))
    with pytest.raises(ValueError, match="period"):
        GridSpec("period", np.array([0.0, 0.5, 1.0]))


def test_gridspec_accepts_legitimate_uniform_grids() -> None:
    GridSpec("frequency", np.linspace(0.05, 6.0, 4001), uniform=True)
    GridSpec("frequency", np.empty(0), uniform=True)  # empty-grid error paths
    GridSpec("frequency", np.array([0.81]), uniform=True)
    # pdot-style narrow window at high frequency: rounding is ~eps*f, not ~df.
    f_center, df = 5000.0, 1e-10
    GridSpec("frequency", f_center + df * np.arange(-50, 51, dtype=np.float64),
             uniform=True)
    grid = uniform_frequency_grid(3000.0, maximum_frequency=20.0, samples_per_peak=10)
    assert grid.uniform
    GridSpec("frequency", grid.values, uniform=True)
    GridSpec("frequency", np.array([0.0, 0.1, 0.2]))  # f = 0 is a legal sample
    GridSpec("frequency", np.array([0.1, 0.2, 0.2, 0.3]))  # repeated shared endpoint


# --- METH-B-8: the opposite-kind view must not have negative strides ----------------


@pytest.mark.parametrize("kind", ["frequency", "period"])
@pytest.mark.parametrize("n", [1, 2, 5])
def test_gridspec_views_have_positive_strides(kind: str, n: int) -> None:
    grid = GridSpec(kind, np.linspace(0.5, 1.5, n))  # type: ignore[arg-type]
    other = grid.period if kind == "frequency" else grid.frequency
    assert all(s >= 0 for s in other.strides)
    assert np.array_equal(other, (1.0 / grid.values)[::-1])


# --- CORE-B-13: identity equality and hashing --------------------------------------


def test_gridspec_equality_and_hash_do_not_raise() -> None:
    a = np.linspace(0.1, 1.0, 10)
    g1 = GridSpec("frequency", a)
    g2 = GridSpec("frequency", a.copy())
    assert (g1 == g2) is False
    assert g1 == g1
    assert g2 in [g1, g2]
    assert [g1, g2].index(g2) == 1
    assert {g1: 1, g2: 2}[g2] == 2


# --- CORE-B-8 / SEC-04: pseudo-Nyquist on multi-exposure cadences ------------------


def _nightly(n: int = 400, span: int = 2000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nights = np.sort(rng.choice(np.arange(span), size=n, replace=False)).astype(float)
    return nights + 0.3 + 0.05 * rng.standard_normal(nights.size)


def _classic(time: np.ndarray, factor: float = 5.0) -> float:
    dt = np.diff(np.sort(time))
    dt = dt[dt > 0.0]
    return float(factor) * 0.5 / float(np.median(dt))


def _ordinary_cadences() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    nightly = _nightly()
    tess = np.arange(0.0, 27.0, 2.0 / 1440.0)
    runs = np.concatenate(
        [n + np.arange(0.0, 2.0 / 24.0, 30.0 / DAY_S) for n in range(0, 1095, 3)]
    )
    return {
        "random": np.sort(rng.uniform(0.0, 1000.0, 500)),
        "nightly": nightly,
        "25% paired": np.sort(np.concatenate([nightly, nightly[::4] + 105.0 / DAY_S])),
        "tess two sectors": np.concatenate([tess, tess + 365.0]),
        "dense nightly runs": runs,
    }


@pytest.mark.parametrize("name", list(_ordinary_cadences()))
def test_pseudo_nyquist_unchanged_for_ordinary_cadences(name: str) -> None:
    """Single visits, continuous space data and dense nightly runs keep the
    classic median-gap value bit-for-bit."""
    time = _ordinary_cadences()[name]
    for factor in (1, 5, 20):
        assert pseudo_nyquist_frequency(time, factor) == _classic(time, factor)


@pytest.mark.parametrize(
    ("exposures", "spacing_s"), [(3, 105.0), (2, 20.0), (5, 72.0), (4, 900.0)]
)
def test_pseudo_nyquist_bounded_on_multi_exposure_visits(
    exposures: int, spacing_s: float
) -> None:
    """Back-to-back exposures no longer set a 2e7-1e8 sample grid (CORE-B-8)."""
    nightly = _nightly()
    time = np.sort(
        np.concatenate([nightly + k * spacing_s / DAY_S for k in range(exposures)])
    )
    baseline = float(time[-1] - time[0])
    classic = _classic(time)
    robust = pseudo_nyquist_frequency(time)
    assert robust * baseline == pytest.approx(G.MULTI_EXPOSURE_RAYLEIGH_BUDGET)
    assert robust < classic
    # Never below the visit cadence's own estimate (one point per night here).
    assert robust >= _classic(nightly)


def test_multi_exposure_default_grid_is_a_prefix_of_the_old_grid() -> None:
    """The fold-method default grid shrinks ~40x but keeps every old sample below
    the new ceiling, so a signal in that range gets exactly the same statistic."""
    from cuperiod.core.lightcurve import LightCurve
    from cuperiod.methods.base import get_method

    nightly = _nightly()
    time = np.sort(np.concatenate([nightly + k * 105.0 / DAY_S for k in range(3)]))
    y = 12.0 + 0.3 * np.sin(2 * np.pi * time / 0.567)
    lc = LightCurve.from_arrays(time, y, np.full(time.size, 0.02))
    method = get_method("PDM")
    settings = method.coerce_settings(None)
    grid = method.default_grid(lc, settings)
    baseline = float(time[-1] - time[0])
    spp = settings.samples_per_peak
    old_size = G.uniform_grid_size(
        baseline,
        maximum_frequency=_classic(time),
        minimum_frequency=1.0 / baseline,
        samples_per_peak=spp,
    )
    assert old_size > 20_000_000  # the regression: ~2e7 trial frequencies
    assert grid.size <= spp * G.MULTI_EXPOSURE_RAYLEIGH_BUDGET + 2
    df = 1.0 / (spp * baseline)
    old_prefix = 1.0 / baseline + df * np.arange(grid.size, dtype=np.float64)
    assert np.array_equal(grid.values, old_prefix)
    assert grid.values[-1] > 1.0 / 0.567  # the planted signal is still in band


def test_nyquist_factor_above_default_widens_the_multi_exposure_cap() -> None:
    """The cap's budget scales with nyquist_factor / 5, so the documented knob still
    widens the search on a multi-exposure cadence; at or below 5 nothing changes."""
    nightly = _nightly()
    time = np.sort(np.concatenate([nightly + k * 105.0 / DAY_S for k in range(3)]))
    baseline = float(time[-1] - time[0])
    budget = G.MULTI_EXPOSURE_RAYLEIGH_BUDGET
    at_default = pseudo_nyquist_frequency(time, 5)
    assert at_default * baseline == pytest.approx(budget)
    for factor in (1, 2):
        assert pseudo_nyquist_frequency(time, factor) == at_default
    assert pseudo_nyquist_frequency(time, 20) * baseline == pytest.approx(4 * budget)
    assert at_default < pseudo_nyquist_frequency(time, 20) < _classic(time, 20)


def test_stacked_near_simultaneous_bands_are_bounded() -> None:
    """Five bands 72 s apart (Stripe 82 drift scans) stack into 5-exposure visits."""
    epochs = _nightly(n=70, span=3300, seed=3)
    stacked = np.sort(np.concatenate([epochs + k * 72.0 / DAY_S for k in range(5)]))
    baseline = float(stacked[-1] - stacked[0])
    fmax = pseudo_nyquist_frequency(stacked)
    assert fmax * baseline == pytest.approx(G.MULTI_EXPOSURE_RAYLEIGH_BUDGET)
    assert fmax > 10.0  # RR Lyrae (1.5-3 c/d) and most delta Scuti stay in band


# --- SEC-04 / CORE-B-8: size guard ----------------------------------------------------


def test_oversized_grid_raises_before_allocating() -> None:
    """One t=0 row in JD data asked for ~1e9 samples: now a clear error,
    raised before anything is allocated."""
    tracemalloc.start()
    try:
        with pytest.raises(G.GridTooLargeError, match="maximum_frequency"):
            uniform_frequency_grid(2.46e6, maximum_frequency=200.0, samples_per_peak=10)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 10_000_000
    assert issubclass(G.GridTooLargeError, ValueError)
    assert issubclass(G.GridTooLargeError, CuPeriodError)


def test_max_size_is_overridable() -> None:
    with pytest.raises(G.GridTooLargeError, match="1,000"):
        uniform_frequency_grid(100.0, maximum_frequency=10.0, max_size=1000)
    grid = uniform_frequency_grid(100.0, maximum_frequency=10.0, max_size=5000)
    assert grid.size == 5000


def test_check_grid_size_names_outlying_timestamps() -> None:
    rng = np.random.default_rng(5)
    time = np.concatenate([[0.0], 58000.0 + np.sort(rng.uniform(0, 1000, 2000))])
    with pytest.raises(G.GridTooLargeError, match="outlying"):
        G.check_grid_size(10**9, time=time)


def test_automatic_grid_with_outlier_timestamp_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below the hard limit an outlier-inflated automatic grid is computed, with the
    diagnosis in the warning: a few early epochs may be genuine archival data."""
    monkeypatch.setattr(G, "AUTO_GRID_WARN_SIZE", 100_000)
    rng = np.random.default_rng(5)
    time = np.concatenate([[0.0], 58000.0 + np.sort(rng.uniform(0, 1000, 2000))])
    with pytest.warns(G.GridSizeWarning, match="outlying"):
        grid = G.default_frequency_grid(time)
    assert grid.size > 100_000
    # An explicit band is the user's call: no warning, just the hard ceiling.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grid = G.default_frequency_grid(time, maximum_frequency=0.5)
    assert grid.size > 0


def test_archival_epochs_with_a_space_sector_warn_rather_than_raise() -> None:
    """A TESS 2-min sector plus 50 ground points from ten years earlier: the baseline
    is ~135x the central span and the automatic fold grid ~3.3e7 samples. That is a
    legitimate combination, so it warns (naming the archival case) and is not
    refused; only the hard limit refuses."""
    rng = np.random.default_rng(9)
    sector = 3650.0 + np.arange(0.0, 27.0, 2.0 / 1440.0)
    archival = np.sort(rng.uniform(0.0, 300.0, 50))
    time = np.concatenate([archival, sector])
    baseline = float(time[-1] - time[0])
    n_points = G.uniform_grid_size(
        baseline,
        maximum_frequency=pseudo_nyquist_frequency(time, 5),
        minimum_frequency=1.0 / baseline,
    )
    assert G.AUTO_GRID_WARN_SIZE < n_points < G.MAX_GRID_SIZE
    with pytest.warns(G.GridSizeWarning, match="archival") as record:
        G.check_grid_size(n_points, automatic=True, time=time, context="PDM")
    assert "outlying" in str(record[0].message)
    with pytest.raises(G.GridTooLargeError, match="archival"):
        G.check_grid_size(G.MAX_GRID_SIZE + 1, automatic=True, time=time)


def test_large_automatic_grid_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(G, "AUTO_GRID_WARN_SIZE", 1000)
    time = np.arange(0.0, 27.0, 2.0 / 1440.0)
    with pytest.warns(G.GridSizeWarning, match="maximum_frequency"):
        G.default_frequency_grid(time)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        G.default_frequency_grid(time, maximum_frequency=5.0)


def test_default_frequency_grid_matches_fold_method_recipe() -> None:
    """The shared helper reproduces the fold methods' default grid sample-for-sample."""
    time = _nightly()
    baseline = float(time.max() - time.min())
    expected = uniform_frequency_grid(
        baseline,
        maximum_frequency=pseudo_nyquist_frequency(time, 5),
        minimum_frequency=1.0 / baseline,
        samples_per_peak=5,
    )
    grid = G.default_frequency_grid(time, samples_per_peak=5, nyquist_factor=5)
    assert np.array_equal(grid.values, expected.values)
    floored = G.default_frequency_grid(time, maximum_frequency_floor=50.0)
    assert floored.values[-1] >= 50.0


def test_default_frequency_grid_needs_a_baseline() -> None:
    with pytest.raises(InsufficientDataError, match="baseline"):
        G.default_frequency_grid(np.array([1.0, 1.0, 1.0]))


# --- regular-cadence Nyquist (helper for METH-A-2 / PW-A-3) ----------------------


def test_regular_cadence_nyquist_on_lattices() -> None:
    lattice = 2460000.5 + np.arange(1296) / 48.0
    assert G.regular_cadence_nyquist(lattice) == pytest.approx(24.0, rel=1e-9)
    # Smooth barycentric timing term (400 s amplitude) over 27 d.
    bary = lattice + 400.0 / DAY_S * np.sin(2 * np.pi * (lattice - lattice[0]) / 365.25)
    assert G.regular_cadence_nyquist(bary) == pytest.approx(24.0, rel=1e-3)
    # 60 % of the cadences dropped: still a lattice with the original step.
    keep = np.random.default_rng(11).random(lattice.size) > 0.6
    assert G.regular_cadence_nyquist(lattice[keep]) == pytest.approx(24.0, rel=1e-9)
    # Gapped TESS-like 2-min sector.
    t = np.arange(0.0, 27.0, 2.0 / 1440.0)
    assert G.regular_cadence_nyquist(t[(t < 13) | (t > 14)]) == pytest.approx(360.0)


def test_regular_cadence_nyquist_rejects_irregular_sampling() -> None:
    rng = np.random.default_rng(2)
    lattice = 2460000.5 + np.arange(1296) / 48.0
    assert G.regular_cadence_nyquist(np.sort(rng.uniform(0, 1000, 500))) is None
    assert G.regular_cadence_nyquist(_nightly()) is None
    jitter = lattice + rng.normal(0.0, 0.05 / 48.0, lattice.size)
    assert G.regular_cadence_nyquist(jitter) is None
    # Nightly runs on a 30-s lattice whose phase restarts at random every night:
    # super-Nyquist frequencies are distinguishable, so this is not a lattice.
    starts = np.arange(100) + rng.uniform(0.0, 0.02, 100)
    runs = np.concatenate([s + np.arange(240) * 30.0 / DAY_S for s in starts])
    assert G.regular_cadence_nyquist(runs) is None
    assert G.regular_cadence_nyquist(np.array([1.0, 2.0, 3.0])) is None
    # Multi-year barycentric drift decorrelates the lattice (super-Nyquist regime).
    long = 2460000.5 + np.arange(0, 1460, 30 / 1440)
    long = long + 400.0 / DAY_S * np.sin(2 * np.pi * (long - long[0]) / 365.25)
    assert G.regular_cadence_nyquist(long) is None
    loose = G.regular_cadence_nyquist(long, min_coherence=0.5)
    assert loose == pytest.approx(24.0, rel=1e-3)


def test_default_frequency_grid_can_stop_at_the_nyquist_frequency() -> None:
    lattice = np.arange(1296) / 48.0
    grid = G.default_frequency_grid(
        lattice, maximum_frequency_floor=50.0, stop_at_nyquist=True
    )
    assert grid.values[-1] == pytest.approx(24.0, abs=grid.values[1] - grid.values[0])
    irregular = np.sort(np.random.default_rng(3).uniform(0.0, 27.0, 1296))
    capped = G.default_frequency_grid(irregular, stop_at_nyquist=True)
    assert np.array_equal(capped.values, G.default_frequency_grid(irregular).values)


def test_size_message_reports_memory() -> None:
    with pytest.raises(G.GridTooLargeError) as info:
        G.check_grid_size(300_000_000, baseline=1000.0, maximum_frequency=60000.0)
    text = str(info.value)
    assert "300,000,000" in text and "GiB" in text and "1000" in text
    assert math.isfinite(float(len(text)))


# --- public re-exports (SEC-04, CORE-B-8, METH-A-2) ---------------------------------

_REEXPORTED = (
    "GridSizeWarning",
    "GridTooLargeError",
    "check_grid_size",
    "default_frequency_grid",
    "pseudo_nyquist_frequency",
    "regular_cadence_nyquist",
    "uniform_grid_size",
)


@pytest.mark.parametrize("name", _REEXPORTED)
def test_grid_policy_is_reachable_from_the_package(name: str) -> None:
    # The refusal had to be caught as cuperiod.core.grid.GridTooLargeError, next to
    # cup.InsufficientDataError at the top level.
    import cuperiod as cup
    import cuperiod.core as core

    for namespace in (cup, core):
        assert name in namespace.__all__
        assert getattr(namespace, name) is getattr(G, name)


def test_oversized_grid_is_caught_from_the_top_level() -> None:
    import cuperiod as cup

    with pytest.raises(cup.GridTooLargeError):
        cup.uniform_frequency_grid(100.0, maximum_frequency=10.0, max_size=1000)
