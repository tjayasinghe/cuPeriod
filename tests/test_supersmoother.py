"""SuperSmoother: reference-package parity, backend parity, and behavior."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import cuperiod as cup
import cuperiod.core.grid as grid_module
from conftest import requires_gpu, requires_numba, requires_torch
from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.grid import (
    GridSizeWarning,
    pseudo_nyquist_frequency,
    uniform_frequency_grid,
)
from cuperiod.methods.supersmoother import span_windows, supersmoother_score
from synth import synthetic_eclipser, synthetic_sine

PERIOD = 0.7365

#: The CPU backends every numerical regression below runs on.
CPU_BACKENDS = ["numpy", pytest.param("numba", marks=requires_numba)]

#: A handful of trial periods spanning wrong, true, and multiple-of-true folds.
TEST_PERIODS = np.array([0.5, PERIOD, 0.9, 1.4, 2.2])


def _reference_star(
    n: int = 115, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A star whose point count makes cuPeriod's windows match the PyPI reference.

    cuPeriod forces span windows to odd point counts (the upstream master
    semantics); the released ``supersmoother`` 0.4 truncates instead. At
    ``n = 115`` (and 75) every default span truncates to an odd count already,
    so the two conventions coincide and parity can be asserted exactly.
    """
    rng = np.random.RandomState(seed)
    t = np.sort(rng.uniform(0, 30, n))
    y = (
        14.0
        + 0.4 * np.sin(2 * np.pi * t / PERIOD)
        + 0.15 * np.sin(4 * np.pi * t / PERIOD + 0.3)
    )
    dy = np.full(n, 0.05)
    return t, y + rng.normal(0, 0.05, n), dy


# --- window semantics ---------------------------------------------------------


def test_span_windows_odd_and_floored() -> None:
    assert span_windows((0.05, 0.2, 0.5), 100) == (5, 21, 51)
    assert span_windows((0.05, 0.2, 0.5), 115) == (5, 23, 57)
    assert span_windows((0.01,), 100) == (3,)


# --- reference parity ---------------------------------------------------------


def test_score_matches_reference_package() -> None:
    ssm = pytest.importorskip("supersmoother")
    t, y, dy = _reference_star()
    mine = supersmoother_score(t, y, dy, TEST_PERIODS)
    w = 1.0 / dy**2
    mu = np.average(y, weights=w)
    baseline = float(np.mean(np.abs((y - mu) / dy)))
    for score, p in zip(mine, TEST_PERIODS, strict=True):
        model = ssm.SuperSmoother(period=p).fit(t, y, dy)
        expected = 1.0 - model.cv_error(skip_endpoints=False) / baseline
        assert score == pytest.approx(expected, abs=1e-9)


def test_score_matches_gatspy() -> None:
    gp = pytest.importorskip("gatspy.periodic")
    t, y, dy = _reference_star()
    mine = supersmoother_score(t, y, dy, TEST_PERIODS)
    ref = gp.SuperSmoother().fit(t, y, dy).score(TEST_PERIODS)
    assert np.allclose(mine, ref, atol=1e-9)


def test_multiband_matches_gatspy() -> None:
    gp = pytest.importorskip("gatspy.periodic")
    tg, yg, eg = _reference_star(n=115, seed=1)
    tr, yr, er = _reference_star(n=75, seed=2)
    yr = yr + 1.0  # distinct band mean
    mb = cup.MultiBandLightCurve.from_light_curves(
        {
            "g": cup.LightCurve.from_arrays(tg, yg, eg),
            "r": cup.LightCurve.from_arrays(tr, yr, er),
        }
    )
    freqs = np.sort(1.0 / TEST_PERIODS)
    grid = cup.GridSpec(kind="frequency", values=freqs, uniform=False)
    pg = cup.periodogram(mb, "SuperSmoother", grid=grid, backend="numpy")
    assert isinstance(pg, cup.Periodogram)

    model = gp.SuperSmootherMultiband().fit(
        np.concatenate([tg, tr]),
        np.concatenate([yg, yr]),
        np.concatenate([eg, er]),
        np.array(["g"] * tg.size + ["r"] * tr.size),
    )
    expected = model.score(1.0 / freqs)
    assert np.allclose(pg.power, expected, atol=1e-9)


# --- period recovery ----------------------------------------------------------


def test_recovers_sine_period() -> None:
    t, mag, err = synthetic_sine(period=0.6234)
    pg = cup.periodogram((t, mag, err), "SuperSmoother", backend="numpy")
    assert isinstance(pg, cup.Periodogram)
    assert pg.method == "SUPERSMOOTHER"
    assert pg.objective_sense == "max"
    assert pg.best_period() == pytest.approx(0.6234, rel=5e-3)


def test_recovers_eclipser_period() -> None:
    # A non-sinusoidal fold is SuperSmoother's home turf; the eclipser may
    # legitimately resolve to P or, with a similar secondary, P/2.
    t, flux, err = synthetic_eclipser(period=2.5, depth=0.08)
    lc = cup.LightCurve.from_arrays(t, flux, err, domain=cup.Domain.FLUX)
    grid = cup.GridSpec(
        kind="frequency", values=np.linspace(0.2, 1.1, 3000), uniform=True
    )
    pg = cup.periodogram(lc, "SuperSmoother", grid=grid, backend="numpy")
    assert isinstance(pg, cup.Periodogram)
    ratio = pg.best_period() / 2.5
    assert min(abs(ratio - 1.0), abs(ratio - 0.5)) < 1e-2


# --- backend parity -----------------------------------------------------------


def _parity_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, y, dy = _reference_star(n=180, seed=3)
    periods = 1.0 / np.linspace(0.8, 2.2, 400)
    return t, y, dy, periods


@requires_numba
@pytest.mark.parametrize("alpha", [None, 8.0])
def test_numba_matches_numpy(alpha: float | None) -> None:
    t, y, dy, periods = _parity_inputs()
    cpu = supersmoother_score(t, y, dy, periods, bass_enhancement=alpha)
    fast = supersmoother_score(
        t, y, dy, periods, bass_enhancement=alpha, backend="numba"
    )
    assert np.allclose(fast, cpu, rtol=1e-9, atol=1e-11)


@requires_torch
def test_torch_cpu_matches_numpy() -> None:
    t, y, dy, periods = _parity_inputs()
    cpu = supersmoother_score(t, y, dy, periods)
    tor = supersmoother_score(t, y, dy, periods, backend="torch:cpu")
    assert np.allclose(tor, cpu, rtol=1e-9, atol=1e-11)


@requires_gpu
def test_cupy_matches_numpy() -> None:
    t, y, dy, periods = _parity_inputs()
    cpu = supersmoother_score(t, y, dy, periods)
    gpu = supersmoother_score(t, y, dy, periods, backend="cupy")
    assert np.allclose(gpu, cpu, rtol=1e-8, atol=1e-10)


# --- tied phases (METH-C-9, MULTIBAND-14, PERF-METHODS-07) --------------------


def _tied_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integer-day timestamps (duplicates included), unequal errors, and a grid
    of commensurate trial frequencies: many phases tie exactly."""
    rng = np.random.default_rng(3)
    t = np.round(np.sort(rng.uniform(0.0, 80.0, 60)))
    y = 15.0 + 0.3 * np.sin(2 * np.pi * t / PERIOD) + rng.normal(0.0, 0.05, 60)
    dy = rng.uniform(0.03, 0.08, 60)
    periods = 1.0 / np.linspace(1.0, 2.5, 400)
    return t, y, dy, periods


@requires_numba
@pytest.mark.parametrize("alpha", [None, 8.0])
def test_numba_matches_numpy_with_tied_phases(alpha: float | None) -> None:
    # The numba kernel sorted with an unstable quicksort, so tied phases were
    # ordered differently from the stable array-API sort (diffs up to ~0.17).
    t, y, dy, periods = _tied_inputs()
    cpu = supersmoother_score(t, y, dy, periods, bass_enhancement=alpha)
    fast = supersmoother_score(
        t, y, dy, periods, bass_enhancement=alpha, backend="numba"
    )
    assert np.allclose(fast, cpu, rtol=1e-9, atol=1e-11)


@requires_torch
def test_torch_cpu_matches_numpy_with_tied_phases() -> None:
    t, y, dy, periods = _tied_inputs()
    cpu = supersmoother_score(t, y, dy, periods)
    tor = supersmoother_score(t, y, dy, periods, backend="torch:cpu")
    assert np.allclose(tor, cpu, rtol=1e-9, atol=1e-11)


@requires_gpu
def test_cupy_tied_phases_are_input_order_invariant() -> None:
    # The GPU sort is stable too, so the canonical row order fixes the tie order.
    # (Scores of windows whose points share one phase can still differ from the
    # CPU at ~1e-3: the degenerate-window test runs on prefix sums whose rounding
    # follows the device's scan order.)
    t, y, dy, periods = _tied_inputs()
    a = supersmoother_score(t, y, dy, periods, backend="cupy")
    perm = np.random.default_rng(1).permutation(t.size)
    b = supersmoother_score(t[perm], y[perm], dy[perm], periods, backend="cupy")
    assert np.array_equal(a, b)


@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize("errors", [True, False])
def test_tied_phases_are_input_order_invariant(backend: str, errors: bool) -> None:
    t, y, dy, periods = _tied_inputs()
    err = dy if errors else None
    a = supersmoother_score(t, y, err, periods, backend=backend)
    for seed in (1, 2):
        perm = np.random.default_rng(seed).permutation(t.size)
        b = supersmoother_score(
            t[perm], y[perm], None if err is None else err[perm], periods,
            backend=backend,
        )
        assert np.array_equal(a, b)


@requires_numba
def test_numba_matches_numpy_on_a_default_grid() -> None:
    # Every samples_per_peak-th default frequency is a multiple of 1/baseline, so
    # the first and last observations fold onto exactly the same phase.
    rng = np.random.default_rng(5)
    t = np.sort(rng.integers(0, 1000, 100) + rng.uniform(0.0, 0.3, 100))
    y = 15.0 + 0.3 * np.sin(2 * np.pi * t / 0.61) + rng.normal(0.0, 0.05, 100)
    dy = np.full(100, 0.05)
    lc = cup.LightCurve.from_arrays(t, y, dy)
    periods = cup.get_method("SuperSmoother").default_grid(
        lc, cup.SuperSmootherSettings()
    ).period
    cpu = supersmoother_score(t, y, dy, periods)
    fast = supersmoother_score(t, y, dy, periods, backend="numba")
    assert np.allclose(fast, cpu, rtol=1e-9, atol=1e-11)


@requires_numba
def test_multiband_numba_matches_numpy_with_a_tied_band() -> None:
    t, y, dy, _ = _tied_inputs()
    tr, yr, er = _reference_star(n=75, seed=2)
    mb = cup.MultiBandLightCurve.from_light_curves(
        {
            "g": cup.LightCurve.from_arrays(t, y, dy),
            "r": cup.LightCurve.from_arrays(tr, yr + 1.0, er),
        }
    )
    grid = cup.GridSpec(
        kind="frequency", values=np.linspace(1.0, 2.5, 300), uniform=True
    )
    cpu = cup.periodogram(mb, "SuperSmoother", grid=grid, backend="numpy")
    fast = cup.periodogram(mb, "SuperSmoother", grid=grid, backend="numba")
    assert isinstance(cpu, cup.Periodogram) and isinstance(fast, cup.Periodogram)
    assert np.allclose(fast.power, cpu.power, rtol=1e-9, atol=1e-11)


@requires_numba
@pytest.mark.parametrize("n", [5, 40, 2000])
def test_numba_stable_sort_handles_degenerate_periods(n: int) -> None:
    # A zero or NaN period folds to NaN phases; the O(N) bucket sort must fall
    # back instead of indexing out of bounds, and the valid periods are unaffected.
    rng = np.random.default_rng(n)
    t = np.sort(rng.uniform(0.0, 50.0, n))
    y = np.sin(2 * np.pi * t / PERIOD) + rng.normal(0.0, 0.1, n)
    periods = np.array([PERIOD, 0.0, np.nan, 1.3, -0.9])
    with np.errstate(divide="ignore", invalid="ignore"):
        fast = supersmoother_score(t, y, None, periods, backend="numba")
        cpu = supersmoother_score(t, y, None, periods)
    assert fast.shape == (5,)
    valid = [0, 3, 4]
    assert np.allclose(fast[valid], cpu[valid], rtol=1e-9, atol=1e-11)


@requires_numba
def test_numba_kernel_closes_over_no_dispatchers() -> None:
    # numba's on-disk cache key hashes the pickled closure; a closure over other
    # jitted helpers embeds a per-process id and missed the cache in every process.
    from cuperiod.methods.supersmoother import _numba_ss_kernel

    kernel = _numba_ss_kernel()
    assert kernel.py_func.__closure__ is None
    assert kernel.py_func.__module__ == "cuperiod.methods._supersmoother_numba"


# --- default grid (METH-C-12) -------------------------------------------------


def test_default_grid_is_the_classic_recipe_on_ordinary_cadence() -> None:
    t, y, dy = _reference_star(n=300, seed=4)
    lc = cup.LightCurve.from_arrays(t, y, dy)
    settings = cup.SuperSmootherSettings()
    grid = cup.get_method("SuperSmoother").default_grid(lc, settings)
    baseline = float(t.max() - t.min())
    expected = uniform_frequency_grid(
        baseline,
        maximum_frequency=pseudo_nyquist_frequency(t, settings.nyquist_factor),
        minimum_frequency=1.0 / baseline,
        samples_per_peak=settings.samples_per_peak,
    )
    assert np.array_equal(grid.values, expected.values)
    # An explicit 0.0 lower edge keeps meaning "1/baseline", as before.
    zero = cup.SuperSmootherSettings(minimum_frequency=0.0)
    assert np.array_equal(
        cup.get_method("SuperSmoother").default_grid(lc, zero).values,
        expected.values,
    )


def test_default_grid_stays_bounded_on_multi_exposure_visits() -> None:
    # 200 visits x 3 exposures within 90 s over ~1000 d: the median gap is an
    # exposure spacing and the classic ceiling reached ~2500 c/d (1e7 trials).
    rng = np.random.default_rng(6)
    visits = np.sort(rng.uniform(0.0, 1000.0, 200))
    t = np.sort((visits[:, None] + np.array([0.0, 45.0, 90.0]) / 86400.0).ravel())
    y = 15.0 + 0.3 * np.sin(2 * np.pi * t / 0.61) + rng.normal(0.0, 0.05, t.size)
    lc = cup.LightCurve.from_arrays(t, y, np.full(t.size, 0.05))
    grid = cup.get_method("SuperSmoother").default_grid(
        lc, cup.SuperSmootherSettings()
    )
    assert grid.values.size < 1_000_000
    assert grid.values[0] < 1.0 / 0.61 < grid.values[-1]


def test_default_grid_warns_about_an_oversized_automatic_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t, y, dy = _reference_star(n=200, seed=5)
    t = np.concatenate([[t.min() - 3000.0], t])
    lc = cup.LightCurve.from_arrays(t, np.append(y[:1], y), np.append(dy[:1], dy))
    monkeypatch.setattr(grid_module, "AUTO_GRID_WARN_SIZE", 1000)
    with pytest.warns(GridSizeWarning, match="SUPERSMOOTHER"):
        cup.get_method("SuperSmoother").default_grid(
            lc, cup.SuperSmootherSettings()
        )


# --- behavior -----------------------------------------------------------------


def test_input_order_invariance() -> None:
    t, y, dy = _reference_star()
    rng = np.random.default_rng(1)
    perm = rng.permutation(t.size)
    a = supersmoother_score(t, y, dy, TEST_PERIODS)
    b = supersmoother_score(t[perm], y[perm], dy[perm], TEST_PERIODS)
    assert np.allclose(a, b, atol=1e-12)


def test_alpha_ten_pins_the_largest_span() -> None:
    # Full bass enhancement forces every point onto the woofer span, which must
    # equal running with that single span alone (crossing both code paths).
    t, y, dy = _reference_star()
    pinned = supersmoother_score(t, y, dy, TEST_PERIODS, bass_enhancement=10.0)
    woofer = supersmoother_score(t, y, dy, TEST_PERIODS, primary_spans=(0.5,))
    assert np.allclose(pinned, woofer, atol=1e-12)


def test_constant_signal_scores_zero() -> None:
    t = np.linspace(0.0, 10.0, 50)
    scores = supersmoother_score(t, np.ones(50), None, np.array([1.0, 2.0]))
    assert np.array_equal(scores, np.zeros(2))


@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize("errors", ["none", "uniform", "hetero"])
@pytest.mark.parametrize("level", [1.0, 3.0, 12.345, 99.99, 1e4])
def test_any_constant_signal_scores_zero(
    level: float, errors: str, backend: str
) -> None:
    # The weighted mean of a constant is rarely bit-equal to it; the ~1e-16
    # residual used to slip past the `baseline <= 0` guard and score 1.0 (the
    # maximum) at every period.
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 60.0, 60))
    dy = {
        "none": None,
        "uniform": np.full(60, 0.05),
        "hetero": rng.uniform(0.01, 0.1, 60),
    }[errors]
    periods = 1.0 / np.linspace(0.05, 2.0, 300)
    scores = supersmoother_score(
        t, np.full(60, level), dy, periods, backend=backend
    )
    assert np.array_equal(scores, np.zeros(periods.size))


def test_long_constant_signal_scores_zero() -> None:
    # The constancy test looks at the data, not at the (length-dependent)
    # rounding of its weighted mean.
    n = 1_000_000
    t = np.linspace(0.0, 1000.0, n)
    scores = supersmoother_score(
        t, np.full(n, 12.345), np.full(n, 0.01), np.array([1.0, 2.0])
    )
    assert np.array_equal(scores, np.zeros(2))


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_constant_multiband_scores_zero(backend: str) -> None:
    rng = np.random.default_rng(2)
    bands = {}
    for name, level in (("g", 99.99), ("r", 12.345)):
        t = np.sort(rng.uniform(0.0, 60.0, 40))
        bands[name] = cup.LightCurve.from_arrays(
            t, np.full(40, level), rng.uniform(0.01, 0.1, 40)
        )
    mb = cup.MultiBandLightCurve.from_light_curves(bands)
    grid = cup.GridSpec(
        kind="frequency", values=np.linspace(0.1, 2.0, 200), uniform=True
    )
    pg = cup.periodogram(mb, "SuperSmoother", grid=grid, backend=backend)
    assert isinstance(pg, cup.Periodogram)
    assert np.array_equal(pg.power, np.zeros(200))


def test_empty_periods() -> None:
    t, y, dy = _reference_star()
    assert supersmoother_score(t, y, dy, np.zeros(0)).size == 0


def test_uniform_weights_when_no_errors() -> None:
    t, y, dy = _reference_star()
    scores = supersmoother_score(t, y, None, TEST_PERIODS)
    assert np.all(np.isfinite(scores))
    assert int(np.argmax(scores)) == 1  # still peaks at the true period


# --- guards and settings ------------------------------------------------------


def test_too_few_points_raises() -> None:
    t, y, dy = _reference_star()
    with pytest.raises(InsufficientDataError):
        cup.periodogram((t[:10], y[:10], dy[:10]), "SuperSmoother")


def test_spans_must_increase() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        cup.SuperSmootherSettings(primary_spans=(0.5, 0.2))
    with pytest.raises(ValidationError, match=r"in \(0, 1\]"):
        cup.SuperSmootherSettings(primary_spans=(0.0, 0.5))


def test_single_span_runs() -> None:
    t, y, dy = _reference_star()
    settings = cup.SuperSmootherSettings(primary_spans=(0.3,))
    pg = cup.periodogram((t, y, dy), "SuperSmoother", settings=settings)
    assert isinstance(pg, cup.Periodogram)
    assert np.all(np.isfinite(pg.power))
