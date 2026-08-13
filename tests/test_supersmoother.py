"""SuperSmoother: reference-package parity, backend parity, and behavior."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import cuperiod as cup
from conftest import requires_gpu, requires_numba, requires_torch
from cuperiod.core.errors import InsufficientDataError
from cuperiod.methods.supersmoother import span_windows, supersmoother_score
from synth import synthetic_eclipser, synthetic_sine

PERIOD = 0.7365

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
