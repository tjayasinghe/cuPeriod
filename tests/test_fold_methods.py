"""Shared behaviour of the fold/AOV methods (PDM, CE, String-Length, MHAOV).

The per-method modules pin each statistic against a transparent reference; this one
pins what the four have in common: the automatic grid recipe and its size guard, the
memory-bounded vectorized chunks, NaN at trial values that cannot be folded, and the
error messages for inputs with nothing usable in them.
"""

from __future__ import annotations

import tracemalloc
import warnings
from collections.abc import Callable

import numpy as np
import pytest

import cuperiod as cup
import cuperiod.core.grid as grid_mod
from conftest import requires_gpu, requires_numba, requires_torch
from cuperiod.core.grid import (
    GridSizeWarning,
    pseudo_nyquist_frequency,
    uniform_frequency_grid,
)
from cuperiod.methods._fold import (
    FOLD_CHUNK_BYTES,
    FOLD_DEVICE_CHUNK_BYTES,
    evaluate_trials,
    finite_angles,
    fold_batch,
    fold_phase,
    foldable_periods,
)
from cuperiod.methods.conditional_entropy import conditional_entropy
from cuperiod.methods.mhaov import aov_power
from cuperiod.methods.pdm import pdm_theta
from cuperiod.methods.string_length import string_length
from synth import synthetic_sine

FOLD_METHODS = ["PDM", "CE", "STRINGLENGTH", "MHAOV"]
#: Message prefix each method uses.
LABELS = {"PDM": "PDM", "CE": "CE", "STRINGLENGTH": "string-length", "MHAOV": "MHAOV"}

Kernel = Callable[..., np.ndarray]
#: The public per-trial kernels, called with trial *periods*.
PERIOD_KERNELS: dict[str, Kernel] = {
    "PDM": pdm_theta,
    "CE": conditional_entropy,
    "STRINGLENGTH": string_length,
}


def _nightly(n: int = 300, span: float = 100.0, seed: int = 0) -> cup.LightCurve:
    """One visit per night (plus jitter): the ordinary ground-based cadence."""
    rng = np.random.default_rng(seed)
    t = 2458000.0 + np.sort(rng.choice(int(span), n, replace=True)) + rng.uniform(
        -0.1, 0.1, n
    )
    y = 12.0 + 0.3 * np.sin(2 * np.pi * t / 0.6234) + rng.normal(0.0, 0.02, n)
    return cup.LightCurve.from_arrays(t, y, np.full(n, 0.02))


# --- automatic grid (METH-B-5) ------------------------------------------------------


@pytest.mark.parametrize("method", FOLD_METHODS)
def test_default_grid_is_the_documented_recipe(method: str) -> None:
    # On an ordinary cadence the shared recipe (cuperiod.core.grid.default_frequency_
    # grid) reproduces the methods' historical grid sample for sample.
    lc = _nightly()
    m = cup.get_method(method)
    settings = m.settings_cls()
    got = m.default_grid(lc, settings)
    finite = lc.finite()
    want = uniform_frequency_grid(
        finite.baseline,
        maximum_frequency=pseudo_nyquist_frequency(
            finite.time, settings.nyquist_factor
        ),
        minimum_frequency=1.0 / finite.baseline,
        samples_per_peak=settings.samples_per_peak,
    )
    assert got.kind == want.kind == "frequency"
    assert got.uniform
    assert np.array_equal(got.values, want.values)


@pytest.mark.parametrize("method", FOLD_METHODS)
def test_default_grid_warns_when_automatic_grid_is_huge(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: the automatic grid had no size guard at all. It now goes through
    # the core check, which warns (naming the method) above AUTO_GRID_WARN_SIZE; the
    # threshold is lowered here so an ordinary curve crosses it.
    monkeypatch.setattr(grid_mod, "AUTO_GRID_WARN_SIZE", 1000)
    lc = _nightly()
    m = cup.get_method(method)
    with pytest.warns(GridSizeWarning, match=f"{m.name} default grid"):
        grid = m.default_grid(lc, m.settings_cls())
    assert grid.size > 1000
    # An explicit ceiling is the user's choice: built silently.
    with warnings.catch_warnings():
        warnings.simplefilter("error", GridSizeWarning)
        m.default_grid(lc, m.settings_cls(maximum_frequency=60.0))


@pytest.mark.parametrize("method", FOLD_METHODS)
def test_default_grid_bounded_on_multi_exposure_visits(method: str) -> None:
    # ASAS-SN-like visits (three exposures 90 s apart, one visit per night over five
    # years): the median gap is the intra-visit one, which used to put the automatic
    # ceiling at 2400 c/d and the grid at 21.8 million samples.
    rng = np.random.default_rng(1)
    nights = np.sort(rng.choice(5 * 365, 700, replace=False)).astype(float)
    t = 2458000.0 + (nights[:, None] + np.array([0.0, 90.0, 180.0]) / 86400.0).ravel()
    y = 12.0 + 0.3 * np.sin(2 * np.pi * t / 0.6234)
    y = y + rng.normal(0.0, 0.02, t.size)
    lc = cup.LightCurve.from_arrays(t, y)
    m = cup.get_method(method)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GridSizeWarning)
        grid = m.default_grid(lc, m.settings_cls())
    assert grid.size <= 1_000_000
    assert grid.frequency[-1] > 1.0 / 0.6234  # the signal stays in band


# --- memory-bounded chunks (METH-B-6) -----------------------------------------------


def test_fold_batch_caps_long_curves_only() -> None:
    # Short curves keep the requested chunk; long ones are capped by the budget.
    assert fold_batch(2048, 300, workspaces=6, on_device=False, default=2048) == 2048
    cap = FOLD_CHUNK_BYTES // (20_000 * 8 * 6)
    assert fold_batch(2048, 20_000, workspaces=6, on_device=False, default=2048) == cap
    dev = FOLD_DEVICE_CHUNK_BYTES // (20_000 * 8 * 6)
    assert fold_batch(4096, 20_000, workspaces=6, on_device=True, default=2048) == dev
    assert fold_batch(0, 300, workspaces=6, on_device=False, default=1024) == 1024
    assert fold_batch(5, 10**9, workspaces=10, on_device=False, default=1024) == 1


@pytest.mark.parametrize("method", list(PERIOD_KERNELS))
def test_chunk_length_does_not_change_the_result(method: str) -> None:
    t, mag, _ = synthetic_sine(n=300)
    periods = np.linspace(0.3, 3.0, 257)
    fn = PERIOD_KERNELS[method]
    whole = fn(t, mag, periods, backend="numpy", batch=4096)
    assert np.array_equal(fn(t, mag, periods, backend="numpy", batch=7), whole)


@pytest.mark.parametrize(
    ("method", "n_periods"), [("PDM", 2048), ("CE", 1024), ("STRINGLENGTH", 600)]
)
def test_numpy_fold_memory_is_bounded_on_long_curves(
    method: str, n_periods: int
) -> None:
    # Regression: a fixed chunk of 2048 / 1024 periods made the numpy (and torch)
    # transients grow linearly with the curve: ~570 MB for PDM at 8000 points, about
    # 4 GiB at 65k. The chunk is now budgeted, so the peak stays near 128 MiB.
    rng = np.random.default_rng(3)
    n = 8000
    t = np.sort(rng.uniform(0.0, 200.0, n))
    y = np.sin(2 * np.pi * t / 0.7) + rng.normal(0.0, 0.1, n)
    periods = np.linspace(0.3, 3.0, n_periods)
    fn = PERIOD_KERNELS[method]
    tracemalloc.start()
    try:
        fn(t, y, periods, backend="numpy")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 160 * 1024**2, f"{method} peak {peak / 1024**2:.0f} MiB"


# --- trial values that cannot be folded (METH-B-4) ----------------------------------


def test_foldable_and_finite_masks() -> None:
    periods = np.array([0.0, np.nan, 1e-320, 0.5, np.inf, -2.0])
    assert foldable_periods(10.0, periods).tolist() == [
        False, False, False, True, True, True,
    ]
    freqs = np.array([np.nan, np.inf, -np.inf, 0.0, 2.0, 1e308])
    assert finite_angles(10.0, freqs).tolist() == [
        False, False, False, True, True, False,
    ]


def test_evaluate_trials_fills_nan_and_skips_copy_when_clean() -> None:
    trials = np.array([1.0, 2.0, 3.0])
    seen: list[np.ndarray] = []

    def compute(x: np.ndarray) -> np.ndarray:
        seen.append(x)
        return 10.0 * x

    out = evaluate_trials(compute, trials, np.array([True, True, True]))
    assert seen[-1] is trials and out.tolist() == [10.0, 20.0, 30.0]
    out = evaluate_trials(compute, trials, np.array([True, False, True]))
    assert seen[-1].tolist() == [1.0, 3.0]
    assert out[0] == 10.0 and np.isnan(out[1]) and out[2] == 30.0
    out = evaluate_trials(compute, trials, np.zeros(3, dtype=bool))
    assert np.all(np.isnan(out)) and len(seen) == 2  # nothing left to compute


def test_fold_phase_numpy_matches_remainder_bitwise() -> None:
    rng = np.random.default_rng(5)
    tau = np.concatenate([[0.0], rng.uniform(0.0, 3000.0, 500), [1e9]])
    periods = np.concatenate(
        [rng.uniform(0.05, 50.0, 64), [np.inf, np.nan, 0.0, -1.3, 1e-300]]
    )
    with np.errstate(all="ignore"):
        want = np.remainder(tau[None, :] / periods[:, None], 1.0)
        got = fold_phase(np, tau, periods)
    assert np.array_equal(got, want, equal_nan=True)


def _bad_period_backends() -> list[object]:
    return [
        "numpy",
        pytest.param("numba", marks=requires_numba),
        pytest.param("torch:cpu", marks=requires_torch),
        # the CUDA string-length sort read pad slots through NaN phases
        pytest.param("cupy", marks=requires_gpu),
    ]


@pytest.mark.parametrize("backend", _bad_period_backends())
@pytest.mark.parametrize("method", FOLD_METHODS)
@pytest.mark.parametrize("where", [0, 120, -1])
def test_unfoldable_trials_give_nan_and_leave_the_rest_intact(
    method: str, backend: str, where: int
) -> None:
    # Regression: a NaN or infinite trial frequency made the numba MHAOV kernel raise
    # inside its parallel loop and return uninitialised memory at ~100 neighbouring
    # samples (or a SystemError); a period of 0 / NaN fed NaN phases to the sorts.
    # Every backend now returns NaN exactly there and the clean values elsewhere.
    t, mag, _ = synthetic_sine(n=200)
    if method == "MHAOV":
        clean = np.linspace(0.2, 3.0, 241)
        bad = clean.copy()
        bad[where] = np.nan
        bad[(where + 7) % bad.size] = np.inf
        fn: Kernel = aov_power
    else:
        clean = np.linspace(0.3, 3.0, 241)
        bad = clean.copy()
        bad[where] = 0.0
        bad[(where + 7) % bad.size] = np.nan
        fn = PERIOD_KERNELS[method]
    with warnings.catch_warnings():
        # no stray divide/invalid warnings from folding the bad trials
        warnings.simplefilter("error", RuntimeWarning)
        want = fn(t, mag, clean, backend=backend)
        got = fn(t, mag, bad, backend=backend)
        again = fn(t, mag, bad, backend=backend)
    broken = ~np.isfinite(bad) | (bad == 0.0)
    assert broken.sum() == 2
    assert np.all(np.isnan(got[broken]))
    if backend == "cupy":
        # the CUDA kernels accumulate with atomics: repeatable to round-off only
        np.testing.assert_allclose(got, again, rtol=1e-12, atol=0.0)
    else:
        assert np.array_equal(got, again, equal_nan=True)
    # MHAOV on numpy sums the clean (uniform) grid by NUFFT; the holed grid is no
    # longer uniform and takes the direct sums, which agree to ~1e-11.
    rtol = 1e-9 if (method, backend) == ("MHAOV", "numpy") else 1e-12
    np.testing.assert_allclose(got[~broken], want[~broken], rtol=rtol, atol=0.0)


# --- messages when nothing is usable (METH-B-10) -----------------------------------


@pytest.mark.parametrize("method", FOLD_METHODS)
def test_unusable_error_column_is_named(method: str) -> None:
    # Regression: an all-zero error column emptied the curve and was reported as "no
    # usable time baseline", although these methods never use the errors.
    t = np.linspace(0.0, 100.0, 300)
    y = np.sin(2 * np.pi * t / 0.7)
    with pytest.raises(cup.InsufficientDataError, match="usable error") as info:
        cup.periodogram((t, y, np.zeros(300)), method, backend="numpy")
    assert "error=None" in str(info.value)
    # The method's own entry points (default grid; power() on a caller-supplied grid)
    # say the same, under the method's message prefix.
    m = cup.get_method(method)
    lc = cup.LightCurve.from_arrays(t, y, np.zeros(300))
    with pytest.raises(cup.InsufficientDataError, match="usable error") as info:
        m.default_grid(lc, m.settings_cls())
    assert str(info.value).startswith(f"{LABELS[method]}: ")
    grid = cup.GridSpec(kind="frequency", values=np.linspace(0.5, 3.0, 50))
    with pytest.raises(cup.InsufficientDataError, match="usable error") as info:
        m.power(grid, lc, m.settings_cls(), "numpy")
    assert str(info.value).startswith(f"{LABELS[method]}: ")


@pytest.mark.parametrize("method", FOLD_METHODS)
def test_no_finite_values_is_named(method: str) -> None:
    t = np.linspace(0.0, 100.0, 300)
    with pytest.raises(cup.InsufficientDataError, match="none of the 300 points"):
        cup.periodogram((t, np.full(300, np.nan)), method, backend="numpy")


@pytest.mark.parametrize("method", FOLD_METHODS)
def test_single_epoch_still_reports_no_baseline(method: str) -> None:
    t = np.full(40, 2458000.5)
    y = np.random.default_rng(0).normal(size=40)
    with pytest.raises(cup.InsufficientDataError, match="no usable time baseline"):
        cup.periodogram((t, y), method, backend="numpy")
