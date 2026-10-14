"""AmplitudeSpectrum.noise_curve(estimator="median") is exact and vectorised.

The median S/N line used to be a per-sample Python loop (seconds on a 1e5-sample grid
with wide boxes, redrawn on every GUI edit). It is now answered for every box at once
by a wavelet-matrix order-statistic descent; these tests pin it to the old loop value
for value -- NaN/inf samples, short boxes near the grid edges, even and odd counts,
ties -- and to ``np.median`` on arbitrary, non-monotone boxes.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cuperiod.prewhiten.spectrum import AmplitudeSpectrum, _range_medians


def _spectrum(amplitude, frequency=None) -> AmplitudeSpectrum:
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if frequency is None:
        frequency = 0.01 + 0.002 * np.arange(amplitude.size)
    return AmplitudeSpectrum(
        frequency=np.asarray(frequency, dtype=np.float64),
        amplitude=amplitude,
        phase=np.zeros(amplitude.size),
        power=np.zeros(amplitude.size),
        t_ref=0.0,
        backend="numpy",
        normalization="lsq",
        n_samples=100,
        baseline=100.0,
    )


def _old_median_curve(spectrum: AmplitudeSpectrum, window: float, min_samples: int):
    """The pre-vectorisation implementation, verbatim in substance."""
    freq = spectrum.frequency
    n = int(freq.size)
    if n == 0:
        return np.zeros(0)
    need = min(min_samples, n)
    lo = np.searchsorted(freq, freq - window, side="left")
    hi = np.searchsorted(freq, freq + window, side="right")
    short = (hi - lo) < need
    if short.any():
        centre = np.searchsorted(freq, freq[short])
        half = max(1, need // 2)
        low = np.maximum(0, np.minimum(centre - half, n - need))
        lo[short] = low
        hi[short] = np.minimum(n, low + need)
    amplitude = spectrum.amplitude
    finite = np.isfinite(amplitude)
    out = np.empty(n)
    for i in range(n):
        box = amplitude[lo[i] : hi[i]]
        box = box[finite[lo[i] : hi[i]]]
        out[i] = float(np.median(box)) if box.size else float("nan")
    return out


def _random_amplitude(rng, n: int, nan_fraction: float, inf_fraction: float = 0.0):
    amp = np.abs(rng.normal(1.0, 0.5, n)) * rng.uniform(0.5, 2.0)
    # A few strong peaks, as in a real residual spectrum.
    peaks = rng.choice(n, size=max(1, n // 200), replace=False)
    amp[peaks] *= 25.0
    amp[rng.random(n) < nan_fraction] = np.nan
    amp[rng.random(n) < inf_fraction] = np.inf
    return amp


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize(
    ("n", "window", "min_samples", "nan_fraction", "inf_fraction"),
    [
        (1, 1.0, 25, 0.0, 0.0),
        (2, 0.0, 25, 0.0, 0.0),
        (7, 0.003, 25, 0.3, 0.0),
        (300, 0.05, 25, 0.0, 0.0),
        (301, 0.05, 25, 0.1, 0.02),
        (500, 0.0001, 25, 0.0, 0.0),  # every box short: the min_samples fallback
        (500, 0.0001, 1, 0.2, 0.0),
        (1000, 0.4, 25, 0.05, 0.0),  # boxes clipped by both grid edges
        (777, 0.02, 50, 0.95, 0.0),  # mostly NaN: many empty boxes
    ],
)
def test_median_curve_equals_the_old_loop(
    seed, n, window, min_samples, nan_fraction, inf_fraction
) -> None:
    rng = np.random.default_rng(1000 + seed)
    spectrum = _spectrum(_random_amplitude(rng, n, nan_fraction, inf_fraction))
    new = spectrum.noise_curve(
        window=window, estimator="median", min_samples=min_samples
    )
    old = _old_median_curve(spectrum, window, min_samples)
    np.testing.assert_array_equal(new, old)


def test_median_curve_with_ties_and_an_all_nan_spectrum() -> None:
    tied = _spectrum(np.repeat([0.5, 0.25, 1.0, 0.25], 40))
    np.testing.assert_array_equal(
        tied.noise_curve(window=0.01, estimator="median"),
        _old_median_curve(tied, 0.01, 25),
    )
    empty = _spectrum(np.full(30, np.nan))
    assert np.all(np.isnan(empty.noise_curve(window=0.01, estimator="median")))
    assert _spectrum(np.zeros(0)).noise_curve(estimator="median").size == 0


def test_median_curve_on_an_irregular_grid() -> None:
    rng = np.random.default_rng(3)
    frequency = np.cumsum(rng.uniform(0.0005, 0.02, 900))
    spectrum = _spectrum(_random_amplitude(rng, 900, 0.05), frequency)
    np.testing.assert_array_equal(
        spectrum.noise_curve(window=0.15, estimator="median", min_samples=10),
        _old_median_curve(spectrum, 0.15, 10),
    )


def test_range_medians_on_arbitrary_boxes() -> None:
    # Nothing assumes monotone or equal-width boxes.
    rng = np.random.default_rng(5)
    values = rng.normal(size=400)
    values[rng.random(400) < 0.1] = np.nan
    lo = rng.integers(0, 400, 3000)
    hi = np.minimum(400, lo + rng.integers(0, 120, 3000))
    got = _range_medians(values, lo, hi)
    for i in range(lo.size):
        box = values[lo[i] : hi[i]]
        box = box[np.isfinite(box)]
        want = float(np.median(box)) if box.size else np.nan
        assert (np.isnan(want) and np.isnan(got[i])) or got[i] == want


def test_median_curve_is_fast_on_a_large_grid() -> None:
    # 1e5 samples with boxes ~4000 samples wide: the loop took seconds here.
    rng = np.random.default_rng(9)
    spectrum = _spectrum(_random_amplitude(rng, 100_000, 0.001))
    start = time.perf_counter()
    curve = spectrum.noise_curve(window=4.0, estimator="median")
    elapsed = time.perf_counter() - start
    assert curve.shape == (100_000,)
    assert np.all(np.isfinite(curve))
    # Spot-check against np.median on a few boxes.
    for i in (0, 1, 2000, 50_000, 99_999):
        lo = np.searchsorted(spectrum.frequency, spectrum.frequency[i] - 4.0, "left")
        hi = np.searchsorted(spectrum.frequency, spectrum.frequency[i] + 4.0, "right")
        box = spectrum.amplitude[lo:hi]
        assert curve[i] == float(np.median(box[np.isfinite(box)]))
    assert elapsed < 2.0
