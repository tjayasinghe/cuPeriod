"""The native Baluev false-alarm probability: astropy parity and sane behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from cuperiod.prewhiten.fap import baluev_fap

_POWERS = (1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.9, 0.999)


def _sampling(n: int, seed: int = 0, weighted: bool = True, offset: float = 0.0):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0.0, 27.4, n)) + offset
    error = rng.uniform(0.001, 0.004, n) if weighted else None
    return time, error


@pytest.mark.parametrize("n", [20, 500, 5000])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("f_max", [10.0, 73.2])
def test_matches_astropy_to_machine_precision(
    n: int, weighted: bool, f_max: float
) -> None:
    astropy = pytest.importorskip("astropy.timeseries")
    time, error = _sampling(n, weighted=weighted)
    rng = np.random.default_rng(1)
    value = rng.normal(10.0, 0.01, n)  # the data values never enter the formula
    ls = astropy.LombScargle(time, value, error, fit_mean=True)
    # astropy quietly snaps the requested band onto its autofrequency grid; feed the
    # band it *actually* used to our implementation so the statistic is compared 1:1.
    _, f_max_used = ls.autofrequency(
        maximum_frequency=f_max, return_freq_limits=True
    )
    for power in _POWERS:
        expected = float(
            ls.false_alarm_probability(
                power, method="baluev", maximum_frequency=f_max
            )
        )
        assert baluev_fap(
            power, time, error, maximum_frequency=float(f_max_used)
        ) == pytest.approx(expected, rel=1e-12, abs=1e-300)


def test_is_invariant_to_the_time_zero_point() -> None:
    # astropy's one-pass time variance loses ~11 digits on full Julian dates; the
    # centred form must give the same FAP wherever the time axis starts.
    time, error = _sampling(500)
    reference = baluev_fap(0.02, time, error, maximum_frequency=25.0)
    for offset in (2458000.0, 2460000.5, -59000.0):
        shifted = baluev_fap(0.02, time + offset, error, maximum_frequency=25.0)
        assert shifted == pytest.approx(reference, rel=1e-9)


def test_stays_close_to_astropy_even_on_raw_julian_dates() -> None:
    # On full Julian dates astropy's one-pass time variance loses precision, so exact
    # agreement is impossible (and ours is the more accurate value); the two must still
    # agree far beyond any scientific use of a false-alarm probability.
    astropy = pytest.importorskip("astropy.timeseries")
    time, error = _sampling(500, offset=2458000.0)
    value = np.random.default_rng(1).normal(10.0, 0.01, time.size)
    ls = astropy.LombScargle(time, value, error, fit_mean=True)
    _, f_max_used = ls.autofrequency(
        maximum_frequency=25.0, return_freq_limits=True
    )
    for power in (1e-3, 0.05, 0.3):
        expected = float(
            ls.false_alarm_probability(
                power, method="baluev", maximum_frequency=25.0
            )
        )
        assert baluev_fap(
            power, time, error, maximum_frequency=float(f_max_used)
        ) == pytest.approx(expected, rel=1e-3)


def test_vectorizes_over_power_and_matches_the_scalar_path() -> None:
    time, error = _sampling(300)
    grid = np.asarray(_POWERS)
    vectored = baluev_fap(grid, time, error, maximum_frequency=25.0)
    assert isinstance(vectored, np.ndarray)
    assert vectored.shape == grid.shape
    scalars = [
        baluev_fap(float(p), time, error, maximum_frequency=25.0) for p in grid
    ]
    assert np.allclose(vectored, scalars, rtol=0.0, atol=0.0)
    assert isinstance(scalars[0], float)


def test_is_monotone_in_power_and_bandwidth() -> None:
    # Small n keeps even the strongest trial power representable (for n in the
    # hundreds, (1 - 0.9)^((n-4)/2) underflows to exactly 0.0). Weak powers are not
    # tested for strict order: there the bound saturates at exactly 1.0, because a
    # weak peak in a wide band is always a false alarm.
    time, error = _sampling(60)
    faps = [
        baluev_fap(p, time, error, maximum_frequency=25.0)
        for p in (0.2, 0.3, 0.5, 0.9, 0.999)
    ]
    assert all(a > b for a, b in zip(faps, faps[1:], strict=False))  # stronger->rarer
    assert baluev_fap(0.05, time, error, maximum_frequency=25.0) == 1.0
    narrow = baluev_fap(0.4, time, error, maximum_frequency=5.0)
    wide = baluev_fap(0.4, time, error, maximum_frequency=50.0)
    assert 0.0 < narrow < wide < 1.0  # a wider band, more chances for a false alarm


def test_edge_cases() -> None:
    time, error = _sampling(200)
    assert baluev_fap(0.0, time, error, maximum_frequency=25.0) == pytest.approx(1.0)
    assert baluev_fap(1.0, time, error, maximum_frequency=25.0) == pytest.approx(0.0)
    # Out-of-range powers are clipped, not propagated as garbage.
    assert baluev_fap(1.5, time, error, maximum_frequency=25.0) == pytest.approx(0.0)
    assert baluev_fap(-0.5, time, error, maximum_frequency=25.0) == pytest.approx(1.0)
    # Fewer than four points cannot constrain the statistic.
    assert np.isnan(baluev_fap(0.5, time[:3], None, maximum_frequency=25.0))
    tiny = baluev_fap(
        np.array([0.5]), time[:3], None, maximum_frequency=25.0
    )
    assert isinstance(tiny, np.ndarray) and np.isnan(tiny).all()


def test_strong_peaks_in_white_noise_are_calibrated() -> None:
    # The bound must be usable as a stopping criterion: a planted signal's peak power
    # in its own spectrum should give FAP ~ 0, and the tallest peak of pure noise
    # should not look wildly significant.
    time, error = _sampling(1000)
    assert baluev_fap(0.25, time, error, maximum_frequency=30.0) < 1e-30
    assert baluev_fap(0.02, time, error, maximum_frequency=30.0) > 1e-4
