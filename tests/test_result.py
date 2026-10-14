"""Periodogram result object: peak ranking and objective sense."""

from __future__ import annotations

import numpy as np
import pytest

from cuperiod.core.result import MultiResult, Periodogram


def _make(power: np.ndarray, sense: str) -> Periodogram:
    freq = np.linspace(0.1, 2.0, power.size)
    return Periodogram.from_spectrum(
        method="TEST",
        backend="numpy",
        frequency=freq,
        power=power,
        objective_sense=sense,  # type: ignore[arg-type]
        n_samples=100,
        baseline=100.0,
    )


def test_best_periods_max_objective() -> None:
    power = np.zeros(200)
    power[50] = 5.0
    power[150] = 3.0
    pg = _make(power, "max")
    peaks = pg.best_periods(2)
    assert peaks[0].power == 5.0
    assert peaks[0].rank == 1
    assert peaks[1].power == 3.0


def test_best_periods_min_objective() -> None:
    stat = np.ones(200)
    stat[70] = 0.1  # a dip is the best peak for a minimized statistic
    pg = _make(stat, "min")
    peaks = pg.best_periods(1)
    assert peaks[0].frequency == pg.frequency[70]


def test_extras_attached_to_peaks() -> None:
    power = np.zeros(50)
    power[10] = 1.0
    freq = np.linspace(0.1, 2.0, 50)
    pg = Periodogram.from_spectrum(
        method="BLS",
        backend="numpy",
        frequency=freq,
        power=power,
        objective_sense="max",
        n_samples=100,
        baseline=100.0,
        extras={"depth": np.linspace(0.0, 0.5, 50)},
    )
    peak = pg.best_periods(1)[0]
    assert "depth" in peak.extra


def test_sorted_ascending_frequency() -> None:
    freq = np.array([2.0, 0.5, 1.0])
    power = np.array([1.0, 2.0, 3.0])
    pg = Periodogram.from_spectrum(
        method="T", backend="n", frequency=freq, power=power,
        objective_sense="max", n_samples=10, baseline=10.0,
    )
    assert np.all(np.diff(pg.frequency) > 0)
    # power must travel with its frequency
    assert pg.power[np.argmin(pg.frequency)] == 2.0


def test_best_period_at_low_frequency_edge() -> None:
    # Regression: the global maximum sitting on the first grid sample (a signal whose
    # period ~ the baseline) must not be discarded in favor of a smaller interior bump.
    power = np.zeros(200)
    power[0] = 10.0   # true peak at the low-frequency / long-period edge
    power[80] = 4.0   # a smaller interior local maximum
    pg = _make(power, "max")
    assert pg.best_period() == pg.period[0]
    assert pg.best_periods(2)[0].power == 10.0


def test_best_period_at_high_frequency_edge() -> None:
    power = np.zeros(200)
    power[-1] = 10.0  # true peak at the high-frequency / short-period edge
    power[80] = 4.0
    pg = _make(power, "max")
    assert pg.best_period() == pg.period[-1]


def test_min_objective_edge_peak() -> None:
    stat = np.ones(200)
    stat[0] = 0.0     # deepest (most significant) minimum at the edge
    stat[120] = 0.5
    pg = _make(stat, "min")
    assert pg.best_period() == pg.period[0]


def test_non_finite_samples_never_selected() -> None:
    # A frequency==0 sample (period inf) and a NaN-power sample must never be reported.
    freq = np.array([0.0, 0.1, 0.2, 0.3])
    power = np.array([np.nan, 1.0, 5.0, 2.0])
    pg = Periodogram.from_spectrum(
        method="X", backend="b", frequency=freq, power=power,
        objective_sense="max", n_samples=10, baseline=10.0,
    )
    peaks = pg.best_periods(4)
    assert all(np.isfinite(p.period) and np.isfinite(p.power) for p in peaks)
    assert peaks[0].power == 5.0


def test_all_non_finite_spectrum_is_safe() -> None:
    freq = np.array([0.1, 0.2, 0.3])
    power = np.array([np.nan, np.nan, np.nan])
    pg = Periodogram.from_spectrum(
        method="X", backend="b", frequency=freq, power=power,
        objective_sense="max", n_samples=10, baseline=10.0,
    )
    assert pg.best_periods(3) == []
    assert np.isnan(pg.best_period())


# --- CORE-B-11: n <= 0 asks for no peaks -------------------------------------------


def _two_peaks() -> Periodogram:
    power = np.zeros(200)
    power[50] = 5.0
    power[150] = 3.0
    return _make(power, "max")


def test_best_periods_non_positive_n_returns_nothing() -> None:
    pg = _two_peaks()
    for n in (0, -1):
        assert pg.best_periods(n) == []
        assert pg.best_periods(n, alias_diverse=True) == []
    assert pg.to_dict(n_best=0)["peaks"] == []
    assert len(pg.best_periods(1)) == 1


# --- CORE-B-6: a non-finite sample must not hide its neighbour ------------------------


def _from(power: list[float], sense: str = "max") -> Periodogram:
    p = np.asarray(power, dtype=float)
    return Periodogram.from_spectrum(
        method="X", backend="b", frequency=np.linspace(0.1, 1.0, p.size), power=p,
        objective_sense=sense,  # type: ignore[arg-type]
        n_samples=10, baseline=1000.0,
    )


def test_peak_next_to_nan_is_still_found() -> None:
    pg = _from([1, 2, 1, 9, np.nan, 1, 3, 1, 2.5, 1])
    assert [p.power for p in pg.best_periods(3)] == [9.0, 3.0, 2.5]
    assert _from([np.nan, 5, 1, 2]).best_periods(1)[0].power == 5.0
    assert _from([1, 1, 9, np.inf, 1, 3, 1]).best_periods(1)[0].power == 9.0
    dip = _from([5, 3, 0.1, np.nan, 4, 2, 5], sense="min")
    assert dip.best_periods(1)[0].power == 0.1


def test_nan_at_zero_frequency_does_not_hide_the_true_maximum() -> None:
    # astropy's LombScargle returns NaN at f = 0; the dominant peak sits next to it.
    freq = np.linspace(0.0, 2.0, 2001)
    power = 0.1 * np.abs(np.sin(7.0 * freq))
    power[1] = 0.984
    power[0] = np.nan
    pg = Periodogram.from_spectrum(
        method="GLS", backend="astropy", frequency=freq, power=power,
        objective_sense="max", n_samples=100, baseline=1000.0,
    )
    assert pg.best_periods(1)[0].power == pytest.approx(0.984)


def test_downsample_ignores_nan_unless_the_whole_bin_is_nan() -> None:
    power = np.ones(40)
    power[12] = 99.0
    power[13] = np.nan
    power[30:40] = np.nan
    pg = _from(list(power))
    _, p_ds = pg.downsample(4)
    assert p_ds[1] == np.float32(99.0)
    assert np.isnan(p_ds[3])


# --- CORE-B-2: min-objective spectra keep their dips when downsampled -----------------


def test_downsample_keeps_dip_of_min_objective_spectrum() -> None:
    rng = np.random.default_rng(0)
    stat = 1.0 + 0.01 * rng.random(20_000)
    stat[12_345] = 0.15
    pg = _make(stat, "min")
    freq, power = pg.downsample(2000)
    assert power.size == 2000
    assert power.min() == np.float32(0.15)
    rebuilt = Periodogram.from_spectrum(
        method="PDM", backend="numpy", frequency=freq, power=power,
        objective_sense="min", n_samples=100, baseline=100.0,
    )
    assert rebuilt.best_period() == pytest.approx(pg.best_period(), rel=1e-6)


# --- CORE-B-17: from_spectrum validates its inputs ------------------------------------


def _spectrum(**overrides: object) -> Periodogram:
    kwargs: dict[str, object] = {
        "method": "X", "backend": "b", "frequency": np.linspace(0.1, 1.0, 100),
        "power": np.ones(100), "objective_sense": "max", "n_samples": 10,
        "baseline": 10.0,
    }
    kwargs.update(overrides)
    return Periodogram.from_spectrum(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"power": np.arange(200.0)}, "power"),
        ({"power": np.arange(50.0)}, "power"),
        ({"power": np.ones((100, 3))}, "power"),
        ({"frequency": np.ones((10, 10))}, "frequency"),
        ({"extras": {"depth": np.ones(101)}}, "depth"),
        ({"objective_sense": "maximize"}, "objective_sense"),
        ({"objective_sense": "MAX"}, "objective_sense"),
        ({"objective_sense": None}, "objective_sense"),
    ],
)
def test_from_spectrum_rejects_malformed_input(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _spectrum(**overrides)


def test_from_spectrum_accepts_an_empty_spectrum() -> None:
    pg = _spectrum(frequency=np.empty(0), power=np.empty(0))
    assert pg.size == 0
    assert pg.best_periods() == []


# --- CORE-B-13: equality / hashing must not raise -------------------------------------


def test_periodogram_equality_and_hash_do_not_raise() -> None:
    pg1 = _two_peaks()
    pg2 = _two_peaks()
    assert (pg1 == pg2) is False
    assert pg1 == pg1
    assert pg2 in [pg1, pg2]
    assert [pg1, pg2].index(pg2) == 1
    assert {pg1: "a", pg2: "b"}[pg1] == "a"


def test_peak_keeps_value_equality_and_becomes_hashable() -> None:
    pk1 = _two_peaks().best_periods(1)[0]
    pk2 = _two_peaks().best_periods(1)[0]
    assert pk1 == pk2
    assert hash(pk1) == hash(pk2)
    assert len({pk1, pk2}) == 1


def test_multiresult_equality_and_hash_do_not_raise() -> None:
    mr1 = MultiResult({"GLS": _two_peaks()})
    mr2 = MultiResult({"GLS": _two_peaks()})
    assert (mr1 == mr2) is False
    assert mr1 == mr1
    assert {mr1: 1, mr2: 2}[mr2] == 2


# --- CORE-B-14: MultiResult is a consistent mapping -----------------------------------


def test_multiresult_membership_matches_getitem() -> None:
    gls, sl = _two_peaks(), _two_peaks()
    mr = MultiResult({"GLS": gls, "STRINGLENGTH": sl})
    assert mr["gls"] is gls and mr["String-Length"] is sl
    assert "gls" in mr
    assert "String-Length" in mr
    assert "nope" not in mr
    assert 3 not in mr
    assert mr.get("gls") is gls
    assert mr.get("nope") is None
    assert mr.get("nope", gls) is gls
    assert list(mr.values()) == [gls, sl]
    assert list(mr.keys()) == ["GLS", "STRINGLENGTH"]
    assert dict(mr.items()) == {"GLS": gls, "STRINGLENGTH": sl}


# --- CORE-B-10: settings-carried peak defaults ----------------------------------------


def test_peak_defaults_drive_best_periods_downsample_and_to_dict() -> None:
    rng = np.random.default_rng(1)
    pg = _make(rng.random(5000), "max")
    assert len(pg.best_periods()) == 10
    assert pg.downsample()[0].size == 2000
    tuned = pg.with_peak_defaults(
        {"n": 3, "min_separation_rayleigh": 50.0, "n_points": 100}
    )
    peaks = tuned.best_periods()
    assert len(peaks) == 3
    freqs = sorted(p.frequency for p in peaks)
    assert min(np.diff(freqs)) >= 50.0 / pg.baseline
    assert tuned.downsample()[0].size == 100
    assert len(tuned.to_dict()["peaks"]) == 3
    assert len(tuned.to_dict(include_raw=True)["frequency"]) == 100
    # Explicit arguments still win, and the original is untouched.
    assert len(tuned.best_periods(5, min_separation_rayleigh=3.0)) == 5
    assert tuned.downsample(50)[0].size == 50
    assert len(pg.best_periods()) == 10


def test_peak_defaults_from_settings_object() -> None:
    from cuperiod.core.config import BLSSettings

    pg = _two_peaks().with_peak_defaults(
        BLSSettings(n_peaks=1, downsample_points=50, alias_freq_tolerance=0.01)
    )
    assert pg.peak_defaults["n"] == 1
    assert pg.peak_defaults["n_points"] == 50
    assert pg.peak_defaults["alias_tolerance"] == 0.01
    assert len(pg.best_periods()) == 1


# --- CORE-B-1: alias tolerance is not the duplicate-peak separation -------------------


def test_alias_diverse_keeps_a_real_second_signal_on_a_short_baseline() -> None:
    """f2 = 1/3.93 sits 1.4 Rayleigh widths from f1/2: outside one Rayleigh width
    (not an alias) but inside the old 3/T band, which discarded it."""
    freq = np.linspace(0.02, 3.0, 6000)
    power = 0.05 * np.random.default_rng(4).random(freq.size)
    for f, height in ((1 / 2.47, 10.0), (1 / 3.93, 8.0), (2 / 2.47, 6.0)):
        power[np.argmin(np.abs(freq - f))] = height
    pg = Periodogram.from_spectrum(
        method="BLS", backend="numpy", frequency=freq, power=power,
        objective_sense="max", n_samples=1000, baseline=27.4,
    )
    periods = [p.period for p in pg.best_periods(5, alias_diverse=True)]
    assert periods[0] == pytest.approx(2.47, rel=1e-2)
    assert periods[1] == pytest.approx(3.93, rel=1e-2)
    # The 2x harmonic of the main signal is still recognised as related.
    assert not any(abs(p - 2.47 / 2) < 0.02 for p in periods)
    # An explicit tolerance (BLS/TLS settings: 0.0035 c/d) fills every slot.
    tight = pg.best_periods(5, alias_diverse=True, alias_tolerance=0.0035)
    assert len(tight) == 5
    assert tight[1].period == pytest.approx(3.93, rel=1e-2)
    assert not any(abs(p.period - 2.47 / 2) < 0.02 for p in tight)
    # ... and so does one carried by the result's settings defaults.
    carried = pg.with_peak_defaults({"alias_tolerance": 0.0035})
    assert [p.period for p in carried.best_periods(5, alias_diverse=True)] == [
        p.period for p in tight
    ]


def _box(
    t: np.ndarray, period: float, t0: float, dur: float, depth: float
) -> np.ndarray:
    phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    return np.where(np.abs(phase) < 0.5 * dur, -depth, 0.0)


def test_bls_alias_diverse_reports_the_second_planet() -> None:
    import cuperiod as cup

    rng = np.random.default_rng(3)
    t = np.arange(0.0, 27.4, 30.0 / 1440.0)
    flux = 1.0 + _box(t, 2.47, 0.7, 0.12, 0.010) + _box(t, 3.93, 1.9, 0.14, 0.008)
    flux += 0.0007 * rng.standard_normal(t.size)
    lc = cup.LightCurve.from_arrays(
        t, flux, np.full(t.size, 0.0007), domain=cup.Domain.FLUX
    )
    pg = cup.periodogram(lc, "BLS", backend="cpu")
    assert isinstance(pg, Periodogram)
    periods = [p.period for p in pg.best_periods(5, alias_diverse=True)]
    assert periods[0] == pytest.approx(2.47, rel=1e-2)
    assert periods[1] == pytest.approx(3.93, rel=1e-2)
    # The BLS setting (0.0035 c/d) fills every slot without P1 harmonics.
    tight = [
        p.period
        for p in pg.best_periods(10, alias_diverse=True, alias_tolerance=0.0035)
    ]
    assert len(tight) == 10
    assert tight[1] == pytest.approx(3.93, rel=1e-2)
    for p in tight[1:]:
        for m in range(2, 9):
            assert abs(p - 2.47 * m) / (2.47 * m) > 0.01
            assert abs(p - 2.47 / m) / (2.47 / m) > 0.01
