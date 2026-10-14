"""period_profile: the model-free P(t) track, and detrending."""

from __future__ import annotations

import numpy as np
import pytest

from cuperiod.core.errors import InsufficientDataError
from cuperiod.core.lightcurve import LightCurve
from cuperiod.pdot.detrend import detrend, running_median
from cuperiod.pdot.profile import period_profile

#: Tylenda et al. (2011) Eq. (1) for V1309 Sco.
V1309 = (1.4456, 15.29, 2455233.5)


def singular_light_curve(
    n: int = 1200,
    noise: float = 0.02,
    seed: int = 11,
    trend: float = 0.0,
    harmonics: tuple[float, float] = (0.30, 0.12),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An eclipsing-like light curve whose period follows the V1309 Sco law.

    The phase is integrated numerically from ``P(t)`` rather than warped, so it
    shares no code with the implementation under test.
    """
    p0, b, t0 = V1309
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2452100.0, 2454300.0, n))
    fine = np.linspace(t.min(), t.max(), 200_001)
    period = p0 * np.exp(b / (fine - t0))
    phase_grid = np.concatenate(
        [[0.0], np.cumsum(np.diff(fine) / (0.5 * (period[1:] + period[:-1])))]
    )
    phase = np.interp(t, fine, phase_grid)
    a1, a2 = harmonics
    mag = (
        12.0
        + a1 * np.sin(2.0 * np.pi * phase)
        + a2 * np.sin(4.0 * np.pi * phase + 0.7)
        + rng.normal(0.0, noise, n)
    )
    mag += trend * (t - t.min()) / (t.max() - t.min())
    return t, mag, np.full(n, noise)


def true_period(time: np.ndarray) -> np.ndarray:
    p0, b, t0 = V1309
    return p0 * np.exp(b / (time - t0))


# --- the track ----------------------------------------------------------------


def test_track_follows_the_true_period() -> None:
    t, mag, err = singular_light_curve()
    prof = period_profile((t, mag, err), period=V1309[0], points_per_interval=50)
    assert prof.n_intervals >= 20
    resid = prof.period - true_period(prof.t_mid)
    assert np.sqrt(np.mean(resid**2)) < 0.001


def test_period_errors_are_calibrated() -> None:
    """The curvature error must be a real 1-sigma, not decorative."""
    t, mag, err = singular_light_curve()
    prof = period_profile((t, mag, err), period=V1309[0], points_per_interval=50)
    pull = (prof.period - true_period(prof.t_mid)) / prof.period_err
    assert np.all(np.isfinite(prof.period_err))
    assert abs(float(np.mean(pull))) < 0.5
    assert 0.6 < float(np.std(pull)) < 1.6


def test_harmonic_order_is_not_overfitted() -> None:
    """A two-harmonic star must not buy ten harmonics off under-stated errors.

    The weighted BIC does exactly that when the quoted errors are optimistic,
    which is the norm on survey photometry; the scale-free criterion does not.
    """
    t, mag, err = singular_light_curve()
    tenth = err / 10.0  # errors ten times too small
    prof = period_profile((t, mag, tenth), period=V1309[0], points_per_interval=50)
    assert prof.n_harmonics <= 4


def test_harmonic_order_is_stable_across_interval_sizes() -> None:
    t, mag, err = singular_light_curve()
    orders = {
        period_profile(
            (t, mag, err), period=V1309[0], points_per_interval=size
        ).n_harmonics
        for size in (40, 50, 60)
    }
    assert len(orders) == 1


def test_pinned_harmonics_are_honoured() -> None:
    t, mag, err = singular_light_curve()
    prof = period_profile((t, mag, err), period=V1309[0], harmonics=3)
    assert prof.n_harmonics == 3


def test_fixed_interval_count() -> None:
    t, mag, err = singular_light_curve()
    prof = period_profile((t, mag, err), period=V1309[0], n_intervals=8)
    assert prof.n_intervals == 8


def test_constant_period_gives_a_flat_track() -> None:
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0.0, 1500.0, 900)) + 2455000.0
    mag = 12.0 + 0.3 * np.sin(2.0 * np.pi * t / 0.7) + rng.normal(0.0, 0.02, t.size)
    prof = period_profile((t, mag, np.full(t.size, 0.02)), period=0.7)
    assert np.allclose(prof.period, 0.7, atol=5e-4)
    assert prof.compare()[0].law == "constant"


def test_window_edge_is_flagged_when_the_change_escapes_the_window() -> None:
    t, mag, err = singular_light_curve()
    narrow = period_profile(
        (t, mag, err), period=V1309[0], points_per_interval=50, period_window=0.002
    )
    assert narrow.any_at_window_edge
    assert any(p.at_window_edge for p in narrow.points)


def test_summaries_and_serialization() -> None:
    t, mag, err = singular_light_curve()
    prof = period_profile((t, mag, err), period=V1309[0])
    exact = true_period(np.array([prof.t_mid[0], prof.t_mid[-1]]))
    assert prof.total_fractional_change() == pytest.approx(
        exact[1] / exact[0] - 1.0, abs=0.004
    )
    rows = prof.to_table()
    assert len(rows) == prof.n_intervals
    assert {"t_mid", "period", "period_err", "at_window_edge"} <= set(rows[0])
    assert prof.to_dict()["n_intervals"] == prof.n_intervals


def test_fit_and_compare_reach_the_law_layer() -> None:
    t, mag, err = singular_light_curve()
    prof = period_profile((t, mag, err), period=V1309[0], points_per_interval=50)
    ranked = prof.compare(laws=("constant", "linear", "singular"))
    assert ranked[0].law == "singular"
    fit = prof.fit("singular")
    lo, hi = prof.points[0].t_start, prof.points[-1].t_end
    exact = true_period(np.array([lo, hi]))
    assert fit.fractional_change(lo, hi) == pytest.approx(
        exact[1] / exact[0] - 1.0, abs=0.003
    )


def test_measure_pdot_populates_the_interval_rates() -> None:
    t, mag, err = singular_light_curve(n=1600)
    prof = period_profile(
        (t, mag, err), period=V1309[0], n_intervals=4, measure_pdot=True, backend="cpu"
    )
    assert np.all(np.isfinite([p.pdot for p in prof.points]))
    assert all(p.pdot < 0.0 for p in prof.points)  # the period is shortening


def test_unseeded_profile_finds_its_own_period() -> None:
    t, mag, err = singular_light_curve()
    prof = period_profile((t, mag, err), backend="cpu", points_per_interval=50)
    assert prof.reference_period == pytest.approx(V1309[0], rel=0.01)


# --- edges --------------------------------------------------------------------


def test_too_few_intervals_raises() -> None:
    t, mag, err = singular_light_curve(n=60)
    with pytest.raises(InsufficientDataError, match="usable intervals"):
        period_profile((t, mag, err), period=V1309[0], points_per_interval=50)


def test_multiband_input_is_refused() -> None:
    from cuperiod.core.lightcurve import MultiBandLightCurve

    t, mag, err = singular_light_curve(n=200)
    lc = MultiBandLightCurve.from_light_curves(
        {"g": LightCurve.from_arrays(t, mag, err)}
    )
    with pytest.raises(ValueError, match="single band"):
        period_profile(lc, period=V1309[0])


@pytest.mark.parametrize("window", [0.0, 1.0, -0.1, np.nan])
def test_bad_period_window_raises(window: float) -> None:
    t, mag, err = singular_light_curve(n=200)
    with pytest.raises(ValueError, match="period_window"):
        period_profile((t, mag, err), period=V1309[0], period_window=window)


def test_unweighted_input_still_measures() -> None:
    t, mag, _ = singular_light_curve()
    prof = period_profile((t, mag), period=V1309[0], points_per_interval=50)
    assert np.all(np.isfinite(prof.period_err))
    assert np.all(prof.period_err > 0.0)


# --- the interval measurement converges on the chi-squared minimum -------------

#: A 6-harmonic, RRab-like template (amplitudes, phases).
RRAB = (
    np.array([0.30, 0.15, 0.10, 0.065, 0.04, 0.025]),
    np.array([0.0, 0.9, 2.1, 3.4, 4.8, 0.1]),
)


def constant_star(
    kind: str, seed: int, noise: float, n: int = 1500, span: float = 1500.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A constant-period (0.7 d) star: two harmonics, or six (RRab-like)."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n)) + 2455000.0
    phi = 2.0 * np.pi * (t - 2455000.0) / 0.7
    if kind == "sine2":
        y = 0.30 * np.sin(phi) + 0.12 * np.sin(2.0 * phi + 0.7)
    else:
        amps, phases = RRAB
        y = sum(
            a * np.sin((m + 1) * phi + p)
            for m, (a, p) in enumerate(zip(amps, phases, strict=True))
        )
    y = 12.0 + y + rng.normal(0.0, noise, n)
    return t, y, np.full(n, noise)


@pytest.mark.parametrize(("kind", "noise"), [("sine2", 0.002), ("rrab6", 0.02)])
def test_errors_are_calibrated_at_high_signal_to_noise(kind: str, noise: float) -> None:
    """One parabolic vertex on the scan stencil is not the minimum.

    The chi-squared is asymmetric at the scale of the scan step, so its vertex
    missed the true minimum by several curvature sigmas whenever that sigma is
    small -- many harmonics, or high signal-to-noise -- and the periods
    scattered 3-4x beyond their own error bars.
    """
    pulls, laws = [], []
    for seed in range(3):
        t, y, err = constant_star(kind, 2000 + seed, noise)
        prof = period_profile((t, y, err), period=0.7)
        free = np.array([not p.at_window_edge for p in prof.points])
        pulls.append(((prof.period - 0.7) / prof.period_err)[free])
        laws.append(prof.compare()[0].law)
    pull = np.concatenate(pulls)
    assert 0.7 < float(np.std(pull)) < 1.4
    assert laws.count("constant") >= 2


def test_interval_frequency_sits_on_the_chi2_minimum() -> None:
    from cuperiod.pdot.profile import _harmonic_chi2, _refine_frequency

    t, y, err = constant_star("sine2", 2001, 0.002, n=60, span=60.0)
    dt = t - t.mean()
    w = 1.0 / err**2
    f_ref = 1.0 / 0.7
    f, sigma, chi2_min, _ = _refine_frequency(
        dt, y, w, f_ref / 1.05, f_ref / 0.95, 2, 5
    )
    fine = np.linspace(f - 3 * sigma, f + 3 * sigma, 601)
    best = min(_harmonic_chi2(dt, y, w, float(g), 2) for g in fine)
    assert chi2_min == pytest.approx(_harmonic_chi2(dt, y, w, f, 2), rel=1e-12)
    assert chi2_min - best < 1e-3


@pytest.mark.parametrize("harmonics", [2, 3, None])
def test_amplitude_is_the_fundamentals(harmonics: int | None) -> None:
    """The fundamental's sine coefficient is coef[1 + H], not coef[2]."""
    t, mag, err = singular_light_curve(noise=0.01)
    prof = period_profile(
        (t, mag, err), period=V1309[0], points_per_interval=60, harmonics=harmonics
    )
    amps = np.array([p.amplitude for p in prof.points])
    assert np.all(np.abs(amps - 0.30) < 0.03)
    assert [row["amplitude"] for row in prof.to_table()] == list(amps)


def test_unweighted_errors_do_not_depend_on_the_units() -> None:
    """Magnitudes and the same light curve in millimagnitudes: same errors."""
    t, mag, err = constant_star("rrab6", 7, 0.05, n=900, span=900.0)
    plain = period_profile((t, mag), period=0.7, harmonics=4, points_per_interval=60)
    small = period_profile(
        (t, 1e-3 * mag), period=0.7, harmonics=4, points_per_interval=60
    )
    assert np.allclose(small.period, plain.period, rtol=1e-12)
    assert np.allclose(small.period_err, plain.period_err, rtol=1e-6)
    weighted = period_profile(
        (t, mag, err), period=0.7, harmonics=4, points_per_interval=60
    )
    ratio = np.median(plain.period_err / weighted.period_err)
    assert ratio == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"harmonics": 0}, "harmonics must be"),
        ({"harmonics": -1}, "harmonics must be"),
        ({"period": np.nan}, "period must be finite"),
        ({"period": np.inf}, "period must be finite"),
        ({"period_window": np.nan}, "period_window must be"),
        ({"points_per_interval": 0}, "points_per_interval must be"),
        ({"n_intervals": 0}, "n_intervals must be"),
    ],
)
def test_bad_scalars_raise_value_errors(kwargs: dict, match: str) -> None:
    t, mag, err = singular_light_curve(n=300)
    args = {"period": V1309[0], **kwargs}
    with pytest.raises(ValueError, match=match):
        period_profile((t, mag, err), **args)


@pytest.mark.parametrize("weighted", [True, False])
def test_constant_light_curve_is_not_measured(weighted: bool) -> None:
    t = np.sort(np.random.default_rng(1).uniform(0.0, 1500.0, 900)) + 2455000.0
    flat = np.full(t.size, 12.0)
    data = (t, flat, np.full(t.size, 0.02)) if weighted else (t, flat)
    with pytest.raises(InsufficientDataError):
        period_profile(data, period=0.7)


@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("harmonics", [1, 3, 10])
def test_batched_chi2_matches_the_scalar_one(weighted: bool, harmonics: int) -> None:
    from cuperiod.pdot.profile import _harmonic_chi2, _harmonic_chi2_many

    rng = np.random.default_rng(harmonics)
    dt = np.sort(rng.uniform(-40.0, 40.0, 70))
    y = np.sin(2 * np.pi * dt / 0.7) + rng.normal(0.0, 0.05, dt.size)
    w = 1.0 / rng.uniform(0.03, 0.08, dt.size) ** 2 if weighted else np.ones(dt.size)
    freqs = np.linspace(1.3, 1.55, 333)
    many = _harmonic_chi2_many(dt, y, w, freqs, harmonics)
    one = np.array([_harmonic_chi2(dt, y, w, float(f), harmonics) for f in freqs])
    assert np.allclose(many, one, rtol=1e-10)
    # A degenerate trial (f = 0 repeats the constant column) is inf in both.
    both = _harmonic_chi2_many(dt, y, w, np.array([0.0, 1.4]), harmonics)
    assert np.isinf(both[0]) and np.isfinite(both[1])
    assert both[1] == pytest.approx(_harmonic_chi2(dt, y, w, 1.4, harmonics), rel=1e-10)


# --- detrending ---------------------------------------------------------------


def test_detrend_removes_a_slow_trend_and_keeps_the_signal() -> None:
    t, mag, err = singular_light_curve(trend=0.8)
    flat = detrend(t, mag, 40.0)
    plain = period_profile((t, mag, err), period=V1309[0], points_per_interval=50)
    clean = period_profile((t, flat, err), period=V1309[0], points_per_interval=50)
    # The trend inflates the residual scatter; removing it must not hurt.
    resid = clean.period - true_period(clean.t_mid)
    assert np.sqrt(np.mean(resid**2)) < 0.001
    assert np.std(np.diff(clean.period)) <= np.std(np.diff(plain.period)) * 1.5


@pytest.mark.parametrize("method", ["median", "binned"])
def test_running_median_tracks_a_known_trend(method: str) -> None:
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0.0, 1000.0, 2000))
    trend = 0.5 * np.sin(2.0 * np.pi * t / 900.0)
    y = trend + 0.2 * np.sin(2.0 * np.pi * t / 1.3) + rng.normal(0.0, 0.02, t.size)
    got = running_median(t, y, 60.0, method=method)
    inner = (t > 60.0) & (t < 940.0)  # the edges see a one-sided window
    assert np.std(got[inner] - trend[inner]) < 0.05


def test_detrend_is_robust_to_outliers() -> None:
    """A median, not a mean: one-sided eclipses must not drag the baseline."""
    rng = np.random.default_rng(9)
    t = np.sort(rng.uniform(0.0, 400.0, 800))
    y = np.full(t.size, 10.0) + rng.normal(0.0, 0.01, t.size)
    y[::10] += 3.0  # deep, one-sided "eclipses"
    trend = running_median(t, y, 50.0)
    assert np.allclose(trend, 10.0, atol=0.05)


def test_detrend_handles_non_finite_points() -> None:
    t = np.linspace(0.0, 100.0, 200)
    y = np.sin(t)
    y[5] = np.nan
    out = detrend(t, y, 20.0)
    assert np.isnan(out[5])
    assert np.all(np.isfinite(np.delete(out, 5)))


def test_detrend_preserves_the_level_by_default() -> None:
    t = np.linspace(0.0, 100.0, 200)
    y = 15.0 + 0.01 * t
    assert float(np.median(detrend(t, y, 20.0))) == pytest.approx(
        float(np.median(y)), abs=0.02
    )
    assert abs(float(np.median(detrend(t, y, 20.0, restore_level=False)))) < 0.02


@pytest.mark.parametrize("width", [0.0, -1.0, np.nan])
def test_bad_detrend_width_raises(width: float) -> None:
    with pytest.raises(ValueError, match="width must be positive"):
        running_median(np.arange(10.0), np.zeros(10), width)


def test_unknown_detrend_method_raises() -> None:
    with pytest.raises(ValueError, match="method must be"):
        running_median(np.arange(10.0), np.zeros(10), 2.0, method="mean")  # type: ignore[arg-type]
