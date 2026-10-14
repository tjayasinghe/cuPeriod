"""TLS: limb-darkened template, transit recovery, and backend wiring."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

import cuperiod as cup
import cuperiod.core.grid as grid_module
import cuperiod.methods.tls as tls_module
from conftest import requires_gpu, requires_numba, requires_torch
from cuperiod.core.errors import BackendUnavailableError
from cuperiod.core.grid import GridSizeWarning, GridTooLargeError
from cuperiod.methods.tls import _duration_bins, limb_darkened_template, tls_power

#: The CPU backends every numerical regression below runs on.
CPU_BACKENDS = ["numpy", pytest.param("numba", marks=requires_numba)]


def test_duration_bins_never_empty() -> None:
    # Regression: fractions too small for n_phase_bins must still yield >=1 valid width,
    # not an empty list (which made tls_power return a silent all-zero spectrum).
    db = _duration_bins(
        cup.TLSSettings(
            n_phase_bins=16, duration_min_frac=0.001,
            duration_max_frac=0.01, n_durations=5,
        )
    )
    assert db and all(1 <= w < 16 for w in db)


def _inject_transit(
    *,
    n: int = 1200,
    span: float = 120.0,
    period: float = 3.0,
    t0: float = 2458001.0,
    depth: float = 0.02,
    dur_frac: float = 0.05,
    u1: float = 0.4,
    u2: float = 0.3,
    noise: float = 0.002,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A flux light curve with a planted limb-darkened transit."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n)) + 2458000.0
    phase = ((t - t0) / period) % 1.0
    x = np.where(phase > 0.5, phase - 1.0, phase)
    inside = np.abs(x) < dur_frac / 2.0
    z = np.clip(2.0 * np.abs(x) / dur_frac, 0.0, 1.0)
    mu = np.sqrt(1.0 - z * z)
    prof = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    prof = prof / prof.max()
    flux = np.ones(n)
    flux[inside] -= depth * prof[inside]
    flux += rng.normal(0.0, noise, n)
    return t, flux, np.full(n, noise)


def test_limb_darkened_template_shape() -> None:
    g = limb_darkened_template(21, 0.4, 0.3)
    assert g.size == 21
    assert g.max() == pytest.approx(1.0)
    assert g[10] == pytest.approx(1.0, abs=1e-6)  # peak at the centre
    assert g[0] < g[10] and g[-1] < g[10]  # tapers toward the limb


def test_tls_recovers_transit() -> None:
    # A single-transit signal also aligns at sub-multiples P/k, so — as with every
    # fold method — the search is bounded to a sensible period range.
    t, flux, err = _inject_transit(period=3.0, depth=0.02)
    pg = cup.periodogram(
        (t, flux, err), "TLS", domain="flux",
        settings=cup.TLSSettings(min_period_days=2.0),
    )
    assert pg.objective_sense == "max"
    peak = pg.best_periods(1, alias_diverse=True)[0]
    assert peak.period == pytest.approx(3.0, rel=5e-3)
    assert peak.extra["depth"] > 0.0
    assert peak.power > 7.0  # SDE of a clear transit


def test_tls_peak_has_transit_parameters() -> None:
    t, flux, err = _inject_transit()
    pg = cup.periodogram(
        (t, flux, err), "tls", domain="flux",
        settings=cup.TLSSettings(min_period_days=2.0),
    )
    peak = pg.best_periods(1, alias_diverse=True)[0]
    assert {"depth", "duration", "t0", "sr"} <= set(peak.extra)
    assert 0.0 < peak.extra["duration"] < 3.0


def test_tls_converts_magnitude() -> None:
    t, flux, _ = _inject_transit(depth=0.03)
    mag = -2.5 * np.log10(flux)
    pg = cup.periodogram(
        (t, mag, 0.003 * np.ones_like(mag)), "TLS", domain="magnitude",
        settings=cup.TLSSettings(min_period_days=2.0),
    )
    peak = pg.best_periods(1, alias_diverse=True)[0]
    assert peak.period == pytest.approx(3.0, rel=1e-2)


def test_tls_gpu_backend_unavailable() -> None:
    from cuperiod.core.backend import cuda_available

    if cuda_available():
        pytest.skip("a GPU is present")
    with pytest.raises(BackendUnavailableError):
        cup.get_method("TLS").resolve_backend("gpu")


@requires_gpu
def test_tls_gpu_matches_cpu() -> None:
    t, flux, err = _inject_transit(period=3.0, depth=0.02)
    settings = cup.TLSSettings(min_period_days=2.0)
    grid = cup.get_method("TLS").default_grid(
        cup.LightCurve.from_arrays(t, flux, err, domain=cup.Domain.FLUX), settings
    )
    periods = grid.period
    cpu = tls_power(t, flux, err, periods, settings=settings, backend="numpy")
    gpu = tls_power(t, flux, err, periods, settings=settings, backend="cupy")
    assert np.allclose(cpu["sr"], gpu["sr"], rtol=1e-6, atol=1e-6)
    assert np.allclose(cpu["sde"], gpu["sde"], rtol=1e-6, atol=1e-6)
    for key in ("depth", "duration", "t0"):
        assert np.allclose(cpu[key], gpu[key], rtol=1e-9, atol=1e-9), key


@requires_gpu
def test_tls_gpu_float32_small_flux_units() -> None:
    # The float32 CUDA kernel used to overflow on unscaled cgs-unit weights.
    t, flux, err = _inject_transit(period=3.0, depth=0.02)
    periods = np.linspace(2.0, 4.0, 400)
    f32 = cup.TLSSettings(precision="float32")
    ref = tls_power(t, flux, err, periods, settings=cup.TLSSettings(),
                    backend="numpy")
    got = tls_power(t, flux * 1e-15, err * 1e-15, periods, settings=f32,
                    backend="cupy")
    assert int(np.argmax(got["sde"])) == int(np.argmax(ref["sde"]))
    assert got["sde"].max() == pytest.approx(ref["sde"].max(), rel=1e-3)


# --- transit epoch (GUI-A-3 / METH-C-2) ------------------------------------------


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_tls_t0_is_an_absolute_mid_transit_time(backend: str) -> None:
    # t0 used to be measured from the first timestamp while BLS (and the GUI's fold,
    # and the 'Transit epoch' column) treat it as an absolute epoch.
    true_t0, period = 2458001.0, 3.0
    t, flux, err = _inject_transit(period=period, t0=true_t0)
    periods = np.linspace(2.9, 3.1, 201)
    res = tls_power(t, flux, err, periods, settings=cup.TLSSettings(),
                    backend=backend)
    best = int(np.argmax(res["sde"]))
    p_best, t0 = float(periods[best]), float(res["t0"][best])
    assert t.min() <= t0 < t.min() + p_best
    wrapped = ((t0 - true_t0) / p_best + 0.5) % 1.0 - 0.5
    assert abs(wrapped) < 2.0 / cup.TLSSettings().n_phase_bins
    # The epoch moves with the time axis, as an epoch must.
    shifted = tls_power(t + 1000.37, flux, err, periods,
                        settings=cup.TLSSettings(), backend=backend)
    assert int(np.argmax(shifted["sde"])) == best
    assert shifted["t0"][best] - t0 == pytest.approx(1000.37, abs=1e-6)


def test_tls_peak_epoch_folds_the_transit_to_phase_zero() -> None:
    from cuperiod.gui.fold import epoch_for_peak, fold_phase

    true_t0 = 2458001.0
    t, flux, err = _inject_transit(period=3.0, t0=true_t0)
    pg = cup.periodogram(
        (t, flux, err), "TLS", domain="flux",
        settings=cup.TLSSettings(min_period_days=2.0),
    )
    peak = pg.best_periods(1, alias_diverse=True)[0]
    phase = fold_phase(np.array([true_t0]), peak.period, epoch_for_peak(peak, t))
    assert min(phase[0], 1.0 - phase[0]) < 0.01


def test_tls_no_dip_reports_the_first_timestamp_as_epoch() -> None:
    t, _, _ = _inject_transit()
    res = tls_power(t, np.full(t.size, 3.0), None, np.array([2.0, 3.0]),
                    settings=cup.TLSSettings())
    assert np.array_equal(res["t0"], np.full(2, t.min()))


# --- depth / signal residue (METH-C-3) ---------------------------------------------


def _template_matched_transit(
    width_bins: int, *, n: int = 6000, period: float = 3.0, depth: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """Noise-free transit whose width equals one of the default template widths."""
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0.0, 120.0, n)) + 2458000.0
    q = width_bins / cup.TLSSettings().n_phase_bins
    t0 = t.min() + period * 100.0 / 256.0  # mid-transit on a phase-bin edge
    phase = ((t - t0) / period) % 1.0
    x = np.where(phase > 0.5, phase - 1.0, phase)
    z = np.clip(2.0 * np.abs(x) / q, 0.0, 1.0)
    mu = np.sqrt(1.0 - z * z)
    prof = 1.0 - 0.4 * (1.0 - mu) - 0.3 * (1.0 - mu) ** 2
    flux = np.ones(n)
    inside = np.abs(x) < q / 2.0
    flux[inside] -= depth * prof[inside]
    return t, flux


@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize("width_bins", [14, 26])  # duty cycles 0.055 and 0.102
def test_tls_depth_is_not_biased_by_the_duty_cycle(
    width_bins: int, backend: str
) -> None:
    # Centring on the mean of all points (transit included) and fitting the depth
    # alone scaled it by ~(1 - duty cycle); the level is now fitted jointly.
    assert width_bins in _duration_bins(cup.TLSSettings())
    t, flux = _template_matched_transit(width_bins)
    res = tls_power(t, flux, None, np.array([3.0]), settings=cup.TLSSettings(),
                    backend=backend)
    assert res["depth"][0] == pytest.approx(0.02, rel=0.015)
    assert res["duration"][0] == pytest.approx(3.0 * width_bins / 256.0)


# --- degenerate input (METH-C-6) ---------------------------------------------------


def _nightly_times(seed: int, n: int = 370) -> np.ndarray:
    rng = np.random.default_rng(seed)
    night = rng.integers(0, 120, n).astype(float)
    return np.sort(night + rng.uniform(0.0, 0.28, n)) + 2459000.0


@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize(
    ("level", "errors"),
    [(12.345, False), (12.345, True), (1234.5, True), (1.0, True),
     (3.3e-13, False), (0.0, False)],
)
def test_tls_constant_flux_gives_an_all_zero_spectrum(
    level: float, errors: bool, backend: str
) -> None:
    # The weighted mean of a constant is an ulp off, so every window looked like a
    # 1e-15 'dip' and the SDE normalization blew that up to SDE ~10.
    periods = 1.0 / np.linspace(1.0 / 60.0, 1.0, 600)[::-1]
    for t in (_nightly_times(0), _nightly_times(1) + 0.123):
        rng = np.random.default_rng(100)
        dy = (abs(level) or 1.0) * rng.uniform(5e-4, 3e-3, t.size) if errors else None
        res = tls_power(t, np.full(t.size, level), dy, periods,
                        settings=cup.TLSSettings(min_period_days=1.0),
                        backend=backend)
        for key in ("sde", "sr", "depth", "duration"):
            assert np.count_nonzero(res[key]) == 0, key
        assert np.array_equal(res["t0"], np.full(periods.size, t.min()))


def test_tls_tiny_real_transit_is_still_searched() -> None:
    t = _nightly_times(7, n=600)
    flux = np.full(t.size, 12.345)
    flux[((t - t[0]) % 4.3) < 0.15] -= 12.345 * 1e-12
    periods = np.linspace(1.0, 20.0, 1500)
    res = tls_power(t, flux, None, periods, settings=cup.TLSSettings())
    assert np.count_nonzero(res["sde"]) > 0
    assert periods[int(np.argmax(res["sde"]))] == pytest.approx(4.3, rel=2e-3)


# --- flux units (METH-C-8) ---------------------------------------------------------


@pytest.mark.parametrize("unit", [1e-15, 1e12])
def test_tls_float64_is_invariant_to_the_flux_unit(unit: float) -> None:
    t, flux, err = _inject_transit()
    periods = np.linspace(2.0, 4.0, 300)
    ref = tls_power(t, flux, err, periods, settings=cup.TLSSettings())
    got = tls_power(t, flux * unit, err * unit, periods, settings=cup.TLSSettings())
    assert np.allclose(got["sde"], ref["sde"], rtol=1e-9, atol=1e-9)
    assert np.allclose(got["sr"], ref["sr"], rtol=1e-9, atol=0.0)
    assert np.allclose(got["depth"], ref["depth"] * unit, rtol=1e-9, atol=0.0)
    # Without errors the signal residue carries the flux unit squared.
    ref_nw = tls_power(t, flux, None, periods, settings=cup.TLSSettings())
    got_nw = tls_power(t, flux * unit, None, periods, settings=cup.TLSSettings())
    assert np.allclose(got_nw["sr"], ref_nw["sr"] * unit**2, rtol=1e-9, atol=0.0)
    assert np.allclose(got_nw["sde"], ref_nw["sde"], rtol=1e-9, atol=1e-9)


@requires_torch
@pytest.mark.parametrize("unit", [1e-15, 1e12])
@pytest.mark.parametrize("errors", [True, False])
def test_tls_torch_float32_survives_small_and_large_flux_units(
    unit: float, errors: bool
) -> None:
    # float32 weights of cgs-unit fluxes (1/dy^2 ~ 1e34) used to overflow into an
    # all-zero SDE spectrum with no error.
    t, flux, err = _inject_transit()
    periods = np.linspace(2.0, 4.0, 300)
    dy = err if errors else None
    ref = tls_power(t, flux, dy, periods, settings=cup.TLSSettings())
    got = tls_power(
        t, flux * unit, None if dy is None else dy * unit, periods,
        settings=cup.TLSSettings(precision="float32"), backend="torch:cpu",
    )
    assert int(np.argmax(got["sde"])) == int(np.argmax(ref["sde"]))
    assert got["sde"].max() == pytest.approx(ref["sde"].max(), rel=1e-3)
    best = int(np.argmax(ref["sde"]))
    assert got["depth"][best] == pytest.approx(ref["depth"][best] * unit, rel=1e-3)
    assert got["t0"].dtype == np.float64


# --- default period grid (METH-C-10, SEC-04) -----------------------------------------


def test_tls_default_grid_resolves_the_narrowest_searched_duration() -> None:
    # With the step tied only to grid_duration_frac (0.05) a transit of the
    # shortest searched duration (0.01) lost ~40% of its SR between grid points.
    settings = cup.TLSSettings()
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 90.0, 12000))
    lc = cup.LightCurve.from_arrays(t, np.ones(t.size), domain=cup.Domain.FLUX)
    freq = cup.get_method("TLS").default_grid(lc, settings).frequency
    i = int(np.searchsorted(freq, 1.0 / 4.7))
    period = 2.0 / (freq[i] + freq[i + 1])  # midway between two grid samples
    q = 0.012
    phase = ((t - 3.1) / period) % 1.0
    x = np.where(phase > 0.5, phase - 1.0, phase)
    z = np.clip(2.0 * np.abs(x) / q, 0.0, 1.0)
    mu = np.sqrt(1.0 - z * z)
    flux = np.ones(t.size)
    inside = np.abs(x) < q / 2.0
    flux[inside] -= 0.01 * (1.0 - 0.4 * (1.0 - mu) - 0.3 * (1.0 - mu) ** 2)[inside]
    on_period = tls_power(t, flux, None, np.array([period]), settings=settings)
    nearby = 1.0 / freq[i - 40 : i + 41][::-1]
    on_grid = tls_power(t, flux, None, nearby, settings=settings)
    assert on_grid["sr"].max() >= 0.85 * on_period["sr"][0]


def test_tls_grid_step_never_exceeds_the_narrowest_duration_budget() -> None:
    lc = cup.LightCurve.from_arrays(
        np.linspace(0.0, 100.0, 500), np.ones(500), domain=cup.Domain.FLUX
    )
    method = cup.get_method("TLS")
    for settings in (
        cup.TLSSettings(),
        cup.TLSSettings(grid_duration_frac=0.2),
        cup.TLSSettings(grid_duration_frac=0.004),
        cup.TLSSettings(duration_min_frac=0.04, duration_max_frac=0.2),
    ):
        freq = method.default_grid(lc, settings).frequency
        step = float(np.median(np.diff(freq)))
        budget = min(settings.grid_duration_frac, 2.0 * settings.duration_min_frac)
        assert step == pytest.approx(budget / (settings.oversample * 100.0), rel=1e-6)


def test_tls_default_grid_warns_about_an_oversized_automatic_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(0)
    t = np.concatenate([[-3000.0], np.sort(rng.uniform(0.0, 30.0, 300))])
    lc = cup.LightCurve.from_arrays(t, np.ones(t.size), domain=cup.Domain.FLUX)
    monkeypatch.setattr(grid_module, "AUTO_GRID_WARN_SIZE", 1000)
    with pytest.warns(GridSizeWarning, match="outlying"):
        cup.get_method("TLS").default_grid(lc, cup.TLSSettings())


def test_tls_default_grid_refuses_before_allocating() -> None:
    # A mis-converted timestamp 1e9 days away asks for ~2e11 trial periods: refused
    # by the size check instead of attempting a terabyte allocation.
    t = np.concatenate([np.linspace(0.0, 30.0, 300), [1e9]])
    lc = cup.LightCurve.from_arrays(t, np.ones(t.size), domain=cup.Domain.FLUX)
    with pytest.raises(GridTooLargeError, match="TLS"):
        cup.get_method("TLS").default_grid(lc, cup.TLSSettings())


def test_tls_default_grid_honours_the_global_size_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lc = cup.LightCurve.from_arrays(
        np.linspace(0.0, 300.0, 400), np.ones(400), domain=cup.Domain.FLUX
    )
    monkeypatch.setattr(grid_module, "MAX_GRID_SIZE", 10_000)
    with pytest.raises(GridTooLargeError, match="TLS"):
        cup.get_method("TLS").default_grid(lc, cup.TLSSettings())


# --- array-API memory bound (METH-C-11) ----------------------------------------------


def test_tls_numpy_batch_is_bounded_by_a_byte_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t, flux, err = _inject_transit(n=20000)
    periods = np.linspace(2.0, 4.0, 64)
    settings = cup.TLSSettings(period_batch=256)
    monkeypatch.setattr(tls_module, "_BATCH_BYTES", 1 << 40)
    unbounded = tls_power(t, flux, err, periods, settings=settings)
    monkeypatch.setattr(tls_module, "_BATCH_BYTES", 4 << 20)
    tracemalloc.start()
    try:
        bounded = tls_power(t, flux, err, periods, settings=settings)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # 64 periods x 20000 points x ~4 float64 workspaces would be ~41 MB.
    assert peak < 16 << 20
    for key in ("sde", "sr", "depth", "duration", "t0"):
        assert np.array_equal(bounded[key], unbounded[key]), key


# --- numba kernel parity on edge configurations (PERF-METHODS-04) ------------------


@requires_numba
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"n_phase_bins": 16},
        {"duration_min_frac": 0.001, "duration_max_frac": 0.002, "n_durations": 1},
        {"duration_min_frac": 0.5, "duration_max_frac": 0.99, "n_durations": 4},
        {"n_phase_bins": 512, "n_durations": 8},
    ],
)
@pytest.mark.parametrize("errors", [True, False])
def test_numba_tls_matches_numpy_on_edge_configurations(
    overrides: dict[str, float], errors: bool
) -> None:
    # A transit centred on the first timestamp straddles fold phase 0, which
    # exercises the circular wrap of the templates.
    t_first = float(_inject_transit()[0].min())
    t, flux, err = _inject_transit(t0=t_first)
    settings = cup.TLSSettings(**overrides)
    periods = np.linspace(2.0, 4.0, 150)
    dy = err if errors else None
    cpu = tls_power(t, flux, dy, periods, settings=settings, backend="numpy")
    fast = tls_power(t, flux, dy, periods, settings=settings, backend="numba")
    for key in ("sde", "sr", "depth", "duration", "t0"):
        assert np.allclose(cpu[key], fast[key], rtol=1e-10, atol=1e-12), key
